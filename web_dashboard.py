#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard Web para Sistema de Contagem de Pessoas
Permite visualização em tempo real via navegador
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import json
import threading
import time
from datetime import datetime
import numpy as np
from ultralytics import YOLO
import supervision as sv

app = Flask(__name__)

# Variáveis globais para compartilhar entre threads
latest_frame = None
frame_lock = threading.Lock()
stats = {
    'current_count': 0,
    'max_count': 0,
    'total_detections': 0,
    'fps': 0,
    'status': 'Inicializando...'
}


class WebPeopleCounter:
    """Versão do contador de pessoas otimizada para web"""

    def __init__(self, config_path="config.json"):
        self.config = self._load_config(config_path)
        self.cap = None
        self.model = None
        self.tracker = None
        self.running = False

        # Estatísticas
        self.current_count = 0
        self.max_count = 0
        self.total_detections = 0
        self.frame_count = 0
        self.start_time = None

        # Cores
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_BLACK = (0, 0, 0)
        self.COLOR_YELLOW = (0, 255, 255)

    def _load_config(self, config_path):
        """Carrega configurações do arquivo JSON"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Erro ao carregar config: {e}")
            return {}

    def initialize(self):
        """Inicializa câmera, modelo e tracker"""
        global stats

        try:
            # Conectar à câmera
            stats['status'] = 'Conectando à câmera...'
            rtsp_url = self.config['camera']['rtsp_url']
            self.cap = cv2.VideoCapture(rtsp_url)

            if not self.cap.isOpened():
                raise Exception("Não foi possível conectar à câmera")

            # Configurações da câmera
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Carregar modelo YOLOv8
            stats['status'] = 'Carregando modelo YOLOv8...'
            model_name = self.config['detection']['model']
            self.model = YOLO(model_name)

            # Inicializar tracker
            stats['status'] = 'Inicializando tracker...'
            self.tracker = sv.ByteTrack()

            stats['status'] = 'Sistema ativo'
            self.running = True
            self.start_time = time.time()

            print("✓ Sistema inicializado com sucesso!")
            return True

        except Exception as e:
            stats['status'] = f'Erro: {str(e)}'
            print(f"Erro ao inicializar: {e}")
            return False

    def process_frame(self):
        """Processa um frame da câmera"""
        global latest_frame, stats

        if not self.cap or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        # Redimensionar frame para melhor performance
        resize_width = self.config['performance']['resize_width']
        resize_height = self.config['performance']['resize_height']
        frame = cv2.resize(frame, (resize_width, resize_height))

        # Detectar pessoas
        confidence = self.config['detection']['confidence_threshold']
        results = self.model(frame, classes=[0], conf=confidence, verbose=False)

        # Converter para formato do supervision
        detections = sv.Detections.from_ultralytics(results[0])

        # Aplicar tracking
        if self.config['tracking']['enabled']:
            detections = self.tracker.update_with_detections(detections)

        # Contar pessoas
        self.current_count = len(detections)
        if self.current_count > self.max_count:
            self.max_count = self.current_count

        self.total_detections += self.current_count
        self.frame_count += 1

        # Desenhar anotações
        annotated_frame = self._draw_detections(frame, detections)
        annotated_frame = self._draw_info_panel(annotated_frame)

        # Atualizar estatísticas globais
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0

        stats.update({
            'current_count': self.current_count,
            'max_count': self.max_count,
            'total_detections': self.total_detections,
            'fps': round(fps, 1),
            'frame_count': self.frame_count
        })

        # Atualizar frame global
        with frame_lock:
            latest_frame = annotated_frame.copy()

        return annotated_frame

    def _draw_detections(self, frame, detections):
        """Desenha caixas delimitadoras e informações das detecções"""
        for i, (bbox, confidence, class_id, tracker_id) in enumerate(
            zip(detections.xyxy, detections.confidence, detections.class_id,
                detections.tracker_id if detections.tracker_id is not None else [None]*len(detections))
        ):
            x1, y1, x2, y2 = map(int, bbox)

            # Desenhar caixa
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_GREEN, 2)

            # Preparar texto
            label_parts = []
            if tracker_id is not None:
                label_parts.append(f"ID:{tracker_id}")
            label_parts.append(f"{confidence:.2f}")
            label = " ".join(label_parts)

            # Desenhar fundo do texto
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                self.COLOR_GREEN,
                -1
            )

            # Desenhar texto
            cv2.putText(
                frame, label, (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_BLACK, 1
            )

        return frame

    def _draw_info_panel(self, frame):
        """Desenha painel de informações no frame"""
        height, width = frame.shape[:2]

        # Painel superior
        panel_height = 100
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), self.COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Título principal
        main_text = f"PESSOAS DETECTADAS: {self.current_count}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.2
        thickness = 2

        text_size = cv2.getTextSize(main_text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = 40

        # Cor baseada em contagem
        max_capacity = self.config.get('room', {}).get('max_capacity', 50)
        alert_threshold = self.config.get('room', {}).get('alert_threshold', 0.8)

        if self.current_count >= max_capacity * alert_threshold:
            text_color = self.COLOR_RED
        else:
            text_color = self.COLOR_GREEN

        # Desenhar texto com sombra
        cv2.putText(frame, main_text, (text_x + 2, text_y + 2),
                   font, font_scale, self.COLOR_BLACK, thickness + 1)
        cv2.putText(frame, main_text, (text_x, text_y),
                   font, font_scale, text_color, thickness)

        # Informações adicionais
        info_y = 70
        info_texts = [
            f"Maximo: {self.max_count}",
            f"FPS: {stats['fps']}",
            f"Frames: {self.frame_count}"
        ]

        total_width = sum(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for t in info_texts) + 40
        start_x = (width - total_width) // 2

        for text in info_texts:
            cv2.putText(frame, text, (start_x, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
            text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            start_x += text_width + 20

        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_WHITE, 1)

        return frame

    def run(self):
        """Loop principal de processamento"""
        while self.running:
            try:
                self.process_frame()
                time.sleep(0.03)  # ~30 FPS
            except Exception as e:
                print(f"Erro no processamento: {e}")
                stats['status'] = f'Erro: {str(e)}'
                time.sleep(1)

    def stop(self):
        """Para o contador"""
        self.running = False
        if self.cap:
            self.cap.release()


# Instância global do contador
counter = None


def generate_frames():
    """Gerador de frames para streaming MJPEG"""
    global latest_frame

    while True:
        with frame_lock:
            if latest_frame is None:
                # Frame de placeholder
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Aguardando frames...", (50, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
            else:
                ret, buffer = cv2.imencode('.jpg', latest_frame,
                                         [cv2.IMWRITE_JPEG_QUALITY, 85])

        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.033)  # ~30 FPS


@app.route('/')
def index():
    """Página principal do dashboard"""
    return render_template('dashboard.html')


@app.route('/video_feed')
def video_feed():
    """Endpoint para streaming de vídeo"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def get_stats():
    """Endpoint para obter estatísticas em JSON"""
    return jsonify(stats)


def run_counter():
    """Thread para executar o contador"""
    global counter
    counter = WebPeopleCounter()

    if counter.initialize():
        counter.run()


if __name__ == '__main__':
    print("\n" + "="*60)
    print(" DASHBOARD WEB - SISTEMA DE CONTAGEM DE PESSOAS")
    print("="*60)

    # Iniciar thread do contador
    counter_thread = threading.Thread(target=run_counter, daemon=True)
    counter_thread.start()

    print("\n✓ Servidor iniciado!")
    print("\n📊 Acesse o dashboard em:")
    print("   http://localhost:5000")
    print("   http://SEU_IP:5000")
    print("\n⚠️  Pressione Ctrl+C para parar\n")
    print("="*60 + "\n")

    # Iniciar servidor Flask
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\n✓ Encerrando servidor...")
        if counter:
            counter.stop()
