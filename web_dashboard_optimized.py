#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard Web OTIMIZADO para Sistema de Contagem de Pessoas
Versão otimizada para máxima acurácia na detecção
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

# Variáveis globais
latest_frame = None
frame_lock = threading.Lock()
stats = {
    'current_count': 0,
    'max_count': 0,
    'total_detections': 0,
    'fps': 0,
    'status': 'Inicializando...',
    'frame_count': 0,
    'rejected_count': 0,
    'model_name': '',
    'confidence_threshold': 0,
    'detection_accuracy': 0
}


class OptimizedWebCounter:
    """Contador otimizado para web com máxima acurácia"""

    def __init__(self, config_path="config_optimized.json"):
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
        self.rejected_count = 0
        self.start_time = None
        self.rejected_detections = []

        # Cores
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_YELLOW = (0, 255, 255)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_BLACK = (0, 0, 0)
        self.COLOR_ORANGE = (0, 165, 255)

    def _load_config(self, config_path):
        """Carrega configurações"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Erro ao carregar config: {e}")
            return {}

    def enhance_frame(self, frame):
        """Aplica pré-processamento para melhorar detecção"""
        if not self.config.get('preprocessing', {}).get('enable'):
            return frame

        enhanced = frame.copy()

        # CLAHE
        if self.config['preprocessing'].get('clahe'):
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=self.config['preprocessing']['clahe_clip_limit'],
                tileGridSize=(self.config['preprocessing']['clahe_tile_size'],
                            self.config['preprocessing']['clahe_tile_size'])
            )
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Sharpening
        if self.config['preprocessing'].get('sharpen'):
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)

        return enhanced

    def apply_roi(self, frame):
        """Aplica ROI"""
        if not self.config.get('roi', {}).get('enable'):
            return frame, 0

        height = frame.shape[0]
        top_margin = int(height * self.config['roi']['top_margin'])
        return frame[top_margin:, :], top_margin

    def is_valid_detection(self, box, confidence, frame_shape):
        """Valida detecção (otimizado para pessoas sentadas)"""
        if not self.config.get('filtering', {}).get('enable'):
            return True, ""

        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = width * height

        if width <= 0 or height <= 0:
            return False, "dimensões inválidas"

        # Área mínima
        if area < self.config['filtering']['min_area']:
            return False, f"área pequena ({area:.0f})"

        # Dimensões mínimas
        if width < self.config['filtering']['min_width']:
            return False, f"largura pequena"
        if height < self.config['filtering']['min_height']:
            return False, f"altura pequena"

        # Aspect ratio para pessoas sentadas
        aspect_ratio = height / width if width > 0 else 0
        if aspect_ratio < self.config['filtering']['min_aspect_ratio']:
            return False, f"muito largo (AR={aspect_ratio:.2f})"
        if aspect_ratio > self.config['filtering']['max_aspect_ratio']:
            return False, f"muito alto (AR={aspect_ratio:.2f})"

        # Área máxima (40% do frame)
        frame_h, frame_w = frame_shape
        if area > (frame_h * frame_w * 0.4):
            return False, f"área grande demais"

        return True, ""

    def initialize(self):
        """Inicializa sistema"""
        global stats

        try:
            # Conectar câmera
            stats['status'] = 'Conectando à câmera...'
            rtsp_url = self.config['camera']['rtsp_url']
            self.cap = cv2.VideoCapture(rtsp_url)

            if not self.cap.isOpened():
                raise Exception("Não foi possível conectar à câmera")

            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Carregar modelo YOLOv8s
            stats['status'] = 'Carregando modelo otimizado...'
            model_name = self.config['detection']['model']
            self.model = YOLO(model_name)

            stats['model_name'] = model_name
            stats['confidence_threshold'] = self.config['detection']['confidence_threshold']

            # Inicializar tracker
            stats['status'] = 'Inicializando tracker...'
            self.tracker = sv.ByteTrack(
                track_activation_threshold=self.config['tracking']['track_high_thresh'],
                lost_track_buffer=self.config['tracking']['track_buffer'],
                minimum_matching_threshold=self.config['tracking']['match_thresh'],
                frame_rate=self.config['tracking']['frame_rate']
            )

            stats['status'] = 'Sistema ativo (OTIMIZADO)'
            self.running = True
            self.start_time = time.time()

            print(f"✓ Sistema otimizado inicializado!")
            print(f"  Modelo: {model_name}")
            print(f"  Confidence: {self.config['detection']['confidence_threshold']}")
            print(f"  IOU: {self.config['detection']['iou_threshold']}")
            print(f"  ROI: {'Sim' if self.config['roi']['enable'] else 'Não'}")
            print(f"  Pré-processamento: {'Sim' if self.config['preprocessing']['enable'] else 'Não'}")

            return True

        except Exception as e:
            stats['status'] = f'Erro: {str(e)}'
            print(f"Erro ao inicializar: {e}")
            return False

    def process_frame(self):
        """Processa frame com otimizações"""
        global latest_frame, stats

        if not self.cap or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        # Redimensionar
        resize_w = self.config['performance']['resize_width']
        resize_h = self.config['performance']['resize_height']
        frame = cv2.resize(frame, (resize_w, resize_h))

        # Aplicar ROI
        roi_frame, roi_offset = self.apply_roi(frame)

        # Pré-processamento
        enhanced = self.enhance_frame(roi_frame)

        # Detectar com parâmetros otimizados
        results = self.model.track(
            enhanced,
            persist=True,
            classes=[0],
            conf=self.config['detection']['confidence_threshold'],
            iou=self.config['detection']['iou_threshold'],
            max_det=self.config['detection']['max_detections'],
            augment=self.config['detection']['augment'],
            verbose=False
        )

        # Converter detecções
        detections = sv.Detections.from_ultralytics(results[0])

        # Filtrar detecções
        valid_indices = []
        self.rejected_detections = []

        for i, (box, confidence) in enumerate(zip(detections.xyxy, detections.confidence)):
            is_valid, reason = self.is_valid_detection(
                box, confidence, (enhanced.shape[0], enhanced.shape[1])
            )
            if is_valid:
                valid_indices.append(i)
            else:
                self.rejected_detections.append((box, confidence, reason))

        # Criar detecções filtradas
        if len(valid_indices) > 0:
            filtered = sv.Detections(
                xyxy=detections.xyxy[valid_indices],
                confidence=detections.confidence[valid_indices],
                class_id=detections.class_id[valid_indices],
                tracker_id=detections.tracker_id[valid_indices] if detections.tracker_id is not None else None
            )
        else:
            filtered = sv.Detections.empty()

        # Aplicar tracking
        if len(filtered) > 0:
            filtered = self.tracker.update_with_detections(filtered)

        # Atualizar contagem
        self.current_count = len(filtered)
        if self.current_count > self.max_count:
            self.max_count = self.current_count

        self.total_detections += self.current_count
        self.frame_count += 1
        self.rejected_count = len(self.rejected_detections)

        # Desenhar anotações
        annotated = self._draw_detections(frame, filtered, roi_offset)
        annotated = self._draw_info_panel(annotated)

        # Atualizar estatísticas globais
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        # Calcular "acurácia" (estimativa baseada em faixa esperada)
        expected_range = self.config['room'].get('expected_count_range', [15, 25])
        if expected_range[0] <= self.current_count <= expected_range[1]:
            accuracy = 100
        elif self.current_count < expected_range[0]:
            accuracy = (self.current_count / expected_range[0]) * 100
        else:
            accuracy = max(0, 100 - ((self.current_count - expected_range[1]) / expected_range[1]) * 50)

        stats.update({
            'current_count': self.current_count,
            'max_count': self.max_count,
            'total_detections': self.total_detections,
            'fps': round(fps, 1),
            'frame_count': self.frame_count,
            'rejected_count': self.rejected_count,
            'detection_accuracy': round(accuracy, 1)
        })

        # Atualizar frame global
        with frame_lock:
            latest_frame = annotated.copy()

        return annotated

    def _draw_detections(self, frame, detections, roi_offset):
        """Desenha detecções"""
        annotated = frame.copy()

        # Detecções válidas (verde)
        for bbox, confidence, tracker_id in zip(
            detections.xyxy,
            detections.confidence,
            detections.tracker_id if detections.tracker_id is not None else [None]*len(detections)
        ):
            x1, y1, x2, y2 = map(int, bbox)
            y1 += roi_offset
            y2 += roi_offset

            cv2.rectangle(annotated, (x1, y1), (x2, y2), self.COLOR_GREEN, 2)

            label = f"ID:{tracker_id} {confidence:.2f}" if tracker_id is not None else f"{confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1-th-5), (x1+tw+5, y1), self.COLOR_GREEN, -1)
            cv2.putText(annotated, label, (x1+2, y1-2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_BLACK, 1)

        # Detecções rejeitadas (vermelho)
        for bbox, confidence, reason in self.rejected_detections:
            x1, y1, x2, y2 = map(int, bbox)
            y1 += roi_offset
            y2 += roi_offset
            cv2.rectangle(annotated, (x1, y1), (x2, y2), self.COLOR_RED, 1)
            cv2.putText(annotated, f"{confidence:.2f}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_RED, 1)

        # Linha ROI
        if roi_offset > 0:
            cv2.line(annotated, (0, roi_offset), (frame.shape[1], roi_offset),
                    self.COLOR_YELLOW, 2)
            cv2.putText(annotated, "ROI START", (10, roi_offset-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_YELLOW, 2)

        return annotated

    def _draw_info_panel(self, frame):
        """Desenha painel de informações"""
        height, width = frame.shape[:2]

        # Painel superior
        panel_h = 130
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_h), self.COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Título
        main_text = f"PESSOAS DETECTADAS: {self.current_count}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.4
        thickness = 2

        text_size = cv2.getTextSize(main_text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = 45

        # Cor baseada em faixa esperada
        expected = self.config['room'].get('expected_count_range', [15, 25])
        if self.current_count < expected[0]:
            color = self.COLOR_ORANGE
        elif self.current_count > expected[1]:
            color = self.COLOR_RED
        else:
            color = self.COLOR_GREEN

        cv2.putText(frame, main_text, (text_x+2, text_y+2),
                   font, font_scale, self.COLOR_BLACK, thickness+1)
        cv2.putText(frame, main_text, (text_x, text_y),
                   font, font_scale, color, thickness)

        # Informações
        info_y = 80
        info_texts = [
            f"Max: {self.max_count}",
            f"FPS: {stats['fps']}",
            f"Frames: {self.frame_count}",
            f"Rejeitados: {self.rejected_count}"
        ]

        start_x = 20
        for text in info_texts:
            cv2.putText(frame, text, (start_x, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
            tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            start_x += tw + 25

        # Faixa esperada
        expected_text = f"Esperado: {expected[0]}-{expected[1]} pessoas"
        cv2.putText(frame, expected_text, (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_YELLOW, 1)

        # Modelo
        model_text = f"{stats['model_name']} | Conf: {stats['confidence_threshold']} | IOU: {self.config['detection']['iou_threshold']}"
        mt_size = cv2.getTextSize(model_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.putText(frame, model_text, (width-mt_size[0]-10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_WHITE, 1)

        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (10, height-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)

        return frame

    def run(self):
        """Loop principal"""
        while self.running:
            try:
                self.process_frame()
                time.sleep(0.03)
            except Exception as e:
                print(f"Erro: {e}")
                stats['status'] = f'Erro: {str(e)}'
                time.sleep(1)

    def stop(self):
        """Para o contador"""
        self.running = False
        if self.cap:
            self.cap.release()


# Instância global
counter = None


def generate_frames():
    """Gerador de frames para streaming"""
    global latest_frame

    while True:
        with frame_lock:
            if latest_frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Aguardando frames...", (50, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
            else:
                ret, buffer = cv2.imencode('.jpg', latest_frame,
                                         [cv2.IMWRITE_JPEG_QUALITY, 85])

        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        time.sleep(0.033)


@app.route('/')
def index():
    """Página principal"""
    return render_template('dashboard_optimized.html')


@app.route('/video_feed')
def video_feed():
    """Streaming de vídeo"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def get_stats():
    """Estatísticas em JSON"""
    return jsonify(stats)


def run_counter():
    """Thread do contador"""
    global counter
    counter = OptimizedWebCounter()

    if counter.initialize():
        counter.run()


if __name__ == '__main__':
    print("\n" + "="*60)
    print(" DASHBOARD WEB OTIMIZADO - CONTAGEM DE PESSOAS")
    print("="*60)
    print("\n🎯 Sistema otimizado para máxima acurácia")
    print("📊 Modelo: YOLOv8s")
    print("⚙️  Confidence: 0.25 | IOU: 0.3")
    print("🔍 ROI + Pré-processamento habilitados")

    # Iniciar contador
    counter_thread = threading.Thread(target=run_counter, daemon=True)
    counter_thread.start()

    print("\n✓ Servidor iniciado!")
    print("\n📊 Acesse o dashboard:")
    print("   http://localhost:5000")
    print("   http://10.50.30.173:5000")
    print("\n⚠️  Pressione Ctrl+C para parar\n")
    print("="*60 + "\n")

    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\n✓ Encerrando...")
        if counter:
            counter.stop()
