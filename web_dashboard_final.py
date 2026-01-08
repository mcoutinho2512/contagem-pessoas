#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard Web FINAL - Sistema de Contagem de Pessoas
✨ COM ROI OTIMIZADO + TROCA DE URL DA CÂMERA
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import json
import threading
import time
from datetime import datetime
import numpy as np
from ultralytics import YOLO
import supervision as sv
from pathlib import Path

app = Flask(__name__)

# Variáveis globais
latest_frame = None
frame_lock = threading.Lock()
counter = None
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
    'detection_accuracy': 0,
    'camera_url': '',
    'camera_connected': False,
    'roi_height': 0
}


class FinalOptimizedCounter:
    """Contador FINAL com ROI + Troca de URL"""

    def __init__(self, config_path="config_optimized.json"):
        self.config = self._load_config(config_path)
        self.config_path = config_path
        self.cap = None
        self.model = None
        self.tracker = None
        self.running = False

        # URL da câmera (pode ser alterada em tempo real)
        self.rtsp_url = self.config['camera']['rtsp_url']

        # Estatísticas
        self.current_count = 0
        self.max_count = 0
        self.total_detections = 0
        self.frame_count = 0
        self.rejected_count = 0
        self.start_time = None
        self.rejected_detections = []

        # ROI
        self.roi_offset = 0
        self.roi_auto_detected = False

        # Cores
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_YELLOW = (0, 255, 255)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_BLACK = (0, 0, 0)
        self.COLOR_ORANGE = (0, 165, 255)
        self.COLOR_CYAN = (255, 255, 0)

    def _load_config(self, config_path):
        """Carrega configurações"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Erro ao carregar config: {e}")
            return {}

    def save_camera_url(self, url):
        """Salva URL da câmera no config"""
        try:
            self.config['camera']['rtsp_url'] = url
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"✓ URL salva no config: {url}")
        except Exception as e:
            print(f"Erro ao salvar URL: {e}")

    def detect_ui_bar(self, frame):
        """
        Detecta automaticamente a altura da barra UI superior
        Retorna o pixel Y onde a barra termina
        """
        height, width = frame.shape[:2]

        # Analisar primeiros 25% da imagem
        top_section = frame[0:int(height*0.25), :]

        # Converter para grayscale
        gray = cv2.cvtColor(top_section, cv2.COLOR_BGR2GRAY)

        # Procurar transição de escuro (barra) para claro (vídeo)
        threshold = 80  # Valor de cinza que separa barra escura de vídeo

        for y in range(10, gray.shape[0]):  # Começar em 10 para pular ruído
            row_mean = np.mean(gray[y, :])

            # Se média da linha for clara, achamos o fim da barra
            if row_mean > threshold:
                print(f"🎯 Barra UI detectada: altura {y} pixels")
                return y

        print("⚠️  Barra UI não detectada, usando configuração manual")
        return 0

    def apply_roi(self, frame):
        """
        Aplica ROI para EXCLUIR completamente a barra superior
        Retorna: (frame_processado, offset_y)
        """
        if not self.config.get('roi', {}).get('enable', True):
            return frame, 0

        height, width = frame.shape[:2]

        # Auto-detectar barra na primeira execução
        if not self.roi_auto_detected and self.config['roi'].get('auto_detect', True):
            detected_height = self.detect_ui_bar(frame)
            if detected_height > 0:
                # Adicionar margem de segurança
                safety_margin = self.config['roi'].get('safety_margin', 10)
                self.roi_offset = detected_height + safety_margin
                self.roi_auto_detected = True
            else:
                # Fallback para configuração manual
                self.roi_offset = int(height * self.config['roi']['top_margin'])
                self.roi_auto_detected = True
        elif not self.roi_auto_detected:
            # Usar configuração manual
            self.roi_offset = int(height * self.config['roi']['top_margin'])
            self.roi_auto_detected = True

        # Garantir que offset é válido
        if self.roi_offset >= height - 50:
            self.roi_offset = int(height * 0.15)  # Fallback seguro

        # Retornar apenas a região ABAIXO da barra
        frame_roi = frame[self.roi_offset:, :]

        return frame_roi, self.roi_offset

    def enhance_frame(self, frame):
        """Pré-processamento de imagem"""
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

    def is_valid_detection(self, box, confidence, frame_shape):
        """Valida detecção"""
        if not self.config.get('filtering', {}).get('enable'):
            return True, ""

        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = width * height

        if width <= 0 or height <= 0:
            return False, "dimensões inválidas"

        if area < self.config['filtering']['min_area']:
            return False, f"área pequena"

        if width < self.config['filtering']['min_width']:
            return False, f"largura pequena"
        if height < self.config['filtering']['min_height']:
            return False, f"altura pequena"

        aspect_ratio = height / width if width > 0 else 0
        if aspect_ratio < self.config['filtering']['min_aspect_ratio']:
            return False, f"muito largo"
        if aspect_ratio > self.config['filtering']['max_aspect_ratio']:
            return False, f"muito alto"

        frame_h, frame_w = frame_shape
        if area > (frame_h * frame_w * 0.4):
            return False, f"área grande demais"

        return True, ""

    def connect_camera(self, url=None):
        """Conecta à câmera"""
        if url:
            self.rtsp_url = url
            self.save_camera_url(url)

        print(f"🎥 Conectando à câmera: {self.rtsp_url}")

        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.rtsp_url)

        if not self.cap.isOpened():
            raise Exception("Falha ao conectar à câmera")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Reset ROI detection
        self.roi_auto_detected = False

        print(f"✓ Câmera conectada com sucesso!")
        return True

    def disconnect_camera(self):
        """Desconecta câmera"""
        if self.cap:
            self.cap.release()
            self.cap = None
        print("📴 Câmera desconectada")

    def initialize(self):
        """Inicializa sistema"""
        global stats

        try:
            # Conectar câmera
            stats['status'] = 'Conectando à câmera...'
            self.connect_camera()

            # Carregar modelo
            stats['status'] = 'Carregando modelo YOLOv8s...'
            model_name = self.config['detection']['model']
            self.model = YOLO(model_name)

            stats['model_name'] = model_name
            stats['confidence_threshold'] = self.config['detection']['confidence_threshold']
            stats['camera_url'] = self.rtsp_url

            # Inicializar tracker
            stats['status'] = 'Inicializando tracker...'
            self.tracker = sv.ByteTrack(
                track_activation_threshold=self.config['tracking']['track_high_thresh'],
                lost_track_buffer=self.config['tracking']['track_buffer'],
                minimum_matching_threshold=self.config['tracking']['match_thresh'],
                frame_rate=self.config['tracking']['frame_rate']
            )

            stats['status'] = 'Sistema ativo (FINAL)'
            stats['camera_connected'] = True
            self.running = True
            self.start_time = time.time()

            print("✓ Sistema FINAL inicializado com sucesso!")
            return True

        except Exception as e:
            stats['status'] = f'Erro: {str(e)}'
            stats['camera_connected'] = False
            print(f"❌ Erro ao inicializar: {e}")
            return False

    def process_frame(self):
        """Processa frame com ROI otimizado"""
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

        # ⚡ APLICAR ROI - EXCLUIR BARRA ANTES DO YOLO
        roi_frame, roi_offset = self.apply_roi(frame)

        # Pré-processamento
        enhanced = self.enhance_frame(roi_frame)

        # Detectar APENAS na área do ROI (sem barra)
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

        # Filtrar
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

        # Tracking
        if len(filtered) > 0:
            filtered = self.tracker.update_with_detections(filtered)

        # Atualizar contagem
        self.current_count = len(filtered)
        if self.current_count > self.max_count:
            self.max_count = self.current_count

        self.total_detections += self.current_count
        self.frame_count += 1
        self.rejected_count = len(self.rejected_detections)

        # Desenhar no frame ORIGINAL (não no ROI)
        annotated = self._draw_detections(frame, filtered, roi_offset)
        annotated = self._draw_roi_indicator(annotated, roi_offset)
        annotated = self._draw_info_panel(annotated, roi_offset)

        # Atualizar stats
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        expected = self.config['room'].get('expected_count_range', [15, 25])
        if expected[0] <= self.current_count <= expected[1]:
            accuracy = 100
        elif self.current_count < expected[0]:
            accuracy = (self.current_count / expected[0]) * 100
        else:
            accuracy = max(0, 100 - ((self.current_count - expected[1]) / expected[1]) * 50)

        stats.update({
            'current_count': self.current_count,
            'max_count': self.max_count,
            'total_detections': self.total_detections,
            'fps': round(fps, 1),
            'frame_count': self.frame_count,
            'rejected_count': self.rejected_count,
            'detection_accuracy': round(accuracy, 1),
            'roi_height': roi_offset,
            'camera_connected': True
        })

        with frame_lock:
            latest_frame = annotated.copy()

        return annotated

    def _draw_detections(self, frame, detections, roi_offset):
        """Desenha detecções com offset do ROI"""
        annotated = frame.copy()

        # Detecções válidas (verde)
        for bbox, confidence, tracker_id in zip(
            detections.xyxy,
            detections.confidence,
            detections.tracker_id if detections.tracker_id is not None else [None]*len(detections)
        ):
            x1, y1, x2, y2 = map(int, bbox)

            # AJUSTAR coordenadas para frame original (adicionar offset do ROI)
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

        return annotated

    def _draw_roi_indicator(self, frame, roi_offset):
        """Desenha linha mostrando onde começa o ROI"""
        if roi_offset > 0:
            # Linha amarela/ciano destacada
            cv2.line(frame, (0, roi_offset), (frame.shape[1], roi_offset),
                    self.COLOR_CYAN, 3)

            # Texto indicativo
            cv2.putText(frame, f"ROI START (Barra ignorada: {roi_offset}px)",
                       (10, roi_offset - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_CYAN, 2)

            # Área da barra com overlay semi-transparente
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], roi_offset),
                         self.COLOR_CYAN, -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        return frame

    def _draw_info_panel(self, frame, roi_offset):
        """Desenha painel de informações"""
        height, width = frame.shape[:2]

        # Painel inferior (para não interferir com ROI)
        panel_h = 80
        panel_y = height - panel_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, panel_y), (width, height), self.COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Informações
        y_pos = panel_y + 25
        texts = [
            f"PESSOAS: {self.current_count}",
            f"MAX: {self.max_count}",
            f"FPS: {stats['fps']}",
            f"ROI: {roi_offset}px",
            f"Rejeitados: {self.rejected_count}"
        ]

        x_pos = 20
        for text in texts:
            cv2.putText(frame, text, (x_pos, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
            x_pos += 180

        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (width - 100, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)

        # Modelo
        model_text = f"{stats['model_name']} | Conf: {stats['confidence_threshold']}"
        cv2.putText(frame, model_text, (20, panel_y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_YELLOW, 1)

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
        self.disconnect_camera()


# === FLASK APIs ===

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
    return render_template('dashboard_final.html')


@app.route('/video_feed')
def video_feed():
    """Streaming de vídeo"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/stats')
def get_stats():
    """Retorna estatísticas"""
    return jsonify(stats)


@app.route('/api/camera/url', methods=['GET', 'POST'])
def camera_url():
    """
    GET: Retorna URL atual
    POST: Troca URL da câmera
    """
    global counter, stats

    if request.method == 'GET':
        return jsonify({
            'url': counter.rtsp_url if counter else '',
            'connected': stats['camera_connected']
        })

    elif request.method == 'POST':
        try:
            data = request.get_json()
            new_url = data.get('url', '').strip()

            if not new_url:
                return jsonify({
                    'success': False,
                    'message': 'URL vazia'
                }), 400

            if not new_url.startswith('rtsp://'):
                return jsonify({
                    'success': False,
                    'message': 'URL deve começar com rtsp://'
                }), 400

            # Reconectar com nova URL
            print(f"\n🔄 Trocando URL da câmera...")
            print(f"   De: {counter.rtsp_url}")
            print(f"   Para: {new_url}")

            counter.disconnect_camera()
            time.sleep(0.5)
            counter.connect_camera(new_url)

            stats['camera_url'] = new_url
            stats['camera_connected'] = True
            stats['status'] = 'Câmera reconectada'

            return jsonify({
                'success': True,
                'message': 'Câmera conectada com sucesso!',
                'url': new_url
            })

        except Exception as e:
            stats['camera_connected'] = False
            stats['status'] = f'Erro: {str(e)}'
            return jsonify({
                'success': False,
                'message': f'Erro ao conectar: {str(e)}'
            }), 500


@app.route('/api/camera/reconnect', methods=['POST'])
def reconnect_camera():
    """Reconecta à câmera atual"""
    global counter, stats

    try:
        counter.disconnect_camera()
        time.sleep(0.5)
        counter.connect_camera()

        stats['camera_connected'] = True
        stats['status'] = 'Câmera reconectada'

        return jsonify({
            'success': True,
            'message': 'Câmera reconectada'
        })

    except Exception as e:
        stats['camera_connected'] = False
        stats['status'] = f'Erro: {str(e)}'
        return jsonify({
            'success': False,
            'message': f'Erro: {str(e)}'
        }), 500


def run_counter():
    """Thread do contador"""
    global counter
    counter = FinalOptimizedCounter()

    if counter.initialize():
        counter.run()


if __name__ == '__main__':
    print("\n" + "="*70)
    print(" 🚀 DASHBOARD WEB FINAL - CONTAGEM DE PESSOAS")
    print("="*70)
    print("\n✨ Funcionalidades:")
    print("   ✅ ROI Otimizado - Ignora barra superior automaticamente")
    print("   ✅ Troca de URL da Câmera - Interface visual")
    print("   ✅ Modelo YOLOv8s - Máxima acurácia")
    print("   ✅ Pré-processamento - CLAHE + Sharpening")

    # Iniciar contador
    counter_thread = threading.Thread(target=run_counter, daemon=True)
    counter_thread.start()

    time.sleep(2)  # Aguardar inicialização

    print("\n✓ Servidor iniciado!")
    print("\n📊 Acesse o dashboard:")
    print("   http://localhost:5000")
    print("   http://10.50.30.173:5000")
    print("\n💡 No dashboard você pode:")
    print("   • Ver vídeo ao vivo com ROI destacado")
    print("   • Trocar URL da câmera em tempo real")
    print("   • Ver estatísticas e acurácia")
    print("\n⚠️  Pressione Ctrl+C para parar\n")
    print("="*70 + "\n")

    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\n✓ Encerrando...")
        if counter:
            counter.stop()
