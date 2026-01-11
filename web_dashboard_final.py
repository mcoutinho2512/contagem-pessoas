#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard Web FINAL - Sistema de Contagem de Pessoas
✨ COM ROI OTIMIZADO + TROCA DE URL DA CÂMERA
"""

from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import cv2
import json
import threading
import time
from datetime import datetime, timedelta
import numpy as np
from ultralytics import YOLO
import supervision as sv
from pathlib import Path
import os
import uuid

# Importar módulos avançados de visão computacional
from modules.line_counter import LineCounter
from modules.zone_detector import ZoneDetector
from modules.abandoned_objects import AbandonedObjectDetector
from modules.report_generator import ReportGenerator
from modules.multi_camera import MultiCameraManager
from modules.visualization import (
    draw_virtual_lines,
    draw_detection_zones,
    draw_abandoned_objects
)

app = Flask(__name__)

# Caminho do arquivo de câmeras
CAMERAS_FILE = 'cameras.json'

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
    'roi_height': 0,
    'class_counts': {
        'person': 0,
        'bicycle': 0,
        'car': 0,
        'motorcycle': 0,
        'bus': 0,
        'truck': 0
    },
    'line_counts': []  # Contadores das linhas virtuais em tempo real
}

# Módulos avançados (inicializados ao startar)
line_counter = None
zone_detector = None
abandoned_detector = None
report_generator = None
multi_camera_manager = None


class FinalOptimizedCounter:
    """Contador FINAL com ROI + Troca de URL"""

    def __init__(self, config_path="config_optimized.json"):
        self.config = self._load_config(config_path)
        self.config_path = config_path
        self.cap = None
        self.model = None
        self.tracker = None
        self.running = False
        self.paused = False  # Flag para pausar processamento durante troca de câmera

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

        # Contagem por categoria (detecções atuais)
        self.class_counts = {
            'person': 0,
            'bicycle': 0,
            'car': 0,
            'motorcycle': 0,
            'bus': 0,
            'truck': 0
        }
        # Mapeamento de class_id COCO para nome
        self.CLASS_MAP = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }

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

    def is_valid_detection(self, box, confidence, frame_shape, class_id=None):
        """
        Valida detecção com thresholds específicos por classe
        ⚡ MELHORIA: Thresholds ajustados para veículos
        """
        if not self.config.get('filtering', {}).get('enable'):
            return True, ""

        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = width * height

        if width <= 0 or height <= 0:
            return False, "dimensões inválidas"

        # Obter nome da classe
        class_name = self.CLASS_MAP.get(int(class_id), 'unknown') if class_id is not None else 'unknown'

        # Usar threshold de área específico por classe
        class_min_areas = self.config.get('detection', {}).get('class_min_areas', {})
        min_area = class_min_areas.get(class_name, self.config['filtering']['min_area'])

        if area < min_area:
            return False, f"área pequena ({class_name})"

        # Validação de confidence específica por classe
        class_conf_thresholds = self.config.get('detection', {}).get('class_confidence_thresholds', {})
        min_conf = class_conf_thresholds.get(class_name, self.config['detection']['confidence_threshold'])

        if confidence < min_conf:
            return False, f"conf baixa ({class_name}: {confidence:.2f} < {min_conf})"

        # Validações dimensionais (diferentes para pessoas vs veículos)
        if class_name == 'person':
            if width < self.config['filtering']['min_width']:
                return False, f"largura pequena"
            if height < self.config['filtering']['min_height']:
                return False, f"altura pequena"

            aspect_ratio = height / width if width > 0 else 0
            if aspect_ratio < self.config['filtering']['min_aspect_ratio']:
                return False, f"muito largo"
            if aspect_ratio > self.config['filtering']['max_aspect_ratio']:
                return False, f"muito alto"
        else:
            # Veículos: mais tolerantes com aspect ratio
            aspect_ratio = width / height if height > 0 else 0
            if aspect_ratio > 6.0:  # Muito largo para veículo
                return False, f"veículo muito largo"
            if aspect_ratio < 0.3:  # Muito alto para veículo
                return False, f"veículo muito alto"

        frame_h, frame_w = frame_shape
        if area > (frame_h * frame_w * 0.5):  # Aumentado para 50% para veículos grandes
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
            classes=[0, 1, 2, 3, 5, 7],  # pessoa, bicicleta, carro, moto, ônibus, caminhão
            conf=self.config['detection']['confidence_threshold'],
            iou=self.config['detection']['iou_threshold'],
            max_det=self.config['detection']['max_detections'],
            augment=self.config['detection']['augment'],
            verbose=False
        )

        # Converter detecções
        detections = sv.Detections.from_ultralytics(results[0])

        # Filtrar com thresholds específicos por classe
        valid_indices = []
        self.rejected_detections = []

        for i, (box, confidence, class_id) in enumerate(
            zip(detections.xyxy, detections.confidence, detections.class_id)
        ):
            is_valid, reason = self.is_valid_detection(
                box, confidence, (enhanced.shape[0], enhanced.shape[1]), class_id
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

        # Atualizar contagem por categoria
        self.class_counts = {k: 0 for k in self.class_counts}
        if len(filtered) > 0 and filtered.class_id is not None:
            for class_id in filtered.class_id:
                class_name = self.CLASS_MAP.get(int(class_id), 'unknown')
                if class_name in self.class_counts:
                    self.class_counts[class_name] += 1

        # ===== PROCESSAR MÓDULOS AVANÇADOS =====
        global line_counter, zone_detector, abandoned_detector, report_generator

        # Preparar lista de objetos rastreados para os módulos
        tracked_objects = []
        if filtered.tracker_id is not None:
            for i, (bbox, class_id, track_id, confidence) in enumerate(
                zip(filtered.xyxy, filtered.class_id, filtered.tracker_id, filtered.confidence)
            ):
                # Ajustar bbox com offset do ROI
                adjusted_bbox = [
                    bbox[0],
                    bbox[1] + roi_offset,
                    bbox[2],
                    bbox[3] + roi_offset
                ]
                tracked_objects.append({
                    'track_id': int(track_id) if track_id is not None else i,
                    'bbox': adjusted_bbox,
                    'class_id': int(class_id),
                    'confidence': float(confidence)
                })

        # Obter ID da câmera ativa para filtrar linhas/zonas
        active_camera_id = None
        try:
            cameras_data = load_cameras()
            active_camera_id = cameras_data.get('active_camera_id')
        except:
            pass

        # Processar linhas virtuais (apenas da câmera ativa)
        if line_counter is not None and len(tracked_objects) > 0:
            line_counter.update(tracked_objects, active_camera_id)

        # Processar zonas de detecção (apenas da câmera ativa)
        if zone_detector is not None and len(tracked_objects) > 0:
            zone_detector.update(tracked_objects, active_camera_id)

        # Processar objetos abandonados (detectar objetos não-pessoas)
        abandoned_objects = []
        if abandoned_detector is not None:
            # Passar todas as detecções do YOLO (incluindo não-pessoas)
            all_detections = []
            for i, (bbox, confidence, class_id) in enumerate(
                zip(detections.xyxy, detections.confidence, detections.class_id)
            ):
                adjusted_bbox = [
                    bbox[0],
                    bbox[1] + roi_offset,
                    bbox[2],
                    bbox[3] + roi_offset
                ]
                all_detections.append({
                    'bbox': adjusted_bbox,
                    'class_id': int(class_id),
                    'confidence': float(confidence)
                })

            abandoned_objects = abandoned_detector.update(all_detections)

        # Desenhar no frame ORIGINAL (não no ROI)
        annotated = self._draw_detections(frame, filtered, roi_offset)
        annotated = self._draw_roi_indicator(annotated, roi_offset)
        annotated = self._draw_info_panel(annotated, roi_offset)

        # Desenhar funcionalidades avançadas (apenas da câmera ativa e se visível)
        if analytics_visibility.get('lines', True):
            if line_counter is not None and len(line_counter.lines) > 0:
                # Filtrar linhas da câmera ativa
                camera_lines = {
                    lid: line for lid, line in line_counter.lines.items()
                    if line.camera_id == active_camera_id or line.camera_id is None
                }
                if camera_lines:
                    annotated = draw_virtual_lines(annotated, camera_lines, line_counter)

        if analytics_visibility.get('zones', True):
            if zone_detector is not None and len(zone_detector.zones) > 0:
                # Filtrar zonas da câmera ativa
                camera_zones = {
                    zid: zone for zid, zone in zone_detector.zones.items()
                    if zone.camera_id == active_camera_id or zone.camera_id is None
                }
                if camera_zones:
                    annotated = draw_detection_zones(annotated, camera_zones, zone_detector)

        if analytics_visibility.get('abandoned', True):
            if abandoned_objects:
                annotated = draw_abandoned_objects(annotated, abandoned_objects)

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

        # Obter contadores das linhas para atualização em tempo real
        line_counts_data = []
        if line_counter is not None:
            line_counts_data = line_counter.get_lines_list(active_camera_id)

        stats.update({
            'current_count': self.current_count,
            'max_count': self.max_count,
            'total_detections': self.total_detections,
            'fps': round(fps, 1),
            'frame_count': self.frame_count,
            'rejected_count': self.rejected_count,
            'detection_accuracy': round(accuracy, 1),
            'roi_height': roi_offset,
            'camera_connected': True,
            'class_counts': self.class_counts,
            'line_counts': line_counts_data  # Atualização em tempo real das linhas
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
                # Se estiver pausado (troca de câmera), aguardar
                if self.paused:
                    time.sleep(0.1)
                    continue

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


# === GERENCIAMENTO DE MÚLTIPLAS CÂMERAS ===

def load_cameras():
    """Carrega lista de câmeras do arquivo JSON"""
    try:
        if os.path.exists(CAMERAS_FILE):
            with open(CAMERAS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Criar arquivo padrão
            default = {
                "cameras": [],
                "active_camera_id": None
            }
            save_cameras(default)
            return default
    except Exception as e:
        print(f"Erro ao carregar câmeras: {e}")
        return {"cameras": [], "active_camera_id": None}


def save_cameras(data):
    """Salva lista de câmeras no arquivo JSON"""
    try:
        with open(CAMERAS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Erro ao salvar câmeras: {e}")
        return False


@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """Lista todas as câmeras cadastradas"""
    cameras_data = load_cameras()
    return jsonify(cameras_data)


@app.route('/api/cameras', methods=['POST'])
def add_camera():
    """Adiciona nova câmera"""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        url = data.get('url', '').strip()
        description = data.get('description', '').strip()

        if not name or not url:
            return jsonify({
                'success': False,
                'message': 'Nome e URL são obrigatórios'
            }), 400

        if not url.startswith('rtsp://'):
            return jsonify({
                'success': False,
                'message': 'URL deve começar com rtsp://'
            }), 400

        cameras_data = load_cameras()

        # Gerar novo ID
        max_id = max([cam['id'] for cam in cameras_data['cameras']], default=0)
        new_id = max_id + 1

        # Adicionar nova câmera
        new_camera = {
            'id': new_id,
            'name': name,
            'url': url,
            'description': description
        }

        cameras_data['cameras'].append(new_camera)

        # Se for a primeira câmera, torná-la ativa
        if len(cameras_data['cameras']) == 1:
            cameras_data['active_camera_id'] = new_id

        save_cameras(cameras_data)

        return jsonify({
            'success': True,
            'message': 'Câmera adicionada com sucesso!',
            'camera': new_camera
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Erro ao adicionar câmera: {str(e)}'
        }), 500


@app.route('/api/cameras/<int:camera_id>', methods=['PUT'])
def update_camera(camera_id):
    """Atualiza câmera existente"""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        url = data.get('url', '').strip()
        description = data.get('description', '').strip()

        if not name or not url:
            return jsonify({
                'success': False,
                'message': 'Nome e URL são obrigatórios'
            }), 400

        if not url.startswith('rtsp://'):
            return jsonify({
                'success': False,
                'message': 'URL deve começar com rtsp://'
            }), 400

        cameras_data = load_cameras()

        # Encontrar e atualizar câmera
        camera_found = False
        for camera in cameras_data['cameras']:
            if camera['id'] == camera_id:
                camera['name'] = name
                camera['url'] = url
                camera['description'] = description
                camera_found = True
                break

        if not camera_found:
            return jsonify({
                'success': False,
                'message': 'Câmera não encontrada'
            }), 404

        save_cameras(cameras_data)

        return jsonify({
            'success': True,
            'message': 'Câmera atualizada com sucesso!',
            'camera': camera
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Erro ao atualizar câmera: {str(e)}'
        }), 500


@app.route('/api/cameras/<int:camera_id>', methods=['DELETE'])
def delete_camera(camera_id):
    """Remove câmera"""
    try:
        cameras_data = load_cameras()

        # Encontrar e remover câmera
        cameras_data['cameras'] = [cam for cam in cameras_data['cameras'] if cam['id'] != camera_id]

        # Se removeu a ativa, resetar
        if cameras_data['active_camera_id'] == camera_id:
            cameras_data['active_camera_id'] = None
            if cameras_data['cameras']:
                cameras_data['active_camera_id'] = cameras_data['cameras'][0]['id']

        save_cameras(cameras_data)

        return jsonify({
            'success': True,
            'message': 'Câmera removida com sucesso!'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Erro ao remover câmera: {str(e)}'
        }), 500


@app.route('/api/cameras/<int:camera_id>/activate', methods=['POST'])
def activate_camera(camera_id):
    """Ativa uma câmera (troca de câmera)"""
    global counter, stats
    import traceback

    try:
        cameras_data = load_cameras()

        # Encontrar câmera
        camera = next((cam for cam in cameras_data['cameras'] if cam['id'] == camera_id), None)

        if not camera:
            return jsonify({
                'success': False,
                'message': 'Câmera não encontrada'
            }), 404

        # Reconectar com nova câmera
        print(f"\n🔄 Ativando câmera: {camera['name']}")
        print(f"   URL: {camera['url']}")

        # Pausar processamento
        counter.paused = True
        time.sleep(0.3)  # Aguardar frame atual terminar

        # Desconectar câmera atual
        counter.disconnect_camera()
        time.sleep(0.5)

        # Resetar ROI para nova câmera
        counter.roi_auto_detected = False
        counter.roi_offset = 0

        # Tentar conectar e capturar erro detalhado
        try:
            counter.connect_camera(camera['url'])
        except Exception as conn_error:
            error_msg = f"Falha ao conectar: {str(conn_error)}"
            print(f"❌ {error_msg}")
            print(traceback.format_exc())
            stats['camera_connected'] = False
            stats['status'] = error_msg
            counter.paused = False  # Resumir processamento mesmo com erro
            return jsonify({
                'success': False,
                'message': f"Erro de conexão com a câmera '{camera['name']}': {str(conn_error)}. Verifique se a URL está correta e se a câmera está acessível."
            }), 500

        # Atualizar câmera ativa
        cameras_data['active_camera_id'] = camera_id
        save_cameras(cameras_data)

        stats['camera_url'] = camera['url']
        stats['camera_connected'] = True
        stats['status'] = f"Conectado: {camera['name']}"

        # Resetar estatísticas
        counter.current_count = 0
        counter.frame_count = 0
        stats['current_count'] = 0
        stats['frame_count'] = 0
        stats['roi_height'] = 0

        print(f"✅ Câmera '{camera['name']}' ativada com sucesso!")

        # Resumir processamento
        counter.paused = False

        return jsonify({
            'success': True,
            'message': f"Câmera '{camera['name']}' ativada com sucesso!",
            'camera': camera
        })

    except Exception as e:
        print(f"❌ Erro ao ativar câmera: {str(e)}")
        print(traceback.format_exc())
        stats['camera_connected'] = False
        stats['status'] = f'Erro: {str(e)}'
        return jsonify({
            'success': False,
            'message': f'Erro ao ativar câmera: {str(e)}'
        }), 500


# ========== API DE PROCESSAMENTO MULTI-CÂMERA ==========

@app.route('/api/background/start', methods=['POST'])
def start_background_processing():
    """Inicia processamento de uma câmera em background"""
    global multi_camera_manager

    if multi_camera_manager is None:
        return jsonify({
            'success': False,
            'message': 'MultiCameraManager não inicializado'
        }), 500

    try:
        data = request.json
        camera_id = data.get('camera_id')

        if not camera_id:
            return jsonify({
                'success': False,
                'message': 'camera_id é obrigatório'
            }), 400

        # Buscar dados da câmera
        cameras_data = load_cameras()
        camera = next((cam for cam in cameras_data['cameras'] if cam['id'] == camera_id), None)

        if not camera:
            return jsonify({
                'success': False,
                'message': 'Câmera não encontrada'
            }), 404

        # Iniciar processamento em background
        if multi_camera_manager.start_camera(camera_id, camera['name'], camera['url']):
            return jsonify({
                'success': True,
                'message': f"Processamento background iniciado para '{camera['name']}'",
                'camera_id': camera_id
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Falha ao iniciar processamento'
            }), 500

    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/background/stop', methods=['POST'])
def stop_background_processing():
    """Para processamento de uma câmera em background"""
    global multi_camera_manager

    if multi_camera_manager is None:
        return jsonify({
            'success': False,
            'message': 'MultiCameraManager não inicializado'
        }), 500

    try:
        data = request.json
        camera_id = data.get('camera_id')

        if not camera_id:
            return jsonify({
                'success': False,
                'message': 'camera_id é obrigatório'
            }), 400

        if multi_camera_manager.stop_camera(camera_id):
            return jsonify({
                'success': True,
                'message': 'Processamento background parado',
                'camera_id': camera_id
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Câmera não estava processando em background'
            }), 404

    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@app.route('/api/background/status', methods=['GET'])
def get_background_status():
    """Retorna status de todas as câmeras em background"""
    global multi_camera_manager

    if multi_camera_manager is None:
        return jsonify({
            'running_cameras': [],
            'stats': []
        })

    return jsonify({
        'running_cameras': multi_camera_manager.get_running_cameras(),
        'stats': multi_camera_manager.get_all_stats()
    })


@app.route('/api/background/stats/<int:camera_id>', methods=['GET'])
def get_background_camera_stats(camera_id):
    """Retorna estatísticas de uma câmera em background"""
    global multi_camera_manager

    if multi_camera_manager is None:
        return jsonify({'error': 'MultiCameraManager não inicializado'}), 500

    stats = multi_camera_manager.get_camera_stats(camera_id)
    if stats:
        return jsonify(stats)
    else:
        return jsonify({'error': 'Câmera não está processando em background'}), 404


# ========== ANALYTICS VISIBILITY ==========

# Controle de visibilidade dos analytics no vídeo
analytics_visibility = {
    'lines': True,
    'zones': True,
    'abandoned': True
}


@app.route('/api/analytics/visibility', methods=['POST'])
def set_analytics_visibility():
    """Define quais analytics mostrar no vídeo"""
    global analytics_visibility
    try:
        data = request.json
        if 'lines' in data:
            analytics_visibility['lines'] = bool(data['lines'])
        if 'zones' in data:
            analytics_visibility['zones'] = bool(data['zones'])
        if 'abandoned' in data:
            analytics_visibility['abandoned'] = bool(data['abandoned'])
        return jsonify({'success': True, 'visibility': analytics_visibility})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400


@app.route('/api/analytics/visibility', methods=['GET'])
def get_analytics_visibility():
    """Retorna visibilidade atual dos analytics"""
    return jsonify(analytics_visibility)


# ========== API DE LINHAS VIRTUAIS ==========

@app.route('/api/lines', methods=['GET'])
def get_lines():
    """Lista todas as linhas virtuais (opcionalmente filtradas por câmera)"""
    global line_counter
    if line_counter is None:
        return jsonify({'lines': []})

    # Filtrar por câmera se especificado
    camera_id = request.args.get('camera_id', type=int)
    return jsonify({'lines': line_counter.get_lines_list(camera_id)})


@app.route('/api/lines', methods=['POST'])
def create_line():
    """Cria nova linha virtual"""
    global line_counter
    if line_counter is None:
        return jsonify({'success': False, 'message': 'LineCounter não inicializado'}), 500

    try:
        data = request.json
        line_id = str(uuid.uuid4())

        # Obter camera_id - obrigatório para associar à câmera
        camera_id = data.get('camera_id')

        line = line_counter.add_line(
            line_id,
            data['name'],
            tuple(data['point1']),
            tuple(data['point2']),
            data.get('direction_mode', 'bidirectional'),
            camera_id
        )

        return jsonify({
            'success': True,
            'message': 'Linha criada com sucesso!',
            'line': line.to_dict()
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/lines/<line_id>', methods=['DELETE'])
def delete_line(line_id):
    """Remove linha virtual"""
    global line_counter
    if line_counter is None:
        return jsonify({'success': False, 'message': 'LineCounter não inicializado'}), 500

    try:
        if line_counter.remove_line(line_id):
            return jsonify({'success': True, 'message': 'Linha removida com sucesso!'})
        else:
            return jsonify({'success': False, 'message': 'Linha não encontrada'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/lines/reset', methods=['POST'])
def reset_line_counts():
    """Reseta contadores de linhas"""
    global line_counter
    if line_counter is None:
        return jsonify({'success': False, 'message': 'LineCounter não inicializado'}), 500

    try:
        line_id = request.json.get('line_id')  # None = resetar todas
        line_counter.reset_counts(line_id)
        return jsonify({'success': True, 'message': 'Contadores resetados!'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ========== API DE ZONAS DE DETECÇÃO ==========

@app.route('/api/zones', methods=['GET'])
def get_zones():
    """Lista todas as zonas de detecção (opcionalmente filtradas por câmera)"""
    global zone_detector
    if zone_detector is None:
        return jsonify({'zones': []})

    # Filtrar por câmera se especificado
    camera_id = request.args.get('camera_id', type=int)
    return jsonify({'zones': zone_detector.get_zones_list(camera_id)})


@app.route('/api/zones', methods=['POST'])
def create_zone():
    """Cria nova zona de detecção"""
    global zone_detector
    if zone_detector is None:
        return jsonify({'success': False, 'message': 'ZoneDetector não inicializado'}), 500

    try:
        data = request.json
        zone_id = str(uuid.uuid4())

        # Obter camera_id - obrigatório para associar à câmera
        camera_id = data.get('camera_id')

        zone = zone_detector.add_zone(
            zone_id,
            data['name'],
            data['polygon'],  # Lista de pontos [[x1,y1], [x2,y2], ...]
            data.get('monitored_classes'),  # None ou lista de class_ids
            camera_id
        )

        return jsonify({
            'success': True,
            'message': 'Zona criada com sucesso!',
            'zone': zone.to_dict()
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/zones/<zone_id>', methods=['DELETE'])
def delete_zone(zone_id):
    """Remove zona de detecção"""
    global zone_detector
    if zone_detector is None:
        return jsonify({'success': False, 'message': 'ZoneDetector não inicializado'}), 500

    try:
        if zone_detector.remove_zone(zone_id):
            return jsonify({'success': True, 'message': 'Zona removida com sucesso!'})
        else:
            return jsonify({'success': False, 'message': 'Zona não encontrada'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ========== API DE RELATÓRIOS ==========

@app.route('/api/reports/generate', methods=['POST'])
def generate_report():
    """Gera relatório no formato especificado"""
    global report_generator
    if report_generator is None:
        return jsonify({'success': False, 'message': 'ReportGenerator não inicializado'}), 500

    try:
        data = request.json
        report_type = data.get('type', 'csv')  # 'csv' ou 'json'

        # Gerar relatório (últimas 24h por padrão)
        if report_type == 'csv':
            filename = report_generator.generate_csv_report()
        elif report_type == 'json':
            filename = report_generator.generate_json_report()
        else:
            return jsonify({'success': False, 'message': 'Tipo inválido'}), 400

        return jsonify({
            'success': True,
            'message': 'Relatório gerado com sucesso!',
            'filename': os.path.basename(filename),
            'download_url': f'/api/reports/download/{os.path.basename(filename)}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/reports/download/<filename>')
def download_report(filename):
    """Download de relatório gerado"""
    try:
        return send_from_directory('data/reports', filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/api/reports/list', methods=['GET'])
def list_reports():
    """Lista relatórios disponíveis"""
    global report_generator
    if report_generator is None:
        return jsonify({'reports': []})

    try:
        reports = report_generator.get_available_reports()
        return jsonify({'reports': reports})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reports/quick/<period>', methods=['GET'])
def quick_report(period):
    """
    Gera relatório rápido para período comum
    GET /api/reports/quick/today
    GET /api/reports/quick/yesterday
    GET /api/reports/quick/week
    GET /api/reports/quick/month
    """
    global line_counter

    try:
        now = datetime.now()

        # Definir período
        periods = {
            'today': (now.replace(hour=0, minute=0, second=0, microsecond=0), now),
            'yesterday': (
                (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0),
                (now - timedelta(days=1)).replace(hour=23, minute=59, second=59)
            ),
            'week': (now - timedelta(days=7), now),
            'month': (now - timedelta(days=30), now)
        }

        if period not in periods:
            return jsonify({'success': False, 'message': 'Período inválido. Use: today, yesterday, week, month'}), 400

        start_date, end_date = periods[period]

        # Calcular totais das linhas
        line_totals = {}
        grand_total_in = 0
        grand_total_out = 0
        class_totals = {}

        if line_counter is not None:
            for line_id, line in line_counter.lines.items():
                line_in = sum(c['in'] for c in line.counts.values())
                line_out = sum(c['out'] for c in line.counts.values())
                grand_total_in += line_in
                grand_total_out += line_out

                line_totals[line.name] = {
                    'in': line_in,
                    'out': line_out,
                    'total': line_in + line_out,
                    'breakdown': line.counts
                }

                # Agregar por classe
                for class_name, counts in line.counts.items():
                    if class_name not in class_totals:
                        class_totals[class_name] = {'in': 0, 'out': 0}
                    class_totals[class_name]['in'] += counts['in']
                    class_totals[class_name]['out'] += counts['out']

        report = {
            'period': {
                'name': period,
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'summary': {
                'total_in': grand_total_in,
                'total_out': grand_total_out,
                'total': grand_total_in + grand_total_out
            },
            'by_class': class_totals,
            'by_line': line_totals,
            'generated_at': now.isoformat()
        }

        return jsonify({'success': True, 'report': report})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/reports/period', methods=['POST'])
def period_report():
    """
    Gera relatório para período específico
    POST /api/reports/period
    Body: {
        "start_date": "2025-01-01T00:00:00",
        "end_date": "2025-01-07T23:59:59",
        "format": "json"  // ou "csv"
    }
    """
    global report_generator, line_counter

    try:
        data = request.json
        start_str = data.get('start_date')
        end_str = data.get('end_date')
        report_format = data.get('format', 'json')

        if not start_str or not end_str:
            return jsonify({'success': False, 'message': 'start_date e end_date são obrigatórios'}), 400

        start_date = datetime.fromisoformat(start_str)
        end_date = datetime.fromisoformat(end_str)

        if report_format == 'csv':
            filename = report_generator.generate_csv_report(start_date, end_date)
            return jsonify({
                'success': True,
                'filename': os.path.basename(filename),
                'download_url': f'/api/reports/download/{os.path.basename(filename)}'
            })
        else:
            filename = report_generator.generate_json_report(start_date, end_date)
            return jsonify({
                'success': True,
                'filename': os.path.basename(filename),
                'download_url': f'/api/reports/download/{os.path.basename(filename)}'
            })

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ========== API DE OBJETOS ABANDONADOS ==========

@app.route('/api/abandoned-objects', methods=['GET'])
def get_abandoned_objects():
    """Lista objetos abandonados atualmente detectados"""
    global abandoned_detector
    if abandoned_detector is None:
        return jsonify({'objects': []})

    try:
        objects = abandoned_detector.get_abandoned_objects()
        stats_data = abandoned_detector.get_stats()

        return jsonify({
            'objects': objects,
            'count': len(objects),
            'stats': stats_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def run_counter():
    """Thread do contador"""
    global counter
    counter = FinalOptimizedCounter()

    # Carregar câmera ativa do cameras.json (prioridade sobre config)
    cameras_data = load_cameras()
    active_camera_url = None

    if cameras_data.get('active_camera_id'):
        active_camera = next(
            (cam for cam in cameras_data['cameras'] if cam['id'] == cameras_data['active_camera_id']),
            None
        )
        if active_camera:
            active_camera_url = active_camera['url']
            print(f"📹 Usando câmera ativa: {active_camera['name']}")
            print(f"   URL: {active_camera_url}")
            # Atualizar URL no config
            counter.config['camera']['rtsp_url'] = active_camera_url
            counter.rtsp_url = active_camera_url

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
    print("   ✅ Linhas Virtuais - Contagem direcional")
    print("   ✅ Zonas Customizadas - Detecção por área")
    print("   ✅ Objetos Abandonados - Alerta automático")
    print("   ✅ Relatórios - Exportação CSV/JSON")

    # Inicializar módulos avançados
    print("\n🔧 Inicializando módulos avançados...")
    line_counter = LineCounter()
    zone_detector = ZoneDetector()
    abandoned_detector = AbandonedObjectDetector(static_time_threshold=30.0)
    report_generator = ReportGenerator()
    print("✓ Módulos inicializados!")

    # Multi-camera será inicializado após o counter (precisa do modelo YOLO)

    # Iniciar contador
    counter_thread = threading.Thread(target=run_counter, daemon=True)
    counter_thread.start()

    time.sleep(3)  # Aguardar inicialização do counter

    # Inicializar MultiCameraManager após counter estar pronto
    if counter is not None and hasattr(counter, 'model'):
        print("🎥 Inicializando gerenciador multi-câmera...")
        multi_camera_manager = MultiCameraManager(
            model=counter.model,
            line_counter=line_counter,
            zone_detector=zone_detector
        )
        print("✓ Gerenciador multi-câmera pronto!")

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
