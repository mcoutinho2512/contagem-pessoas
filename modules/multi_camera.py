"""
Módulo de Processamento Multi-Câmera
Permite processar múltiplas câmeras simultaneamente em background
Cada câmera tem seu próprio thread de processamento
"""

import cv2
import time
import threading
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import supervision as sv
from ultralytics import YOLO


@dataclass
class CameraStats:
    """Estatísticas de uma câmera"""
    camera_id: int
    camera_name: str
    is_running: bool = False
    is_connected: bool = False
    current_count: int = 0
    total_detections: int = 0
    frame_count: int = 0
    fps: float = 0.0
    last_update: float = 0.0
    error_message: str = ""
    counts_by_class: Dict[str, int] = field(default_factory=dict)


class CameraProcessor:
    """Processador individual de câmera - roda em thread separada"""

    # Classes COCO para detecção
    DETECTION_CLASSES = [0, 1, 2, 3, 5, 7]  # pessoa, bicicleta, carro, moto, ônibus, caminhão
    CLASS_NAMES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

    def __init__(self, camera_id: int, camera_name: str, camera_url: str,
                 model: YOLO, line_counter, zone_detector,
                 confidence: float = 0.25):
        """
        Args:
            camera_id: ID da câmera
            camera_name: Nome da câmera
            camera_url: URL RTSP da câmera
            model: Modelo YOLO compartilhado
            line_counter: LineCounter compartilhado
            zone_detector: ZoneDetector compartilhado
            confidence: Threshold de confiança
        """
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.camera_url = camera_url
        self.model = model
        self.line_counter = line_counter
        self.zone_detector = zone_detector
        self.confidence = confidence

        # Estado
        self.is_running = False
        self.is_connected = False
        self.should_stop = False

        # Conexão
        self.cap = None
        self.thread = None

        # Tracker (cada câmera precisa do seu próprio tracker)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )

        # Estatísticas
        self.stats = CameraStats(camera_id=camera_id, camera_name=camera_name)
        self.start_time = None
        self.frame_count = 0

        # Lock para thread safety
        self.lock = threading.Lock()

    def start(self) -> bool:
        """Inicia processamento da câmera em thread separada"""
        if self.is_running:
            return True

        self.should_stop = False
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        """Para o processamento da câmera"""
        self.should_stop = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        self._disconnect()

    def _connect(self) -> bool:
        """Conecta à câmera"""
        try:
            self.cap = cv2.VideoCapture(self.camera_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Tentar ler um frame para verificar conexão
            for _ in range(3):
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.is_connected = True
                    self.stats.is_connected = True
                    self.stats.error_message = ""
                    print(f"✓ Câmera {self.camera_name} (ID {self.camera_id}) conectada em background")
                    return True
                time.sleep(0.5)

            self.stats.error_message = "Não foi possível ler frames da câmera"
            return False

        except Exception as e:
            self.stats.error_message = str(e)
            print(f"✗ Erro ao conectar câmera {self.camera_name}: {e}")
            return False

    def _disconnect(self):
        """Desconecta da câmera"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_connected = False
        self.stats.is_connected = False

    def _processing_loop(self):
        """Loop principal de processamento (roda em thread separada)"""
        self.is_running = True
        self.stats.is_running = True
        self.start_time = time.time()

        # Conectar à câmera
        if not self._connect():
            self.is_running = False
            self.stats.is_running = False
            return

        print(f"🎬 Iniciando processamento background: {self.camera_name}")

        consecutive_errors = 0
        max_errors = 10

        while not self.should_stop:
            try:
                ret, frame = self.cap.read()

                if not ret or frame is None:
                    consecutive_errors += 1
                    if consecutive_errors >= max_errors:
                        print(f"⚠️ Muitos erros consecutivos na câmera {self.camera_name}, reconectando...")
                        self._disconnect()
                        time.sleep(2)
                        if not self._connect():
                            break
                        consecutive_errors = 0
                    time.sleep(0.1)
                    continue

                consecutive_errors = 0

                # Processar frame
                self._process_frame(frame)

                # Pequena pausa para não sobrecarregar CPU
                time.sleep(0.03)  # ~30 FPS max

            except Exception as e:
                print(f"Erro no processamento da câmera {self.camera_name}: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_errors:
                    break
                time.sleep(0.5)

        # Cleanup
        self._disconnect()
        self.is_running = False
        self.stats.is_running = False
        print(f"⏹️ Processamento background parado: {self.camera_name}")

    def _process_frame(self, frame: np.ndarray):
        """Processa um frame (detecção + tracking + analytics)"""
        try:
            # Detecção
            results = self.model.track(
                frame,
                persist=True,
                classes=self.DETECTION_CLASSES,
                conf=self.confidence,
                verbose=False
            )

            # Converter detecções
            detections = sv.Detections.from_ultralytics(results[0])

            # Tracking
            if len(detections) > 0:
                detections = self.tracker.update_with_detections(detections)

            # Preparar objetos rastreados
            tracked_objects = []
            if detections.tracker_id is not None:
                for bbox, class_id, track_id, confidence in zip(
                    detections.xyxy, detections.class_id,
                    detections.tracker_id, detections.confidence
                ):
                    tracked_objects.append({
                        'track_id': int(track_id) if track_id is not None else 0,
                        'bbox': list(bbox),
                        'class_id': int(class_id),
                        'confidence': float(confidence)
                    })

            # Atualizar analytics (linhas e zonas desta câmera)
            if self.line_counter and len(tracked_objects) > 0:
                self.line_counter.update(tracked_objects, self.camera_id)

            if self.zone_detector and len(tracked_objects) > 0:
                self.zone_detector.update(tracked_objects, self.camera_id)

            # Atualizar estatísticas
            with self.lock:
                self.frame_count += 1
                self.stats.frame_count = self.frame_count
                self.stats.current_count = len(detections)
                self.stats.total_detections += len(detections)
                self.stats.last_update = time.time()

                elapsed = time.time() - self.start_time
                self.stats.fps = self.frame_count / elapsed if elapsed > 0 else 0

                # Contagem por classe
                if detections.class_id is not None:
                    for class_id in detections.class_id:
                        class_name = self.CLASS_NAMES.get(int(class_id), 'unknown')
                        self.stats.counts_by_class[class_name] = \
                            self.stats.counts_by_class.get(class_name, 0) + 1

        except Exception as e:
            print(f"Erro ao processar frame da câmera {self.camera_name}: {e}")

    def get_stats(self) -> dict:
        """Retorna estatísticas da câmera"""
        with self.lock:
            return {
                'camera_id': self.camera_id,
                'camera_name': self.camera_name,
                'is_running': self.is_running,
                'is_connected': self.is_connected,
                'current_count': self.stats.current_count,
                'total_detections': self.stats.total_detections,
                'frame_count': self.stats.frame_count,
                'fps': round(self.stats.fps, 1),
                'last_update': self.stats.last_update,
                'error_message': self.stats.error_message,
                'counts_by_class': dict(self.stats.counts_by_class)
            }


class MultiCameraManager:
    """Gerenciador de múltiplos processadores de câmera"""

    def __init__(self, model: YOLO, line_counter, zone_detector):
        """
        Args:
            model: Modelo YOLO compartilhado
            line_counter: LineCounter compartilhado
            zone_detector: ZoneDetector compartilhado
        """
        self.model = model
        self.line_counter = line_counter
        self.zone_detector = zone_detector
        self.processors: Dict[int, CameraProcessor] = {}
        self.lock = threading.Lock()

    def start_camera(self, camera_id: int, camera_name: str, camera_url: str) -> bool:
        """
        Inicia processamento de uma câmera em background

        Args:
            camera_id: ID da câmera
            camera_name: Nome da câmera
            camera_url: URL RTSP

        Returns:
            True se iniciou com sucesso
        """
        with self.lock:
            # Verificar se já está rodando
            if camera_id in self.processors:
                if self.processors[camera_id].is_running:
                    print(f"Câmera {camera_name} já está processando em background")
                    return True
                else:
                    # Remover processador parado
                    del self.processors[camera_id]

            # Criar novo processador
            processor = CameraProcessor(
                camera_id=camera_id,
                camera_name=camera_name,
                camera_url=camera_url,
                model=self.model,
                line_counter=self.line_counter,
                zone_detector=self.zone_detector
            )

            # Iniciar
            if processor.start():
                self.processors[camera_id] = processor
                return True

            return False

    def stop_camera(self, camera_id: int) -> bool:
        """Para processamento de uma câmera"""
        with self.lock:
            if camera_id in self.processors:
                self.processors[camera_id].stop()
                del self.processors[camera_id]
                return True
            return False

    def stop_all(self):
        """Para todos os processadores"""
        with self.lock:
            for processor in self.processors.values():
                processor.stop()
            self.processors.clear()

    def get_running_cameras(self) -> List[int]:
        """Retorna IDs das câmeras em processamento"""
        with self.lock:
            return [cid for cid, proc in self.processors.items() if proc.is_running]

    def get_all_stats(self) -> List[dict]:
        """Retorna estatísticas de todas as câmeras em processamento"""
        with self.lock:
            return [proc.get_stats() for proc in self.processors.values()]

    def get_camera_stats(self, camera_id: int) -> Optional[dict]:
        """Retorna estatísticas de uma câmera específica"""
        with self.lock:
            if camera_id in self.processors:
                return self.processors[camera_id].get_stats()
            return None

    def is_camera_running(self, camera_id: int) -> bool:
        """Verifica se câmera está processando"""
        with self.lock:
            return camera_id in self.processors and self.processors[camera_id].is_running
