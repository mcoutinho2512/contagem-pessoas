"""
Módulo de Detecção de Objetos Abandonados/Esquecidos
Detecta objetos que ficam estáticos por tempo prolongado
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional


class AbandonedObjectDetector:
    """Detector de objetos abandonados/esquecidos"""

    # Classes que podem ser abandonadas (COCO dataset)
    ABANDONABLE_CLASSES = {
        24: 'backpack',
        25: 'umbrella',
        26: 'handbag',
        27: 'tie',
        28: 'suitcase',
        31: 'skis',
        32: 'snowboard',
        33: 'sports ball',
        34: 'kite',
        35: 'baseball bat',
        36: 'baseball glove',
        37: 'skateboard',
        38: 'surfboard',
        39: 'tennis racket'
    }

    def __init__(self, static_time_threshold: float = 30.0, iou_threshold: float = 0.7,
                 excluded_classes: Optional[List[int]] = None):
        """
        Args:
            static_time_threshold: Tempo em segundos para considerar objeto abandonado (padrão: 30s)
            iou_threshold: IoU mínimo para considerar mesmo objeto (padrão: 0.7)
            excluded_classes: Classes a ignorar (ex: [0] para ignorar pessoas)
        """
        self.static_time_threshold = static_time_threshold
        self.iou_threshold = iou_threshold
        self.excluded_classes = excluded_classes or [0]  # Ignorar pessoas por padrão

        # Tracking de objetos: {id: {bbox, class, first_seen, last_seen, is_abandoned, alerted}}
        self.tracked_objects: Dict[int, dict] = {}
        self.next_id = 0

    def update(self, detections: List[dict]) -> List[dict]:
        """
        Atualiza tracking de objetos e retorna objetos abandonados

        Args:
            detections: Lista de detecções [{bbox, class_id, confidence}]

        Returns:
            Lista de objetos abandonados
        """
        current_time = time.time()
        matched_ids = set()

        # Para cada detecção
        for detection in detections:
            bbox = detection.get('bbox')
            class_id = detection.get('class_id')
            confidence = detection.get('confidence', 0)

            if bbox is None or class_id is None:
                continue

            # Ignorar classes excluídas (pessoas, etc)
            if class_id in self.excluded_classes:
                continue

            # Tentar fazer match com objetos existentes
            matched_id = self._find_matching_object(bbox)

            if matched_id is not None:
                # Atualizar objeto existente
                obj = self.tracked_objects[matched_id]
                obj['last_seen'] = current_time
                obj['bbox'] = bbox  # Atualizar posição (pode ter movido um pouco)
                obj['confidence'] = confidence

                # Verificar se virou abandonado
                time_static = current_time - obj['first_seen']
                if time_static >= self.static_time_threshold:
                    if not obj['is_abandoned']:
                        obj['is_abandoned'] = True
                        obj['abandoned_at'] = current_time
                        print(f"⚠️  Objeto abandonado detectado: {self.ABANDONABLE_CLASSES.get(class_id, 'Desconhecido')}")

                matched_ids.add(matched_id)
            else:
                # Novo objeto detectado
                obj_id = self._generate_id()
                self.tracked_objects[obj_id] = {
                    'bbox': bbox,
                    'class': class_id,
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'confidence': confidence,
                    'is_abandoned': False,
                    'abandoned_at': None,
                    'alerted': False
                }
                matched_ids.add(obj_id)

        # Remover objetos que não foram vistos recentemente (5s)
        self._cleanup_old_objects(current_time, timeout=5.0)

        return self.get_abandoned_objects()

    def _find_matching_object(self, bbox: Tuple[float, float, float, float]) -> Optional[int]:
        """
        Encontra objeto existente com maior IoU

        Args:
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            ID do objeto correspondente ou None
        """
        best_match = None
        best_iou = 0

        for obj_id, obj in self.tracked_objects.items():
            iou = self._calculate_iou(bbox, obj['bbox'])
            if iou > best_iou and iou >= self.iou_threshold:
                best_iou = iou
                best_match = obj_id

        return best_match

    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """
        Calcula Intersection over Union entre duas bounding boxes

        Args:
            box1: (x1, y1, x2, y2)
            box2: (x1, y1, x2, y2)

        Returns:
            IoU score (0 a 1)
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Área de interseção
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Área de união
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def _generate_id(self) -> int:
        """Gera ID único para novo objeto"""
        obj_id = self.next_id
        self.next_id += 1
        return obj_id

    def _cleanup_old_objects(self, current_time: float, timeout: float = 5.0):
        """
        Remove objetos que não foram vistos recentemente

        Args:
            current_time: Timestamp atual
            timeout: Tempo sem ver o objeto para removê-lo (segundos)
        """
        to_remove = []

        for obj_id, obj in self.tracked_objects.items():
            time_unseen = current_time - obj['last_seen']
            if time_unseen > timeout:
                to_remove.append(obj_id)

        for obj_id in to_remove:
            del self.tracked_objects[obj_id]

    def get_abandoned_objects(self) -> List[dict]:
        """
        Retorna lista de objetos abandonados

        Returns:
            Lista de dicionários com informações dos objetos abandonados
        """
        current_time = time.time()
        abandoned = []

        for obj_id, obj in self.tracked_objects.items():
            if obj['is_abandoned']:
                abandoned.append({
                    'id': obj_id,
                    'bbox': obj['bbox'],
                    'class': obj['class'],
                    'class_name': self.ABANDONABLE_CLASSES.get(obj['class'], 'Objeto'),
                    'time_abandoned': current_time - obj['first_seen'],
                    'confidence': obj['confidence']
                })

        return abandoned

    def reset(self):
        """Reseta o detector (limpa todos os objetos rastreados)"""
        self.tracked_objects = {}
        self.next_id = 0

    def get_stats(self) -> dict:
        """Retorna estatísticas do detector"""
        current_time = time.time()
        abandoned_count = sum(1 for obj in self.tracked_objects.values() if obj['is_abandoned'])

        return {
            'total_tracked': len(self.tracked_objects),
            'abandoned_count': abandoned_count,
            'threshold': self.static_time_threshold,
            'longest_static': max(
                (current_time - obj['first_seen'] for obj in self.tracked_objects.values()),
                default=0
            )
        }
