"""
Módulo de Zonas de Detecção Customizadas
Permite criar zonas poligonais e detectar objetos apenas dentro delas
Suporta múltiplas câmeras com zonas independentes por câmera
"""

import time
import json
import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


class DetectionZone:
    """Representa uma zona poligonal de detecção"""

    def __init__(self, zone_id: str, name: str, polygon_points: List[Tuple[int, int]],
                 monitored_classes: Optional[List[int]] = None, camera_id: int = None):
        """
        Args:
            zone_id: ID único da zona
            name: Nome da zona (ex: "Área Restrita")
            polygon_points: Lista de pontos [(x1,y1), (x2,y2), ...]
            monitored_classes: Classes a monitorar [0, 2, 3] ou None para todas
            camera_id: ID da câmera associada
        """
        self.id = zone_id
        self.name = name
        self.polygon = np.array(polygon_points, dtype=np.int32)
        self.monitored_classes = monitored_classes
        self.camera_id = camera_id

        self.current_count = 0
        self.max_count = 0
        self.history = []  # [{timestamp, count, objects_detected}]
        self.alerts = []

    def to_dict(self) -> dict:
        """Serializa para dicionário"""
        return {
            'id': self.id,
            'name': self.name,
            'polygon': self.polygon.tolist(),
            'monitored_classes': self.monitored_classes,
            'camera_id': self.camera_id,
            'current_count': self.current_count,
            'max_count': self.max_count
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Cria zona a partir de dicionário"""
        zone = cls(
            data['id'],
            data['name'],
            data['polygon'],
            data.get('monitored_classes'),
            data.get('camera_id')
        )
        zone.max_count = data.get('max_count', 0)
        return zone


class ZoneDetector:
    """Gerenciador de zonas de detecção"""

    # Mapeamento de class_id COCO para nome
    CLASS_MAP = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }

    def __init__(self, config_file: str = 'config/zones.json'):
        """
        Args:
            config_file: Caminho para arquivo de configuração das zonas
        """
        self.config_file = config_file
        self.zones: Dict[str, DetectionZone] = {}

        # Carregar configuração
        self.load_config()

    def add_zone(self, zone_id: str, name: str, polygon_points: List[Tuple[int, int]],
                 monitored_classes: Optional[List[int]] = None, camera_id: int = None) -> DetectionZone:
        """Adiciona nova zona de detecção"""
        zone = DetectionZone(zone_id, name, polygon_points, monitored_classes, camera_id)
        self.zones[zone_id] = zone
        self.save_config()
        return zone

    def remove_zone(self, zone_id: str) -> bool:
        """Remove zona"""
        if zone_id in self.zones:
            del self.zones[zone_id]
            self.save_config()
            return True
        return False

    def get_zones_for_camera(self, camera_id: int) -> List[DetectionZone]:
        """Retorna zonas de uma câmera específica"""
        return [zone for zone in self.zones.values() if zone.camera_id == camera_id]

    def update(self, tracked_objects: List[dict], camera_id: int = None):
        """
        Atualiza contagens por zona

        Args:
            tracked_objects: Lista de {track_id, bbox, class_id, confidence}
            camera_id: ID da câmera (para filtrar zonas)
        """
        current_time = time.time()

        # Resetar contagens atuais apenas para zonas desta câmera
        for zone in self.zones.values():
            if camera_id is None or zone.camera_id == camera_id or zone.camera_id is None:
                zone.current_count = 0

        # Para cada objeto detectado
        for obj in tracked_objects:
            bbox = obj.get('bbox')
            class_id = obj.get('class_id')

            if bbox is None:
                continue

            # Centro do objeto
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            point = (int(center_x), int(center_y))

            # Verificar cada zona (apenas desta câmera)
            for zone in self.zones.values():
                # Filtrar por câmera
                if camera_id is not None and zone.camera_id != camera_id and zone.camera_id is not None:
                    continue

                # Filtrar por classe se especificado
                if zone.monitored_classes is not None and class_id not in zone.monitored_classes:
                    continue

                # Verificar se ponto está dentro do polígono
                if self._point_in_polygon(point, zone.polygon):
                    zone.current_count += 1

        # Atualizar máximos e histórico
        for zone in self.zones.values():
            if camera_id is not None and zone.camera_id != camera_id and zone.camera_id is not None:
                continue

            if zone.current_count > zone.max_count:
                zone.max_count = zone.current_count

            # Registrar no histórico (a cada 5 segundos)
            if not zone.history or (current_time - zone.history[-1]['timestamp']) >= 5:
                zone.history.append({
                    'timestamp': current_time,
                    'count': zone.current_count
                })

                # Manter apenas últimos 1000 registros
                if len(zone.history) > 1000:
                    zone.history = zone.history[-1000:]

    def _point_in_polygon(self, point: Tuple[int, int], polygon: np.ndarray) -> bool:
        """
        Verifica se ponto está dentro do polígono usando cv2

        Args:
            point: (x, y)
            polygon: Array numpy com pontos do polígono

        Returns:
            True se ponto está dentro ou na borda
        """
        result = cv2.pointPolygonTest(polygon, point, False)
        return result >= 0  # >= 0 significa dentro ou na borda

    def get_zones_list(self, camera_id: int = None) -> List[dict]:
        """Retorna lista de zonas com contadores, opcionalmente filtradas por câmera"""
        if camera_id is not None:
            # Inclui zonas desta câmera OU zonas sem câmera associada (compatibilidade)
            return [zone.to_dict() for zone in self.zones.values()
                    if zone.camera_id == camera_id or zone.camera_id is None]
        return [zone.to_dict() for zone in self.zones.values()]

    def reset_counts(self, zone_id: Optional[str] = None):
        """Reseta contadores (de uma zona específica ou todas)"""
        if zone_id:
            if zone_id in self.zones:
                self.zones[zone_id].current_count = 0
                self.zones[zone_id].max_count = 0
                self.zones[zone_id].history = []
        else:
            # Resetar todas
            for zone in self.zones.values():
                zone.current_count = 0
                zone.max_count = 0
                zone.history = []

    def save_config(self):
        """Salva configuração das zonas em JSON"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)

            data = {
                zone_id: {
                    'name': zone.name,
                    'polygon': zone.polygon.tolist(),
                    'monitored_classes': zone.monitored_classes,
                    'camera_id': zone.camera_id,
                    'max_count': zone.max_count
                }
                for zone_id, zone in self.zones.items()
            }

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erro ao salvar configuração de zonas: {e}")

    def load_config(self):
        """Carrega configuração das zonas do JSON"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for zone_id, zone_data in data.items():
                    zone = DetectionZone(
                        zone_id,
                        zone_data['name'],
                        zone_data['polygon'],
                        zone_data.get('monitored_classes'),
                        zone_data.get('camera_id')
                    )
                    zone.max_count = zone_data.get('max_count', 0)

                    self.zones[zone_id] = zone

                print(f"✓ {len(self.zones)} zona(s) de detecção carregada(s)")
        except Exception as e:
            print(f"Erro ao carregar configuração de zonas: {e}")
