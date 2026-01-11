"""
Módulos Avançados de Visão Computacional
Sistema de Contagem de Pessoas
"""

from .line_counter import LineCounter, VirtualLine
from .zone_detector import ZoneDetector, DetectionZone
from .abandoned_objects import AbandonedObjectDetector
from .multi_camera import MultiCameraManager, CameraProcessor
from .visualization import (
    draw_virtual_lines,
    draw_detection_zones,
    draw_abandoned_objects,
    draw_hud_overlay
)

__all__ = [
    'LineCounter',
    'VirtualLine',
    'ZoneDetector',
    'DetectionZone',
    'AbandonedObjectDetector',
    'MultiCameraManager',
    'CameraProcessor',
    'draw_virtual_lines',
    'draw_detection_zones',
    'draw_abandoned_objects',
    'draw_hud_overlay'
]

__version__ = '1.1.0'
