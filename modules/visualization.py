"""
Módulo de Visualização para Linhas e Zonas
Funções para desenhar elementos visuais nos frames de vídeo
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple


def draw_virtual_lines(frame: np.ndarray, lines: Dict, line_counter) -> np.ndarray:
    """
    Desenha linhas virtuais e contadores no frame

    Args:
        frame: Frame de vídeo (numpy array)
        lines: Dicionário de linhas {line_id: VirtualLine}
        line_counter: Instância do LineCounter

    Returns:
        Frame com linhas desenhadas
    """
    # Cores diferentes para cada linha
    line_colors = [
        (255, 212, 0),   # Ciano
        (0, 255, 128),   # Verde claro
        (255, 128, 0),   # Azul claro
        (128, 0, 255),   # Rosa
    ]

    for idx, line in enumerate(lines.values()):
        color = line_colors[idx % len(line_colors)]
        thickness = 3

        # Desenhar linha
        cv2.line(frame, line.point1, line.point2, color, thickness)

        # Desenhar pontos nos extremos
        cv2.circle(frame, line.point1, 8, color, -1)
        cv2.circle(frame, line.point2, 8, color, -1)

        # Calcular ponto médio para label
        mid_x = (line.point1[0] + line.point2[0]) // 2
        mid_y = (line.point1[1] + line.point2[1]) // 2

        # Configurações de fonte
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2

        # Calcular totais IN/OUT
        total_in = sum(counts['in'] for counts in line.counts.values())
        total_out = sum(counts['out'] for counts in line.counts.values())

        # Label principal: Nome da linha
        label = line.name
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

        # Background do nome
        bg_pt1 = (mid_x - 5, mid_y - text_h - 10)
        bg_pt2 = (mid_x + text_w + 10, mid_y + 5)
        cv2.rectangle(frame, bg_pt1, bg_pt2, color, -1)
        cv2.putText(frame, label, (mid_x, mid_y), font, font_scale, (0, 0, 0), font_thickness)

        # Mostrar totais IN/OUT de forma compacta
        y_offset = mid_y + 35
        font_scale_small = 0.6

        # Total geral
        total_text = f"IN:{total_in} | OUT:{total_out}"
        (tw, th), _ = cv2.getTextSize(total_text, font, font_scale_small, 2)

        # Background semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (mid_x - 5, y_offset - th - 5), (mid_x + tw + 10, y_offset + 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, total_text, (mid_x, y_offset), font, font_scale_small, (255, 255, 255), 2)

        # Mostrar detalhes por categoria em linha compacta
        y_offset += 28
        details = []
        class_abbrev = {'person': 'PES', 'car': 'CAR', 'motorcycle': 'MOT', 'bus': 'BUS', 'truck': 'CAM', 'bicycle': 'BIC'}

        for cls_name, counts in line.counts.items():
            total = counts['in'] + counts['out']
            if total > 0:
                abbrev = class_abbrev.get(cls_name, cls_name[:3].upper())
                details.append(f"{abbrev}:{total}")

        if details:
            # Mostrar em uma única linha compacta
            detail_text = " | ".join(details)
            (dw, dh), _ = cv2.getTextSize(detail_text, font, 0.45, 1)

            overlay = frame.copy()
            cv2.rectangle(overlay, (mid_x - 5, y_offset - dh - 5), (mid_x + dw + 10, y_offset + 5), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, detail_text, (mid_x, y_offset), font, 0.45, color, 1)

    return frame


def draw_detection_zones(frame: np.ndarray, zones: Dict, zone_detector=None) -> np.ndarray:
    """
    Desenha zonas de detecção no frame

    Args:
        frame: Frame de vídeo (numpy array)
        zones: Dicionário de zonas {zone_id: DetectionZone}
        zone_detector: Instância do ZoneDetector (opcional)

    Returns:
        Frame com zonas desenhadas
    """
    overlay = frame.copy()

    for zone in zones.values():
        # Determinar cor baseada em ocupação
        current_count = zone.current_count if hasattr(zone, 'current_count') else 0

        if current_count == 0:
            color = (255, 212, 0)  # Ciano - vazio
            alpha = 0.2
        elif current_count < 5:
            color = (0, 255, 0)  # Verde - normal
            alpha = 0.3
        elif current_count < 10:
            color = (0, 165, 255)  # Laranja - atenção
            alpha = 0.4
        else:
            color = (0, 0, 255)  # Vermelho - cheio
            alpha = 0.5

        # Desenhar polígono preenchido no overlay
        cv2.fillPoly(overlay, [zone.polygon], color)

        # Desenhar borda no frame original
        cv2.polylines(frame, [zone.polygon], True, color, 3)

        # Calcular centróide do polígono
        moments = cv2.moments(zone.polygon)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])

            # Desenhar label
            label = f"{zone.name}: {current_count}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2

            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

            # Background do label
            bg_pt1 = (cx - text_w // 2 - 5, cy - text_h - 10)
            bg_pt2 = (cx + text_w // 2 + 5, cy)
            cv2.rectangle(frame, bg_pt1, bg_pt2, color, -1)

            # Texto
            text_pos = (cx - text_w // 2, cy - 5)
            cv2.putText(frame, label, text_pos, font, font_scale, (255, 255, 255), font_thickness)

    # Blend overlay com frame
    cv2.addWeighted(overlay, alpha if 'alpha' in locals() else 0.3, frame, 0.7, 0, frame)

    return frame


def draw_abandoned_objects(frame: np.ndarray, abandoned_objects: List[dict],
                          class_names: Dict[int, str] = None) -> np.ndarray:
    """
    Desenha objetos abandonados com alerta vermelho

    Args:
        frame: Frame de vídeo
        abandoned_objects: Lista de objetos abandonados
        class_names: Mapeamento de class_id para nome

    Returns:
        Frame com alertas desenhados
    """
    if class_names is None:
        class_names = {
            24: 'Mochila',
            25: 'Guarda-chuva',
            26: 'Bolsa',
            27: 'Gravata',
            28: 'Mala'
        }

    for obj in abandoned_objects:
        bbox = obj['bbox']
        x1, y1, x2, y2 = map(int, bbox)

        # Caixa vermelha piscante (alternância baseada em segundos)
        color = (0, 0, 255)  # Vermelho
        thickness = 4

        # Desenhar retângulo com borda grossa
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label de alerta
        class_id = obj.get('class', 24)
        class_name = class_names.get(class_id, 'Objeto')
        time_abandoned = int(obj.get('time_abandoned', 0))

        label = f"ALERTA: {class_name} - {time_abandoned}s"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        # Background do label (vermelho sólido)
        bg_pt1 = (x1, y1 - text_h - 10)
        bg_pt2 = (x1 + text_w, y1)
        cv2.rectangle(frame, bg_pt1, bg_pt2, color, -1)

        # Texto branco
        text_pos = (x1, y1 - 5)
        cv2.putText(frame, label, text_pos, font, font_scale, (255, 255, 255), font_thickness)

        # Adicionar ícone de alerta
        icon_center = (x1 + 15, y1 - 30)
        cv2.putText(frame, "⚠", icon_center, font, 1.0, color, 2)

    return frame


def draw_hud_overlay(frame: np.ndarray, stats: dict) -> np.ndarray:
    """
    Desenha HUD (Heads-Up Display) com informações gerais

    Args:
        frame: Frame de vídeo
        stats: Dicionário com estatísticas

    Returns:
        Frame com HUD
    """
    height, width = frame.shape[:2]

    # Configurações de fonte
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    color = (255, 212, 0)  # Ciano

    # Info no canto superior direito
    info_lines = [
        f"FPS: {stats.get('fps', 0):.1f}",
        f"Pessoas: {stats.get('current_count', 0)}",
        f"Max: {stats.get('max_count', 0)}",
    ]

    y_offset = 30
    x_pos = width - 150

    for line in info_lines:
        (text_w, text_h), _ = cv2.getTextSize(line, font, font_scale, font_thickness)

        # Background semi-transparente
        overlay = frame.copy()
        bg_pt1 = (x_pos - 5, y_offset - text_h - 5)
        bg_pt2 = (x_pos + text_w + 5, y_offset + 5)
        cv2.rectangle(overlay, bg_pt1, bg_pt2, (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Texto
        cv2.putText(frame, line, (x_pos, y_offset), font, font_scale, color, font_thickness)
        y_offset += 25

    return frame
