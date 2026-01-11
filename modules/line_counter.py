"""
Módulo de Linhas Virtuais para Contagem Direcional
Permite criar linhas virtuais e contar objetos que as cruzam (entrada/saída)
Suporta múltiplas câmeras com linhas independentes por câmera
"""

import time
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional


class VirtualLine:
    """Representa uma linha virtual para contagem de objetos"""

    def __init__(self, line_id: str, name: str, point1: Tuple[int, int],
                 point2: Tuple[int, int], direction_mode: str = 'bidirectional',
                 camera_id: int = None):
        """
        Args:
            line_id: ID único da linha
            name: Nome da linha (ex: "Entrada Principal")
            point1: (x1, y1) - primeiro ponto
            point2: (x2, y2) - segundo ponto
            direction_mode: 'bidirectional', 'up_to_down', 'down_to_up', 'left_to_right', 'right_to_left'
            camera_id: ID da câmera associada
        """
        self.id = line_id
        self.name = name
        self.point1 = tuple(point1)
        self.point2 = tuple(point2)
        self.direction_mode = direction_mode
        self.camera_id = camera_id

        # Contadores por classe
        self.counts = {
            'person': {'in': 0, 'out': 0},
            'car': {'in': 0, 'out': 0},
            'motorcycle': {'in': 0, 'out': 0},
            'bus': {'in': 0, 'out': 0},
            'truck': {'in': 0, 'out': 0},
            'bicycle': {'in': 0, 'out': 0}
        }

        # Tracking de objetos que já cruzaram (evitar contar 2x)
        self.crossed_objects = set()

        # Histórico
        self.history = []  # [{timestamp, class, direction, track_id}]

    def to_dict(self) -> dict:
        """Serializa para dicionário"""
        return {
            'id': self.id,
            'name': self.name,
            'point1': list(self.point1),
            'point2': list(self.point2),
            'direction_mode': self.direction_mode,
            'camera_id': self.camera_id,
            'counts': self.counts,
            'total': sum(c['in'] + c['out'] for c in self.counts.values())
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Cria linha a partir de dicionário"""
        line = cls(
            data['id'],
            data['name'],
            tuple(data['point1']),
            tuple(data['point2']),
            data.get('direction_mode', 'bidirectional'),
            data.get('camera_id')
        )
        if 'counts' in data:
            line.counts = data['counts']
        return line


class LineCounter:
    """Gerenciador de linhas virtuais e contagem de cruzamentos - OTIMIZADO"""

    # Mapeamento de class_id COCO para nome
    CLASS_MAP = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }

    def __init__(self, config_file: str = 'config/lines.json'):
        """
        Args:
            config_file: Caminho para arquivo de configuração das linhas
        """
        self.config_file = config_file
        self.lines: Dict[str, VirtualLine] = {}
        self.object_positions: Dict[str, Dict[int, List[Tuple[float, float, float]]]] = {}  # camera_id -> track_id -> positions

        # ⚡ OTIMIZAÇÃO: Salvar config em batches, não a cada cruzamento
        self._config_dirty = False
        self._last_save_time = time.time()
        self._save_interval = 5.0  # Salvar a cada 5 segundos se houver mudanças

        # ⚡ OTIMIZAÇÃO: Cache de bounding boxes das linhas
        self._line_bboxes: Dict[str, Tuple[int, int, int, int]] = {}

        # Carregar configuração
        self.load_config()
        self._update_line_bboxes()

    def _update_line_bboxes(self):
        """⚡ OTIMIZAÇÃO: Atualiza cache de bounding boxes das linhas"""
        self._line_bboxes = {}
        for line_id, line in self.lines.items():
            x1, y1 = line.point1
            x2, y2 = line.point2
            # Bounding box da linha com margem GRANDE de 200 pixels
            # Aumentado para garantir detecção de linhas em todas as posições
            margin = 200
            self._line_bboxes[line_id] = (
                min(x1, x2) - margin,
                min(y1, y2) - margin,
                max(x1, x2) + margin,
                max(y1, y2) + margin
            )

    def _get_nearby_lines(self, pos: Tuple[float, float], camera_id: int = None) -> List[str]:
        """⚡ OTIMIZAÇÃO: Retorna apenas linhas próximas à posição (spatial indexing)"""
        x, y = pos
        nearby = []
        for line_id, bbox in self._line_bboxes.items():
            line = self.lines.get(line_id)
            if line is None:
                continue
            # Filtrar por câmera
            if camera_id is not None and line.camera_id is not None and line.camera_id != camera_id:
                continue
            # Verificar se posição está dentro do bbox da linha
            if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                nearby.append(line_id)
        return nearby

    def _maybe_save_config(self, current_time: float):
        """⚡ OTIMIZAÇÃO: Salva config apenas periodicamente"""
        if self._config_dirty and (current_time - self._last_save_time) >= self._save_interval:
            self.save_config()
            self._config_dirty = False
            self._last_save_time = current_time

    def add_line(self, line_id: str, name: str, point1: Tuple[int, int],
                 point2: Tuple[int, int], direction_mode: str = 'bidirectional',
                 camera_id: int = None) -> VirtualLine:
        """Adiciona nova linha virtual"""
        line = VirtualLine(line_id, name, point1, point2, direction_mode, camera_id)
        self.lines[line_id] = line
        self._update_line_bboxes()  # Atualizar cache
        self.save_config()  # Salvar imediatamente ao adicionar linha
        return line

    def remove_line(self, line_id: str) -> bool:
        """Remove linha virtual"""
        if line_id in self.lines:
            del self.lines[line_id]
            self._update_line_bboxes()  # Atualizar cache
            self.save_config()  # Salvar imediatamente ao remover linha
            return True
        return False

    def get_lines_for_camera(self, camera_id: int) -> List[VirtualLine]:
        """Retorna linhas de uma câmera específica"""
        return [line for line in self.lines.values() if line.camera_id == camera_id]

    def update(self, tracked_objects: List[dict], camera_id: int = None):
        """
        Atualiza contadores com novos objetos rastreados
        ⚡ OTIMIZADO: Spatial indexing + batched save

        Args:
            tracked_objects: Lista de {track_id, bbox, class_id, confidence}
            camera_id: ID da câmera (para filtrar linhas)
        """
        current_time = time.time()

        # Inicializar posições para esta câmera se não existir
        # FIX: usar 'is not None' para permitir camera_id=0
        cam_key = str(camera_id) if camera_id is not None else 'default'
        if cam_key not in self.object_positions:
            self.object_positions[cam_key] = {}

        for obj in tracked_objects:
            track_id = obj.get('track_id')
            bbox = obj.get('bbox')
            class_id = obj.get('class_id')

            if track_id is None or bbox is None:
                continue

            # Centro do objeto
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            center = (center_x, center_y)

            # Atualizar histórico de posições
            if track_id not in self.object_positions[cam_key]:
                self.object_positions[cam_key][track_id] = []

            positions = self.object_positions[cam_key][track_id]

            # ⚡ OTIMIZAÇÃO: Pular se objeto não se moveu significativamente (< 5 pixels)
            if positions:
                last_pos = positions[-1][:2]
                dx = center_x - last_pos[0]
                dy = center_y - last_pos[1]
                if dx*dx + dy*dy < 25:  # 5 pixels ao quadrado
                    continue

            positions.append((*center, current_time))

            # Manter apenas últimas 30 posições (reduzido para economia de memória)
            if len(positions) > 30:
                self.object_positions[cam_key][track_id] = positions[-30:]

            # ⚡ OTIMIZAÇÃO: Spatial indexing - verificar apenas linhas próximas
            nearby_line_ids = self._get_nearby_lines(center, camera_id)

            for line_id in nearby_line_ids:
                line = self.lines.get(line_id)
                if line:
                    self._check_line_crossing(track_id, class_id, line, current_time, cam_key)

        # Limpar posições antigas (objetos que não aparecem há mais de 10s)
        self._cleanup_old_positions(current_time, cam_key)

        # ⚡ OTIMIZAÇÃO: Salvar config apenas periodicamente
        self._maybe_save_config(current_time)

    def _check_line_crossing(self, track_id: int, class_id: int, line: VirtualLine, current_time: float, cam_key: str):
        """Verifica se objeto cruzou a linha"""
        positions = self.object_positions.get(cam_key, {}).get(track_id, [])

        if len(positions) < 2:
            return

        # Criar ID único para este cruzamento (evitar contar 2x)
        crossing_id = f"{track_id}_{line.id}"

        # Verificar se já contou esse cruzamento recentemente (reduzido para 1.5s)
        if self._already_counted(line, crossing_id, current_time, window=1.5):
            return

        # MÉTODO 1: Verificar cruzamento usando múltiplas posições recentes
        # Verifica os últimos 10 pares de posições para objetos rápidos
        num_positions = min(len(positions), 12)

        for i in range(num_positions - 1):
            prev_pos = positions[-(i+2)][:2]
            curr_pos = positions[-(i+1)][:2]

            # Verificar se segmento (prev_pos -> curr_pos) cruza a linha
            if self._segments_intersect(prev_pos, curr_pos, line.point1, line.point2):
                # Determinar direção usando primeira e última posição conhecida
                first_pos = positions[0][:2]
                last_pos = positions[-1][:2]
                direction = self._get_crossing_direction(first_pos, last_pos, line)

                # Atualizar contador
                class_name = self._get_class_name(class_id)
                if class_name in line.counts:
                    line.counts[class_name][direction] += 1

                    # Adicionar ao histórico
                    line.history.append({
                        'timestamp': current_time,
                        'class': class_name,
                        'direction': direction,
                        'track_id': track_id
                    })

                    # Marcar como cruzado com timestamp
                    line.crossed_objects.add((crossing_id, current_time))

                    # Limpar cruzamentos antigos (> 5s)
                    self._cleanup_crossed_objects(line, current_time)

                    # ⚡ OTIMIZAÇÃO: Marcar para salvar (não salvar imediatamente)
                    self._config_dirty = True

                    print(f"[LINE] {line.name}: {class_name} cruzou -> {direction.upper()} (track_id={track_id})")

                    # Sair após detectar cruzamento
                    return

        # MÉTODO 2: Verificar usando posição inicial vs posição atual (para objetos muito rápidos)
        if len(positions) >= 3:
            first_pos = positions[0][:2]
            last_pos = positions[-1][:2]

            # Verificar se o objeto estava de um lado e agora está do outro
            if self._crossed_line_sides(first_pos, last_pos, line):
                if not self._already_counted(line, crossing_id, current_time, window=1.5):
                    direction = self._get_crossing_direction(first_pos, last_pos, line)
                    class_name = self._get_class_name(class_id)

                    if class_name in line.counts:
                        line.counts[class_name][direction] += 1
                        line.history.append({
                            'timestamp': current_time,
                            'class': class_name,
                            'direction': direction,
                            'track_id': track_id
                        })
                        line.crossed_objects.add((crossing_id, current_time))
                        self._cleanup_crossed_objects(line, current_time)
                        # ⚡ OTIMIZAÇÃO: Marcar para salvar (não salvar imediatamente)
                        self._config_dirty = True
                        print(f"[LINE-FAST] {line.name}: {class_name} cruzou -> {direction.upper()} (track_id={track_id})")
                        return

    def _already_counted(self, line: VirtualLine, crossing_id: str, current_time: float, window: float = 2.0) -> bool:
        """Verifica se cruzamento já foi contado recentemente"""
        for crossed_id, crossed_time in line.crossed_objects:
            if crossed_id == crossing_id and (current_time - crossed_time) < window:
                return True
        return False

    def _cleanup_crossed_objects(self, line: VirtualLine, current_time: float, timeout: float = 5.0):
        """Remove cruzamentos antigos do set"""
        line.crossed_objects = {
            (crossing_id, timestamp)
            for crossing_id, timestamp in line.crossed_objects
            if (current_time - timestamp) < timeout
        }

    def _segments_intersect(self, p1: Tuple[float, float], p2: Tuple[float, float],
                           p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
        """
        Verifica se dois segmentos de reta se cruzam
        p1-p2: trajetória do objeto
        p3-p4: linha virtual
        """
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def _crossed_line_sides(self, pos1: Tuple[float, float], pos2: Tuple[float, float],
                           line: VirtualLine) -> bool:
        """
        Verifica se duas posições estão em lados opostos da linha virtual
        Útil para detectar objetos muito rápidos que "pulam" sobre a linha
        """
        # Vetor da linha
        dx = line.point2[0] - line.point1[0]
        dy = line.point2[1] - line.point1[1]

        # Calcular de que lado cada posição está (usando produto vetorial)
        side1 = (pos1[0] - line.point1[0]) * dy - (pos1[1] - line.point1[1]) * dx
        side2 = (pos2[0] - line.point1[0]) * dy - (pos2[1] - line.point1[1]) * dx

        # Se os sinais são diferentes, estão em lados opostos
        return (side1 > 0) != (side2 > 0)

    def _get_crossing_direction(self, prev_pos: Tuple[float, float],
                               curr_pos: Tuple[float, float],
                               line: VirtualLine) -> str:
        """
        Determina direção do cruzamento (in ou out)
        Usa produto vetorial para determinar de que lado da linha o objeto estava
        """
        # Vetor da linha
        dx = line.point2[0] - line.point1[0]
        dy = line.point2[1] - line.point1[1]

        # Produto vetorial para determinar "lado" da linha
        prev_side = (prev_pos[0] - line.point1[0]) * dy - (prev_pos[1] - line.point1[1]) * dx
        curr_side = (curr_pos[0] - line.point1[0]) * dy - (curr_pos[1] - line.point1[1]) * dx

        # Se mudou de negativo para positivo = "in", contrário = "out"
        if prev_side < 0 and curr_side >= 0:
            return 'in'
        elif prev_side >= 0 and curr_side < 0:
            return 'out'
        else:
            # Fallback robusto: usar direção do movimento combinando X e Y
            movement_x = curr_pos[0] - prev_pos[0]
            movement_y = curr_pos[1] - prev_pos[1]

            # Calcular ângulo da linha para determinar direção principal
            line_angle = abs(dy) > abs(dx)  # True = linha mais vertical

            if line_angle:
                # Linha mais vertical: usar movimento horizontal
                return 'in' if movement_x > 0 else 'out'
            else:
                # Linha mais horizontal: usar movimento vertical
                return 'in' if movement_y > 0 else 'out'

    def _get_class_name(self, class_id: int) -> str:
        """Mapeia class_id para nome"""
        return self.CLASS_MAP.get(class_id, 'unknown')

    def _cleanup_old_positions(self, current_time: float, cam_key: str, timeout: float = 10.0):
        """Remove posições de objetos que não aparecem há muito tempo"""
        if cam_key not in self.object_positions:
            return

        to_remove = []

        for track_id, positions in self.object_positions[cam_key].items():
            if positions and (current_time - positions[-1][2]) > timeout:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.object_positions[cam_key][track_id]

    def get_lines_list(self, camera_id: int = None) -> List[dict]:
        """Retorna lista de linhas com contadores, opcionalmente filtradas por câmera"""
        if camera_id is not None:
            # Inclui linhas desta câmera OU linhas sem câmera associada (compatibilidade)
            return [line.to_dict() for line in self.lines.values()
                    if line.camera_id == camera_id or line.camera_id is None]
        return [line.to_dict() for line in self.lines.values()]

    def reset_counts(self, line_id: Optional[str] = None):
        """Reseta contadores (de uma linha específica ou todas)"""
        if line_id:
            if line_id in self.lines:
                for class_name in self.lines[line_id].counts:
                    self.lines[line_id].counts[class_name] = {'in': 0, 'out': 0}
                self.lines[line_id].history = []
                self.lines[line_id].crossed_objects = set()
        else:
            # Resetar todas
            for line in self.lines.values():
                for class_name in line.counts:
                    line.counts[class_name] = {'in': 0, 'out': 0}
                line.history = []
                line.crossed_objects = set()

    def save_config(self):
        """Salva configuração das linhas em JSON"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)

            data = {
                line_id: {
                    'name': line.name,
                    'point1': list(line.point1),
                    'point2': list(line.point2),
                    'direction_mode': line.direction_mode,
                    'camera_id': line.camera_id,
                    'counts': line.counts
                }
                for line_id, line in self.lines.items()
            }

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erro ao salvar configuração de linhas: {e}")

    def load_config(self):
        """Carrega configuração das linhas do JSON"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for line_id, line_data in data.items():
                    line = VirtualLine(
                        line_id,
                        line_data['name'],
                        tuple(line_data['point1']),
                        tuple(line_data['point2']),
                        line_data.get('direction_mode', 'bidirectional'),
                        line_data.get('camera_id')
                    )
                    if 'counts' in line_data:
                        line.counts = line_data['counts']

                    self.lines[line_id] = line

                print(f"✓ {len(self.lines)} linha(s) virtual(is) carregada(s)")
        except Exception as e:
            print(f"Erro ao carregar configuração de linhas: {e}")
