#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema OTIMIZADO de Contagem de Pessoas em Tempo Real
Versão melhorada para detectar pessoas sentadas com oclusão parcial
Otimizado para salas de reunião com mesas e múltiplas pessoas
Autor: Sistema desenvolvido via Claude Code
Data: 2026-01-08
"""

import cv2
import json
import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from ultralytics import YOLO
import supervision as sv


class OptimizedPeopleCounter:
    """
    Contador de pessoas OTIMIZADO para detectar pessoas sentadas
    com oclusão parcial em salas de reunião
    """

    def __init__(self, config_path: str = "config_optimized.json"):
        """
        Inicializa o contador otimizado

        Args:
            config_path: Caminho para arquivo de configuração otimizado
        """
        # Carregar configurações
        self.config = self._load_config(config_path)

        # Configurar logging
        self._setup_logging()

        # Variáveis de estado
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
        self.fps_history = []

        # Debug e análise
        self.rejected_detections = []
        self.detection_history = []

        # Cores
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_YELLOW = (0, 255, 255)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_BLACK = (0, 0, 0)
        self.COLOR_ORANGE = (0, 165, 255)

        self.logger.info("OptimizedPeopleCounter inicializado")

    def _load_config(self, config_path: str) -> Dict:
        """Carrega configurações do arquivo JSON"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✓ Configurações otimizadas carregadas de {config_path}")
            return config
        except FileNotFoundError:
            print(f"✗ Arquivo {config_path} não encontrado")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"✗ Erro ao decodificar JSON: {e}")
            sys.exit(1)

    def _setup_logging(self):
        """Configura sistema de logging"""
        if not self.config['logging']['enabled']:
            self.logger = logging.getLogger('DummyLogger')
            self.logger.addHandler(logging.NullHandler())
            return

        log_dir = Path(self.config['logging']['log_directory'])
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"optimized_counter_{timestamp}.log"

        logging.basicConfig(
            level=logging.DEBUG if self.config['logging'].get('debug_mode') else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("Sistema de logging inicializado")

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Aplica pré-processamento para melhorar detecção

        Args:
            frame: Frame original

        Returns:
            Frame processado
        """
        if not self.config['preprocessing']['enable']:
            return frame

        enhanced = frame.copy()

        # CLAHE - Melhorar contraste adaptativo
        if self.config['preprocessing']['clahe']:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            clip_limit = self.config['preprocessing']['clahe_clip_limit']
            tile_size = self.config['preprocessing']['clahe_tile_size']
            clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                   tileGridSize=(tile_size, tile_size))
            l = clahe.apply(l)

            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Sharpening - Realçar bordas
        if self.config['preprocessing']['sharpen']:
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)

        # Denoising - Reduzir ruído (opcional)
        if self.config['preprocessing']['denoise']:
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

        return enhanced

    def apply_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Aplica ROI para ignorar áreas irrelevantes

        Args:
            frame: Frame completo

        Returns:
            (frame_roi, roi_y_offset)
        """
        if not self.config['roi']['enable']:
            return frame, 0

        height, width = frame.shape[:2]

        # Calcular margens
        top_margin = int(height * self.config['roi']['top_margin'])
        bottom_margin = int(height * self.config['roi']['bottom_margin'])
        left_margin = int(width * self.config['roi']['left_margin'])
        right_margin = int(width * self.config['roi']['right_margin'])

        # Aplicar ROI
        roi = frame[top_margin:height-bottom_margin,
                   left_margin:width-right_margin]

        return roi, top_margin

    def is_valid_detection(self, box: np.ndarray, confidence: float,
                          frame_shape: Tuple) -> Tuple[bool, str]:
        """
        Valida se uma detecção é uma pessoa real
        Otimizado para pessoas sentadas

        Args:
            box: Coordenadas [x1, y1, x2, y2]
            confidence: Confiança da detecção
            frame_shape: (height, width) do frame

        Returns:
            (is_valid, rejection_reason)
        """
        if not self.config['filtering']['enable']:
            return True, ""

        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = width * height

        frame_height, frame_width = frame_shape

        # Verificações básicas
        if width <= 0 or height <= 0:
            return False, "dimensões inválidas"

        # Área mínima
        min_area = self.config['filtering']['min_area']
        if area < min_area:
            return False, f"área muito pequena ({area:.0f} < {min_area})"

        # Dimensões mínimas
        min_width = self.config['filtering']['min_width']
        min_height = self.config['filtering']['min_height']
        if width < min_width:
            return False, f"largura muito pequena ({width:.0f} < {min_width})"
        if height < min_height:
            return False, f"altura muito pequena ({height:.0f} < {min_height})"

        # Aspect ratio - AJUSTADO para pessoas sentadas
        aspect_ratio = height / width if width > 0 else 0
        min_ratio = self.config['filtering']['min_aspect_ratio']
        max_ratio = self.config['filtering']['max_aspect_ratio']

        if self.config['filtering']['seated_person_mode']:
            # Pessoas sentadas tem aspect ratio menor (0.8 - 3.0)
            if aspect_ratio < min_ratio:
                return False, f"muito largo (AR={aspect_ratio:.2f} < {min_ratio})"
            if aspect_ratio > max_ratio:
                return False, f"muito alto (AR={aspect_ratio:.2f} > {max_ratio})"
        else:
            # Pessoas em pé tem aspect ratio maior (1.5 - 4.0)
            if aspect_ratio < 1.5 or aspect_ratio > 4.0:
                return False, f"aspect ratio inválido ({aspect_ratio:.2f})"

        # Rejeitar detecções muito grandes (provavelmente múltiplas pessoas)
        max_area_ratio = 0.4  # 40% do frame
        if area > (frame_height * frame_width * max_area_ratio):
            return False, f"área muito grande ({area / (frame_height * frame_width):.1%})"

        return True, ""

    def connect_camera(self) -> bool:
        """Conecta à câmera RTSP"""
        rtsp_url = self.config['camera']['rtsp_url']
        camera_ip = rtsp_url.split('@')[1].split(':')[0] if '@' in rtsp_url else 'unknown'

        self.logger.info(f"Tentando conectar à câmera: {camera_ip}")

        self.cap = cv2.VideoCapture(rtsp_url)

        if not self.cap.isOpened():
            self.logger.error("Falha ao conectar à câmera")
            return False

        # Configurar propriedades
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config['camera']['buffer_size'])

        # Obter informações do stream
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.logger.info(f"✓ Conectado! Resolução: {width}x{height}, FPS: {fps}")

        return True

    def load_model(self) -> bool:
        """Carrega modelo YOLOv8 otimizado"""
        model_name = self.config['detection']['model']
        self.logger.info(f"Carregando modelo {model_name}...")

        try:
            self.model = YOLO(model_name)
            device = self.config['detection']['device']
            self.logger.info(f"✓ Modelo carregado em {device}")

            # Configurar modelo
            if hasattr(self.model, 'overrides'):
                self.model.overrides['conf'] = self.config['detection']['confidence_threshold']
                self.model.overrides['iou'] = self.config['detection']['iou_threshold']
                self.model.overrides['max_det'] = self.config['detection']['max_detections']

            return True

        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {e}")
            return False

    def initialize_tracker(self) -> bool:
        """Inicializa tracker ByteTrack otimizado"""
        if not self.config['tracking']['enabled']:
            self.logger.info("Tracking desabilitado")
            return True

        try:
            # ByteTrack com parâmetros otimizados
            self.tracker = sv.ByteTrack(
                track_activation_threshold=self.config['tracking']['track_high_thresh'],
                lost_track_buffer=self.config['tracking']['track_buffer'],
                minimum_matching_threshold=self.config['tracking']['match_thresh'],
                frame_rate=self.config['tracking']['frame_rate']
            )

            self.logger.info("✓ Tracker ByteTrack inicializado (otimizado)")
            return True

        except Exception as e:
            self.logger.error(f"Erro ao inicializar tracker: {e}")
            return False

    def detect_people(self, frame: np.ndarray) -> sv.Detections:
        """
        Detecta pessoas no frame com configurações otimizadas

        Args:
            frame: Frame a processar

        Returns:
            Detecções filtradas
        """
        # Detectar com YOLOv8
        results = self.model.track(
            frame,
            persist=True,
            classes=[self.config['detection']['person_class_id']],
            conf=self.config['detection']['confidence_threshold'],
            iou=self.config['detection']['iou_threshold'],
            max_det=self.config['detection']['max_detections'],
            augment=self.config['detection']['augment'],
            half=self.config['detection']['half_precision'],
            verbose=False
        )

        # Converter para formato supervision
        detections = sv.Detections.from_ultralytics(results[0])

        # Log de debug
        if self.config['logging'].get('debug_mode'):
            self.logger.debug(f"Raw detections: {len(detections)}")

        return detections

    def filter_detections(self, detections: sv.Detections,
                         frame_shape: Tuple) -> sv.Detections:
        """
        Filtra detecções inválidas

        Args:
            detections: Detecções brutas
            frame_shape: (height, width) do frame

        Returns:
            Detecções válidas
        """
        if not self.config['filtering']['enable'] or len(detections) == 0:
            return detections

        valid_indices = []
        self.rejected_detections = []

        for i, (box, confidence) in enumerate(zip(detections.xyxy, detections.confidence)):
            is_valid, reason = self.is_valid_detection(box, confidence, frame_shape)

            if is_valid:
                valid_indices.append(i)
            else:
                if self.config['logging'].get('debug_mode'):
                    self.logger.debug(f"Rejected: conf={confidence:.2f}, reason={reason}")
                self.rejected_detections.append((box, confidence, reason))

        # Filtrar detecções
        if len(valid_indices) > 0:
            filtered = sv.Detections(
                xyxy=detections.xyxy[valid_indices],
                confidence=detections.confidence[valid_indices],
                class_id=detections.class_id[valid_indices],
                tracker_id=detections.tracker_id[valid_indices] if detections.tracker_id is not None else None
            )
        else:
            filtered = sv.Detections.empty()

        if self.config['logging'].get('debug_mode'):
            self.logger.debug(f"After filtering: {len(filtered)} valid, {len(self.rejected_detections)} rejected")

        return filtered

    def draw_detections(self, frame: np.ndarray, detections: sv.Detections,
                       roi_offset: int = 0) -> np.ndarray:
        """
        Desenha caixas delimitadoras e informações

        Args:
            frame: Frame original
            detections: Detecções válidas
            roi_offset: Offset Y do ROI

        Returns:
            Frame anotado
        """
        annotated = frame.copy()

        # Desenhar detecções aceitas (verde)
        for i, (bbox, confidence, class_id, tracker_id) in enumerate(
            zip(detections.xyxy, detections.confidence, detections.class_id,
                detections.tracker_id if detections.tracker_id is not None else [None]*len(detections))
        ):
            x1, y1, x2, y2 = map(int, bbox)

            # Ajustar coordenadas para ROI
            y1 += roi_offset
            y2 += roi_offset

            # Desenhar caixa
            cv2.rectangle(annotated, (x1, y1), (x2, y2), self.COLOR_GREEN, 2)

            # Preparar label
            label_parts = []
            if tracker_id is not None and self.config['display']['show_tracker_id']:
                label_parts.append(f"ID:{tracker_id}")
            if self.config['display']['show_confidence']:
                label_parts.append(f"{confidence:.2f}")
            label = " ".join(label_parts)

            # Desenhar label
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width + 5, y1),
                self.COLOR_GREEN,
                -1
            )
            cv2.putText(
                annotated, label, (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_BLACK, 1
            )

        # Desenhar detecções rejeitadas (vermelho) - modo debug
        if self.config['display'].get('show_rejected_detections'):
            for bbox, confidence, reason in self.rejected_detections:
                x1, y1, x2, y2 = map(int, bbox)
                y1 += roi_offset
                y2 += roi_offset

                cv2.rectangle(annotated, (x1, y1), (x2, y2), self.COLOR_RED, 1)
                cv2.putText(
                    annotated, f"{confidence:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_RED, 1
                )

        return annotated

    def draw_info_panel(self, frame: np.ndarray, count: int, fps: float) -> np.ndarray:
        """
        Desenha painel de informações otimizado

        Args:
            frame: Frame a anotar
            count: Contagem atual
            fps: FPS atual

        Returns:
            Frame com painel
        """
        height, width = frame.shape[:2]

        # Painel superior com transparência
        panel_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), self.COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Título principal - PESSOAS DETECTADAS
        main_text = f"PESSOAS DETECTADAS: {count}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.3
        thickness = 2

        text_size = cv2.getTextSize(main_text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = 40

        # Cor baseada em faixa esperada
        expected_range = self.config['room'].get('expected_count_range', [15, 25])
        if count < expected_range[0]:
            text_color = self.COLOR_ORANGE  # Abaixo do esperado
        elif count > expected_range[1]:
            text_color = self.COLOR_RED  # Acima do esperado
        else:
            text_color = self.COLOR_GREEN  # Dentro do esperado

        # Desenhar texto com sombra
        cv2.putText(frame, main_text, (text_x + 2, text_y + 2),
                   font, font_scale, self.COLOR_BLACK, thickness + 1)
        cv2.putText(frame, main_text, (text_x, text_y),
                   font, font_scale, text_color, thickness)

        # Informações secundárias
        info_y = 75
        info_texts = [
            f"Maximo: {self.max_count}",
            f"FPS: {fps:.1f}",
            f"Frames: {self.frame_count}",
            f"Rejeitados: {len(self.rejected_detections)}"
        ]

        total_width = sum(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for t in info_texts) + 60
        start_x = (width - total_width) // 2

        for text in info_texts:
            cv2.putText(frame, text, (start_x, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
            text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            start_x += text_width + 20

        # Indicador de faixa esperada
        expected_text = f"Esperado: {expected_range[0]}-{expected_range[1]} pessoas"
        cv2.putText(frame, expected_text, (10, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_YELLOW, 1)

        # Modelo usado
        model_text = f"Modelo: {self.config['detection']['model']} | Conf: {self.config['detection']['confidence_threshold']}"
        model_size = cv2.getTextSize(model_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.putText(frame, model_text, (width - model_size[0] - 10, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_WHITE, 1)

        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)

        # ROI indicator
        if self.config['roi']['enable']:
            roi_text = f"ROI: Top {self.config['roi']['top_margin']:.0%}"
            cv2.putText(frame, roi_text, (width - 150, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_YELLOW, 1)

        return frame

    def process_frame(self) -> Optional[np.ndarray]:
        """
        Processa um frame completo com todas as otimizações

        Returns:
            Frame anotado ou None se erro
        """
        # Capturar frame
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Redimensionar para performance
        resize_width = self.config['performance']['resize_width']
        resize_height = self.config['performance']['resize_height']
        frame = cv2.resize(frame, (resize_width, resize_height))

        # Aplicar ROI
        roi_frame, roi_offset = self.apply_roi(frame)

        # Pré-processamento
        enhanced_frame = self.enhance_frame(roi_frame)

        # Detectar pessoas
        detections = self.detect_people(enhanced_frame)

        # Filtrar detecções
        filtered_detections = self.filter_detections(
            detections,
            (enhanced_frame.shape[0], enhanced_frame.shape[1])
        )

        # Aplicar tracking
        if self.config['tracking']['enabled'] and len(filtered_detections) > 0:
            filtered_detections = self.tracker.update_with_detections(filtered_detections)

        # Atualizar contagem
        self.current_count = len(filtered_detections)
        if self.current_count > self.max_count:
            self.max_count = self.current_count
            self.logger.info(f"⬆ Novo máximo: {self.max_count} pessoas")

        self.total_detections += self.current_count
        self.frame_count += 1

        # Calcular FPS
        if self.start_time:
            elapsed = time.time() - self.start_time
            current_fps = self.frame_count / elapsed if elapsed > 0 else 0
            self.fps_history.append(current_fps)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            display_fps = sum(self.fps_history) / len(self.fps_history)
        else:
            display_fps = 0.0

        # Desenhar anotações no frame ORIGINAL (não no ROI)
        annotated_frame = self.draw_detections(frame, filtered_detections, roi_offset)
        annotated_frame = self.draw_info_panel(annotated_frame, self.current_count, display_fps)

        # Desenhar linha de ROI
        if self.config['roi']['enable']:
            cv2.line(annotated_frame, (0, roi_offset), (frame.shape[1], roi_offset),
                    self.COLOR_YELLOW, 2)

        return annotated_frame

    def run(self):
        """Loop principal otimizado"""
        print("\n" + "="*60)
        print(" SISTEMA OTIMIZADO DE CONTAGEM DE PESSOAS")
        print("="*60)
        print(f"Modelo: {self.config['detection']['model']}")
        print(f"Confidence: {self.config['detection']['confidence_threshold']}")
        print(f"IOU: {self.config['detection']['iou_threshold']}")
        print(f"ROI: {'Habilitado' if self.config['roi']['enable'] else 'Desabilitado'}")
        print(f"Pré-processamento: {'Habilitado' if self.config['preprocessing']['enable'] else 'Desabilitado'}")
        print("="*60 + "\n")

        # Inicializar componentes
        if not self.connect_camera():
            return

        if not self.load_model():
            return

        if not self.initialize_tracker():
            return

        print("✓ Sistema inicializado com sucesso!\n")
        print("Controles:")
        print("  [Q] - Sair")
        print("  [S] - Salvar estatísticas")
        print("  [Ctrl+C] - Interromper\n")
        print("="*60 + "\n")

        self.running = True
        self.start_time = time.time()

        try:
            while self.running:
                frame = self.process_frame()

                if frame is None:
                    self.logger.warning("Falha ao capturar frame, tentando reconectar...")
                    time.sleep(self.config['camera']['reconnect_delay'])
                    if not self.connect_camera():
                        break
                    continue

                # Mostrar frame se habilitado
                if self.config['display']['show_video']:
                    cv2.imshow(self.config['display']['window_name'], frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        break
                    elif key == ord('s') or key == ord('S'):
                        self.save_statistics()

                # Logging periódico
                if self.frame_count % (self.config['logging']['interval_seconds'] * 20) == 0:
                    self.logger.info(
                        f"Pessoas: {self.current_count} | Máximo: {self.max_count} | "
                        f"Frames: {self.frame_count} | FPS: {sum(self.fps_history)/len(self.fps_history):.1f}"
                    )

        except KeyboardInterrupt:
            print("\n\n✓ Interrompido pelo usuário")

        except Exception as e:
            self.logger.error(f"Erro durante execução: {e}", exc_info=True)

        finally:
            self.stop()

    def save_statistics(self):
        """Salva estatísticas detalhadas"""
        if not self.config['logging']['save_statistics']:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_file = Path(self.config['logging']['log_directory']) / f"stats_optimized_{timestamp}.json"

        elapsed = time.time() - self.start_time if self.start_time else 0
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0

        stats = {
            'timestamp': timestamp,
            'duration_seconds': round(elapsed, 2),
            'frames_processed': self.frame_count,
            'current_count': self.current_count,
            'max_count': self.max_count,
            'total_detections': self.total_detections,
            'average_fps': round(avg_fps, 2),
            'config': {
                'model': self.config['detection']['model'],
                'confidence_threshold': self.config['detection']['confidence_threshold'],
                'iou_threshold': self.config['detection']['iou_threshold'],
                'roi_enabled': self.config['roi']['enable'],
                'preprocessing_enabled': self.config['preprocessing']['enable']
            }
        }

        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            self.logger.info(f"✓ Estatísticas salvas em {stats_file}")
            print(f"\n✓ Estatísticas salvas em {stats_file}\n")
        except Exception as e:
            self.logger.error(f"Erro ao salvar estatísticas: {e}")

    def stop(self):
        """Para o sistema e libera recursos"""
        self.logger.info("Encerrando sistema...")
        self.running = False

        if self.config['logging']['save_statistics']:
            self.save_statistics()

        if self.cap:
            self.cap.release()

        if self.config['display']['show_video']:
            cv2.destroyAllWindows()

        # Resumo final
        elapsed = time.time() - self.start_time if self.start_time else 0
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0

        print("\n" + "="*60)
        print(" RESUMO DA SESSÃO")
        print("="*60)
        print(f"Tempo de execução: {elapsed/60:.1f} minutos")
        print(f"Frames processados: {self.frame_count}")
        print(f"Contagem atual: {self.current_count} pessoas")
        print(f"Máximo detectado: {self.max_count} pessoas")
        print(f"FPS médio: {avg_fps:.1f}")
        print("="*60 + "\n")

        print("✓ Sistema encerrado com sucesso!\n")


def main():
    """Função principal"""
    import argparse

    parser = argparse.ArgumentParser(description='Sistema Otimizado de Contagem de Pessoas')
    parser.add_argument('--config', type=str, default='config_optimized.json',
                       help='Caminho para arquivo de configuração')

    args = parser.parse_args()

    counter = OptimizedPeopleCounter(config_path=args.config)
    counter.run()


if __name__ == '__main__':
    main()
