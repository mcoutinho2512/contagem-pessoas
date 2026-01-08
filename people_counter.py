#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Contagem de Pessoas em Tempo Real
Monitora sala de reunião através de câmera IP CCTV usando YOLOv8
Autor: Sistema desenvolvido via Claude Code
Data: 2025-01-07
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


class PeopleCounter:
    """
    Classe principal para contagem de pessoas em tempo real
    usando detecção por YOLOv8 e rastreamento de objetos
    """

    def __init__(self, config_path: str = "config.json"):
        """
        Inicializa o contador de pessoas

        Args:
            config_path: Caminho para o arquivo de configuração JSON
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
        self.total_frames = 0
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.fps_list = []

        # Cores para interface (BGR)
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_YELLOW = (0, 255, 255)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_BLACK = (0, 0, 0)

        logging.info("PeopleCounter inicializado com sucesso")

    def _load_config(self, config_path: str) -> Dict:
        """Carrega configurações do arquivo JSON"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✓ Configurações carregadas de {config_path}")
            return config
        except FileNotFoundError:
            print(f"✗ Erro: Arquivo {config_path} não encontrado")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"✗ Erro ao decodificar JSON: {e}")
            sys.exit(1)

    def _setup_logging(self):
        """Configura o sistema de logging"""
        log_dir = Path(self.config['logging']['log_directory'])
        log_dir.mkdir(exist_ok=True)

        # Configurar logging para console e arquivo
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )

    def connect_camera(self) -> bool:
        """
        Conecta à câmera RTSP com reconexão automática

        Returns:
            True se conectado com sucesso, False caso contrário
        """
        rtsp_url = self.config['camera']['rtsp_url']
        timeout = self.config['camera']['timeout']

        logging.info(f"Tentando conectar à câmera: {rtsp_url.split('@')[1] if '@' in rtsp_url else rtsp_url}")

        try:
            # Criar captura de vídeo
            self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

            # Configurar buffer reduzido para menor latência
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config['camera']['buffer_size'])

            # Tentar ler um frame para verificar conexão
            ret, frame = self.cap.read()

            if ret and frame is not None:
                height, width = frame.shape[:2]
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                logging.info(f"✓ Conectado! Resolução: {width}x{height}, FPS: {fps:.1f}")
                return True
            else:
                logging.error("✗ Falha ao ler frame da câmera")
                if self.cap:
                    self.cap.release()
                return False

        except Exception as e:
            logging.error(f"✗ Erro ao conectar à câmera: {e}")
            return False

    def reconnect_camera(self):
        """Tenta reconectar à câmera em caso de falha"""
        logging.warning("Tentando reconectar à câmera...")

        if self.cap:
            self.cap.release()

        delay = self.config['camera']['reconnect_delay']

        while self.running:
            time.sleep(delay)

            if self.connect_camera():
                logging.info("✓ Reconexão bem-sucedida!")
                return True

            logging.warning(f"Tentando novamente em {delay} segundos...")

        return False

    def load_model(self):
        """Carrega o modelo YOLOv8 para detecção"""
        model_name = self.config['detection']['model']
        device = self.config['detection']['device']

        logging.info(f"Carregando modelo {model_name}...")

        try:
            self.model = YOLO(model_name)
            logging.info(f"✓ Modelo carregado em {device}")

            # Configurar tracker do supervision
            self.tracker = sv.ByteTrack()
            logging.info("✓ Tracker ByteTrack inicializado")

        except Exception as e:
            logging.error(f"✗ Erro ao carregar modelo: {e}")
            sys.exit(1)

    def process_detections(self, frame: np.ndarray) -> Tuple[int, np.ndarray, List]:
        """
        Processa detecções de pessoas no frame

        Args:
            frame: Frame de vídeo para processar

        Returns:
            Tupla com (número de pessoas, frame anotado, lista de detecções)
        """
        # Executar detecção
        results = self.model(
            frame,
            conf=self.config['detection']['confidence_threshold'],
            classes=[self.config['detection']['person_class_id']],  # Apenas classe pessoa
            verbose=False,
            device=self.config['detection']['device']
        )[0]

        # Converter para formato supervision
        detections = sv.Detections.from_ultralytics(results)

        # Aplicar rastreamento se habilitado
        if self.config['tracking']['enabled']:
            detections = self.tracker.update_with_detections(detections)

        # Contar pessoas detectadas
        people_count = len(detections)

        # Desenhar bounding boxes se configurado
        if self.config['display']['show_bounding_boxes']:
            # Criar anotadores
            box_annotator = sv.BoxAnnotator(
                thickness=2,
                color=sv.Color.from_hex("#00FF00")  # Verde
            )

            label_annotator = sv.LabelAnnotator(
                text_thickness=1,
                text_scale=0.5,
                text_color=sv.Color.from_hex("#FFFFFF")
            )

            # Anotar frame
            frame = box_annotator.annotate(scene=frame.copy(), detections=detections)

            # Adicionar labels com confiança se configurado
            if self.config['display']['show_confidence'] and len(detections) > 0:
                labels = [
                    f"ID:{tracker_id} {confidence:.2f}"
                    for confidence, tracker_id in zip(detections.confidence, detections.tracker_id)
                ]
                frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        return people_count, frame, detections

    def draw_info_panel(self, frame: np.ndarray, count: int, fps: float) -> np.ndarray:
        """
        Desenha painel de informações no frame

        Args:
            frame: Frame de vídeo
            count: Número atual de pessoas
            fps: Taxa de quadros por segundo

        Returns:
            Frame com painel de informações
        """
        height, width = frame.shape[:2]

        # Calcular percentual de ocupação
        max_capacity = self.config['room']['max_capacity']
        occupancy_percent = (count / max_capacity * 100) if max_capacity > 0 else 0

        # Determinar cor baseada na ocupação
        alert_threshold = self.config['room']['alert_threshold']
        if occupancy_percent >= (alert_threshold * 100):
            status_color = self.COLOR_RED
            status_text = "ALERTA: CAPACIDADE ALTA!"
        elif occupancy_percent >= 60:
            status_color = self.COLOR_YELLOW
            status_text = "Ocupação Moderada"
        else:
            status_color = self.COLOR_GREEN
            status_text = "Ocupação Normal"

        # Criar painel semi-transparente
        panel_height = 200
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), self.COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Texto principal - PESSOAS NA SALA
        main_text = f"PESSOAS NA SALA: {count}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.5
        thickness = 3

        text_size = cv2.getTextSize(main_text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = 50

        # Sombra do texto
        cv2.putText(frame, main_text, (text_x + 2, text_y + 2), font, font_scale, self.COLOR_BLACK, thickness + 2)
        # Texto principal
        cv2.putText(frame, main_text, (text_x, text_y), font, font_scale, self.COLOR_WHITE, thickness)

        # Informações adicionais
        info_y = 90
        info_font_scale = 0.6
        info_thickness = 2

        info_lines = [
            f"Maximo da Sessao: {self.max_count}",
            f"Capacidade Total: {max_capacity}",
            f"Ocupacao: {occupancy_percent:.1f}%",
            f"Status: {status_text}",
            f"FPS: {fps:.1f}"
        ]

        for i, line in enumerate(info_lines):
            y_pos = info_y + (i * 22)

            # Cor especial para linha de status
            text_color = status_color if i == 3 else self.COLOR_WHITE

            cv2.putText(frame, line, (12, y_pos + 2), cv2.FONT_HERSHEY_SIMPLEX,
                       info_font_scale, self.COLOR_BLACK, info_thickness + 1)
            cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                       info_font_scale, text_color, info_thickness)

        # Alerta visual se acima do threshold
        if occupancy_percent >= (alert_threshold * 100):
            # Borda vermelha piscante
            border_thickness = 10
            if int(time.time() * 2) % 2 == 0:  # Pisca a cada 0.5s
                cv2.rectangle(frame, (0, 0), (width - 1, height - 1), self.COLOR_RED, border_thickness)

        return frame

    def log_count(self, count: int):
        """
        Registra contagem em arquivo de log

        Args:
            count: Número de pessoas a registrar
        """
        if not self.config['logging']['enabled']:
            return

        # Nome do arquivo baseado na data
        log_dir = Path(self.config['logging']['log_directory'])
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"count_log_{date_str}.txt"

        # Formatar linha de log
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"{timestamp} | Pessoas: {count:2d} | Máximo: {self.max_count:2d}\n"

        # Escrever no arquivo
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_line)
            logging.info(f"Log registrado: {count} pessoas")
        except Exception as e:
            logging.error(f"Erro ao escrever log: {e}")

    def save_statistics(self):
        """Salva estatísticas da sessão em arquivo JSON"""
        if not self.config['logging']['save_statistics']:
            return

        # Calcular estatísticas
        uptime = time.time() - self.start_time
        avg_fps = np.mean(self.fps_list) if self.fps_list else 0

        stats = {
            "timestamp": datetime.now().isoformat(),
            "session_start": datetime.fromtimestamp(self.start_time).isoformat(),
            "total_frames": self.total_frames,
            "max_people": self.max_count,
            "current_people": self.current_count,
            "avg_fps": round(avg_fps, 2),
            "uptime_seconds": round(uptime, 2),
            "uptime_formatted": self._format_uptime(uptime),
            "camera_url": self.config['camera']['rtsp_url'].split('@')[1] if '@' in self.config['camera']['rtsp_url'] else "hidden"
        }

        # Nome do arquivo com timestamp
        log_dir = Path(self.config['logging']['log_directory'])
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_file = log_dir / f"statistics_{timestamp_str}.json"

        # Salvar JSON
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            logging.info(f"✓ Estatísticas salvas em {stats_file}")
            print(f"\n✓ Estatísticas salvas em {stats_file}")
        except Exception as e:
            logging.error(f"Erro ao salvar estatísticas: {e}")

    def _format_uptime(self, seconds: float) -> str:
        """Formata tempo de execução em formato legível"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def run(self):
        """Loop principal do sistema"""
        print("\n" + "="*60)
        print(" SISTEMA DE CONTAGEM DE PESSOAS EM TEMPO REAL")
        print("="*60)

        # Conectar à câmera
        if not self.connect_camera():
            logging.error("Não foi possível conectar à câmera. Encerrando.")
            return

        # Carregar modelo
        self.load_model()

        # Iniciar sistema
        self.running = True
        frame_count = 0
        fps_start_time = time.time()

        print("\n✓ Sistema iniciado com sucesso!")
        print("\nControles:")
        print("  [Q] - Sair")
        print("  [S] - Salvar estatísticas")
        print("\n" + "="*60 + "\n")

        try:
            while self.running:
                # Ler frame
                ret, frame = self.cap.read()

                if not ret or frame is None:
                    logging.warning("Falha ao ler frame. Tentando reconectar...")
                    if not self.reconnect_camera():
                        break
                    continue

                # Redimensionar se configurado
                target_width = self.config['performance']['resize_width']
                target_height = self.config['performance']['resize_height']

                if target_width > 0 and target_height > 0:
                    frame = cv2.resize(frame, (target_width, target_height))

                # Processar detecções
                self.current_count, annotated_frame, detections = self.process_detections(frame)

                # Atualizar máximo
                if self.current_count > self.max_count:
                    self.max_count = self.current_count
                    logging.info(f"⬆ Novo máximo: {self.max_count} pessoas")

                # Calcular FPS
                frame_count += 1
                self.total_frames += 1

                if frame_count >= 30:
                    elapsed = time.time() - fps_start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    self.fps_list.append(current_fps)

                    # Manter apenas últimas 100 medições
                    if len(self.fps_list) > 100:
                        self.fps_list.pop(0)

                    frame_count = 0
                    fps_start_time = time.time()

                # FPS para display
                display_fps = np.mean(self.fps_list[-10:]) if self.fps_list else 0

                # Desenhar painel de informações
                if self.config['display']['show_info_panel']:
                    annotated_frame = self.draw_info_panel(annotated_frame, self.current_count, display_fps)

                # Mostrar vídeo se configurado
                if self.config['display']['show_video']:
                    cv2.imshow(self.config['display']['window_name'], annotated_frame)

                # Logging periódico
                current_time = time.time()
                log_interval = self.config['logging']['interval_seconds']

                if current_time - self.last_log_time >= log_interval:
                    self.log_count(self.current_count)
                    self.last_log_time = current_time

                # Processar teclas
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == ord('Q'):
                    logging.info("Saída solicitada pelo usuário")
                    break
                elif key == ord('s') or key == ord('S'):
                    self.save_statistics()

        except KeyboardInterrupt:
            logging.info("\nInterrompido pelo usuário (Ctrl+C)")

        except Exception as e:
            logging.error(f"Erro durante execução: {e}", exc_info=True)

        finally:
            self.cleanup()

    def cleanup(self):
        """Limpa recursos e encerra sistema"""
        logging.info("Encerrando sistema...")

        self.running = False

        # Salvar estatísticas finais
        self.save_statistics()

        # Liberar recursos
        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()

        # Resumo final
        uptime = time.time() - self.start_time
        print("\n" + "="*60)
        print(" RESUMO DA SESSÃO")
        print("="*60)
        print(f"Tempo de execução: {self._format_uptime(uptime)}")
        print(f"Total de frames processados: {self.total_frames}")
        print(f"Máximo de pessoas detectadas: {self.max_count}")
        print(f"FPS médio: {np.mean(self.fps_list):.2f}" if self.fps_list else "FPS médio: N/A")
        print("="*60)
        print("\n✓ Sistema encerrado com sucesso!\n")


def main():
    """Função principal"""
    # Verificar se arquivo de configuração existe
    config_file = "config.json"

    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if not os.path.exists(config_file):
        print(f"✗ Erro: Arquivo de configuração '{config_file}' não encontrado")
        print(f"✓ Uso: python {sys.argv[0]} [config.json]")
        sys.exit(1)

    # Criar e executar contador
    counter = PeopleCounter(config_file)
    counter.run()


if __name__ == "__main__":
    main()
