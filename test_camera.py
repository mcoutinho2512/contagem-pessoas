#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Teste de Conexão RTSP
Testa a conexão com a câmera IP e exibe informações do stream
Autor: Sistema desenvolvido via Claude Code
"""

import cv2
import json
import sys
import time
from datetime import datetime


def load_config(config_path="config.json"):
    """Carrega configurações do arquivo JSON"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"✗ Erro: Arquivo {config_path} não encontrado")
        return None
    except json.JSONDecodeError as e:
        print(f"✗ Erro ao decodificar JSON: {e}")
        return None


def test_camera_connection(rtsp_url, duration=10):
    """
    Testa conexão com a câmera RTSP

    Args:
        rtsp_url: URL RTSP da câmera
        duration: Duração do teste em segundos
    """
    print("\n" + "="*70)
    print(" TESTE DE CONEXÃO RTSP - CÂMERA IP CCTV")
    print("="*70)

    # Ocultar senha na exibição
    display_url = rtsp_url
    if '@' in rtsp_url:
        parts = rtsp_url.split('@')
        user_pass = parts[0].split('//')[1]
        username = user_pass.split(':')[0]
        display_url = rtsp_url.replace(user_pass, f"{username}:****")

    print(f"\nCâmera: {display_url}")
    print(f"Duração do teste: {duration} segundos")
    print("\n" + "-"*70)

    # Tentar conectar
    print("\n[1/4] Iniciando conexão...")

    try:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            print("✗ FALHA: Não foi possível abrir o stream RTSP")
            print("\nPossíveis causas:")
            print("  • URL RTSP incorreta")
            print("  • Credenciais inválidas (usuário/senha)")
            print("  • Câmera offline ou inacessível")
            print("  • Firewall bloqueando conexão")
            print("  • Formato RTSP não suportado")
            return False

        print("✓ Conexão estabelecida")

        # Ler primeiro frame
        print("\n[2/4] Testando captura de frame...")
        ret, frame = cap.read()

        if not ret or frame is None:
            print("✗ FALHA: Não foi possível capturar frame")
            cap.release()
            return False

        print("✓ Frame capturado com sucesso")

        # Obter informações do stream
        print("\n[3/4] Obtendo informações do stream...")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))

        # Decodificar codec
        codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])

        print(f"\n  Resolução: {width} x {height} pixels")
        print(f"  FPS (nominal): {fps:.2f}")
        print(f"  Codec: {codec_str}")
        print(f"  Tamanho do frame: {frame.nbytes / 1024:.2f} KB")

        # Teste de streaming
        print(f"\n[4/4] Testando streaming por {duration} segundos...")
        print("\nPressione 'Q' para sair antes do tempo\n")

        start_time = time.time()
        frame_count = 0
        error_count = 0

        cv2.namedWindow('Teste RTSP - Pressione Q para sair', cv2.WINDOW_NORMAL)

        while True:
            elapsed = time.time() - start_time

            if elapsed >= duration:
                break

            ret, frame = cap.read()

            if not ret or frame is None:
                error_count += 1
                print(f"✗ Erro ao capturar frame #{frame_count + 1}")

                if error_count > 10:
                    print("\n✗ FALHA: Muitos erros consecutivos. Stream instável.")
                    break

                time.sleep(0.1)
                continue

            frame_count += 1
            error_count = 0  # Reset contador de erros

            # Calcular FPS real
            actual_fps = frame_count / elapsed if elapsed > 0 else 0

            # Adicionar informações no frame
            info_text = [
                f"Tempo: {elapsed:.1f}s / {duration}s",
                f"Frames: {frame_count}",
                f"FPS real: {actual_fps:.1f}",
                f"Resolucao: {width}x{height}",
                f"Pressione Q para sair"
            ]

            y_offset = 30
            for i, text in enumerate(info_text):
                cv2.putText(frame, text, (10, y_offset + i*30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Exibir frame
            cv2.imshow('Teste RTSP - Pressione Q para sair', frame)

            # Verificar tecla
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n⚠ Teste interrompido pelo usuário")
                break

        # Limpar
        cap.release()
        cv2.destroyAllWindows()

        # Resultados
        actual_fps = frame_count / elapsed if elapsed > 0 else 0

        print("\n" + "-"*70)
        print(" RESULTADOS DO TESTE")
        print("-"*70)
        print(f"Duração: {elapsed:.2f} segundos")
        print(f"Frames capturados: {frame_count}")
        print(f"FPS médio: {actual_fps:.2f}")
        print(f"Erros: {error_count}")

        if actual_fps < 5:
            print("\n⚠ AVISO: FPS muito baixo. Possíveis causas:")
            print("  • Conexão de rede lenta")
            print("  • Câmera sobrecarregada")
            print("  • Configuração de bitrate muito alta")
        elif actual_fps < fps * 0.7:
            print(f"\n⚠ AVISO: FPS real ({actual_fps:.1f}) está abaixo do nominal ({fps:.1f})")
        else:
            print("\n✓ Stream estável e funcionando corretamente")

        print("\n" + "="*70)
        print(" TESTE CONCLUÍDO COM SUCESSO")
        print("="*70 + "\n")

        return True

    except KeyboardInterrupt:
        print("\n\n⚠ Teste interrompido pelo usuário (Ctrl+C)")
        return False

    except Exception as e:
        print(f"\n✗ ERRO durante o teste: {e}")
        print("\nDetalhes técnicos:")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Função principal"""
    # Carregar configuração
    config = load_config()

    if config is None:
        print("\n✗ Não foi possível carregar configurações")
        print("✓ Certifique-se de que config.json existe no diretório atual")
        sys.exit(1)

    # Obter URL RTSP
    rtsp_url = config.get('camera', {}).get('rtsp_url')

    if not rtsp_url:
        print("✗ URL RTSP não encontrada no arquivo de configuração")
        sys.exit(1)

    # Duração do teste (padrão: 10 segundos)
    duration = 10
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except ValueError:
            print(f"⚠ Duração inválida, usando padrão: {duration}s")

    # Executar teste
    success = test_camera_connection(rtsp_url, duration)

    if success:
        print("✓ A câmera está pronta para uso com o sistema de contagem!")
        print("\nPróximo passo: Execute 'python people_counter.py' para iniciar\n")
    else:
        print("\n✗ Teste falhou. Verifique:")
        print("  1. URL RTSP no config.json está correta")
        print("  2. Credenciais (usuário e senha) estão corretas")
        print("  3. Câmera está ligada e acessível na rede")
        print("  4. Não há firewall bloqueando a conexão")
        print("  5. FFmpeg está instalado corretamente\n")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
