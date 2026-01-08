#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizador de Estatísticas - Sistema de Contagem de Pessoas
Interface interativa para análise de logs e estatísticas
Autor: Sistema desenvolvido via Claude Code
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
import csv


class StatsViewer:
    """Visualizador interativo de estatísticas"""

    def __init__(self, log_dir="logs"):
        """
        Inicializa o visualizador

        Args:
            log_dir: Diretório contendo os logs
        """
        self.log_dir = Path(log_dir)

        if not self.log_dir.exists():
            print(f"✗ Diretório de logs '{log_dir}' não encontrado")
            sys.exit(1)

    def list_log_files(self) -> List[Path]:
        """Lista todos os arquivos de log de contagem"""
        return sorted(self.log_dir.glob("count_log_*.txt"))

    def list_stats_files(self) -> List[Path]:
        """Lista todos os arquivos de estatísticas JSON"""
        return sorted(self.log_dir.glob("statistics_*.json"))

    def parse_log_file(self, log_file: Path) -> List[Dict]:
        """
        Parse arquivo de log de contagem

        Args:
            log_file: Caminho do arquivo de log

        Returns:
            Lista de dicionários com os dados
        """
        data = []

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Formato: 2025-01-07 14:30:00 | Pessoas: 12 | Máximo: 15
                    parts = line.split('|')

                    if len(parts) >= 3:
                        timestamp_str = parts[0].strip()
                        people_str = parts[1].strip()
                        max_str = parts[2].strip()

                        # Extrair números
                        people = int(people_str.split(':')[1].strip())
                        max_people = int(max_str.split(':')[1].strip())

                        # Parse timestamp
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

                        data.append({
                            'timestamp': timestamp,
                            'people': people,
                            'max': max_people
                        })

        except Exception as e:
            print(f"✗ Erro ao ler arquivo {log_file}: {e}")

        return data

    def show_daily_summary(self, log_file: Path):
        """Mostra resumo do dia"""
        print("\n" + "="*70)
        print(f" RESUMO DO DIA - {log_file.name}")
        print("="*70)

        data = self.parse_log_file(log_file)

        if not data:
            print("\n✗ Nenhum dado encontrado no arquivo")
            return

        # Estatísticas
        people_counts = [d['people'] for d in data]
        max_overall = max(d['max'] for d in data)
        avg_people = sum(people_counts) / len(people_counts)
        min_people = min(people_counts)
        max_people = max(people_counts)

        # Período
        start_time = data[0]['timestamp']
        end_time = data[-1]['timestamp']

        print(f"\nPeríodo: {start_time.strftime('%Y-%m-%d %H:%M')} até {end_time.strftime('%H:%M')}")
        print(f"Total de registros: {len(data)}")

        print("\n" + "-"*70)
        print(" ESTATÍSTICAS DE OCUPAÇÃO")
        print("-"*70)
        print(f"  Média de pessoas: {avg_people:.1f}")
        print(f"  Mínimo: {min_people}")
        print(f"  Máximo: {max_people}")
        print(f"  Máximo absoluto da sessão: {max_overall}")

        # Encontrar picos
        print("\n" + "-"*70)
        print(" HORÁRIOS DE PICO")
        print("-"*70)

        peak_records = sorted(data, key=lambda x: x['people'], reverse=True)[:5]

        for i, record in enumerate(peak_records, 1):
            time_str = record['timestamp'].strftime('%H:%M:%S')
            print(f"  {i}. {time_str} - {record['people']} pessoas")

        # Análise por hora
        print("\n" + "-"*70)
        print(" OCUPAÇÃO POR HORA")
        print("-"*70)

        hourly_data = defaultdict(list)
        for record in data:
            hour = record['timestamp'].hour
            hourly_data[hour].append(record['people'])

        for hour in sorted(hourly_data.keys()):
            counts = hourly_data[hour]
            avg = sum(counts) / len(counts)
            max_h = max(counts)

            # Criar barra visual
            bar_length = int(avg / 2)  # Escala: 2 pessoas = 1 caractere
            bar = '█' * bar_length

            print(f"  {hour:02d}:00 - Média: {avg:5.1f} | Máx: {max_h:3d} | {bar}")

        print("\n" + "="*70)

    def show_hourly_analysis(self, log_file: Path):
        """Mostra análise detalhada hora a hora"""
        print("\n" + "="*70)
        print(f" ANÁLISE HORA A HORA - {log_file.name}")
        print("="*70)

        data = self.parse_log_file(log_file)

        if not data:
            print("\n✗ Nenhum dado encontrado")
            return

        hourly_data = defaultdict(list)
        for record in data:
            hour = record['timestamp'].hour
            hourly_data[hour].append(record['people'])

        print("\n" + "-"*70)

        for hour in sorted(hourly_data.keys()):
            counts = hourly_data[hour]
            avg = sum(counts) / len(counts)
            min_h = min(counts)
            max_h = max(counts)
            samples = len(counts)

            print(f"\n⏰ {hour:02d}:00 - {hour:02d}:59")
            print(f"   Registros: {samples}")
            print(f"   Média: {avg:.1f} | Mín: {min_h} | Máx: {max_h}")

            # Gráfico de barras visual
            print(f"   Gráfico: ", end="")
            max_bar_length = 50
            max_count = max(hourly_data[h] for h in hourly_data for _ in [0])  # Máximo global
            if isinstance(max_count, list):
                max_count = max(max_count)

            bar_length = int((avg / max_count) * max_bar_length) if max_count > 0 else 0
            print('█' * bar_length + f" {avg:.1f}")

        print("\n" + "="*70)

    def show_all_statistics(self):
        """Mostra todas as estatísticas salvas"""
        print("\n" + "="*70)
        print(" TODAS AS ESTATÍSTICAS SALVAS")
        print("="*70)

        stats_files = self.list_stats_files()

        if not stats_files:
            print("\n✗ Nenhum arquivo de estatísticas encontrado")
            return

        print(f"\nEncontrados {len(stats_files)} arquivo(s) de estatísticas:\n")

        for i, stats_file in enumerate(stats_files, 1):
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)

                timestamp = datetime.fromisoformat(stats['timestamp'])

                print(f"{i}. {stats_file.name}")
                print(f"   Data/Hora: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Máximo de pessoas: {stats['max_people']}")
                print(f"   Pessoas atuais: {stats['current_people']}")
                print(f"   FPS médio: {stats['avg_fps']:.2f}")
                print(f"   Tempo de execução: {stats['uptime_formatted']}")
                print(f"   Total de frames: {stats['total_frames']}")
                print()

            except Exception as e:
                print(f"   ✗ Erro ao ler {stats_file.name}: {e}\n")

        print("="*70)

    def export_to_csv(self, log_file: Path):
        """Exporta dados para CSV"""
        data = self.parse_log_file(log_file)

        if not data:
            print("\n✗ Nenhum dado para exportar")
            return

        # Nome do arquivo CSV
        csv_file = self.log_dir / f"export_{log_file.stem}.csv"

        try:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # Cabeçalho
                writer.writerow(['Data', 'Hora', 'Pessoas', 'Máximo da Sessão'])

                # Dados
                for record in data:
                    writer.writerow([
                        record['timestamp'].strftime('%Y-%m-%d'),
                        record['timestamp'].strftime('%H:%M:%S'),
                        record['people'],
                        record['max']
                    ])

            print(f"\n✓ Dados exportados para: {csv_file}")
            print(f"  Total de registros: {len(data)}")

        except Exception as e:
            print(f"\n✗ Erro ao exportar: {e}")

    def show_menu(self):
        """Mostra menu interativo"""
        while True:
            print("\n" + "="*70)
            print(" VISUALIZADOR DE ESTATÍSTICAS - CONTAGEM DE PESSOAS")
            print("="*70)

            # Listar arquivos de log disponíveis
            log_files = self.list_log_files()

            if not log_files:
                print("\n✗ Nenhum arquivo de log encontrado no diretório 'logs'")
                print("✓ Execute o sistema de contagem primeiro para gerar logs")
                return

            print("\nArquivos de log disponíveis:")
            for i, log_file in enumerate(log_files, 1):
                # Extrair data do nome do arquivo
                date_str = log_file.stem.replace('count_log_', '')
                try:
                    date = datetime.strptime(date_str, '%Y%m%d')
                    date_formatted = date.strftime('%d/%m/%Y')
                except:
                    date_formatted = date_str

                print(f"  {i}. {date_formatted}")

            print("\nOpções:")
            print("  [1-9] - Selecionar arquivo de log")
            print("  [A]   - Análise hora a hora")
            print("  [S]   - Todas as estatísticas salvas")
            print("  [E]   - Exportar para CSV")
            print("  [Q]   - Sair")

            choice = input("\nEscolha uma opção: ").strip().upper()

            if choice == 'Q':
                print("\n✓ Encerrando visualizador\n")
                break

            elif choice == 'S':
                self.show_all_statistics()
                input("\nPressione ENTER para continuar...")

            elif choice == 'A':
                print("\nSelecione o arquivo de log:")
                for i, log_file in enumerate(log_files, 1):
                    print(f"  {i}. {log_file.name}")

                try:
                    idx = int(input("\nNúmero do arquivo: ").strip()) - 1
                    if 0 <= idx < len(log_files):
                        self.show_hourly_analysis(log_files[idx])
                    else:
                        print("✗ Número inválido")
                except ValueError:
                    print("✗ Entrada inválida")

                input("\nPressione ENTER para continuar...")

            elif choice == 'E':
                print("\nSelecione o arquivo de log para exportar:")
                for i, log_file in enumerate(log_files, 1):
                    print(f"  {i}. {log_file.name}")

                try:
                    idx = int(input("\nNúmero do arquivo: ").strip()) - 1
                    if 0 <= idx < len(log_files):
                        self.export_to_csv(log_files[idx])
                    else:
                        print("✗ Número inválido")
                except ValueError:
                    print("✗ Entrada inválida")

                input("\nPressione ENTER para continuar...")

            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(log_files):
                    self.show_daily_summary(log_files[idx])
                    input("\nPressione ENTER para continuar...")
                else:
                    print("✗ Número inválido")

            else:
                print("✗ Opção inválida")


def main():
    """Função principal"""
    # Diretório de logs
    log_dir = "logs"

    if len(sys.argv) > 1:
        log_dir = sys.argv[1]

    # Criar e executar visualizador
    viewer = StatsViewer(log_dir)
    viewer.show_menu()


if __name__ == "__main__":
    main()
