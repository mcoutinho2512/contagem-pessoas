"""
Módulo de Geração de Relatórios
Exporta dados em PDF, CSV e Excel
"""

import os
import csv
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time


class ReportGenerator:
    """Gerador de relatórios de monitoramento"""

    def __init__(self, data_log_file: str = 'data/logs/monitoring_log.json'):
        """
        Args:
            data_log_file: Arquivo com logs históricos
        """
        self.data_log_file = data_log_file
        self.ensure_directories()

    def ensure_directories(self):
        """Garante que diretórios existem"""
        os.makedirs('data/logs', exist_ok=True)
        os.makedirs('data/reports', exist_ok=True)

    def log_data(self, data: dict):
        """
        Adiciona entrada ao log histórico

        Args:
            data: Dados a logar (timestamp, counts, etc)
        """
        try:
            # Carregar log existente
            log_data = []
            if os.path.exists(self.data_log_file):
                with open(self.data_log_file, 'r') as f:
                    try:
                        log_data = json.load(f)
                    except json.JSONDecodeError:
                        log_data = []

            # Adicionar nova entrada
            entry = {
                'timestamp': data.get('timestamp', time.time()),
                'datetime': datetime.now().isoformat(),
                'people_count': data.get('people_count', 0),
                'max_count': data.get('max_count', 0),
                'fps': data.get('fps', 0),
                'camera': data.get('camera', 'unknown'),
                'lines': data.get('lines', {}),
                'zones': data.get('zones', {}),
                'abandoned_objects': data.get('abandoned_objects', 0)
            }

            log_data.append(entry)

            # Manter apenas últimos 10000 registros (aprox. 2-3 dias com log a cada segundo)
            if len(log_data) > 10000:
                log_data = log_data[-10000:]

            # Salvar log
            with open(self.data_log_file, 'w') as f:
                json.dump(log_data, f, indent=2)

        except Exception as e:
            print(f"Erro ao fazer log de dados: {e}")

    def generate_csv_report(self, start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           output_filename: Optional[str] = None) -> str:
        """
        Gera relatório em CSV

        Args:
            start_date: Data inicial (default: últimas 24h)
            end_date: Data final (default: agora)
            output_filename: Nome do arquivo (default: auto-gerado)

        Returns:
            Caminho do arquivo gerado
        """
        # Definir período
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=1)

        # Nome do arquivo
        if output_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"relatorio_{timestamp}.csv"

        output_path = os.path.join('data/reports', output_filename)

        # Carregar dados
        log_data = self._load_log_data(start_date, end_date)

        # Gerar CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Cabeçalho
            writer.writerow([
                'Data/Hora',
                'Pessoas',
                'Máximo',
                'FPS',
                'Câmera',
                'Objetos Abandonados',
                'Linhas (Total)',
                'Zonas (Total)'
            ])

            # Dados
            for entry in log_data:
                # Contadores de linhas
                lines_total = 0
                if 'lines' in entry and isinstance(entry['lines'], dict):
                    for line_data in entry['lines'].values():
                        if isinstance(line_data, dict):
                            lines_total += line_data.get('total', 0)

                # Contadores de zonas
                zones_total = 0
                if 'zones' in entry and isinstance(entry['zones'], dict):
                    zones_total = sum(z.get('current_count', 0) for z in entry['zones'].values()
                                     if isinstance(z, dict))

                writer.writerow([
                    entry.get('datetime', ''),
                    entry.get('people_count', 0),
                    entry.get('max_count', 0),
                    f"{entry.get('fps', 0):.1f}",
                    entry.get('camera', ''),
                    entry.get('abandoned_objects', 0),
                    lines_total,
                    zones_total
                ])

        print(f"✓ Relatório CSV gerado: {output_path}")
        return output_path

    def generate_json_report(self, start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            output_filename: Optional[str] = None) -> str:
        """
        Gera relatório em JSON (mais detalhado)

        Args:
            start_date: Data inicial
            end_date: Data final
            output_filename: Nome do arquivo

        Returns:
            Caminho do arquivo gerado
        """
        # Definir período
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=1)

        # Nome do arquivo
        if output_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"relatorio_{timestamp}.json"

        output_path = os.path.join('data/reports', output_filename)

        # Carregar dados
        log_data = self._load_log_data(start_date, end_date)

        # Calcular estatísticas
        stats = self._calculate_statistics(log_data)

        # Gerar relatório
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'period_start': start_date.isoformat(),
                'period_end': end_date.isoformat(),
                'total_records': len(log_data)
            },
            'statistics': stats,
            'data': log_data
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"✓ Relatório JSON gerado: {output_path}")
        return output_path

    def _load_log_data(self, start_date: datetime, end_date: datetime) -> List[dict]:
        """Carrega dados do log dentro do período especificado"""
        try:
            if not os.path.exists(self.data_log_file):
                return []

            with open(self.data_log_file, 'r') as f:
                all_data = json.load(f)

            # Filtrar por período
            filtered_data = []
            for entry in all_data:
                try:
                    entry_date = datetime.fromisoformat(entry.get('datetime', ''))
                    if start_date <= entry_date <= end_date:
                        filtered_data.append(entry)
                except (ValueError, TypeError):
                    continue

            return filtered_data

        except Exception as e:
            print(f"Erro ao carregar log: {e}")
            return []

    def _calculate_statistics(self, log_data: List[dict]) -> dict:
        """Calcula estatísticas a partir dos dados"""
        if not log_data:
            return {
                'total_people': 0,
                'max_occupancy': 0,
                'avg_occupancy': 0,
                'avg_fps': 0,
                'total_abandoned_objects': 0
            }

        people_counts = [entry.get('people_count', 0) for entry in log_data]
        fps_values = [entry.get('fps', 0) for entry in log_data]
        abandoned_counts = [entry.get('abandoned_objects', 0) for entry in log_data]

        return {
            'total_people': sum(people_counts),
            'max_occupancy': max(people_counts) if people_counts else 0,
            'avg_occupancy': sum(people_counts) / len(people_counts) if people_counts else 0,
            'avg_fps': sum(fps_values) / len(fps_values) if fps_values else 0,
            'total_abandoned_objects': sum(abandoned_counts)
        }

    def get_available_reports(self) -> List[dict]:
        """Lista relatórios disponíveis"""
        reports = []
        reports_dir = 'data/reports'

        if os.path.exists(reports_dir):
            for filename in os.listdir(reports_dir):
                if filename.endswith(('.csv', '.json', '.pdf', '.xlsx')):
                    filepath = os.path.join(reports_dir, filename)
                    stat = os.stat(filepath)

                    reports.append({
                        'filename': filename,
                        'path': filepath,
                        'size': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })

        # Ordenar por data de modificação (mais recente primeiro)
        reports.sort(key=lambda x: x['modified'], reverse=True)

        return reports

    def cleanup_old_reports(self, days: int = 30):
        """Remove relatórios antigos"""
        cutoff_date = datetime.now() - timedelta(days=days)
        reports_dir = 'data/reports'

        if os.path.exists(reports_dir):
            for filename in os.listdir(reports_dir):
                filepath = os.path.join(reports_dir, filename)
                stat = os.stat(filepath)
                file_date = datetime.fromtimestamp(stat.st_mtime)

                if file_date < cutoff_date:
                    try:
                        os.remove(filepath)
                        print(f"✓ Relatório antigo removido: {filename}")
                    except Exception as e:
                        print(f"Erro ao remover {filename}: {e}")
