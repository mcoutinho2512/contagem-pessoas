# Sistema de Contagem de Pessoas em Tempo Real

Sistema completo de monitoramento e contagem de pessoas em salas de reunião usando visão computacional (YOLOv8) e câmeras IP CCTV com protocolo RTSP.

## Características Principais

- **Detecção em Tempo Real**: Usa YOLOv8 (modelo nano) para alta performance
- **Rastreamento Inteligente**: ByteTrack para evitar contagem duplicada
- **Interface Visual**: Painel overlay com informações detalhadas
- **Sistema de Alertas**: Notificações visuais quando capacidade excede 80%
- **Logging Automático**: Registros periódicos em arquivo texto
- **Estatísticas JSON**: Salvamento de métricas da sessão
- **Reconexão Automática**: Mantém sistema ativo se câmera desconectar
- **Ferramentas Auxiliares**: Scripts para teste e visualização de dados

## Requisitos

### Hardware
- **CPU**: Qualquer processador moderno (2+ cores recomendado)
- **RAM**: Mínimo 4GB (8GB recomendado)
- **GPU**: Opcional - NVIDIA com CUDA acelera processamento
- **Rede**: Conexão estável com a câmera IP

### Software
- **Python**: 3.8 ou superior
- **OpenCV**: Para processamento de vídeo
- **FFmpeg**: Para protocolo RTSP (geralmente já incluído no OpenCV)
- **Sistema Operacional**: Linux, Windows ou macOS

### Câmera
- Câmera IP com suporte a protocolo RTSP
- Visão completa da sala a monitorar
- Credenciais de acesso (usuário/senha)

## Instalação

### 1. Clone ou baixe o projeto

```bash
cd /home/servicedesk
# O projeto já está em: people-counter/
```

### 2. Instale as dependências

```bash
cd people-counter
pip install -r requirements.txt
```

**Nota**: Na primeira execução, o YOLOv8 baixará automaticamente o modelo (~6MB). Isso acontece apenas uma vez.

### 3. Configure a câmera

Edite o arquivo `config.json` e atualize a URL RTSP:

```json
{
  "camera": {
    "rtsp_url": "rtsp://usuario:senha@IP:porta/caminho"
  }
}
```

**Formato comum**: `rtsp://admin:senha123@192.168.1.100:554/stream1`

### 4. Teste a conexão

```bash
python test_camera.py
```

Se o teste passar, você está pronto para usar o sistema!

## Uso

### Iniciar Sistema de Contagem

```bash
python people_counter.py
```

### Controles Durante Execução

- **Q**: Sair do programa
- **S**: Salvar estatísticas imediatamente
- **Ctrl+C**: Interromper (salva estatísticas antes de sair)

### Visualizar Estatísticas

```bash
python view_stats.py
```

Menu interativo oferece:
- Resumo diário
- Análise hora a hora com gráficos
- Todas as estatísticas salvas
- Exportação para CSV

## Estrutura de Arquivos

```
people-counter/
├── people_counter.py       # Script principal
├── config.json             # Configurações (⚠️ NÃO COMPARTILHAR - contém senha)
├── requirements.txt        # Dependências Python
├── README.md              # Esta documentação
├── test_camera.py         # Teste de conexão RTSP
├── view_stats.py          # Visualizador de estatísticas
└── logs/                  # Gerado automaticamente
    ├── count_log_YYYYMMDD.txt      # Logs diários
    └── statistics_YYYYMMDD_HHMMSS.json  # Estatísticas das sessões
```

## Configurações Avançadas

### Arquivo config.json

#### Câmera
```json
"camera": {
  "rtsp_url": "rtsp://admin:123456@10.50.14.135:554/ch01",
  "reconnect_delay": 5,      // Segundos entre tentativas de reconexão
  "buffer_size": 1,          // Buffer mínimo para baixa latência
  "timeout": 10              // Timeout de conexão em segundos
}
```

#### Detecção
```json
"detection": {
  "model": "yolov8n.pt",           // Modelo: n (nano), s (small), m (medium)
  "confidence_threshold": 0.5,     // 0.0 a 1.0 - aumentar reduz falsos positivos
  "device": "cpu",                 // "cpu" ou "cuda" (GPU NVIDIA)
  "person_class_id": 0             // ID da classe pessoa no COCO dataset
}
```

#### Sala
```json
"room": {
  "max_capacity": 50,        // Capacidade máxima da sala
  "alert_threshold": 0.8     // Alerta quando > 80% da capacidade
}
```

#### Logging
```json
"logging": {
  "enabled": true,
  "interval_seconds": 60,    // Frequência de logging (60s = 1 minuto)
  "log_directory": "logs",
  "save_statistics": true
}
```

#### Performance
```json
"performance": {
  "resize_width": 1280,      // Redimensionar frame (0 = manter original)
  "resize_height": 720,
  "skip_frames": 0,          // Pular N frames (para melhorar FPS)
  "max_fps": 30
}
```

## Formato dos Logs

### Arquivo de Contagem (count_log_YYYYMMDD.txt)
```
2025-01-07 14:30:00 | Pessoas: 12 | Máximo: 15
2025-01-07 14:31:00 | Pessoas: 14 | Máximo: 15
2025-01-07 14:32:00 | Pessoas: 11 | Máximo: 15
```

### Arquivo de Estatísticas (statistics_*.json)
```json
{
  "timestamp": "2025-01-07T14:30:00",
  "session_start": "2025-01-07T13:00:00",
  "total_frames": 54000,
  "max_people": 18,
  "current_people": 12,
  "avg_fps": 30.5,
  "uptime_seconds": 5400,
  "uptime_formatted": "01:30:00"
}
```

## Troubleshooting

### Problema: Não conecta à câmera

**Soluções**:
1. Verifique se a URL RTSP está correta
2. Teste credenciais (usuário/senha)
3. Confirme que a câmera está acessível: `ping IP_DA_CAMERA`
4. Verifique se a porta RTSP está aberta (geralmente 554)
5. Tente acessar via VLC: Media > Open Network Stream

### Problema: FPS muito baixo (< 10)

**Soluções**:
1. Reduza resolução em `config.json`:
   ```json
   "resize_width": 640,
   "resize_height": 480
   ```
2. Use GPU se disponível: `"device": "cuda"`
3. Aumente threshold de confiança: `"confidence_threshold": 0.7`
4. Verifique conexão de rede com a câmera

### Problema: Muitos falsos positivos

**Soluções**:
1. Aumente threshold de confiança:
   ```json
   "confidence_threshold": 0.6  // ou 0.7
   ```
2. Melhore iluminação da sala
3. Ajuste ângulo da câmera para melhor visibilidade

### Problema: Perdendo detecções (pessoas não detectadas)

**Soluções**:
1. Diminua threshold de confiança:
   ```json
   "confidence_threshold": 0.3  // ou 0.4
   ```
2. Melhore iluminação
3. Verifique se pessoas estão visíveis por inteiro
4. Use modelo maior (mais lento mas mais preciso):
   ```json
   "model": "yolov8s.pt"  // small ao invés de nano
   ```

### Problema: Sistema trava ou usa muita memória

**Soluções**:
1. Reduza resolução de processamento
2. Adicione skip de frames:
   ```json
   "skip_frames": 1  // Processa 1 frame, pula 1
   ```
3. Feche outros programas pesados
4. Considere adicionar mais RAM

### Problema: Erro "Module not found"

**Solução**:
```bash
pip install -r requirements.txt --upgrade
```

## Otimização de Performance

### Para Máxima Velocidade (FPS alto)
```json
{
  "detection": {
    "model": "yolov8n.pt",
    "confidence_threshold": 0.6,
    "device": "cuda"  // Se tiver GPU NVIDIA
  },
  "performance": {
    "resize_width": 640,
    "resize_height": 480,
    "skip_frames": 1
  }
}
```

### Para Máxima Precisão (detecção melhor)
```json
{
  "detection": {
    "model": "yolov8m.pt",  // Modelo medium
    "confidence_threshold": 0.4,
    "device": "cuda"
  },
  "performance": {
    "resize_width": 1920,
    "resize_height": 1080,
    "skip_frames": 0
  }
}
```

## Casos de Uso

### 1. Monitoramento de Capacidade COVID
- Configure capacidade reduzida
- Alerta automático quando próximo do limite
- Histórico para compliance

### 2. Análise de Ocupação
- Identifique horários de pico
- Otimize agendamento de salas
- Relatórios de uso

### 3. Segurança e Controle de Acesso
- Monitoramento em tempo real
- Alertas de superlotação
- Logs para auditoria

### 4. Otimização de Recursos
- HVAC baseado em ocupação
- Iluminação inteligente
- Economia de energia

## Melhorias Futuras

O sistema foi projetado para ser extensível. Possíveis melhorias:

- Dashboard web com Flask/Django
- Gráficos interativos com matplotlib/plotly
- Notificações por email/SMS
- Suporte a múltiplas câmeras
- API REST para integração
- Detecção de zona específica (ROI)
- Machine Learning para previsão de ocupação
- Integração com sistemas de reserva de salas

## Segurança

**⚠️ IMPORTANTE**:

1. **NÃO compartilhe** o arquivo `config.json` - contém credenciais
2. Use senhas fortes para suas câmeras
3. Mantenha câmeras em rede isolada (VLAN)
4. Considere usar HTTPS para acesso remoto
5. Altere senhas padrão de câmeras

## Informações Técnicas

### Modelos YOLOv8 Disponíveis

| Modelo | Tamanho | Velocidade | Precisão |
|--------|---------|------------|----------|
| yolov8n.pt | ~6 MB | Muito rápido | Boa |
| yolov8s.pt | ~22 MB | Rápido | Muito boa |
| yolov8m.pt | ~52 MB | Moderado | Excelente |
| yolov8l.pt | ~87 MB | Lento | Excelente |
| yolov8x.pt | ~136 MB | Muito lento | Máxima |

**Recomendação**: Use `yolov8n.pt` para tempo real em CPU

### Requisitos de Largura de Banda

- **720p (HD)**: ~2-4 Mbps
- **1080p (Full HD)**: ~4-8 Mbps
- **4K**: ~15-25 Mbps

### Performance Esperada

| Hardware | Resolução | FPS Esperado |
|----------|-----------|--------------|
| CPU i5/Ryzen 5 | 720p | 15-25 FPS |
| CPU i7/Ryzen 7 | 1080p | 20-30 FPS |
| GPU GTX 1660 | 1080p | 60+ FPS |
| GPU RTX 3060 | 4K | 60+ FPS |

## Suporte e Contribuições

Este sistema foi desenvolvido como solução completa e pronta para produção.

Para reportar bugs ou sugerir melhorias, documente:
- Versão do Python
- Sistema operacional
- Modelo da câmera
- Logs de erro completos

## Licença

Sistema desenvolvido para uso interno. Todos os direitos reservados.

## Autor

Sistema desenvolvido via Claude Code em 2025-01-07.

## Changelog

### v1.0.0 (2025-01-07)
- Versão inicial completa
- Detecção com YOLOv8
- Rastreamento com ByteTrack
- Interface visual com overlay
- Sistema de alertas
- Logging automático
- Estatísticas JSON
- Reconexão automática
- Scripts auxiliares (teste e visualização)
- Documentação completa

---

**🎯 Sistema pronto para uso em produção!**

Para começar:
1. `pip install -r requirements.txt`
2. Configure `config.json` com sua câmera
3. `python test_camera.py` para testar
4. `python people_counter.py` para iniciar

Boa contagem! 📊👥
