# Guia Rápido de Início

## Instalação em 4 Passos

### 1. Instalar Dependências
```bash
cd /home/servicedesk/people-counter
pip install -r requirements.txt
```

### 2. Configurar Câmera
Edite o arquivo `config.json` e configure a URL RTSP da sua câmera:
```json
{
  "camera": {
    "rtsp_url": "rtsp://admin:123456@10.50.14.135:554/ch01"
  }
}
```

**Formato**: `rtsp://usuario:senha@IP:porta/caminho`

### 3. Testar Conexão
```bash
python test_camera.py
```

Se aparecer "✓ TESTE CONCLUÍDO COM SUCESSO", pode prosseguir!

### 4. Iniciar Sistema
```bash
python people_counter.py
```

## Controles

Durante a execução:
- **Q** - Sair
- **S** - Salvar estatísticas
- **Ctrl+C** - Interromper

## Visualizar Dados

```bash
python view_stats.py
```

## Ajustes Rápidos

### FPS Baixo?
Edite `config.json`:
```json
{
  "performance": {
    "resize_width": 640,
    "resize_height": 480
  }
}
```

### Muitos Falsos Positivos?
Edite `config.json`:
```json
{
  "detection": {
    "confidence_threshold": 0.7
  }
}
```

### Perdendo Detecções?
Edite `config.json`:
```json
{
  "detection": {
    "confidence_threshold": 0.3
  }
}
```

## Arquivos Importantes

- `config.json` - Configurações (⚠️ contém senha)
- `logs/` - Logs e estatísticas
- `README.md` - Documentação completa

## Problemas Comuns

**Erro de conexão**: Verifique URL, usuário e senha
**FPS baixo**: Reduza resolução ou use GPU
**Falsos positivos**: Aumente confidence_threshold

---

**Documentação completa**: Veja [README.md](README.md)
