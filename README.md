# Sistema de Detecção de Objetos Perigosos em Vídeos

## Descrição
Este projeto implementa um sistema automatizado para detecção de objetos perigosos em vídeos utilizando técnicas de visão computacional e aprendizado de máquina. O sistema monitora uma pasta específica para novos arquivos de vídeo, processa-os automaticamente para identificar objetos potencialmente perigosos (como armas) e notifica os usuários quando conteúdo prejudicial é detectado.

## Arquitetura
O sistema é composto por:

1. **API FastAPI**: Fornece endpoints para treinamento, predição e processamento de arquivos
2. **Monitor de Arquivos**: Observa um diretório específico para novos arquivos de vídeo
3. **Motor de Detecção**: Utiliza modelos YOLO e EfficientNet para detectar objetos perigosos
4. **Fluxo de Trabalho n8n**: Automatiza o processo de notificação quando objetos perigosos são detectados

## Requisitos
- Python 3.8+
- Docker e Docker Compose
- CUDA (opcional, para aceleração por GPU)

## Dependências Principais
- FastAPI: Framework web para criação da API
- Ultralytics YOLO: Detecção de objetos
- PyTorch: Framework de aprendizado de máquina
- OpenCV: Processamento de imagens e vídeos
- Watchdog: Monitoramento de diretórios
- n8n: Automação de fluxos de trabalho

## Instalação

### Configuração do Ambiente
1. Clone o repositório:
   ```
   git clone https://github.com/seu-usuario/fiap-hackaton.git
   cd fiap-hackaton
   ```

2. Crie e ative um ambiente virtual:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

4. Inicie o serviço n8n:
   ```
   docker-compose up -d
   ```

5. Configure os diretórios de observação:
   - Por padrão, o sistema monitora `C:\Users\aniba\Videos\hackaton\listener`
   - Você pode modificar este caminho no arquivo `main.py`

## Uso

### Iniciar o Servidor
```
uvicorn app.main:app --reload
```

### Endpoints da API

#### Treinamento
- **Endpoint**: `/train`
- **Método**: POST
- **Parâmetros**: 
  - `video`: Arquivo de vídeo (form-data)
  - `label`: Rótulo ("harmful" ou "harmless")
- **Descrição**: Treina o modelo com um vídeo rotulado

#### Predição
- **Endpoint**: `/predict`
- **Método**: POST
- **Parâmetros**: 
  - `video`: Arquivo de vídeo (form-data)
- **Descrição**: Analisa um vídeo e retorna se contém objetos perigosos

#### Predição por Nome de Arquivo
- **Endpoint**: `/predict_filename`
- **Método**: POST
- **Parâmetros**: 
  - `filename`: Nome do arquivo no diretório monitorado
- **Descrição**: Analisa um vídeo existente no diretório monitorado

#### Processamento
- **Endpoint**: `/processed`
- **Método**: POST
- **Parâmetros**: 
  - `filename`: Nome do arquivo a ser movido
- **Descrição**: Move um arquivo do diretório monitorado para a pasta "processed"

### Monitoramento Automático
O sistema inicia automaticamente um observador de arquivos que monitora o diretório configurado. Quando um novo arquivo é adicionado:

1. O sistema detecta o arquivo
2. Envia uma notificação para o webhook n8n
3. O n8n inicia o fluxo de trabalho para análise do vídeo
4. Se um objeto perigoso for detectado, um e-mail de alerta é enviado
5. O arquivo é movido para a pasta "processed" após a análise

## Fluxo de Trabalho n8n
O fluxo de trabalho n8n é configurado para:
1. Receber notificações de novos arquivos
2. Chamar a API para analisar o vídeo
3. Verificar se o conteúdo é perigoso
4. Enviar e-mail de alerta se necessário
5. Marcar o arquivo como processado

## Configuração
As principais configurações estão em:
- `app/main.py`: Diretórios de observação e endpoints da API
- `app/model.py`: Configurações dos modelos de ML
- `docker-compose.yml`: Configuração do serviço n8n
