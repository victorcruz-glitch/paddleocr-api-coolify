# Base image com Python 3.10 slim para economizar recursos
FROM python:3.10-slim

# Instala dependências de sistema necessárias para o OpenCV e paddleocr
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Cria diretório de trabalho
WORKDIR /app

# Instala a stack atual do PaddleOCR 3.x com PP-OCRv5 real
RUN pip install --no-cache-dir paddlepaddle==3.2.2
RUN pip install --no-cache-dir "paddleocr==3.4.0"

# Instala FastAPI e Uvicorn para o servidor REST
RUN pip install --no-cache-dir fastapi uvicorn python-multipart pydantic numpy

# Baixa os modelos FSRCNN de Super Resolução
RUN wget -q https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb -O /app/FSRCNN_x2.pb
RUN wget -q https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb -O /app/FSRCNN_x4.pb

# Copia o código da nossa API
COPY api.py /app/api.py

# Porta padrão que a API vai rodar
EXPOSE 8868

# Variáveis de ambiente para o PaddleOCR baixar modelos por padrão
ENV PYTHONPATH=/app

# Inicia o servidor com Uvicorn (usando workers reduzidos para salvar CPU/RAM)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8868", "--workers", "1"]
