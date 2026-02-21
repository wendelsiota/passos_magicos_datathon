# ============================================================
# Passos Mágicos — API de Predição de Defasagem Escolar
# ============================================================
# Build:   docker build -t passos-magicos .
# Run API: docker run -p 8000:8000 passos-magicos
# Docs:    http://localhost:8000/docs
#
# MLflow UI (rodar separadamente, fora do container da API):
#   mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
#   Acesse: http://localhost:5000
# ============================================================

FROM python:3.11-slim

# Evita prompts interativos durante instalação de pacotes do sistema
ENV DEBIAN_FRONTEND=noninteractive

# Variáveis de ambiente Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instala dependências do sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala dependências Python (camada cacheável separada)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código-fonte e o modelo treinado
COPY src/       ./src/
COPY app/       ./app/
COPY models/    ./models/

# Variável de ambiente para o MLflow tracking URI dentro do container.
# Por padrão usa armazenamento local (./mlruns). Para usar um servidor
# MLflow remoto, sobrescreva em runtime:
#   docker run -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 ...
# MLflow: usa SQLite dentro do container por padrão (MLflow 3.x)
# Para servidor remoto: docker run -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 ...
ENV MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db

# Cria diretório de logs com permissão de escrita
RUN mkdir -p logs && chmod 777 logs

# Porta exposta pela API
EXPOSE 8000

# Healthcheck automático do Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Comando de inicialização
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
