"""
main.py
Ponto de entrada da aplicação FastAPI.

Execução local:
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Via Docker:
  docker build -t passos-magicos . && docker run -p 8000:8000 passos-magicos
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.utils import setup_logging, load_model
from app.routes import router

# Configuração de logging
setup_logging(
    level=logging.INFO,
    log_file=os.path.join("logs", "api.log"),
)

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join("models", "model.pkl")

# Estado global compartilhado com as rotas
app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo uma vez na inicialização da aplicação."""
    logger.info("Inicializando aplicação — carregando modelo...")
    app_state["model"] = load_model(MODEL_PATH)
    logger.info("Modelo carregado com sucesso.")
    yield
    logger.info("Encerrando aplicação.")
    app_state.clear()


app = FastAPI(
    title="Passos Mágicos — Predição de Defasagem Escolar",
    description=(
        "API para classificação binária de defasagem escolar de alunos "
        "da ONG Passos Mágicos. "
        "Target: 1 = defasado (Defas < 0), 0 = no nível."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Injeta estado nas rotas via app
app.state.app_state = app_state

app.include_router(router)
