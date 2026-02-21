"""
routes.py
Endpoints da API FastAPI para predição de defasagem escolar.

Endpoints:
  GET  /          → healthcheck
  GET  /health    → status do modelo
  POST /predict   → classificação de um aluno
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.preprocessing import preprocess_inference
from src.feature_engineering import engineer_features, get_model_features

logger = logging.getLogger(__name__)

router = APIRouter()


# --------------------------------------------------------------------------- #
# Schemas de entrada e saída
# --------------------------------------------------------------------------- #

class StudentFeatures(BaseModel):
    """Features de entrada para predição. Valores na escala 0–10."""

    inde: float = Field(..., ge=0.0, le=10.0, description="INDE 22 — Índice de Desenvolvimento Educacional")
    ida: float = Field(..., ge=0.0, le=10.0, description="IDA — Indicador de Desempenho Acadêmico")
    ieg: float = Field(..., ge=0.0, le=10.0, description="IEG — Indicador de Engajamento")
    ipv: float = Field(..., ge=0.0, le=10.0, description="IPV — Indicador de Ponto de Virada")
    iaa: float = Field(..., ge=0.0, le=10.0, description="IAA — Indicador de Auto-Avaliação")
    nota_mat: Optional[float] = Field(None, ge=0.0, le=10.0, description="Nota de Matemática (opcional)")
    nota_port: Optional[float] = Field(None, ge=0.0, le=10.0, description="Nota de Português (opcional)")

    model_config = {"json_schema_extra": {
        "example": {
            "inde": 6.2,
            "ida": 7.1,
            "ieg": 5.8,
            "ipv": 6.5,
            "iaa": 7.0,
            "nota_mat": 6.0,
            "nota_port": 6.5,
        }
    }}


class PredictionResponse(BaseModel):
    """Resposta da predição."""

    prediction: int = Field(..., description="0 = no nível | 1 = defasado")
    label: str = Field(..., description="Rótulo legível da predição")
    probability_defasado: float = Field(..., description="Probabilidade de estar defasado (classe 1)")
    probability_no_nivel: float = Field(..., description="Probabilidade de estar no nível (classe 0)")
    model_version: str = Field(default="1.0.0")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@router.get("/", tags=["status"])
def root():
    """Raiz — confirma que a API está no ar."""
    return {"message": "Passos Mágicos — API de Predição de Defasagem Escolar", "version": "1.0.0"}


@router.get("/health", response_model=HealthResponse, tags=["status"])
def health(request: Request):
    """Healthcheck — verifica se o modelo está carregado."""
    app_state = request.app.state.app_state
    model_loaded = "model" in app_state and app_state["model"] is not None
    return HealthResponse(status="ok" if model_loaded else "degraded", model_loaded=model_loaded)


@router.post("/predict", response_model=PredictionResponse, tags=["predição"])
def predict(features: StudentFeatures, request: Request):
    """
    Classifica um aluno como defasado (1) ou no nível (0).

    Recebe os indicadores de desempenho e retorna a predição binária
    junto com as probabilidades de cada classe.
    """
    start = time.time()
    app_state = request.app.state.app_state

    if "model" not in app_state or app_state["model"] is None:
        raise HTTPException(status_code=503, detail="Modelo não disponível. Tente novamente em instantes.")

    model = app_state["model"]

    # Pré-processamento e feature engineering
    input_dict = features.model_dump()
    df = preprocess_inference(input_dict)
    df = engineer_features(df)
    X = df[get_model_features()]

    # Predição
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    prob_defasado = float(proba[1])
    prob_no_nivel = float(proba[0])

    elapsed_ms = (time.time() - start) * 1000
    label = "defasado" if pred == 1 else "no nível"

    logger.info(
        f"Predição: {label} | prob_defasado={prob_defasado:.3f} | latência={elapsed_ms:.1f}ms"
    )

    return PredictionResponse(
        prediction=pred,
        label=label,
        probability_defasado=round(prob_defasado, 4),
        probability_no_nivel=round(prob_no_nivel, 4),
    )
