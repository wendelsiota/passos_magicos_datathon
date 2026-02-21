"""
train.py
Script principal de treino do modelo de classificação de defasagem escolar.

Fluxo:
  1. Carrega dataset 2024
  2. Cria target binário (Defas < 0 → 1)
  3. Pré-processa e faz feature engineering
  4. Split treino/teste (estratificado)
  5. Treina GradientBoostingClassifier com busca de hiperparâmetros
  6. Avalia e salva o modelo em models/model.pkl
  7. Registra experimento, métricas e modelo no MLflow (aditivo)
"""

import os
import sys
import logging
import json

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Garante que src/ seja importável ao rodar o script diretamente
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import load_data_2024, create_binary_target, preprocess
from src.feature_engineering import engineer_features, get_model_features
from src.evaluate import evaluate_model, print_evaluation_report, is_model_production_ready
from src.utils import setup_logging, save_model

# Caminhos
DATA_PATH = os.path.join("data", "BASE DE DADOS PEDE 2024 - DATATHON.xlsx")
MODEL_PATH = os.path.join("models", "model.pkl")
METRICS_PATH = os.path.join("models", "metrics.json")

# MLflow — usa SQLite por padrão (recomendado no MLflow 3.x)
# Pode ser sobrescrito via variável de ambiente MLFLOW_TRACKING_URI
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")
MLFLOW_EXPERIMENT = "passos-magicos-defasagem"
MLFLOW_MODEL_NAME = "PassosMagicos-DefasagemClassifier"

logger = logging.getLogger(__name__)


def build_pipeline() -> Pipeline:
    """Constrói o pipeline sklearn: scaler + GradientBoosting."""
    clf = GradientBoostingClassifier(
        random_state=42,
        n_iter_no_change=10,
        validation_fraction=0.1,
    )
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])
    return pipeline


def get_param_grid() -> dict:
    """Grade de hiperparâmetros para GridSearchCV."""
    return {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [3, 4],
        "clf__learning_rate": [0.05, 0.1],
        "clf__subsample": [0.8, 1.0],
    }


def _log_to_mlflow(
    grid_search: GridSearchCV,
    metrics: dict,
    model_path: str,
    n_samples: int,
    n_features: int,
    tracking_uri: str = None,
) -> str | None:
    """
    Registra o experimento no MLflow de forma aditiva e não-bloqueante.

    Loga: hiperparâmetros, métricas de avaliação, artefatos (model.pkl,
    metrics.json) e registra o modelo no Model Registry.

    Parâmetros
    ----------
    tracking_uri : str, optional
        URI do MLflow Tracking Server. Se None, usa MLFLOW_TRACKING_URI
        (padrão: sqlite:///mlruns/mlflow.db). Útil para testes com tmp_path.

    Retorna o run_id MLflow ou None se o tracking falhar.
    """
    try:
        uri = tracking_uri or MLFLOW_TRACKING_URI
        os.makedirs("mlruns", exist_ok=True)
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        with mlflow.start_run(run_name="GradientBoosting") as run:
            # --- Tags descritivas do experimento ---
            mlflow.set_tags({
                "dataset": "PEDE 2024",
                "algorithm": "GradientBoostingClassifier",
                "target": "Defas < 0 → defasado",
                "n_samples": str(n_samples),
                "n_features": str(n_features),
                "feature_selection": "sem IAN (data leakage removido)",
            })

            # --- Hiperparâmetros do GridSearchCV ---
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("cv_scoring", "f1")
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("imputer_strategy", "median")

            # --- Score de validação cruzada ---
            mlflow.log_metric("cv_f1_best", grid_search.best_score_)

            # --- Métricas no conjunto de teste ---
            mlflow.log_metric("f1_score", metrics["f1_score"])
            mlflow.log_metric("roc_auc", metrics["roc_auc"])
            mlflow.log_metric("precision", metrics["precision"])
            mlflow.log_metric("recall", metrics["recall"])
            mlflow.log_metric("accuracy", metrics["accuracy"])

            # --- Artefatos locais ---
            if os.path.exists(model_path):
                mlflow.log_artifact(model_path, artifact_path="joblib")
            if os.path.exists(METRICS_PATH):
                mlflow.log_artifact(METRICS_PATH, artifact_path="reports")

            # --- Modelo sklearn no formato MLflow + registro ---
            mlflow.sklearn.log_model(
                sk_model=grid_search.best_estimator_,
                artifact_path="model",
                registered_model_name=MLFLOW_MODEL_NAME,
                input_example=None,
            )

            run_id = run.info.run_id
            logger.info(
                f"MLflow — experimento: '{MLFLOW_EXPERIMENT}' | "
                f"run_id: {run_id} | "
                f"modelo registrado: '{MLFLOW_MODEL_NAME}'"
            )
            return run_id

    except Exception as exc:
        logger.warning(f"MLflow tracking falhou (não crítico, treino continua): {exc}")
        return None


def train(data_path: str = DATA_PATH, model_path: str = MODEL_PATH) -> dict:
    """
    Executa o pipeline completo de treino e retorna as métricas do modelo.
    """
    # 1. Carregar dados
    df_raw = load_data_2024(data_path)

    # 2. Criar target antes do pré-processamento (usa coluna original "Defas")
    y_full = create_binary_target(df_raw)

    # 3. Pré-processar features
    df_processed = preprocess(df_raw)

    # Alinhar target com linhas restantes após drop de nulos
    y = y_full.loc[df_processed.index]

    # 4. Feature engineering
    df_engineered = engineer_features(df_processed)
    feature_cols = get_model_features()
    X = df_engineered[feature_cols]

    # 5. Split treino/teste estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(
        f"Split — treino: {len(X_train)} amostras | teste: {len(X_test)} amostras"
    )

    # 6. Treinar com GridSearchCV
    pipeline = build_pipeline()
    param_grid = get_param_grid()

    logger.info("Iniciando GridSearchCV (cv=5, scoring=f1)...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    logger.info(f"Melhores hiperparâmetros: {grid_search.best_params_}")
    logger.info(f"Melhor F1 (CV): {grid_search.best_score_:.4f}")

    # 7. Avaliar no conjunto de teste
    metrics = evaluate_model(best_pipeline, X_test, y_test)
    print_evaluation_report(metrics)

    # 8. Verificar prontidão para produção
    is_model_production_ready(metrics)

    # 9. Salvar modelo e métricas
    save_model(best_pipeline, model_path)

    metrics_to_save = {k: v for k, v in metrics.items() if k != "classification_report"}
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    logger.info(f"Métricas salvas em: {METRICS_PATH}")

    # 10. Registrar no MLflow (aditivo — não bloqueia o fluxo)
    _log_to_mlflow(
        grid_search=grid_search,
        metrics=metrics,
        model_path=model_path,
        n_samples=len(X),
        n_features=len(feature_cols),
    )

    return metrics


if __name__ == "__main__":
    setup_logging(log_file=os.path.join("logs", "train.log"))
    train()
