"""
utils.py
Utilitários compartilhados: configuração de logging e carregamento de modelo.
Suporta carregamento via joblib (local) e MLflow Model Registry.
"""

import logging
import os
import joblib


def setup_logging(level: int = logging.INFO, log_file: str = None) -> None:
    """
    Configura o logging do projeto.
    Se log_file for fornecido, grava também em arquivo.
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def save_model(pipeline, filepath: str) -> None:
    """Salva o pipeline treinado em disco com joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    logging.getLogger(__name__).info(f"Modelo salvo em: {filepath}")


def load_model(filepath: str):
    """Carrega o pipeline salvo com joblib. Lança FileNotFoundError se não existir."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Modelo não encontrado: {filepath}")
    pipeline = joblib.load(filepath)
    logging.getLogger(__name__).info(f"Modelo carregado de: {filepath}")
    return pipeline


def load_model_from_mlflow(model_name: str, version: str = "latest"):
    """
    Carrega o modelo diretamente do MLflow Model Registry.

    Compatível com MLflow 3.x — usa número de versão ou alias.

    Parâmetros
    ----------
    model_name : str
        Nome registrado no MLflow Registry.
        Ex: "PassosMagicos-DefasagemClassifier"
    version : str
        Versão numérica ("1", "2", …) ou alias configurado no Registry
        ("champion", "challenger", "latest").
        Padrão: "latest" (versão mais recente registrada).

    Exemplos
    --------
    # Versão mais recente
    model = load_model_from_mlflow("PassosMagicos-DefasagemClassifier")

    # Versão específica
    model = load_model_from_mlflow("PassosMagicos-DefasagemClassifier", version="1")

    # Via alias (configure no MLflow UI ou CLI)
    model = load_model_from_mlflow("PassosMagicos-DefasagemClassifier", version="champion")
    """
    import mlflow.sklearn

    model_uri = f"models:/{model_name}/{version}"
    pipeline = mlflow.sklearn.load_model(model_uri)
    logging.getLogger(__name__).info(
        f"Modelo carregado do MLflow Registry: '{model_name}' versão '{version}'"
    )
    return pipeline
