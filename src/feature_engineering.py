"""
feature_engineering.py
Criação e transformação de features para o modelo.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def add_inde_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria feature categórica baseada na classificação PEDRA do INDE.
    Quartzo: 2.405–5.506 → 0
    Ágata:   5.506–6.868 → 1
    Ametista:6.868–8.230 → 2
    Topázio: 8.230–9.294 → 3
    """
    df = df.copy()
    bins = [0, 5.506, 6.868, 8.230, 10.0]
    labels = [0, 1, 2, 3]
    df["pedra_num"] = pd.cut(df["inde"], bins=bins, labels=labels, include_lowest=True).astype(float)
    logger.info("Feature 'pedra_num' criada (classificação INDE→PEDRA)")
    return df


def add_avg_notas(df: pd.DataFrame) -> pd.DataFrame:
    """Cria média das notas de Matemática e Português."""
    df = df.copy()
    nota_cols = [c for c in ["nota_mat", "nota_port"] if c in df.columns]
    if nota_cols:
        df["avg_notas"] = df[nota_cols].mean(axis=1)
        logger.info("Feature 'avg_notas' criada (média das notas)")
    return df


def add_engajamento_aprendizagem(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria índice combinado de engajamento * aprendizagem.
    Intuitivamente: aluno engajado e com boa aprendizagem tem menor risco.
    """
    df = df.copy()
    if "ieg" in df.columns and "ida" in df.columns:
        df["eng_aprendizagem"] = df["ieg"] * df["ida"] / 10.0
        logger.info("Feature 'eng_aprendizagem' criada (IEG × IDA / 10)")
    return df


def add_gap_ian_inde(df: pd.DataFrame) -> pd.DataFrame:
    """
    Diferença entre IAN e INDE.
    IAN alto + INDE baixo pode indicar aluno que se adapta mas não aprende.
    """
    df = df.copy()
    if "ian" in df.columns and "inde" in df.columns:
        df["gap_ian_inde"] = df["ian"] - df["inde"]
        logger.info("Feature 'gap_ian_inde' criada (IAN - INDE)")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completa de feature engineering.
    Executa todas as transformações na ordem correta.
    Nota: gap_ian_inde foi removido pois IAN é data leakage (derivado de Defas).
    """
    logger.info("Iniciando feature engineering...")
    df = add_inde_category(df)
    df = add_avg_notas(df)
    df = add_engajamento_aprendizagem(df)
    logger.info(f"Feature engineering concluído: {df.shape[1]} colunas totais")
    return df


def get_model_features() -> list:
    """Retorna lista das features usadas no modelo (ordem fixa)."""
    return [
        "inde", "ida", "ieg", "ipv", "iaa",
        "nota_mat", "nota_port",
        "pedra_num", "avg_notas", "eng_aprendizagem",
    ]