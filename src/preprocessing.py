"""
preprocessing.py
Funções de limpeza e preparação dos dados.
Dataset de referência: BASE DE DADOS PEDE 2024 - DATATHON.xlsx
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Colunas do dataset 2024 que serão usadas como features
# NOTA: IAN foi removido — é derivado diretamente de Defas (data leakage)
FEATURE_COLS = ["INDE 22", "IDA", "IEG", "IPV", "IAA", "Matem", "Portug"]
TARGET_COL_2024 = "Defas"

# Mapeamento para nomes internos padronizados
RENAME_MAP = {
    "INDE 22": "inde",
    "IDA": "ida",
    "IEG": "ieg",
    "IPV": "ipv",
    "IAA": "iaa",
    "Matem": "nota_mat",
    "Portug": "nota_port",
}

# Nomes padronizados finais (usados em todo o projeto)
STANDARD_FEATURES = ["inde", "ida", "ieg", "ipv", "iaa", "nota_mat", "nota_port"]


def load_data_2024(filepath: str) -> pd.DataFrame:
    """Carrega o dataset 2024 e retorna um DataFrame bruto."""
    logger.info(f"Carregando dataset 2024: {filepath}")
    df = pd.read_excel(filepath)
    logger.info(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    return df


def create_binary_target(df: pd.DataFrame, target_col: str = TARGET_COL_2024) -> pd.Series:
    """
    Cria o target binário a partir da coluna de defasagem.
    1 = defasado (valor < 0)
    0 = no nível ou adiantado (valor >= 0)
    """
    target = (df[target_col] < 0).astype(int)
    logger.info(f"Target binário criado — defasados: {target.sum()} ({target.mean()*100:.1f}%)")
    return target


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia colunas para nomes padronizados internos."""
    df = df.rename(columns=RENAME_MAP)
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Seleciona apenas as colunas de features padronizadas."""
    available = [c for c in STANDARD_FEATURES if c in df.columns]
    missing = [c for c in STANDARD_FEATURES if c not in df.columns]
    if missing:
        logger.warning(f"Colunas ausentes (serão ignoradas): {missing}")
    return df[available]


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trata valores nulos:
    - Notas (nota_mat, nota_port): imputa com mediana
    - Indicadores principais (ian, inde, ida, ieg, ipv, iaa): remove linhas com nulos
    """
    df = df.copy()

    # Imputação com mediana para notas (alta taxa de nulos)
    for col in ["nota_mat", "nota_port"]:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Imputando {col} com mediana={median_val:.2f}")

    # Remove linhas com nulos nos indicadores críticos
    critical_cols = [c for c in ["inde", "ida", "ieg", "ipv", "iaa"] if c in df.columns]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    after = len(df)
    if before != after:
        logger.info(f"Removidas {before - after} linhas com nulos nos indicadores críticos")

    return df


def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Converte todas as colunas de features para float, coercing erros."""
    df = df.copy()
    for col in STANDARD_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clip_outliers(df: pd.DataFrame, lower: float = 0.0, upper: float = 10.0) -> pd.DataFrame:
    """
    Garante que os indicadores (escala 0-10) estejam dentro dos limites esperados.
    Valores fora do range são clipados.
    """
    df = df.copy()
    indicator_cols = [c for c in ["inde", "ida", "ieg", "ipv", "iaa"] if c in df.columns]
    for col in indicator_cols:
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completa de pré-processamento.
    Executa: rename → select → convert → handle_missing → clip.
    Retorna DataFrame pronto para feature_engineering.
    """
    logger.info("Iniciando pré-processamento...")
    df = rename_columns(df)
    df = select_features(df)
    df = convert_to_numeric(df)
    df = handle_missing_values(df)
    df = clip_outliers(df)
    logger.info(f"Pré-processamento concluído: {df.shape[0]} linhas, {df.shape[1]} colunas")
    return df


def preprocess_inference(input_dict: dict) -> pd.DataFrame:
    """
    Pré-processa um único registro recebido via API para inferência.
    input_dict: dicionário com as features no formato padronizado (nomes internos).
    Retorna DataFrame com uma linha pronto para predict().
    """
    df = pd.DataFrame([input_dict])
    df = convert_to_numeric(df)
    df = clip_outliers(df)

    # Garante que todas as features estão presentes
    for col in STANDARD_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    df = df[STANDARD_FEATURES]
    return df