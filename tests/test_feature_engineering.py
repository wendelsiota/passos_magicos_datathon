"""
test_feature_engineering.py
Testes unitários para src/feature_engineering.py
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    add_inde_category,
    add_avg_notas,
    add_engajamento_aprendizagem,
    add_gap_ian_inde,
    engineer_features,
    get_model_features,
)


# --------------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------------- #

@pytest.fixture
def base_df():
    """DataFrame com colunas padronizadas para testar feature engineering."""
    return pd.DataFrame({
        "inde":      [3.0,  5.5,  7.0,  8.5,  9.5],
        "ida":       [4.0,  6.0,  7.5,  8.0,  9.0],
        "ieg":       [5.0,  6.0,  7.0,  8.0,  9.0],
        "ipv":       [4.5,  5.5,  6.5,  7.5,  8.5],
        "iaa":       [5.0,  6.0,  7.0,  8.0,  9.0],
        "nota_mat":  [3.0,  5.0,  7.0,  8.0,  9.0],
        "nota_port": [4.0,  5.0,  6.0,  8.0,  9.5],
    })


# --------------------------------------------------------------------------- #
# add_inde_category
# --------------------------------------------------------------------------- #

class TestAddIndeCategory:
    def test_creates_pedra_num_column(self, base_df):
        df = add_inde_category(base_df)
        assert "pedra_num" in df.columns

    def test_quartzo_range(self, base_df):
        # inde=3.0 → Quartzo → label 0
        df = add_inde_category(base_df)
        assert df.loc[0, "pedra_num"] == 0.0

    def test_agata_range(self):
        # inde=5.5 é exatamente o limite — usa 5.6 para garantir Ágata (5.506–6.868 → label 1)
        df = pd.DataFrame({"inde": [5.6]})
        result = add_inde_category(df)
        assert result.loc[0, "pedra_num"] == 1.0

    def test_ametista_range(self, base_df):
        # inde=7.0 → Ametista → label 2
        df = add_inde_category(base_df)
        assert df.loc[2, "pedra_num"] == 2.0

    def test_topazio_range(self, base_df):
        # inde=8.5 → Topázio → label 3
        df = add_inde_category(base_df)
        assert df.loc[3, "pedra_num"] == 3.0

    def test_does_not_modify_original(self, base_df):
        original_cols = list(base_df.columns)
        add_inde_category(base_df)
        assert list(base_df.columns) == original_cols

    def test_no_nulls_for_valid_inde(self, base_df):
        df = add_inde_category(base_df)
        assert df["pedra_num"].isna().sum() == 0


# --------------------------------------------------------------------------- #
# add_avg_notas
# --------------------------------------------------------------------------- #

class TestAddAvgNotas:
    def test_creates_avg_notas_column(self, base_df):
        df = add_avg_notas(base_df)
        assert "avg_notas" in df.columns

    def test_average_is_correct(self, base_df):
        df = add_avg_notas(base_df)
        expected = (base_df["nota_mat"] + base_df["nota_port"]) / 2
        pd.testing.assert_series_equal(df["avg_notas"], expected, check_names=False)

    def test_skips_if_no_nota_columns(self):
        df = pd.DataFrame({"inde": [7.0]})
        result = add_avg_notas(df)
        assert "avg_notas" not in result.columns

    def test_handles_single_nota_column(self):
        df = pd.DataFrame({"nota_mat": [6.0]})
        result = add_avg_notas(df)
        assert result["avg_notas"].iloc[0] == 6.0


# --------------------------------------------------------------------------- #
# add_engajamento_aprendizagem
# --------------------------------------------------------------------------- #

class TestAddEngajamentoAprendizagem:
    def test_creates_column(self, base_df):
        df = add_engajamento_aprendizagem(base_df)
        assert "eng_aprendizagem" in df.columns

    def test_formula_is_correct(self, base_df):
        df = add_engajamento_aprendizagem(base_df)
        expected = base_df["ieg"] * base_df["ida"] / 10.0
        pd.testing.assert_series_equal(df["eng_aprendizagem"], expected, check_names=False)

    def test_skips_if_missing_columns(self):
        df = pd.DataFrame({"ieg": [7.0]})
        result = add_engajamento_aprendizagem(df)
        assert "eng_aprendizagem" not in result.columns


# --------------------------------------------------------------------------- #
# add_gap_ian_inde (mantido para compatibilidade, mas não usado no modelo)
# --------------------------------------------------------------------------- #

class TestAddGapIanInde:
    def test_creates_column_when_both_present(self):
        df = pd.DataFrame({"ian": [5.0, 10.0], "inde": [3.0, 7.0]})
        result = add_gap_ian_inde(df)
        assert "gap_ian_inde" in result.columns

    def test_formula_is_correct(self):
        df = pd.DataFrame({"ian": [5.0], "inde": [3.0]})
        result = add_gap_ian_inde(df)
        assert result["gap_ian_inde"].iloc[0] == 2.0

    def test_skips_if_missing_column(self, base_df):
        # base_df não tem 'ian'
        result = add_gap_ian_inde(base_df)
        assert "gap_ian_inde" not in result.columns


# --------------------------------------------------------------------------- #
# engineer_features
# --------------------------------------------------------------------------- #

class TestEngineerFeatures:
    def test_returns_dataframe(self, base_df):
        result = engineer_features(base_df)
        assert isinstance(result, pd.DataFrame)

    def test_adds_expected_features(self, base_df):
        result = engineer_features(base_df)
        for col in ["pedra_num", "avg_notas", "eng_aprendizagem"]:
            assert col in result.columns

    def test_preserves_original_columns(self, base_df):
        result = engineer_features(base_df)
        for col in base_df.columns:
            assert col in result.columns

    def test_row_count_unchanged(self, base_df):
        result = engineer_features(base_df)
        assert len(result) == len(base_df)


# --------------------------------------------------------------------------- #
# get_model_features
# --------------------------------------------------------------------------- #

class TestGetModelFeatures:
    def test_returns_list(self):
        features = get_model_features()
        assert isinstance(features, list)

    def test_returns_nonempty_list(self):
        features = get_model_features()
        assert len(features) > 0

    def test_no_ian_in_features(self):
        # IAN é data leakage — não deve estar na lista
        features = get_model_features()
        assert "ian" not in features

    def test_inde_in_features(self):
        features = get_model_features()
        assert "inde" in features

    def test_engineered_features_included(self):
        features = get_model_features()
        for col in ["pedra_num", "avg_notas", "eng_aprendizagem"]:
            assert col in features
