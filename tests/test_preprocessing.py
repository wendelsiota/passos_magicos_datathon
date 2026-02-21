"""
test_preprocessing.py
Testes unitários para src/preprocessing.py
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    STANDARD_FEATURES,
    create_binary_target,
    rename_columns,
    select_features,
    handle_missing_values,
    convert_to_numeric,
    clip_outliers,
    preprocess,
    preprocess_inference,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def raw_df():
    """DataFrame mínimo simulando o dataset 2024 (colunas originais)."""
    return pd.DataFrame({
        "INDE 22": [7.0, 5.0, 3.5, 8.0, 6.0],
        "IDA":     [8.0, 4.0, 3.0, 9.0, 6.5],
        "IEG":     [7.5, 5.0, 4.0, 8.5, 6.0],
        "IPV":     [6.0, 4.5, 3.0, 7.0, 5.5],
        "IAA":     [7.0, 5.0, 4.0, 8.0, 6.0],
        "Matem":   [7.0, 4.0, None, 9.0, 6.0],
        "Portug":  [6.5, 3.5, None, 8.0, 5.5],
        "Defas":   [0,   -1,   -2,   1,  -1],
    })


@pytest.fixture
def std_df(raw_df):
    """DataFrame já com colunas renomeadas e no formato padrão."""
    df = rename_columns(raw_df)
    return df


# --------------------------------------------------------------------------- #
# create_binary_target
# --------------------------------------------------------------------------- #

class TestCreateBinaryTarget:
    def test_positive_defas_is_zero(self, raw_df):
        target = create_binary_target(raw_df)
        assert target.iloc[3] == 0  # Defas=1 → no nível

    def test_zero_defas_is_zero(self, raw_df):
        target = create_binary_target(raw_df)
        assert target.iloc[0] == 0  # Defas=0 → no nível

    def test_negative_defas_is_one(self, raw_df):
        target = create_binary_target(raw_df)
        assert target.iloc[1] == 1  # Defas=-1 → defasado
        assert target.iloc[2] == 1  # Defas=-2 → defasado

    def test_dtype_is_int(self, raw_df):
        target = create_binary_target(raw_df)
        assert target.dtype == int

    def test_only_zero_and_one(self, raw_df):
        target = create_binary_target(raw_df)
        assert set(target.unique()).issubset({0, 1})


# --------------------------------------------------------------------------- #
# rename_columns
# --------------------------------------------------------------------------- #

class TestRenameColumns:
    def test_inde_renamed(self, raw_df):
        df = rename_columns(raw_df)
        assert "inde" in df.columns
        assert "INDE 22" not in df.columns

    def test_notas_renamed(self, raw_df):
        df = rename_columns(raw_df)
        assert "nota_mat" in df.columns
        assert "nota_port" in df.columns

    def test_all_standard_features_present(self, raw_df):
        df = rename_columns(raw_df)
        for col in STANDARD_FEATURES:
            assert col in df.columns


# --------------------------------------------------------------------------- #
# select_features
# --------------------------------------------------------------------------- #

class TestSelectFeatures:
    def test_returns_only_standard_features(self, std_df):
        df = select_features(std_df)
        assert set(df.columns) == set(STANDARD_FEATURES)

    def test_shape_rows_preserved(self, std_df):
        df = select_features(std_df)
        assert len(df) == len(std_df)

    def test_ignores_extra_columns(self):
        df = pd.DataFrame({c: [1.0] for c in STANDARD_FEATURES + ["extra_col"]})
        result = select_features(df)
        assert "extra_col" not in result.columns


# --------------------------------------------------------------------------- #
# convert_to_numeric
# --------------------------------------------------------------------------- #

class TestConvertToNumeric:
    def test_string_numbers_converted(self, std_df):
        std_df["inde"] = std_df["inde"].astype(str)
        df = convert_to_numeric(std_df)
        assert pd.api.types.is_float_dtype(df["inde"])

    def test_non_numeric_strings_become_nan(self, std_df):
        std_df["inde"] = "abc"
        df = convert_to_numeric(std_df)
        assert df["inde"].isna().all()


# --------------------------------------------------------------------------- #
# handle_missing_values
# --------------------------------------------------------------------------- #

class TestHandleMissingValues:
    def test_imputes_nota_mat_with_median(self, std_df):
        std_df["nota_mat"] = std_df["nota_mat"].where(std_df["nota_mat"].notna(), other=np.nan)
        df = handle_missing_values(std_df)
        assert df["nota_mat"].isna().sum() == 0

    def test_imputes_nota_port_with_median(self, std_df):
        df = handle_missing_values(std_df)
        assert df["nota_port"].isna().sum() == 0

    def test_drops_rows_with_null_critical_indicators(self):
        df = pd.DataFrame({
            "inde": [7.0, np.nan, 5.0],
            "ida":  [6.0, 5.0, 4.0],
            "ieg":  [7.0, 6.0, 5.0],
            "ipv":  [6.0, 5.0, 4.0],
            "iaa":  [7.0, 6.0, 5.0],
            "nota_mat":  [6.0, 5.0, 4.0],
            "nota_port": [6.0, 5.0, 4.0],
        })
        result = handle_missing_values(df)
        assert len(result) == 2
        assert result["inde"].isna().sum() == 0


# --------------------------------------------------------------------------- #
# clip_outliers
# --------------------------------------------------------------------------- #

class TestClipOutliers:
    def test_clips_above_10(self, std_df):
        std_df["inde"] = 15.0
        df = clip_outliers(std_df)
        assert (df["inde"] <= 10.0).all()

    def test_clips_below_0(self, std_df):
        std_df["inde"] = -5.0
        df = clip_outliers(std_df)
        assert (df["inde"] >= 0.0).all()

    def test_normal_values_unchanged(self, std_df):
        original_inde = std_df["inde"].copy()
        df = clip_outliers(std_df)
        pd.testing.assert_series_equal(df["inde"], original_inde)


# --------------------------------------------------------------------------- #
# preprocess (pipeline completa)
# --------------------------------------------------------------------------- #

class TestPreprocess:
    def test_returns_dataframe(self, raw_df):
        result = preprocess(raw_df)
        assert isinstance(result, pd.DataFrame)

    def test_output_has_standard_features(self, raw_df):
        result = preprocess(raw_df)
        for col in STANDARD_FEATURES:
            assert col in result.columns

    def test_no_nulls_in_result(self, raw_df):
        result = preprocess(raw_df)
        assert result.isna().sum().sum() == 0

    def test_values_within_range(self, raw_df):
        result = preprocess(raw_df)
        indicator_cols = [c for c in ["inde", "ida", "ieg", "ipv", "iaa"] if c in result.columns]
        for col in indicator_cols:
            assert result[col].between(0, 10).all()


# --------------------------------------------------------------------------- #
# preprocess_inference
# --------------------------------------------------------------------------- #

class TestPreprocessInference:
    def test_returns_single_row_dataframe(self):
        input_dict = {c: 5.0 for c in STANDARD_FEATURES}
        result = preprocess_inference(input_dict)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_missing_fields_filled_with_nan(self):
        input_dict = {"inde": 6.0, "ida": 7.0, "ieg": 6.0, "ipv": 5.0, "iaa": 7.0}
        result = preprocess_inference(input_dict)
        assert "nota_mat" in result.columns
        assert "nota_port" in result.columns

    def test_column_order_is_standard(self):
        input_dict = {c: 5.0 for c in STANDARD_FEATURES}
        result = preprocess_inference(input_dict)
        assert list(result.columns) == STANDARD_FEATURES
