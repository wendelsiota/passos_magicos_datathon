"""
test_train.py
Testes unitários para src/train.py
"""

import os
import json
import joblib
import pytest
import numpy as np
import pandas as pd

from src.train import build_pipeline, get_param_grid, train
from src.feature_engineering import get_model_features


# --------------------------------------------------------------------------- #
# build_pipeline
# --------------------------------------------------------------------------- #

class TestBuildPipeline:
    def test_returns_pipeline(self):
        from sklearn.pipeline import Pipeline
        pipeline = build_pipeline()
        assert isinstance(pipeline, Pipeline)

    def test_has_imputer_step(self):
        pipeline = build_pipeline()
        assert "imputer" in pipeline.named_steps

    def test_has_scaler_step(self):
        pipeline = build_pipeline()
        assert "scaler" in pipeline.named_steps

    def test_has_clf_step(self):
        pipeline = build_pipeline()
        assert "clf" in pipeline.named_steps

    def test_pipeline_fits_and_predicts(self):
        """Pipeline deve funcionar com dados sintéticos simples."""
        pipeline = build_pipeline()
        n_features = len(get_model_features())
        X = np.random.rand(50, n_features)
        y = np.random.randint(0, 2, 50)
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert len(preds) == 50
        assert set(preds).issubset({0, 1})

    def test_pipeline_predict_proba(self):
        """predict_proba deve retornar probabilidades entre 0 e 1."""
        pipeline = build_pipeline()
        n_features = len(get_model_features())
        X = np.random.rand(30, n_features)
        y = np.random.randint(0, 2, 30)
        pipeline.fit(X, y)
        proba = pipeline.predict_proba(X)
        assert proba.shape == (30, 2)
        assert (proba >= 0).all() and (proba <= 1).all()


# --------------------------------------------------------------------------- #
# get_param_grid
# --------------------------------------------------------------------------- #

class TestGetParamGrid:
    def test_returns_dict(self):
        grid = get_param_grid()
        assert isinstance(grid, dict)

    def test_contains_clf_params(self):
        grid = get_param_grid()
        for key in grid:
            assert key.startswith("clf__")

    def test_n_estimators_in_grid(self):
        grid = get_param_grid()
        assert "clf__n_estimators" in grid

    def test_all_values_are_lists(self):
        grid = get_param_grid()
        for key, values in grid.items():
            assert isinstance(values, list)
            assert len(values) >= 1


# --------------------------------------------------------------------------- #
# train (integração leve — usa o dataset real)
# --------------------------------------------------------------------------- #

@pytest.mark.integration
class TestTrain:
    DATA_PATH = os.path.join("data", "BASE DE DADOS PEDE 2024 - DATATHON.xlsx")
    MODEL_PATH = os.path.join("models", "model_test.pkl")

    def test_train_returns_metrics_dict(self):
        if not os.path.exists(self.DATA_PATH):
            pytest.skip("Dataset não disponível")
        metrics = train(data_path=self.DATA_PATH, model_path=self.MODEL_PATH)
        assert isinstance(metrics, dict)

    def test_train_metrics_contain_f1(self):
        if not os.path.exists(self.DATA_PATH):
            pytest.skip("Dataset não disponível")
        metrics = train(data_path=self.DATA_PATH, model_path=self.MODEL_PATH)
        assert "f1_score" in metrics

    def test_train_f1_above_threshold(self):
        if not os.path.exists(self.DATA_PATH):
            pytest.skip("Dataset não disponível")
        metrics = train(data_path=self.DATA_PATH, model_path=self.MODEL_PATH)
        assert metrics["f1_score"] >= 0.70, f"F1 abaixo de 0.70: {metrics['f1_score']:.4f}"

    def test_train_saves_model_file(self):
        if not os.path.exists(self.DATA_PATH):
            pytest.skip("Dataset não disponível")
        train(data_path=self.DATA_PATH, model_path=self.MODEL_PATH)
        assert os.path.exists(self.MODEL_PATH)

    def test_saved_model_is_loadable(self):
        if not os.path.exists(self.DATA_PATH):
            pytest.skip("Dataset não disponível")
        train(data_path=self.DATA_PATH, model_path=self.MODEL_PATH)
        model = joblib.load(self.MODEL_PATH)
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def teardown_method(self):
        """Limpa o modelo de teste após cada teste."""
        if os.path.exists(self.MODEL_PATH):
            os.remove(self.MODEL_PATH)
