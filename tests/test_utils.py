"""
test_utils.py
Testes unitários para src/utils.py
"""

import os
import logging
import tempfile

import joblib
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from unittest.mock import patch, MagicMock

from src.utils import setup_logging, save_model, load_model, load_model_from_mlflow


# --------------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------------- #

@pytest.fixture
def simple_pipeline():
    """Pipeline simples treinado para usar nos testes."""
    import numpy as np
    p = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    X = np.random.rand(20, 2)
    y = [0] * 10 + [1] * 10
    p.fit(X, y)
    return p


# --------------------------------------------------------------------------- #
# setup_logging
# --------------------------------------------------------------------------- #

class TestSetupLogging:
    def test_sets_root_logger_level(self):
        setup_logging(level=logging.DEBUG)
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_creates_log_file(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        setup_logging(log_file=log_file)
        logger = logging.getLogger("test_setup")
        logger.info("Mensagem de teste")
        assert os.path.exists(log_file)

    def test_creates_log_directory_if_missing(self, tmp_path):
        log_file = str(tmp_path / "subdir" / "app.log")
        setup_logging(log_file=log_file)
        assert os.path.exists(os.path.dirname(log_file))

    def test_runs_without_log_file(self):
        """setup_logging sem arquivo deve funcionar sem erro."""
        setup_logging(level=logging.INFO)


# --------------------------------------------------------------------------- #
# save_model
# --------------------------------------------------------------------------- #

class TestSaveModel:
    def test_saves_file(self, simple_pipeline, tmp_path):
        model_path = str(tmp_path / "model.pkl")
        save_model(simple_pipeline, model_path)
        assert os.path.exists(model_path)

    def test_creates_parent_directory(self, simple_pipeline, tmp_path):
        model_path = str(tmp_path / "nested" / "model.pkl")
        save_model(simple_pipeline, model_path)
        assert os.path.exists(model_path)

    def test_saved_file_is_loadable(self, simple_pipeline, tmp_path):
        model_path = str(tmp_path / "model.pkl")
        save_model(simple_pipeline, model_path)
        loaded = joblib.load(model_path)
        assert hasattr(loaded, "predict")


# --------------------------------------------------------------------------- #
# load_model
# --------------------------------------------------------------------------- #

class TestLoadModel:
    def test_loads_saved_model(self, simple_pipeline, tmp_path):
        model_path = str(tmp_path / "model.pkl")
        joblib.dump(simple_pipeline, model_path)
        loaded = load_model(model_path)
        assert hasattr(loaded, "predict")

    def test_raises_if_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_model("/tmp/nonexistent_model_xyz.pkl")

    def test_loaded_model_predicts(self, simple_pipeline, tmp_path):
        import numpy as np
        model_path = str(tmp_path / "model.pkl")
        joblib.dump(simple_pipeline, model_path)
        loaded = load_model(model_path)
        X = np.random.rand(5, 2)
        preds = loaded.predict(X)
        assert len(preds) == 5


# --------------------------------------------------------------------------- #
# load_model_from_mlflow
# --------------------------------------------------------------------------- #

class TestLoadModelFromMlflow:
    def test_calls_mlflow_load_model(self, simple_pipeline):
        """Deve chamar mlflow.sklearn.load_model com a URI correta."""
        with patch("mlflow.sklearn.load_model", return_value=simple_pipeline) as mock_load:
            result = load_model_from_mlflow("MeuModelo", version="1")
        mock_load.assert_called_once_with("models:/MeuModelo/1")
        assert result is simple_pipeline

    def test_uri_with_latest_version(self, simple_pipeline):
        """Versão padrão 'latest' deve compor a URI corretamente."""
        with patch("mlflow.sklearn.load_model", return_value=simple_pipeline) as mock_load:
            load_model_from_mlflow("PassosMagicos-DefasagemClassifier")
        mock_load.assert_called_once_with("models:/PassosMagicos-DefasagemClassifier/latest")

    def test_uri_with_alias(self, simple_pipeline):
        """Alias 'champion' deve ser incluído na URI."""
        with patch("mlflow.sklearn.load_model", return_value=simple_pipeline) as mock_load:
            load_model_from_mlflow("MeuModelo", version="champion")
        mock_load.assert_called_once_with("models:/MeuModelo/champion")

    def test_loaded_model_has_predict(self, simple_pipeline):
        """Modelo retornado deve ter o método predict."""
        with patch("mlflow.sklearn.load_model", return_value=simple_pipeline):
            model = load_model_from_mlflow("MeuModelo")
        assert hasattr(model, "predict")

    def test_propagates_mlflow_exception(self):
        """Deve propagar exceções do MLflow (ex: modelo não encontrado)."""
        with patch("mlflow.sklearn.load_model", side_effect=Exception("Not found")):
            with pytest.raises(Exception, match="Not found"):
                load_model_from_mlflow("ModeloInexistente")
