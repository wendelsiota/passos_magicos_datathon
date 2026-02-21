"""
test_mlflow_integration.py
Testes de integração para o rastreamento MLflow em src/train.py.

Todos os testes usam um banco SQLite temporário como MLflow tracking URI,
sem necessidade de servidor MLflow em execução.
"""

import os
import pytest
import mlflow
from unittest.mock import patch

from src.train import (
    _log_to_mlflow,
    MLFLOW_EXPERIMENT,
    MLFLOW_MODEL_NAME,
    build_pipeline,
    get_param_grid,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def mock_grid_search():
    """GridSearchCV mock com os atributos mínimos necessários."""
    from unittest.mock import MagicMock
    gs = MagicMock()
    gs.best_params_ = {
        "clf__n_estimators": 200,
        "clf__max_depth": 3,
        "clf__learning_rate": 0.1,
        "clf__subsample": 1.0,
    }
    gs.best_score_ = 0.929
    gs.best_estimator_ = build_pipeline()
    return gs


@pytest.fixture
def sample_metrics():
    return {
        "f1_score": 0.9249,
        "roc_auc": 0.9245,
        "precision": 0.8797,
        "recall": 0.9750,
        "accuracy": 0.8895,
        "confusion_matrix": [[36, 16], [3, 117]],
        "classification_report": {},
    }


@pytest.fixture
def mlflow_uri(tmp_path):
    """Tracking URI SQLite em diretório temporário isolado por teste."""
    return f"sqlite:///{tmp_path}/mlflow.db"


# --------------------------------------------------------------------------- #
# Testes unitários — _log_to_mlflow() com MLflow isolado em tmp_path
# --------------------------------------------------------------------------- #

class TestLogToMlflow:
    def test_returns_run_id_on_success(self, mlflow_uri, mock_grid_search, sample_metrics):
        run_id = _log_to_mlflow(mock_grid_search, sample_metrics, "/tmp/model.pkl", 860, 10, mlflow_uri)
        assert run_id is not None
        assert isinstance(run_id, str)
        assert len(run_id) == 32  # UUID sem hífens

    def test_creates_experiment(self, mlflow_uri, mock_grid_search, sample_metrics):
        _log_to_mlflow(mock_grid_search, sample_metrics, "/tmp/model.pkl", 860, 10, mlflow_uri)
        mlflow.set_tracking_uri(mlflow_uri)
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
        assert experiment is not None

    def test_logs_best_hyperparams(self, mlflow_uri, mock_grid_search, sample_metrics):
        run_id = _log_to_mlflow(mock_grid_search, sample_metrics, "/tmp/model.pkl", 860, 10, mlflow_uri)
        mlflow.set_tracking_uri(mlflow_uri)
        run = mlflow.get_run(run_id)
        params = run.data.params
        assert params["clf__n_estimators"] == "200"
        assert params["clf__max_depth"] == "3"
        assert params["clf__learning_rate"] == "0.1"
        assert params["clf__subsample"] == "1.0"

    def test_logs_static_params(self, mlflow_uri, mock_grid_search, sample_metrics):
        run_id = _log_to_mlflow(mock_grid_search, sample_metrics, "/tmp/model.pkl", 860, 10, mlflow_uri)
        mlflow.set_tracking_uri(mlflow_uri)
        run = mlflow.get_run(run_id)
        params = run.data.params
        assert params["cv_folds"] == "5"
        assert params["test_size"] == "0.2"
        assert params["random_state"] == "42"
        assert params["imputer_strategy"] == "median"

    def test_logs_evaluation_metrics(self, mlflow_uri, mock_grid_search, sample_metrics):
        run_id = _log_to_mlflow(mock_grid_search, sample_metrics, "/tmp/model.pkl", 860, 10, mlflow_uri)
        mlflow.set_tracking_uri(mlflow_uri)
        run = mlflow.get_run(run_id)
        logged = run.data.metrics
        assert abs(logged["f1_score"] - 0.9249) < 1e-4
        assert abs(logged["roc_auc"] - 0.9245) < 1e-4
        assert abs(logged["recall"] - 0.9750) < 1e-4
        assert abs(logged["accuracy"] - 0.8895) < 1e-4

    def test_logs_cv_score(self, mlflow_uri, mock_grid_search, sample_metrics):
        run_id = _log_to_mlflow(mock_grid_search, sample_metrics, "/tmp/model.pkl", 860, 10, mlflow_uri)
        mlflow.set_tracking_uri(mlflow_uri)
        run = mlflow.get_run(run_id)
        assert abs(run.data.metrics["cv_f1_best"] - 0.929) < 1e-4

    def test_logs_dataset_tags(self, mlflow_uri, mock_grid_search, sample_metrics):
        run_id = _log_to_mlflow(mock_grid_search, sample_metrics, "/tmp/model.pkl", 860, 10, mlflow_uri)
        mlflow.set_tracking_uri(mlflow_uri)
        run = mlflow.get_run(run_id)
        tags = run.data.tags
        assert tags["dataset"] == "PEDE 2024"
        assert tags["algorithm"] == "GradientBoostingClassifier"
        assert tags["n_samples"] == "860"
        assert tags["n_features"] == "10"

    def test_logs_artifact_when_model_exists(self, tmp_path, mock_grid_search, sample_metrics):
        """Se model.pkl existe, deve ser logado como artefato."""
        import joblib
        model_path = str(tmp_path / "model.pkl")
        joblib.dump(build_pipeline(), model_path)
        mlflow_uri = f"sqlite:///{tmp_path}/mlflow.db"
        run_id = _log_to_mlflow(mock_grid_search, sample_metrics, model_path, 860, 10, mlflow_uri)
        assert run_id is not None

    def test_returns_none_on_mlflow_failure(self, mock_grid_search, sample_metrics):
        """Se MLflow falhar, deve retornar None sem lançar exceção."""
        with patch("mlflow.set_tracking_uri", side_effect=Exception("MLflow indisponível")):
            run_id = _log_to_mlflow(mock_grid_search, sample_metrics, "/tmp/m.pkl", 860, 10)
        assert run_id is None

    def test_multiple_runs_accumulate(self, mlflow_uri, mock_grid_search, sample_metrics):
        """Cada chamada deve criar um run separado no mesmo experimento."""
        _log_to_mlflow(mock_grid_search, sample_metrics, "/tmp/m.pkl", 860, 10, mlflow_uri)
        _log_to_mlflow(mock_grid_search, sample_metrics, "/tmp/m.pkl", 860, 10, mlflow_uri)
        mlflow.set_tracking_uri(mlflow_uri)
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) == 2


# --------------------------------------------------------------------------- #
# Testes de integração — train() completo com MLflow
# --------------------------------------------------------------------------- #

@pytest.mark.integration
class TestTrainWithMlflow:
    DATA_PATH = os.path.join("data", "BASE DE DADOS PEDE 2024 - DATATHON.xlsx")

    def test_train_creates_mlflow_run(self, tmp_path):
        if not os.path.exists(self.DATA_PATH):
            pytest.skip("Dataset não disponível")
        from src.train import train
        mlflow_uri = f"sqlite:///{tmp_path}/mlflow.db"
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
        model_path = str(tmp_path / "model_test.pkl")
        train(data_path=self.DATA_PATH, model_path=model_path)
        mlflow.set_tracking_uri(mlflow_uri)
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
        assert experiment is not None
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) == 1

    def test_train_run_has_f1_above_threshold(self, tmp_path):
        if not os.path.exists(self.DATA_PATH):
            pytest.skip("Dataset não disponível")
        from src.train import train
        mlflow_uri = f"sqlite:///{tmp_path}/mlflow.db"
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
        model_path = str(tmp_path / "model_test.pkl")
        train(data_path=self.DATA_PATH, model_path=model_path)
        mlflow.set_tracking_uri(mlflow_uri)
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        f1 = runs.iloc[0]["metrics.f1_score"]
        assert f1 >= 0.70, f"F1 abaixo de 0.70: {f1:.4f}"
