"""
test_evaluate.py
Testes unitários para src/evaluate.py
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from src.evaluate import (
    evaluate_model,
    print_evaluation_report,
    is_model_production_ready,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def perfect_metrics():
    return {
        "accuracy": 1.0,
        "f1_score": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "roc_auc": 1.0,
        "confusion_matrix": [[10, 0], [0, 20]],
        "classification_report": {},
    }


@pytest.fixture
def weak_metrics():
    return {
        "accuracy": 0.60,
        "f1_score": 0.55,
        "precision": 0.60,
        "recall": 0.50,
        "roc_auc": 0.62,
        "confusion_matrix": [[5, 5], [8, 12]],
        "classification_report": {},
    }


def make_mock_pipeline(y_pred, y_prob):
    """Cria um pipeline mock com predict e predict_proba."""
    pipeline = MagicMock()
    pipeline.predict.return_value = np.array(y_pred)
    pipeline.predict_proba.return_value = np.column_stack([1 - np.array(y_prob), np.array(y_prob)])
    return pipeline


# --------------------------------------------------------------------------- #
# evaluate_model
# --------------------------------------------------------------------------- #

class TestEvaluateModel:
    def test_returns_dict(self):
        y_test = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.2, 0.85])
        pipeline = make_mock_pipeline(y_pred, y_prob)
        metrics = evaluate_model(pipeline, np.zeros((5, 3)), y_test)
        assert isinstance(metrics, dict)

    def test_contains_required_keys(self):
        y_test = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.2, 0.85])
        pipeline = make_mock_pipeline(y_pred, y_prob)
        metrics = evaluate_model(pipeline, np.zeros((5, 3)), y_test)
        for key in ["accuracy", "f1_score", "precision", "recall", "roc_auc",
                    "confusion_matrix", "classification_report"]:
            assert key in metrics

    def test_perfect_predictions(self):
        y_test = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.05, 0.1, 0.9, 0.95, 0.88])
        pipeline = make_mock_pipeline(y_pred, y_prob)
        metrics = evaluate_model(pipeline, np.zeros((5, 3)), y_test)
        assert metrics["f1_score"] == pytest.approx(1.0)
        assert metrics["accuracy"] == pytest.approx(1.0)

    def test_confusion_matrix_is_list(self):
        y_test = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.4, 0.6])
        pipeline = make_mock_pipeline(y_pred, y_prob)
        metrics = evaluate_model(pipeline, np.zeros((4, 3)), y_test)
        assert isinstance(metrics["confusion_matrix"], list)

    def test_metrics_in_valid_range(self):
        y_test = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.6])
        pipeline = make_mock_pipeline(y_pred, y_prob)
        metrics = evaluate_model(pipeline, np.zeros((6, 3)), y_test)
        for key in ["accuracy", "f1_score", "precision", "recall", "roc_auc"]:
            assert 0.0 <= metrics[key] <= 1.0


# --------------------------------------------------------------------------- #
# print_evaluation_report
# --------------------------------------------------------------------------- #

class TestPrintEvaluationReport:
    def test_runs_without_error(self, perfect_metrics, capsys):
        print_evaluation_report(perfect_metrics)
        captured = capsys.readouterr()
        assert "F1-Score" in captured.out

    def test_shows_f1_value(self, perfect_metrics, capsys):
        print_evaluation_report(perfect_metrics)
        captured = capsys.readouterr()
        assert "1.0000" in captured.out

    def test_shows_confusion_matrix(self, perfect_metrics, capsys):
        print_evaluation_report(perfect_metrics)
        captured = capsys.readouterr()
        assert "TN" in captured.out
        assert "TP" in captured.out


# --------------------------------------------------------------------------- #
# is_model_production_ready
# --------------------------------------------------------------------------- #

class TestIsModelProductionReady:
    def test_approved_with_perfect_metrics(self, perfect_metrics):
        assert is_model_production_ready(perfect_metrics) is True

    def test_rejected_with_weak_metrics(self, weak_metrics):
        assert is_model_production_ready(weak_metrics) is False

    def test_rejected_when_only_f1_fails(self):
        metrics = {"f1_score": 0.60, "roc_auc": 0.85}
        assert is_model_production_ready(metrics, f1_threshold=0.70) is False

    def test_rejected_when_only_auc_fails(self):
        metrics = {"f1_score": 0.80, "roc_auc": 0.70}
        assert is_model_production_ready(metrics, auc_threshold=0.75) is False

    def test_custom_thresholds(self):
        metrics = {"f1_score": 0.85, "roc_auc": 0.90}
        assert is_model_production_ready(metrics, f1_threshold=0.80, auc_threshold=0.85) is True

    def test_exactly_at_threshold(self):
        metrics = {"f1_score": 0.70, "roc_auc": 0.75}
        assert is_model_production_ready(metrics, f1_threshold=0.70, auc_threshold=0.75) is True
