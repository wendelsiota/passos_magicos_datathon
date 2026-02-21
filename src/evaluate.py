"""
evaluate.py
Funções de avaliação do modelo.

Métricas principais:
- F1-Score: balanceia precisão e recall — ideal para dados com leve desbalanceamento
- AUC-ROC: mede separabilidade entre classes — referência de qualidade geral
- Precisão e Recall separados: para entender tradeoffs de negócio
"""

import logging
import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
)

logger = logging.getLogger(__name__)


def evaluate_model(pipeline, X_test, y_test) -> dict:
    """
    Avalia o pipeline no conjunto de teste.
    Retorna dicionário com todas as métricas.
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    logger.info(
        f"Avaliação — F1: {metrics['f1_score']:.4f} | "
        f"AUC-ROC: {metrics['roc_auc']:.4f} | "
        f"Precisão: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f}"
    )
    return metrics


def print_evaluation_report(metrics: dict) -> None:
    """Imprime relatório de avaliação formatado."""
    print("\n" + "=" * 50)
    print("RELATÓRIO DE AVALIAÇÃO DO MODELO")
    print("=" * 50)
    print(f"  Acurácia:  {metrics['accuracy']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}  ← métrica principal")
    print(f"  AUC-ROC:   {metrics['roc_auc']:.4f}  ← qualidade geral")
    print(f"  Precisão:  {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print("=" * 50)
    print("\nMatriz de Confusão:")
    cm = metrics["confusion_matrix"]
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print("\nInterpretação:")
    print("  - F1 > 0.75: modelo confiável para produção")
    print("  - AUC > 0.80: boa separabilidade entre classes")
    print("  - Recall alto: poucos defasados deixam de ser identificados")
    print("=" * 50)


def is_model_production_ready(metrics: dict, f1_threshold: float = 0.70, auc_threshold: float = 0.75) -> bool:
    """
    Verifica se o modelo atende aos critérios mínimos para produção.
    Retorna True se aprovado.
    """
    f1_ok = metrics["f1_score"] >= f1_threshold
    auc_ok = metrics["roc_auc"] >= auc_threshold
    approved = f1_ok and auc_ok
    status = "✅ APROVADO" if approved else "❌ REPROVADO"
    logger.info(
        f"Validação produção: {status} | "
        f"F1={metrics['f1_score']:.4f} (min={f1_threshold}) | "
        f"AUC={metrics['roc_auc']:.4f} (min={auc_threshold})"
    )
    return approved