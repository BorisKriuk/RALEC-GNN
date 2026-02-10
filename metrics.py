#!/usr/bin/env python3
"""Evaluation metrics for binary crisis prediction."""

from typing import Dict, Tuple
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
)


@dataclass
class Metrics:
    auc_roc: float
    avg_precision: float
    precision: float
    recall: float
    f1: float
    brier_score: float
    accuracy: float
    positive_rate: float
    optimal_threshold: float = 0.5

    def to_dict(self) -> Dict:
        return {
            "AUC-ROC": self.auc_roc,
            "Avg Precision": self.avg_precision,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1": self.f1,
            "Brier Score": self.brier_score,
            "Accuracy": self.accuracy,
            "Positive Rate": self.positive_rate,
            "Optimal Threshold": self.optimal_threshold,
        }


# ------------------------------------------------------------------
def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-10)
    idx = np.argmax(f1s[:-1])
    if len(thresholds) > 0:
        return float(np.clip(thresholds[idx], 0.05, 0.95))
    return 0.5


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Metrics:
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    if len(np.unique(y_true)) < 2:
        return Metrics(
            auc_roc=0.5,
            avg_precision=float(np.mean(y_true)),
            precision=0.0,
            recall=0.0,
            f1=0.0,
            brier_score=0.25,
            accuracy=float(np.mean(y_true == (y_prob >= 0.5).astype(int))),
            positive_rate=float(np.mean(y_true)),
        )

    thr = find_optimal_threshold(y_true, y_prob)
    yp = (y_prob >= thr).astype(int)

    return Metrics(
        auc_roc=roc_auc_score(y_true, y_prob),
        avg_precision=average_precision_score(y_true, y_prob),
        precision=precision_score(y_true, yp, zero_division=0),
        recall=recall_score(y_true, yp, zero_division=0),
        f1=f1_score(y_true, yp, zero_division=0),
        brier_score=brier_score_loss(y_true, y_prob),
        accuracy=accuracy_score(y_true, yp),
        positive_rate=float(np.mean(y_true)),
        optimal_threshold=thr,
    )


# ------------------------------------------------------------------
def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    rng = np.random.RandomState(42)
    n = len(y_true)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    aucs = np.array(aucs)
    alpha = (1 - confidence) / 2
    return (
        roc_auc_score(y_true, y_prob),
        float(np.percentile(aucs, alpha * 100)),
        float(np.percentile(aucs, (1 - alpha) * 100)),
    )


def significance_test(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    n_bootstrap: int = 2000,
) -> Dict:
    rng = np.random.RandomState(42)
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(
            roc_auc_score(y_true[idx], probs_a[idx])
            - roc_auc_score(y_true[idx], probs_b[idx])
        )
    diffs = np.array(diffs)
    p_value = float(np.mean(diffs <= 0))
    return {
        "auc_a": roc_auc_score(y_true, probs_a),
        "auc_b": roc_auc_score(y_true, probs_b),
        "auc_diff": roc_auc_score(y_true, probs_a)
        - roc_auc_score(y_true, probs_b),
        "ci_lower": float(np.percentile(diffs, 2.5)),
        "ci_upper": float(np.percentile(diffs, 97.5)),
        "p_value": p_value,
        "significant_05": p_value < 0.05,
        "significant_01": p_value < 0.01,
    }