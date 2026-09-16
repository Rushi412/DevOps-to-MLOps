"""Metrics, reported together so no single number can mislead.

Accuracy is included only to be argued with.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import DECISION_THRESHOLD


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, float]:
    """Score a set of predicted probabilities.

    y_score is a probability, not a label. The label is y_score >= threshold,
    and the threshold is config -- what you deploy is model + threshold.
    """
    y_pred = (y_score >= threshold).astype(int)

    # A constant score returns exactly 0.5 here -- sklearn does not raise.
    # That is the correct answer and the whole lesson: 0.5 is a coin flip,
    # so a baseline that "looks" 81% accurate ranks nobody above anybody.
    # The guard is for the degenerate case of a single class in y_true.
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "positive_rate": float(y_pred.mean()),
    }


def format_table(results: dict[str, dict[str, float]]) -> str:
    """Render {model_name: metrics} as a fixed-width table."""
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "positive_rate"]
    name_w = max(len(n) for n in results) + 2
    head = "model".ljust(name_w) + "".join(c.rjust(14) for c in cols)
    lines = [head, "-" * len(head)]
    for name, m in results.items():
        row = name.ljust(name_w)
        for c in cols:
            v = m[c]
            row += ("nan" if v != v else f"{v:.4f}").rjust(14)
        lines.append(row)
    return "\n".join(lines)
