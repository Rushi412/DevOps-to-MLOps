"""Candidate models and the training step.

Note the division of labour, because it is the whole point of this stage:

  src/features.py   PURE. No fitted state. Shared with serving verbatim.
  this module       FITTED state (a scaler's mean, a tree's splits), learned
                    from the training split only.

Anything learned from data must be fitted inside the Pipeline, on train only.
Fitting a scaler on the full dataset before splitting is leakage: the test
rows contributed to the mean, so your held-out score is no longer held out.
This is the quiet version of the bug -- nothing crashes, the number is just
optimistic.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED


def make_logreg() -> Pipeline:
    """Linear model. Needs scaling; tells you the direction of each feature.

    Worth noting for this dataset: the generator builds churn from a logistic
    function of the features, so a logistic regression is *well specified*
    here. Expect it to do well. That is a property of synthetic data, not a
    general result -- real churn is not this tidy.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def make_hgb() -> Pipeline:
    """Gradient-boosted trees. No scaling needed, handles interactions.

    The default reach for tabular problems. If it cannot beat a linear model
    here, that is information about the data, not a bug.
    """
    return Pipeline(
        [
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=4,
                    learning_rate=0.08,
                    max_iter=200,
                    l2_regularization=1.0,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


CANDIDATES: dict[str, Callable[[], Pipeline]] = {
    "logreg": make_logreg,
    "hgb": make_hgb,
}


def hyperparams(pipe: Pipeline) -> dict[str, object]:
    """Flatten the estimator's params for logging.

    Logged in full rather than hand-picked, because the one you forgot to
    record is always the one that mattered.
    """
    clf = pipe.named_steps["clf"]
    return {f"clf__{k}": v for k, v in clf.get_params().items()}


def fit(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    return pipe.fit(X, y)


def score(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Probability of the positive class, not a label.

    The label is score >= threshold, and the threshold is config -- what you
    deploy is model + threshold.
    """
    return pipe.predict_proba(X)[:, 1]
