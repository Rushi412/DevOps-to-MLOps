"""Baselines. The thing every model must beat before anyone is allowed to
be impressed by it.

Each exposes fit/predict_proba, so stage 02's real estimator drops into the
same harness with no changes to the runner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class MajorityClass:
    """Always predicts the majority label. The accuracy trap, made concrete.

    Scores ~85% accuracy on this dataset and is worth exactly nothing: it
    never flags a single churner. Its AUC is undefined because it ranks
    nobody above anybody.
    """

    name = "majority_class"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MajorityClass":
        self.majority_ = int(pd.Series(y).mode().iloc[0])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), float(self.majority_))


class BaseRate:
    """Predicts the training base rate for everyone.

    Better calibrated than MajorityClass -- if 15% churn it says 0.15 -- but
    still ranks nothing, so still AUC-undefined. Useful as a sanity check on
    calibration later.
    """

    name = "base_rate"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BaseRate":
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.rate_)


class RuleBased:
    """A hand-written heuristic of the kind a business already has.

    This is the real bar. If the stage-02 model cannot clearly beat this,
    the correct decision is to ship the rule: no retraining cost, no drift,
    no GPU bill, and anyone can read it.

    Operates on RAW columns, not transformed features, because that is how
    such a rule actually arrives -- someone in the room says "month-to-month
    customers who open tickets and stop logging in".
    """

    name = "rule_based"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RuleBased":
        return self  # nothing learned, by design

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        score = np.full(len(X), 0.10)
        score += 0.25 * (X["contract_type"] == "month_to_month")
        score += 0.15 * (X["support_tickets_90d"] >= 2)
        score += 0.12 * (X["logins_30d"] < 6)
        score += 0.08 * (X["tenure_months"] < 6)
        score += 0.05 * (X["payment_method"] == "electronic_check")
        return np.clip(score, 0.0, 1.0)


ALL_BASELINES = (MajorityClass, BaseRate, RuleBased)
