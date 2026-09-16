"""THE shared transform. One implementation, imported by every path.

This module exists in stage 01 -- before there is a model to serve -- on
purpose. Training/serving skew is the signature production ML bug, and the
fix is structural, not vigilant: if the training job and the API import the
same function, they cannot drift apart.

Stage 05's FastAPI handler will `from src.features import transform`.
Nothing else is allowed to compute a feature.
"""
from __future__ import annotations

import pandas as pd

RAW_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "support_tickets_90d",
    "logins_30d",
    "contract_type",
    "payment_method",
]

CONTRACT_LEVELS = ["month_to_month", "one_year", "two_year"]
PAYMENT_LEVELS = ["card", "bank_transfer", "electronic_check"]

FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "support_tickets_90d",
    "logins_30d",
    "charges_per_tenure_month",
    "tickets_per_tenure_month",
    "logins_per_ticket",
    *[f"contract_{c}" for c in CONTRACT_LEVELS],
    *[f"payment_{p}" for p in PAYMENT_LEVELS],
]


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Raw columns -> model-ready feature matrix.

    Pure: no fitted state, no global lookups, no I/O. Anything that needs to
    be *learned* from the training set (a mean to impute with, a scaler)
    belongs in a fitted pipeline in stage 02, not here -- computing it here
    would leak the test set into training.

    Returns columns in FEATURE_COLUMNS order, always. Column order silently
    changing is a real way to ship a broken model.
    """
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing raw columns: {sorted(missing)}")

    out = pd.DataFrame(index=df.index)

    out["tenure_months"] = df["tenure_months"].astype(float)
    out["monthly_charges"] = df["monthly_charges"].astype(float)
    out["support_tickets_90d"] = df["support_tickets_90d"].astype(float)
    out["logins_30d"] = df["logins_30d"].astype(float)

    # Guard every denominator. A null quietly becoming a zero, or an inf
    # reaching the model, is how a prediction shifts without anything failing.
    safe_tenure = out["tenure_months"].clip(lower=0.5)
    out["charges_per_tenure_month"] = out["monthly_charges"] / safe_tenure
    out["tickets_per_tenure_month"] = out["support_tickets_90d"] / safe_tenure
    out["logins_per_ticket"] = out["logins_30d"] / (out["support_tickets_90d"] + 1.0)

    # Explicit levels, not pd.get_dummies. get_dummies infers columns from
    # whatever it happens to see -- a serving batch with no "two_year"
    # customers would produce a narrower matrix and a shape error at best.
    for c in CONTRACT_LEVELS:
        out[f"contract_{c}"] = (df["contract_type"] == c).astype(float)
    for p in PAYMENT_LEVELS:
        out[f"payment_{p}"] = (df["payment_method"] == p).astype(float)

    return out[FEATURE_COLUMNS]
