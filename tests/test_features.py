"""Unit tests on the transform.

Level 1 of the ML test pyramid. Feature code is ordinary code -- fast,
deterministic, testable -- and it is where a surprising share of production
ML bugs actually live.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import generate
from src.features import FEATURE_COLUMNS, transform


@pytest.fixture
def raw() -> pd.DataFrame:
    return generate(n=200, seed=7)


def test_column_order_is_stable(raw):
    assert list(transform(raw).columns) == FEATURE_COLUMNS


def test_no_nulls_or_infinities(raw):
    out = transform(raw)
    assert not out.isna().any().any()
    assert np.isfinite(out.to_numpy()).all()


def test_rejects_missing_raw_columns(raw):
    with pytest.raises(ValueError, match="missing raw columns"):
        transform(raw.drop(columns=["logins_30d"]))


def test_zero_tenure_does_not_divide_by_zero():
    """The edge case that produces inf in production and nowhere else."""
    row = pd.DataFrame(
        [{
            "tenure_months": 0.0,
            "monthly_charges": 80.0,
            "support_tickets_90d": 3,
            "logins_30d": 0,
            "contract_type": "month_to_month",
            "payment_method": "card",
        }]
    )
    out = transform(row)
    assert np.isfinite(out.to_numpy()).all()


def test_unseen_category_yields_all_zero_dummies():
    """A serving payload with a category training never saw must not crash
    and must not silently land in another category's column."""
    row = pd.DataFrame(
        [{
            "tenure_months": 12.0,
            "monthly_charges": 60.0,
            "support_tickets_90d": 1,
            "logins_30d": 9,
            "contract_type": "three_year",   # does not exist
            "payment_method": "crypto",      # does not exist
        }]
    )
    out = transform(row)
    contract_cols = [c for c in out.columns if c.startswith("contract_")]
    payment_cols = [c for c in out.columns if c.startswith("payment_")]
    assert out[contract_cols].sum().sum() == 0
    assert out[payment_cols].sum().sum() == 0


def test_single_row_matches_batch(raw):
    """Anti-skew: transforming one row must equal that row's slice from a
    batch transform. Stage 05 serves single rows; training sees batches. If
    this ever fails, the two paths have diverged."""
    batch = transform(raw)
    single = transform(raw.iloc[[0]])
    pd.testing.assert_frame_equal(single, batch.iloc[[0]])


def test_generator_is_deterministic():
    pd.testing.assert_frame_equal(generate(n=50, seed=1), generate(n=50, seed=1))
