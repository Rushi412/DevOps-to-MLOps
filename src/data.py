"""Dataset generation and time-based splitting.

Why synthetic rather than a Kaggle download:
  1. The repo clones and runs with no account, no network, no 200 MB file.
  2. Stage 08 needs a *drifted* version of the same distribution to make PSI
     and Evidently mean anything. With a static CSV that is awkward; here it
     is one argument.
The generator is a fixed-seed process, so it behaves like a real snapshot:
same seed in, same bytes out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import N_CUSTOMERS, RANDOM_SEED, TRAIN_FRAC, VAL_FRAC

CONTRACTS = ("month_to_month", "one_year", "two_year")
PAYMENTS = ("card", "bank_transfer", "electronic_check")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(
    n: int = N_CUSTOMERS,
    seed: int = RANDOM_SEED,
    drift: float = 0.0,
) -> pd.DataFrame:
    """Generate a customer-churn snapshot.

    Args:
        n: number of customers.
        seed: RNG seed. Same seed -> same frame.
        drift: 0.0 reproduces the reference distribution. Raising it shifts
            tenure down and support load up, which is what stage 08 will
            detect. Unused before then.
    """
    rng = np.random.default_rng(seed)

    tenure = rng.gamma(shape=2.0, scale=12.0 * (1.0 - 0.35 * drift), size=n)
    tenure = np.clip(tenure, 0.5, 72.0)

    monthly_charges = np.clip(rng.normal(70.0, 22.0, size=n), 18.0, 145.0)

    tickets = rng.poisson(lam=0.9 + 1.4 * drift, size=n)

    # Engagement falls off for customers who are already unhappy.
    logins = rng.poisson(lam=np.clip(14.0 - 1.6 * tickets, 1.0, None), size=n)

    contract = rng.choice(CONTRACTS, size=n, p=[0.52, 0.28, 0.20])
    payment = rng.choice(PAYMENTS, size=n, p=[0.40, 0.33, 0.27])

    signup = pd.Timestamp("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 900, size=n), unit="D"
    )

    # The true generating process. A model that recovers roughly this is doing
    # its job; anything scoring far above it is almost certainly leaking.
    logit = (
        -1.20
        - 0.035 * tenure
        + 0.420 * tickets
        - 0.055 * logins
        + 1.100 * (contract == "month_to_month")
        + 0.300 * (payment == "electronic_check")
        + 0.006 * (monthly_charges - 70.0)
    )
    churned = rng.binomial(1, _sigmoid(logit))

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n)],
            "signup_date": signup,
            "tenure_months": tenure.round(1),
            "monthly_charges": monthly_charges.round(2),
            "support_tickets_90d": tickets,
            "logins_30d": logins,
            "contract_type": contract,
            "payment_method": payment,
            "churned": churned,
        }
    )
    return df.sort_values("signup_date").reset_index(drop=True)


def time_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically: oldest -> train, newest -> test.

    Deliberately NOT train_test_split(shuffle=True). Churn is time-ordered;
    a random split trains on customers who signed up after the ones it is
    scored on, which inflates every metric you will report.
    """
    df = df.sort_values("signup_date").reset_index(drop=True)
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return (
        df.iloc[:i_train].copy(),
        df.iloc[i_train:i_val].copy(),
        df.iloc[i_val:].copy(),
    )
