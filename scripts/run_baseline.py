"""Stage 01 entry point.

    python scripts/run_baseline.py

Generates the snapshot, splits it by time, fits every baseline on train and
scores them on validation. No model is saved -- stage 01 produces a number to
beat, not an artifact.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline import ALL_BASELINES
from src.config import DATA_RAW, DECISION_THRESHOLD, RANDOM_SEED, TARGET
from src.data import generate, time_split
from src.evaluate import evaluate, format_table
from src.features import FEATURE_COLUMNS, transform


def main() -> None:
    print(f"seed={RANDOM_SEED}  threshold={DECISION_THRESHOLD}\n")

    df = generate()
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    snapshot = DATA_RAW / "churn_snapshot.csv"
    df.to_csv(snapshot, index=False)
    print(f"snapshot        {len(df):,} rows -> data/raw/churn_snapshot.csv")

    train, val, test = time_split(df)
    print(f"time split      train={len(train):,}  val={len(val):,}  test={len(test):,}")
    print(
        f"                train {train['signup_date'].min().date()} .. "
        f"{train['signup_date'].max().date()}   "
        f"test {test['signup_date'].min().date()} .. "
        f"{test['signup_date'].max().date()}"
    )
    print(
        f"churn rate      train={train[TARGET].mean():.3f}  "
        f"val={val[TARGET].mean():.3f}  test={test[TARGET].mean():.3f}"
    )

    # Transform runs here, but the baselines take raw columns. Both paths are
    # exercised so the contract is proven from day one.
    X_train, y_train = transform(train), train[TARGET].to_numpy()
    X_val, y_val = transform(val), val[TARGET].to_numpy()
    print(f"features        {len(FEATURE_COLUMNS)} columns\n")

    results: dict[str, dict[str, float]] = {}
    for cls in ALL_BASELINES:
        model = cls().fit(train, y_train)
        results[cls.name] = evaluate(y_val, model.predict_proba(val))

    print(format_table(results))
    maj = results["majority_class"]
    rule = results["rule_based"]
    print(
        "\nRead this table, not the accuracy column:\n"
        f"  majority_class  {maj['accuracy']:.1%} accurate and catches zero churners\n"
        f"                  (recall {maj['recall']:.3f}). Its AUC is exactly 0.500 --\n"
        "                  a constant score ranks nobody, which is a coin flip.\n"
        f"  rule_based      LOWER accuracy ({rule['accuracy']:.1%}) and far more useful:\n"
        f"                  recall {rule['recall']:.3f}, AUC {rule['roc_auc']:.3f}.\n"
        "                  This is the bar stage 02 has to clear."
    )
    print(
        f"\nBAR TO BEAT -> rule_based roc_auc = "
        f"{results['rule_based']['roc_auc']:.4f} (validation)"
    )
    print("The test split stays untouched until stage 06.")


if __name__ == "__main__":
    main()
