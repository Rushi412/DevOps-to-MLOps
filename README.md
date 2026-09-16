# MLOps, built from scratch by a DevOps engineer

I'm a DevOps engineer, about a year in. I know how to build, ship and watch a
service. I do not know machine learning. This repo is me closing that gap in
public — eight stages, every commit, every wall I hit.

The companion study map is
[MLOps for Ops Engineers](https://claude.ai/artifact/Y2jeRbVZaFmwGEhrY78eft) —
the concepts, translated into ops terms, written before any code.

---

## Stage 01 — skeleton + baseline

**Status:** done · **Next:** stage 02, training pipeline with MLflow

### Run it

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt
python scripts/run_baseline.py
pytest -q
```

No network, no Kaggle account, no download. The dataset is generated from a
fixed seed.

### What it produces

```
snapshot        12,000 rows -> data/raw/churn_snapshot.csv
time split      train=8,400  val=1,800  test=1,800
                train 2023-01-01 .. 2024-09-23   test 2025-01-30 .. 2025-06-18
churn rate      train=0.182  val=0.188  test=0.199
features        13 columns

model                 accuracy     precision        recall            f1       roc_auc
--------------------------------------------------------------------------------------
majority_class          0.8122        0.0000        0.0000        0.0000        0.5000
base_rate               0.8122        0.0000        0.0000        0.0000        0.5000
rule_based              0.7972        0.4328        0.2574        0.3228        0.6899
```

**The bar for stage 02 is AUC 0.6899.** Not zero. Not "better than nothing."
A hand-written rule that took four lines.

### The thing I got wrong first

I assumed the baseline was a formality — generate data, predict the majority
class, move on. Then I looked at the table.

`majority_class` is **81.2% accurate**. If I'd reported that number to a
stakeholder with no other context, it would have sounded like a working model.
It catches **zero** churners. Recall is 0.000. It is a function that returns
`False`.

And `rule_based`, the one that's actually useful, is **less accurate** —
79.7%. It looks worse on the metric everyone instinctively reaches for, while
being the only one of the three that does anything at all.

The AUC column is where the truth is: 0.500 for both constant baselines,
0.6899 for the rule. 0.5 means the model ranks nobody above anybody — a coin
flip. That's the number I'll be trying to beat, and it's why the runner prints
a warning telling you not to read the accuracy column.

Small thing, but it bit me: I'd written in the output text that AUC would come
back as `nan` for a constant score. sklearn returns exactly 0.5 and doesn't
raise. Had to fix my own explanation to match what the code actually did.

### Decisions worth flagging

**Time-based split, not `train_test_split(shuffle=True)`.** Churn is
time-ordered. Splitting randomly trains on customers who signed up *after* the
ones you score, which inflates every metric. Train ends 2024-09-23; test
starts 2025-01-30. The model only ever sees the past.

**`src/features.py` exists before there is a model.** This is the one
transform, and stage 05's API will import this exact function. Training/serving
skew is the signature production ML bug and the fix is structural — if both
paths import the same code they cannot drift apart. There's a test
(`test_single_row_matches_batch`) that fails the moment they do.

**Explicit category levels, not `pd.get_dummies`.** get_dummies infers columns
from whatever it happens to see. A serving batch with no `two_year` customers
would silently produce a narrower matrix. `test_unseen_category_yields_all_zero_dummies`
covers the reverse case.

**Every denominator is guarded.** `tenure_months` can be near zero. An `inf`
reaching a model doesn't crash anything — it just makes a prediction wrong.

**The test split has not been touched.** It stays sealed until stage 06. Every
look at it leaks information and inflates the final estimate.

**Synthetic data, deliberately.** Trade-off: weaker portfolio signal than a
real dataset, but the repo runs anywhere with no setup, and stage 08 needs a
*drifted* version of the same distribution to make PSI mean anything.
`generate(drift=0.4)` already does that; nothing uses it yet.

### Layout

```
src/config.py      seeds, paths, split fractions, decision threshold
src/data.py        generator + time_split
src/features.py    the one transform. imported by everything.
src/baseline.py    majority / base-rate / hand-written rule
src/evaluate.py    metrics, reported together
scripts/run_baseline.py
tests/test_features.py
```

`requirements.txt` uses loose pins. That's a known gap — stage 02 replaces it
with a lockfile, because "same code, different library version, different
model" is one of the assumptions ML breaks.

---

## Stages

| # | Stage | Status |
|---|---|---|
| 01 | Skeleton + baseline | done |
| 02 | Training pipeline, MLflow tracking | next |
| 03 | Data contracts (Pandera) + DVC | |
| 04 | Shared transform module / anti-skew | |
| 05 | FastAPI serving | |
| 06 | ML test pyramid + eval gate | |
| 07 | Docker + GitHub Actions | |
| 08 | Drift monitoring (PSI / Evidently) | |

Environment: Python 3.13.3, numpy 2.5.3, pandas 2.3.3, scikit-learn 1.9.1.
