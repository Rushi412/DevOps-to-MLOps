"""Single source of truth for paths, seeds and split boundaries.

Stage 02 note: everything in here becomes an MLflow *param*, so that a run
can be tied back to the exact configuration that produced it. Config is one
of the four things that must be versioned together (curriculum module 02).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# Pinning the seed does not make training bit-identical, but it removes the
# single largest source of run-to-run variance. Record it either way.
RANDOM_SEED = 42

N_CUSTOMERS = 12_000

# Time-ordered split. Train on the past, test on the future -- a random split
# would let the model peek at tomorrow and the offline score becomes fiction.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test = the remaining 0.15, i.e. the most recent slice

TARGET = "churned"

# The decision threshold is config, not part of the model. You will want to
# tune it without retraining (curriculum module 01).
DECISION_THRESHOLD = 0.50
