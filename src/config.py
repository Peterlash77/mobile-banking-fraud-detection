"""
Central configuration for the fraud detection project.

Everything that might need changing lives here, so that no other module
contains a hard-coded path or magic number. This keeps the experiments
reproducible and makes the handover to the web application straightforward.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_FILE = PROJECT_ROOT / "creditcard.csv"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------

# A single seed used everywhere. Reported in the dissertation so that a reader
# can reproduce the exact split and the exact model weights.
RANDOM_SEED = 42

# Proportion of the data held out for final testing. The split is stratified,
# so the 0.17% fraud rate is preserved in both halves.
TEST_SIZE = 0.20

# Proportion of the *training* data held out for validation (threshold tuning
# and early stopping). Kept separate from the test set so that the test set is
# touched exactly once, at the very end.
VALIDATION_SIZE = 0.20

# The target column and the columns that require scaling.
# V1-V28 are already the output of a PCA transformation and are therefore
# approximately centred and comparable in magnitude. 'Time' and 'Amount' are
# raw values on completely different scales and must be treated separately.
TARGET_COLUMN = "Class"
COLUMNS_TO_SCALE = ["Time", "Amount"]

# Whether to drop exactly duplicated rows before splitting.
# This matters: 1,081 duplicate rows exist in the raw file. If they are kept,
# the same record can appear in both the training and the test set, which
# leaks information and inflates the reported scores.
DROP_DUPLICATES = True
