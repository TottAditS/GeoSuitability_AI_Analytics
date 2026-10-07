"""Shared test setup: import paths of the experiment and dashboard modules, and whether the (git-ignored) data exist."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("scripts/modeling", "scripts/dashboard"):
    path = str(ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)

FEATURES_CSV = ROOT / "datasets" / "processed" / "features" / "tabular_features.csv"
DATA_READY = FEATURES_CSV.exists()
SKIP_REASON = "the warehouse table is not built (datasets/ is not part of the repository; run the pipeline first)"
