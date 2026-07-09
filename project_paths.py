"""
Central path configuration for the GitHub version of the BUSI project.
All scripts should be run from the project root, for example:
    python scripts/01_visualize_ours_prediction_boundary.py --device cpu
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = SRC_DIR / "data"
BUSI_ROOT = DATA_DIR / "Dataset_BUSI_with_GT"
if not (BUSI_ROOT / "benign").exists() and (BUSI_ROOT / "Dataset_BUSI_with_GT" / "benign").exists():
    BUSI_ROOT = BUSI_ROOT / "Dataset_BUSI_with_GT"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
UTILS_DIR = PROJECT_ROOT / "utils"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "results"
WEIGHTS_DIR = RESULTS_DIR / "weights"
LOGS_DIR = RESULTS_DIR / "logs"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"
VIS_OUTPUT_DIR = OUTPUTS_DIR / "visualization" / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

def setup_paths() -> Path:
    """Make imports stable after folders were moved to src/scripts/utils."""
    for p in [PROJECT_ROOT, SRC_DIR, SCRIPTS_DIR, UTILS_DIR]:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return PROJECT_ROOT

def as_path(path) -> Path:
    """Resolve a user-provided path relative to PROJECT_ROOT."""
    p = Path(str(path))
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p

def ensure_dir(path) -> Path:
    p = as_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
