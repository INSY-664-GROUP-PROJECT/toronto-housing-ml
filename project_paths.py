from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
FEATURES_DIR = ARTIFACTS_DIR / "features"

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def model_artifact_path(name: str) -> Path:
    return MODELS_DIR / name


def feature_artifact_path(name: str) -> Path:
    return FEATURES_DIR / name
