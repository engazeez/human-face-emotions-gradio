from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"

# Defaults
DEFAULT_MODEL_PATH = MODELS_DIR / "hfe_emotion_cnn.keras"
DEFAULT_CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

# Image preprocessing (must match training)
IMG_SIZE = 128
