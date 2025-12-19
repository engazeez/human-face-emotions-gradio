from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image

# Keras (TensorFlow backend)
from tensorflow import keras


def load_class_names(class_names_path: Path) -> List[str]:
    """
    Loads class names from JSON.
    Accepted formats:
      - ["Angry","Fear","Happy","Sad","Suprise"]
      - {"classes":[...]}
    """
    if not class_names_path.exists():
        raise FileNotFoundError(
            f"Missing class names file: {class_names_path}. "
            "Create models/class_names.json to match training label order."
        )

    data = json.loads(class_names_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict) and "classes" in data and isinstance(data["classes"], list):
        return [str(x) for x in data["classes"]]

    raise ValueError("class_names.json must be a list or a dict with key 'classes'.")


def load_model(model_path: Path) -> keras.Model:
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    return keras.models.load_model(model_path)


def preprocess_pil(img: Image.Image, img_size: int) -> np.ndarray:
    """
    Preprocess to match training:
      - RGB
      - resize to (img_size, img_size)
      - normalize to [0,1]
      - add batch dim
    """
    img = img.convert("RGB").resize((img_size, img_size))
    x = np.asarray(img).astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    return x


def predict_image(
    model: keras.Model,
    class_names: List[str],
    img: Image.Image,
    img_size: int,
) -> Tuple[Dict[str, float], str]:
    if img is None:
        return {}, "No image provided."

    x = preprocess_pil(img, img_size)
    probs = model.predict(x, verbose=0)[0]
    probs = probs.astype(float)

    pred_id = int(np.argmax(probs))
    pred_label = class_names[pred_id]

    scores = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    return scores, pred_label
