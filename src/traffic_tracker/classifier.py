"""
classifier.py
-------------
MobileNetV3-Large classifier for vehicle color and body type.

Two separate single-task models (color, type) — simpler and more reliable
than multi-task on limited training data.

Fallback behaviour (before Colab training):
  - Color: HSV-histogram KNN classifier (no neural net needed)
  - Body type: returns "Unknown" — requires fine-tuned weights

NOTE: Training notebook uses MobileNetV3-Large. This file MUST stay in sync.
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)


# ── Transform used for all MobileNetV3 inference ───────────────────────────
def _make_transform(input_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── Colour lookup (HSV-based fallback) ─────────────────────────────────────
# (hue_low, hue_high, sat_min, val_min) — works in OpenCV HSV space
_HSV_RANGES = {
    "Red":    [(0, 10, 80, 60), (160, 180, 80, 60)],   # wraps around 0/180
    "Orange": [(10, 25, 100, 80)],
    "Yellow": [(25, 35, 100, 80)],
    "Green":  [(35, 85, 60, 60)],
    "Blue":   [(85, 130, 60, 60)],
    "Purple": [(130, 160, 60, 60)],
    "White":  [(0, 180, 0, 200)],
    "Silver": [(0, 180, 0, 140)],
    "Gray":   [(0, 180, 0, 80)],
    "Black":  [(0, 180, 0, 0)],
    "Brown":  [(10, 20, 60, 40)],
}


def _hsv_color_fallback(crop_bgr: np.ndarray) -> Tuple[str, float]:
    """Estimate dominant colour from an HSV histogram. Returns (color, confidence)."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    # Build per-pixel colour label map
    votes: Counter = Counter()
    mask_all = np.zeros(hsv.shape[:2], dtype=bool)

    ordered = [
        "Black", "White", "Silver", "Gray",
        "Red", "Orange", "Yellow", "Green", "Blue", "Purple", "Brown",
    ]
    for color in ordered:
        ranges = _HSV_RANGES.get(color, [])
        mask = np.zeros(hsv.shape[:2], dtype=bool)
        for r in ranges:
            h_lo, h_hi, s_min, v_min = r
            m = (
                (hsv[:, :, 0] >= h_lo) & (hsv[:, :, 0] <= h_hi) &
                (hsv[:, :, 1] >= s_min) &
                (hsv[:, :, 2] >= v_min)
            ) & ~mask_all
            mask |= m
            mask_all |= m
        votes[color] = int(mask.sum())

    total = hsv.shape[0] * hsv.shape[1]
    best_color, best_count = votes.most_common(1)[0]
    confidence = best_count / max(total, 1)
    return best_color, round(confidence, 3)


# ── MobileNetV3 Builder ─────────────────────────────────────────────────────
def _build_mobilenet(num_classes: int) -> nn.Module:
    """Build a MobileNetV3-Large model matching torchvision layout.

    Uses Large to match the training notebook (mobilenet_v3_large).
    Large has a 1280-dim classifier head vs 576 in Small — state dict
    will fail to load if the wrong variant is used here.
    """
    base = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
    )
    in_features = base.classifier[-1].in_features
    base.classifier[-1] = nn.Linear(in_features, num_classes)
    return base


# ── Public classifier class ─────────────────────────────────────────────────
class VehicleClassifier:
    """
    Classifies vehicle color (10 classes) and body type (7 classes).

    Usage:
        clf = VehicleClassifier(config)
        color, color_conf = clf.predict_color(crop_bgr)
        vtype, type_conf  = clf.predict_type(crop_bgr)
    """

    def __init__(self, config: dict):
        self.cfg_clf = config.get("classification", {})
        self.cfg_paths = config.get("paths", {})
        self.input_size = self.cfg_clf.get("input_size", 224)
        self.conf_cutoff = self.cfg_clf.get("confidence_cutoff", 0.40)

        # Default class lists — must EXACTLY match the JSON files in alphabetical order
        # (torchvision ImageFolder sorts class folders alphabetically)
        self.color_classes = [
            "Beige", "Black", "Blue", "Brown", "Gold", "Green", "Grey",
            "Orange", "Pink", "Purple", "Red", "Silver", "Tan", "White", "Yellow",
        ]
        self.type_classes = ["Convertible", "Coupe", "Hatchback", "SUV", "Sedan", "Truck", "Van"]

        # Try loading exact class names from JSON files if present
        models_dir = Path(self.cfg_paths.get("models_dir", "models"))
        color_json = models_dir / "color_classes.json"
        type_json = models_dir / "type_classes.json"

        if color_json.exists():
            try:
                import json
                with open(color_json, "r") as f:
                    self.color_classes = json.load(f)
                logger.info(f"Loaded {len(self.color_classes)} color classes from {color_json}")
            except Exception as e:
                logger.warning(f"Could not load color_classes.json: {e}")

        if type_json.exists():
            try:
                import json
                with open(type_json, "r") as f:
                    self.type_classes = json.load(f)
                logger.info(f"Loaded {len(self.type_classes)} type classes from {type_json}")
            except Exception as e:
                logger.warning(f"Could not load type_classes.json: {e}")

        self.device = torch.device(
            config.get("detection", {}).get("device", "cpu")
        )
        self.transform = _make_transform(self.input_size)

        self.color_model: Optional[nn.Module] = self._load_model(
            self.cfg_paths.get("color_weights", ""),
            len(self.color_classes),
            "color",
        )
        self.type_model: Optional[nn.Module] = self._load_model(
            self.cfg_paths.get("type_weights", ""),
            len(self.type_classes),
            "type",
        )

    # ── Public methods ──────────────────────────────────────────────────

    def predict_color(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Returns (color_label, confidence).
        Uses neural model if available, else HSV-histogram fallback.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return "Unknown", 0.0

        if self.color_model is not None:
            return self._nn_predict(crop_bgr, self.color_model, self.color_classes)

        # Fallback: HSV histogram
        return _hsv_color_fallback(crop_bgr)

    def predict_type(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Returns (type_label, confidence).
        Requires fine-tuned weights; returns 'Unknown' without them.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return "Unknown", 0.0

        if self.type_model is not None:
            return self._nn_predict(crop_bgr, self.type_model, self.type_classes)

        return "Unknown", 0.0

    # ── Internal helpers ────────────────────────────────────────────────

    @torch.no_grad()
    def _nn_predict(
        self,
        crop_bgr: np.ndarray,
        model: nn.Module,
        classes: list,
    ) -> Tuple[str, float]:
        """Run a single crop through a MobileNetV3 model."""
        img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = classes[idx] if conf >= self.conf_cutoff else "Unknown"
        return label, round(conf, 3)

    def _load_model(
        self,
        weights_path: str,
        num_classes: int,
        name: str,
    ) -> Optional[nn.Module]:
        """Load MobileNetV3 from weights file, or None if not found."""
        if not weights_path or not Path(weights_path).exists():
            logger.warning(
                f"No {name} classifier weights found at '{weights_path}'. "
                f"{'Using HSV fallback.' if name == 'color' else 'Will return Unknown.'}"
            )
            return None

        try:
            state = torch.load(weights_path, map_location=self.device, weights_only=False)
            
            # Clean keys if saved with wrapper prefix (e.g. 'net.')
            cleaned_state = {}
            for k, v in state.items():
                new_key = k[4:] if k.startswith("net.") else k
                cleaned_state[new_key] = v

            # Detect output dimension from classifier head weight shape
            if "classifier.3.weight" in cleaned_state:
                num_classes = cleaned_state["classifier.3.weight"].shape[0]

            model = _build_mobilenet(num_classes)
            model.load_state_dict(cleaned_state)
            model.to(self.device)
            model.eval()
            logger.info(f"Loaded {name} classifier ({num_classes} classes) from: {weights_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load {name} classifier: {e}")
            return None
