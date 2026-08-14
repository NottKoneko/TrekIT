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

def _hsv_color_fallback(crop_bgr: np.ndarray) -> Tuple[str, float]:
    """Estimate dominant vehicle body colour from HSV analysis of central region."""
    if crop_bgr is None or crop_bgr.size == 0:
        return "Unknown", 0.0

    h, w = crop_bgr.shape[:2]
    # Crop central region (avoid road, wheels, outer background shadows, windshields)
    margin_h = int(h * 0.20)
    margin_w = int(w * 0.20)
    body_crop = crop_bgr[margin_h:max(margin_h + 1, h - margin_h), margin_w:max(margin_w + 1, w - margin_w)]
    if body_crop.size == 0:
        body_crop = crop_bgr

    hsv = cv2.cvtColor(body_crop, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    total_pixels = body_crop.shape[0] * body_crop.shape[1]
    if total_pixels == 0:
        return "Unknown", 0.0

    # True chromatic (vibrant color) vs achromatic (neutral) mask
    chromatic_mask = (S >= 60) & (V >= 45)
    achromatic_mask = ~chromatic_mask

    counts = {
        "Beige": 0, "Black": 0, "Blue": 0, "Brown": 0, "Gold": 0,
        "Green": 0, "Grey": 0, "Orange": 0, "Pink": 0, "Purple": 0,
        "Red": 0, "Silver": 0, "Tan": 0, "White": 0, "Yellow": 0,
    }

    # Chromatic classification by Hue & Value
    counts["Red"] = int((chromatic_mask & (((H >= 0) & (H <= 10)) | ((H >= 165) & (H <= 180)))).sum())
    counts["Pink"] = int((chromatic_mask & (H > 150) & (H < 165)).sum())
    counts["Orange"] = int((chromatic_mask & ((H > 10) & (H <= 25) & (V >= 120))).sum())
    counts["Brown"] = int((chromatic_mask & ((H > 10) & (H <= 25) & (V < 120) & (S > 80))).sum())
    counts["Gold"] = int((chromatic_mask & ((H > 25) & (H <= 38) & (V >= 150) & (S < 150))).sum())
    counts["Yellow"] = int((chromatic_mask & ((H > 25) & (H <= 38) & ((V < 150) | (S >= 150)))).sum())
    counts["Green"] = int((chromatic_mask & ((H > 38) & (H <= 85))).sum())
    counts["Blue"] = int((chromatic_mask & ((H > 85) & (H <= 135))).sum())
    counts["Purple"] = int((chromatic_mask & ((H > 135) & (H <= 150))).sum())

    # Achromatic / Neutral classification by Value & Saturation
    counts["White"] = int((achromatic_mask & (V >= 185)).sum())
    counts["Silver"] = int((achromatic_mask & (V >= 120) & (V < 185) & (S <= 40)).sum())
    counts["Beige"] = int((achromatic_mask & (V >= 140) & (S > 40) & (H >= 15) & (H <= 40)).sum())
    counts["Tan"] = int((achromatic_mask & (V >= 90) & (V < 140) & (S > 35) & (H >= 10) & (H <= 40)).sum())
    counts["Grey"] = int((achromatic_mask & (V >= 60) & (V < 120) & (S <= 40)).sum())
    counts["Black"] = int((achromatic_mask & (V < 60)).sum())

    total_chromatic = sum(counts[c] for c in ["Red", "Pink", "Orange", "Brown", "Gold", "Yellow", "Green", "Blue", "Purple"])

    if total_chromatic >= total_pixels * 0.20:
        chromatic_colors = ["Red", "Pink", "Orange", "Brown", "Gold", "Yellow", "Green", "Blue", "Purple"]
        best_color = max(chromatic_colors, key=lambda c: counts[c])
        conf = counts[best_color] / max(total_chromatic, 1)
    else:
        neutral_colors = ["White", "Silver", "Beige", "Tan", "Grey", "Black"]
        best_color = max(neutral_colors, key=lambda c: counts[c])
        conf = counts[best_color] / max(total_pixels - total_chromatic, 1)

    return best_color, round(conf, 3)


# ── MobileNetV3 Builders ───────────────────────────────────────────────────
def _build_mobilenet_large(num_classes: int) -> nn.Module:
    """Build a MobileNetV3-Large model."""
    base = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
    )
    in_features = base.classifier[-1].in_features
    base.classifier[-1] = nn.Linear(in_features, num_classes)
    return base


def _build_mobilenet_small(num_classes: int) -> nn.Module:
    """Build a MobileNetV3-Small model."""
    base = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )
    in_features = base.classifier[-1].in_features
    base.classifier[-1] = nn.Linear(in_features, num_classes)
    return base


# ── Public classifier class ─────────────────────────────────────────────────
class VehicleClassifier:
    """
    Classifies vehicle color (15 classes) and body type (7 classes).

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
        Ensembles neural classifier with physical HSV body panel analyzer.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return "Unknown", 0.0

        hsv_color, hsv_conf = _hsv_color_fallback(crop_bgr)

        if self.color_model is not None:
            nn_color, nn_conf = self._nn_predict(crop_bgr, self.color_model, self.color_classes)
            if nn_color == "Gray":
                nn_color = "Grey"

            # If neural model is highly confident (>= 0.60), trust neural model prediction
            if nn_conf >= 0.60:
                return nn_color, nn_conf

            # If physical HSV body panel analyzer has strong color (>= 0.25), use physical color
            if hsv_conf >= 0.25:
                return hsv_color, hsv_conf

            return nn_color if nn_color != "Unknown" else hsv_color, max(nn_conf, hsv_conf)

        return hsv_color, hsv_conf

    def predict_type(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Returns (type_label, confidence).
        Uses neural model if available, else geometric aspect-ratio fallback.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return "Unknown", 0.0

        if self.type_model is not None:
            return self._nn_predict(crop_bgr, self.type_model, self.type_classes)

        # Fallback: Aspect ratio geometric profile
        h, w = crop_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return "Unknown", 0.0

        ar = w / float(h)   # width / height ratio
        if ar < 1.08:
            return "SUV", 0.75
        elif 1.08 <= ar < 1.24:
            return "Hatchback", 0.70
        elif 1.24 <= ar < 1.44:
            return "Sedan", 0.78
        elif 1.44 <= ar < 1.68:
            return "Coupe", 0.72
        else:
            return "Truck", 0.80

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
        """Load MobileNetV3 from weights file, auto-detecting Small vs Large architecture."""
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

            # Detect output class dimension from classifier head
            if "classifier.3.weight" in cleaned_state:
                num_classes = cleaned_state["classifier.3.weight"].shape[0]

            # Detect Small vs Large variant from classifier.0 input dimension (576 vs 960)
            is_small = False
            if "classifier.0.weight" in cleaned_state:
                in_feat = cleaned_state["classifier.0.weight"].shape[1]
                if in_feat == 576:
                    is_small = True

            if is_small:
                model = _build_mobilenet_small(num_classes)
                arch_name = "MobileNetV3-Small"
            else:
                model = _build_mobilenet_large(num_classes)
                arch_name = "MobileNetV3-Large"

            model.load_state_dict(cleaned_state)
            model.to(self.device)
            model.eval()
            logger.info(f"Loaded {name} classifier ({arch_name}, {num_classes} classes) from: {weights_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load {name} classifier: {e}")
            return None
