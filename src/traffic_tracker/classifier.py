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


def preprocess_crop(crop_bgr: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> Optional[Image.Image]:
    """Properly convert BGR crop from OpenCV to RGB PIL Image for torchvision consistency."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    if len(crop_bgr.shape) == 2:
        crop_bgr = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2BGR)
    elif len(crop_bgr.shape) == 3 and crop_bgr.shape[2] == 4:
        crop_bgr = cv2.cvtColor(crop_bgr, cv2.COLOR_BGRA2BGR)
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


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

    # True chromatic (vibrant color) mask requires significant saturation
    chromatic_mask = (S >= 70) & (V >= 50)
    achromatic_mask = ~chromatic_mask

    counts = {
        "Beige": 0, "Black": 0, "Blue": 0, "Brown": 0, "Gold": 0,
        "Green": 0, "Grey": 0, "Orange": 0, "Pink": 0, "Purple": 0,
        "Red": 0, "Silver": 0, "Tan": 0, "White": 0, "Yellow": 0,
    }

    # Chromatic classification by Hue & Value
    counts["Red"] = int((chromatic_mask & (((H >= 0) & (H <= 10)) | ((H >= 165) & (H <= 180)))).sum())
    counts["Pink"] = int((chromatic_mask & (H > 150) & (H < 165) & (S < 140)).sum())
    counts["Orange"] = int((chromatic_mask & ((H > 10) & (H <= 25) & (V >= 130))).sum())
    counts["Brown"] = int((chromatic_mask & ((H > 10) & (H <= 25) & (V < 130) & (S > 80))).sum())
    counts["Gold"] = int((chromatic_mask & ((H > 22) & (H <= 35) & (V >= 150) & (S >= 90) & (S < 160))).sum())
    counts["Yellow"] = int((chromatic_mask & ((H > 22) & (H <= 38) & (V >= 150) & (S >= 160))).sum())
    counts["Green"] = int((chromatic_mask & ((H > 38) & (H <= 85))).sum())
    counts["Blue"] = int((chromatic_mask & ((H > 85) & (H <= 140))).sum())
    counts["Purple"] = int((chromatic_mask & ((H > 140) & (H <= 160) & (S >= 90))).sum())

    # Achromatic / Neutral classification by Value & Saturation
    counts["White"] = int((achromatic_mask & (V >= 190)).sum())
    counts["Silver"] = int((achromatic_mask & (V >= 130) & (V < 190) & (S <= 35)).sum())
    counts["Beige"] = int((achromatic_mask & (V >= 140) & (S > 35) & (H >= 15) & (H <= 40)).sum())
    counts["Tan"] = int((achromatic_mask & (V >= 90) & (V < 140) & (S > 35) & (H >= 10) & (H <= 40)).sum())
    counts["Grey"] = int((achromatic_mask & (V >= 60) & (V < 130) & (S <= 35)).sum())
    counts["Black"] = int((achromatic_mask & (V < 60)).sum())

    total_chromatic = sum(counts[c] for c in ["Red", "Pink", "Orange", "Brown", "Gold", "Yellow", "Green", "Blue", "Purple"])

    # Require at least 35% chromatic pixels to consider the car a vibrant color
    if total_chromatic >= total_pixels * 0.35:
        chromatic_colors = ["Red", "Pink", "Orange", "Brown", "Gold", "Yellow", "Green", "Blue", "Purple"]
        best_color = max(chromatic_colors, key=lambda c: counts[c])
        conf = counts[best_color] / max(total_chromatic, 1)
    else:
        neutral_colors = ["White", "Silver", "Beige", "Tan", "Grey", "Black"]
        best_color = max(neutral_colors, key=lambda c: counts[c])
        conf = counts[best_color] / max(total_pixels - total_chromatic, 1)

    return best_color, round(conf, 3)


# ── Unified Multi-Task Network ────────────────────────────────────────────
class VehicleAttributeNet(nn.Module):
    """
    Unified Multi-Task Network for vehicle color (15 classes) and body type (7 classes).
    Uses a shared MobileNetV3-Large backbone and two specialized classification heads.
    """
    def __init__(self, n_colors: int = 15, n_types: int = 7, pretrained: bool = False):
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v3_large(weights=weights)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_feat = 960  # MobileNetV3-Large output feature channels

        self.color_head = nn.Sequential(
            nn.Linear(in_feat, 256),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(256, n_colors),
        )
        self.type_head = nn.Sequential(
            nn.Linear(in_feat, 256),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(256, n_types),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x)
        feat = self.pool(feat)
        feat = torch.flatten(feat, 1)
        color_logits = self.color_head(feat)
        type_logits = self.type_head(feat)
        return color_logits, type_logits


# ── MobileNetV3 Builders (Single-Task Legacy) ──────────────────────────────
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
    Supports unified multi-task network (VehicleAttributeNet) or legacy dual models.

    Usage:
        clf = VehicleClassifier(config)
        color_probs, type_probs = clf.predict_attributes_probs(crop_bgr)
        color_probs = clf.predict_color_probs(crop_bgr)
        type_probs  = clf.predict_type_probs(crop_bgr)
    """

    def __init__(self, config: dict):
        self.cfg_clf = config.get("classification", {})
        self.cfg_paths = config.get("paths", {})
        self.input_size = self.cfg_clf.get("input_size", 224)
        self.conf_cutoff = self.cfg_clf.get("confidence_cutoff", 0.35)

        # Default class lists — must EXACTLY match the JSON files in alphabetical order
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

        dev_str = config.get("detection", {}).get("device", "cpu")
        if dev_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested for classifier but not available. Falling back to CPU.")
            dev_str = "cpu"
        self.device = torch.device(dev_str)
        self.use_fp16 = (self.device.type == "cuda")
        self.transform = _make_transform(self.input_size)

        # 1. Attempt to load unified multi-task attribute network
        multi_task_path = self.cfg_paths.get("vehicle_attributes_weights", str(models_dir / "vehicle_attributes.pt"))
        self.multitask_model: Optional[VehicleAttributeNet] = self._load_multitask_model(multi_task_path)

        # 2. If no multi-task model, load legacy dual models
        self.color_model: Optional[nn.Module] = None
        self.type_model: Optional[nn.Module] = None
        if self.multitask_model is None:
            self.color_model = self._load_model(
                self.cfg_paths.get("color_weights", ""),
                len(self.color_classes),
                "color",
            )
            self.type_model = self._load_model(
                self.cfg_paths.get("type_weights", ""),
                len(self.type_classes),
                "type",
            )

    # ── Probability distribution API (for continuous EMA smoothing) ─────

    @torch.no_grad()
    def predict_attributes_probs(self, crop_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run a single forward pass through the unified multi-task network.
        Returns (color_probs, type_probs) simultaneously.
        """
        n_c = len(self.color_classes)
        n_t = len(self.type_classes)
        if crop_bgr is None or crop_bgr.size == 0:
            return (
                np.ones(n_c, dtype=np.float32) / max(n_c, 1),
                np.ones(n_t, dtype=np.float32) / max(n_t, 1),
            )

        if self.multitask_model is not None:
            pil_img = preprocess_crop(crop_bgr)
            if pil_img is None:
                return (
                    np.ones(n_c, dtype=np.float32) / max(n_c, 1),
                    np.ones(n_t, dtype=np.float32) / max(n_t, 1),
                )
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            if self.use_fp16:
                with torch.cuda.amp.autocast():
                    c_logits, t_logits = self.multitask_model(tensor)
            else:
                c_logits, t_logits = self.multitask_model(tensor)

            c_probs = torch.softmax(c_logits.float(), dim=1)[0].cpu().numpy()
            t_probs = torch.softmax(t_logits.float(), dim=1)[0].cpu().numpy()
            return c_probs, t_probs

        # Fallback to separate models or heuristics
        return self.predict_color_probs(crop_bgr), self.predict_type_probs(crop_bgr)

    # ── Probability distribution API (for continuous EMA smoothing) ─────

    def predict_color_probs(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Return a 1D probability distribution over color_classes."""
        n_classes = len(self.color_classes) if self.color_classes is not None else 15
        if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
            return np.ones(n_classes, dtype=np.float32) / max(n_classes, 1)

        if self.multitask_model is not None:
            return self.predict_attributes_probs(crop_bgr)[0]

        if self.color_model is not None:
            return self._nn_predict_probs(crop_bgr, self.color_model)

        # Fallback to HSV histogram distribution
        hsv_color, hsv_conf = _hsv_color_fallback(crop_bgr)
        probs = np.ones(n_classes, dtype=np.float32) * ((1.0 - hsv_conf) / max(n_classes - 1, 1))
        if self.color_classes:
            lower_classes = [c.lower() for c in self.color_classes]
            if hsv_color.lower() in lower_classes:
                probs[lower_classes.index(hsv_color.lower())] = hsv_conf
        return probs

    def predict_type_probs(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Return a 1D probability distribution over type_classes."""
        n_classes = len(self.type_classes) if self.type_classes is not None else 7
        if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
            return np.ones(n_classes, dtype=np.float32) / max(n_classes, 1)

        if self.multitask_model is not None:
            return self.predict_attributes_probs(crop_bgr)[1]

        if self.type_model is not None:
            return self._nn_predict_probs(crop_bgr, self.type_model)

        # Fallback to aspect ratio geometric distribution
        probs = np.ones(n_classes, dtype=np.float32) * (0.20 / max(n_classes - 1, 1))
        h, w = crop_bgr.shape[:2]
        if h > 0 and w > 0:
            ar = w / float(h)
            if ar < 1.10 and "SUV" in self.type_classes:
                probs[self.type_classes.index("SUV")] = 0.80
            elif 1.10 <= ar < 1.25 and "Hatchback" in self.type_classes:
                probs[self.type_classes.index("Hatchback")] = 0.80
            elif 1.25 <= ar < 1.45 and "Sedan" in self.type_classes:
                probs[self.type_classes.index("Sedan")] = 0.80
            elif 1.45 <= ar < 1.68 and "Coupe" in self.type_classes:
                probs[self.type_classes.index("Coupe")] = 0.80
            elif "Truck" in self.type_classes:
                probs[self.type_classes.index("Truck")] = 0.80
        return probs / np.sum(probs)

    # ── Single-frame prediction API ─────────────────────────────────────

    def predict_color(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """Returns (color_label, confidence)."""
        probs = self.predict_color_probs(crop_bgr)
        if probs is None or len(probs) == 0:
            return "Unknown", 0.0
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = self.color_classes[idx] if (0 <= idx < len(self.color_classes) and conf >= self.conf_cutoff) else "Unknown"
        return label, round(conf, 3)

    def predict_type(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """Returns (type_label, confidence)."""
        probs = self.predict_type_probs(crop_bgr)
        if probs is None or len(probs) == 0:
            return "Unknown", 0.0
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = self.type_classes[idx] if (0 <= idx < len(self.type_classes) and conf >= self.conf_cutoff) else "Unknown"
        return label, round(conf, 3)

    # ── Internal helpers ────────────────────────────────────────────────

    @torch.no_grad()
    def _nn_predict_probs(
        self,
        crop_bgr: np.ndarray,
        model: nn.Module,
    ) -> np.ndarray:
        """Run a BGR crop through a MobileNetV3 model and return softmax probability vector."""
        n_classes = len(self.color_classes) if model == self.color_model else len(self.type_classes)
        pil_img = preprocess_crop(crop_bgr)
        if pil_img is None:
            return np.ones(n_classes, dtype=np.float32) / max(n_classes, 1)

        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        if len(probs) != n_classes:
            if len(probs) > n_classes:
                probs = probs[:n_classes]
                probs = probs / (np.sum(probs) + 1e-9)
            else:
                padded = np.zeros(n_classes, dtype=np.float32)
                padded[:len(probs)] = probs
                probs = padded
        return probs

    def _load_multitask_model(self, weights_path: str) -> Optional[VehicleAttributeNet]:
        """Load unified Multi-Task VehicleAttributeNet weights."""
        if not weights_path or not Path(weights_path).exists():
            return None

        try:
            state = torch.load(weights_path, map_location=self.device, weights_only=False)
            cleaned_state = {}
            for k, v in state.items():
                new_key = k[4:] if k.startswith("net.") else k
                cleaned_state[new_key] = v

            n_c = len(self.color_classes)
            n_t = len(self.type_classes)
            if "color_head.3.weight" in cleaned_state:
                n_c = cleaned_state["color_head.3.weight"].shape[0]
            if "type_head.3.weight" in cleaned_state:
                n_t = cleaned_state["type_head.3.weight"].shape[0]

            model = VehicleAttributeNet(n_colors=n_c, n_types=n_t, pretrained=False)
            model.load_state_dict(cleaned_state)
            model.to(self.device)
            model.eval()
            logger.info(f"Loaded unified Multi-Task VehicleAttributeNet ({n_c} colors, {n_t} types) from: {weights_path}")
            return model
        except Exception as e:
            logger.warning(f"Could not load multi-task attribute weights from {weights_path}: {e}")
            return None

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
