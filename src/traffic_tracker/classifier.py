"""
classifier.py
-------------
Multi-Head Vehicle Classifier for Color, Body Type, and View Orientation.
Supports PyTorch (CUDA FP16 / CPU) and ONNX Runtime acceleration with batched inference.
"""

import logging
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)

ORIENTATION_CLASSES = ["Rear", "Front", "Side/Angle"]


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


def _make_transform(input_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_body_crop(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    Focus on the vehicle body sheet-metal (middle-lower region),
    stripping away windows, tinted glass roofs, tree reflections, road asphalt, and shadows.
    """
    if image is None or image.size == 0:
        return None
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    bh = y2 - y1
    bw = x2 - x1
    if bh <= 20 or bw <= 20:
        return None

    crop_y1 = int(y1 + 0.25 * bh)
    crop_y2 = int(y2 - 0.10 * bh)
    crop_x1 = int(x1 + 0.15 * bw)
    crop_x2 = int(x2 - 0.15 * bw)

    if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
        return None
    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    return crop if crop.size > 0 else None


def _hsv_color_fallback(crop_bgr: np.ndarray) -> Tuple[str, float, float]:
    """
    Estimate dominant vehicle body colour from HSV analysis of central region.
    Returns (best_color, confidence, chromatic_ratio).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "Unknown", 0.0, 0.0

    h, w = crop_bgr.shape[:2]
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
        return "Unknown", 0.0, 0.0

    chromatic_mask = (S >= 110) & (V >= 50)
    achromatic_mask = ~chromatic_mask

    counts = {
        "Beige": 0, "Black": 0, "Blue": 0, "Brown": 0, "Gold": 0,
        "Green": 0, "Grey": 0, "Orange": 0, "Pink": 0, "Purple": 0,
        "Red": 0, "Silver": 0, "Tan": 0, "White": 0, "Yellow": 0,
    }

    counts["Red"] = int((chromatic_mask & (((H >= 0) & (H <= 10)) | ((H >= 165) & (H <= 180)))).sum())
    counts["Pink"] = int((chromatic_mask & (H > 150) & (H < 165) & (V >= 130)).sum())
    counts["Orange"] = int((chromatic_mask & ((H > 10) & (H <= 25) & (V >= 130))).sum())
    counts["Brown"] = int((chromatic_mask & ((H > 10) & (H <= 25) & (V < 130))).sum())
    counts["Gold"] = int((chromatic_mask & ((H > 22) & (H <= 35) & (V >= 150) & (S < 160))).sum())
    counts["Yellow"] = int((chromatic_mask & ((H > 22) & (H <= 38) & (V >= 150) & (S >= 160))).sum())
    counts["Green"] = int((chromatic_mask & ((H > 38) & (H <= 85))).sum())
    counts["Blue"] = int((chromatic_mask & ((H > 85) & (H <= 140))).sum())
    counts["Purple"] = int((chromatic_mask & ((H > 140) & (H <= 160))).sum())

    counts["White"] = int((achromatic_mask & (V >= 185) & (S <= 45)).sum())
    counts["Silver"] = int((achromatic_mask & (V >= 125) & (V < 185) & (S <= 45)).sum())
    counts["Beige"] = int((achromatic_mask & (V >= 135) & (S > 45) & (H >= 15) & (H <= 40)).sum())
    counts["Tan"] = int((achromatic_mask & (V >= 75) & (V < 135) & (S > 45) & (H >= 10) & (H <= 40)).sum())
    counts["Grey"] = int((achromatic_mask & (V >= 50) & (V < 125) & (S <= 45)).sum())
    counts["Black"] = int((achromatic_mask & (V < 50)).sum())

    total_chromatic = sum(counts[c] for c in ["Red", "Pink", "Orange", "Brown", "Gold", "Yellow", "Green", "Blue", "Purple"])
    chromatic_ratio = float(total_chromatic / max(total_pixels, 1))

    if total_chromatic >= total_pixels * 0.35:
        chromatic_colors = ["Red", "Pink", "Orange", "Brown", "Gold", "Yellow", "Green", "Blue", "Purple"]
        best_color = max(chromatic_colors, key=lambda c: counts[c])
        conf = counts[best_color] / max(total_chromatic, 1)
    else:
        neutral_colors = ["White", "Silver", "Beige", "Tan", "Grey", "Black"]
        best_color = max(neutral_colors, key=lambda c: counts[c])
        conf = counts[best_color] / max(total_pixels - total_chromatic, 1)

    return best_color, round(conf, 3), round(chromatic_ratio, 3)


class VehicleAttributeNet(nn.Module):
    """
    Unified Multi-Task Network for vehicle color (15 classes), body type (7 classes),
    and view orientation (3 classes).
    """
    def __init__(self, n_colors: int = 15, n_types: int = 7, n_orientations: int = 3, pretrained: bool = False):
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v3_large(weights=weights)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_feat = 960

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
        self.orientation_head = nn.Sequential(
            nn.Linear(in_feat, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, n_orientations),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.features(x)
        feat = self.pool(feat)
        feat = torch.flatten(feat, 1)
        return self.color_head(feat), self.type_head(feat), self.orientation_head(feat)


ExtendedVehicleAttributeNet = VehicleAttributeNet  # Backward-compatible alias


class VehicleClassifier:
    """
    Production Multi-Head Vehicle Classifier (Color, Type, Orientation).
    Supports ONNX Runtime and PyTorch with batch processing.
    """

    def __init__(self, config: dict):
        self.cfg_clf = config.get("classification", {})
        self.cfg_paths = config.get("paths", {})
        self.input_size = self.cfg_clf.get("input_size", 224)
        self.conf_cutoff = self.cfg_clf.get("confidence_cutoff", 0.35)

        self.color_classes = [
            "Beige", "Black", "Blue", "Brown", "Gold", "Green", "Grey",
            "Orange", "Pink", "Purple", "Red", "Silver", "Tan", "White", "Yellow",
        ]
        self.type_classes = ["Convertible", "Coupe", "Hatchback", "SUV", "Sedan", "Truck", "Van"]
        self.orientation_classes = ORIENTATION_CLASSES

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

        # ── ONNX Session Initialization ─────────────────────────────────────
        self.ort_session = None
        onnx_path = self.cfg_paths.get("vehicle_attributes_onnx", str(models_dir / "vehicle_attributes.onnx"))
        if Path(onnx_path).exists():
            self.ort_session = self._init_onnx_session(onnx_path)

        # ── PyTorch Multi-Task Model Fallback ────────────────────────────────
        self.multitask_model: Optional[VehicleAttributeNet] = None
        if self.ort_session is None:
            multi_task_path = self.cfg_paths.get("vehicle_attributes_weights", str(models_dir / "vehicle_attributes.pt"))
            self.multitask_model = self._load_multitask_model(multi_task_path)

    def _init_onnx_session(self, onnx_path: str):
        """Initializes ONNX Runtime session with hardware acceleration providers."""
        try:
            import onnxruntime as ort
            avail = ort.get_available_providers()
            providers = []
            if "CUDAExecutionProvider" in avail and torch.cuda.is_available():
                providers.append("CUDAExecutionProvider")
            if "TensorrtExecutionProvider" in avail and torch.cuda.is_available():
                providers.append("TensorrtExecutionProvider")
            providers.append("CPUExecutionProvider")

            session = ort.InferenceSession(onnx_path, providers=providers)
            logger.info(f"Loaded ONNX VehicleAttributeNet from {onnx_path} (Providers: {session.get_providers()})")
            return session
        except Exception as e:
            logger.warning(f"Could not initialize ONNX session for {onnx_path}: {e}")
            return None

    @torch.no_grad()
    def predict_attributes_probs(
        self, crop_bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs forward inference on a single crop.
        Returns: (color_probs, type_probs, orientation_probs)
        """
        n_c, n_t, n_o = len(self.color_classes), len(self.type_classes), len(self.orientation_classes)
        if crop_bgr is None or crop_bgr.size == 0:
            return (
                np.ones(n_c, dtype=np.float32) / max(n_c, 1),
                np.ones(n_t, dtype=np.float32) / max(n_t, 1),
                np.ones(n_o, dtype=np.float32) / max(n_o, 1),
            )

        c_probs_b, t_probs_b, o_probs_b = self.predict_attributes_batch([crop_bgr])
        return c_probs_b[0], t_probs_b[0], o_probs_b[0]

    @torch.no_grad()
    def predict_attributes_batch(
        self, crops_bgr: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs batched forward inference on a list of BGR crops [B, 3, 224, 224].
        Returns: (batch_color_probs, batch_type_probs, batch_orientation_probs)
        """
        n_c, n_t, n_o = len(self.color_classes), len(self.type_classes), len(self.orientation_classes)
        batch_size = len(crops_bgr)
        if batch_size == 0:
            return (
                np.zeros((0, n_c), dtype=np.float32),
                np.zeros((0, n_t), dtype=np.float32),
                np.zeros((0, n_o), dtype=np.float32),
            )

        # Build tensor batch
        tensors = []
        valid_indices = []
        for i, crop in enumerate(crops_bgr):
            if crop is not None and crop.size > 0:
                pil_img = preprocess_crop(crop)
                if pil_img is not None:
                    tensors.append(self.transform(pil_img))
                    valid_indices.append(i)

        if not tensors:
            return (
                np.ones((batch_size, n_c), dtype=np.float32) / max(n_c, 1),
                np.ones((batch_size, n_t), dtype=np.float32) / max(n_t, 1),
                np.ones((batch_size, n_o), dtype=np.float32) / max(n_o, 1),
            )

        batch_tensor = torch.stack(tensors)

        # 1. ONNX Runtime Path
        if self.ort_session is not None:
            try:
                ort_inputs = {self.ort_session.get_inputs()[0].name: batch_tensor.numpy()}
                ort_outs = self.ort_session.run(None, ort_inputs)
                
                # Softmax conversion
                def _softmax(x):
                    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
                    return e / np.sum(e, axis=-1, keepdims=True)

                c_probs = _softmax(ort_outs[0])
                t_probs = _softmax(ort_outs[1])
                o_probs = _softmax(ort_outs[2]) if len(ort_outs) > 2 else np.full((len(valid_indices), n_o), 1.0 / n_o, dtype=np.float32)
                
                out_c = np.ones((batch_size, n_c), dtype=np.float32) / max(n_c, 1)
                out_t = np.ones((batch_size, n_t), dtype=np.float32) / max(n_t, 1)
                out_o = np.ones((batch_size, n_o), dtype=np.float32) / max(n_o, 1)
                for vi, src_idx in enumerate(valid_indices):
                    out_c[src_idx] = c_probs[vi]
                    out_t[src_idx] = t_probs[vi]
                    out_o[src_idx] = o_probs[vi]
                return out_c, out_t, out_o
            except Exception as e:
                logger.debug(f"ONNX batch inference failed: {e}")

        # 2. PyTorch Path
        if self.multitask_model is not None:
            batch_tensor = batch_tensor.to(self.device)
            if self.use_fp16 and self.device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    outs = self.multitask_model(batch_tensor)
            else:
                outs = self.multitask_model(batch_tensor)

            c_logits, t_logits = outs[0], outs[1]
            o_logits = outs[2] if len(outs) > 2 else None

            c_probs = torch.softmax(c_logits.float(), dim=1).cpu().numpy()
            t_probs = torch.softmax(t_logits.float(), dim=1).cpu().numpy()
            o_probs = torch.softmax(o_logits.float(), dim=1).cpu().numpy() if o_logits is not None else np.full((len(valid_indices), n_o), 1.0 / n_o, dtype=np.float32)

            out_c = np.ones((batch_size, n_c), dtype=np.float32) / max(n_c, 1)
            out_t = np.ones((batch_size, n_t), dtype=np.float32) / max(n_t, 1)
            out_o = np.ones((batch_size, n_o), dtype=np.float32) / max(n_o, 1)
            for vi, src_idx in enumerate(valid_indices):
                out_c[src_idx] = c_probs[vi]
                out_t[src_idx] = t_probs[vi]
                out_o[src_idx] = o_probs[vi]
            return out_c, out_t, out_o

        # Heuristic fallback
        out_c = np.ones((batch_size, n_c), dtype=np.float32) / max(n_c, 1)
        out_t = np.ones((batch_size, n_t), dtype=np.float32) / max(n_t, 1)
        out_o = np.ones((batch_size, n_o), dtype=np.float32) / max(n_o, 1)
        return out_c, out_t, out_o

    def predict_color_probs(self, crop_bgr: np.ndarray) -> np.ndarray:
        return self.predict_attributes_probs(crop_bgr)[0]

    def predict_type_probs(self, crop_bgr: np.ndarray) -> np.ndarray:
        return self.predict_attributes_probs(crop_bgr)[1]

    def predict_orientation_probs(self, crop_bgr: np.ndarray) -> np.ndarray:
        return self.predict_attributes_probs(crop_bgr)[2]

    def predict_color(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        probs = self.predict_color_probs(crop_bgr)
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = self.color_classes[idx] if (0 <= idx < len(self.color_classes) and conf >= self.conf_cutoff) else "Unknown"
        return label, round(conf, 3)

    def predict_type(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        probs = self.predict_type_probs(crop_bgr)
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = self.type_classes[idx] if (0 <= idx < len(self.type_classes) and conf >= self.conf_cutoff) else "Unknown"
        return label, round(conf, 3)

    def predict_orientation(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        probs = self.predict_orientation_probs(crop_bgr)
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = self.orientation_classes[idx] if 0 <= idx < len(self.orientation_classes) else "Rear"
        return label, round(conf, 3)

    def _load_multitask_model(self, weights_path: str) -> Optional[VehicleAttributeNet]:
        """Loads unified Multi-Task weights with backward compatibility for missing orientation heads."""
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
            n_o = len(self.orientation_classes)

            if "color_head.3.weight" in cleaned_state:
                n_c = cleaned_state["color_head.3.weight"].shape[0]
            if "type_head.3.weight" in cleaned_state:
                n_t = cleaned_state["type_head.3.weight"].shape[0]
            if "orientation_head.3.weight" in cleaned_state:
                n_o = cleaned_state["orientation_head.3.weight"].shape[0]

            model = VehicleAttributeNet(n_colors=n_c, n_types=n_t, n_orientations=n_o, pretrained=False)
            # Load matching weights without crashing on missing orientation head in legacy checkpoints
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in cleaned_state.items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
            model.to(self.device)
            model.eval()
            logger.info(f"Loaded unified Multi-Task VehicleAttributeNet ({n_c} colors, {n_t} types, {n_o} orientations) from: {weights_path}")
            return model
        except Exception as e:
            logger.warning(f"Could not load multi-task attribute weights from {weights_path}: {e}")
            return None
