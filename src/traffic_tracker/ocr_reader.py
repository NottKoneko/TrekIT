"""
ocr_reader.py
-------------
License Plate OCR using dedicated LPRNet sequence model with Prefix Beam Search CTC,
Regional Syntax Disambiguation, and EasyOCR fallback.
Supports PyTorch (CUDA FP16 / CPU) and ONNX Runtime acceleration with batched inference.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from .syntax_validator import RegionalSyntaxValidator, default_syntax_validator
from .utils import assess_image_quality, rectify_plate_quad

logger = logging.getLogger(__name__)


def estimate_sharpness(img: np.ndarray) -> float:
    """Legacy sharpness estimator returning Laplacian variance."""
    return assess_image_quality(img)[0]

# Standard alphanumeric CTC character list (matching train_lprnet.py)
LPR_CHARS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "-"
]


class SmallBasicBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
        self.shortcut = nn.Conv2d(ch_in, ch_out, kernel_size=1) if ch_in != ch_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.block(x)


class LPRNet(nn.Module):
    def __init__(self, class_num: int = len(LPR_CHARS), dropout_rate: float = 0.5):
        super().__init__()
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            SmallBasicBlock(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            SmallBasicBlock(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SmallBasicBlock(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(256, class_num, kernel_size=1, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)  # (N, class_num, 1, seq_len)
        return logits.squeeze(2)   # (N, class_num, seq_len)


def beam_search_ctc(
    logits: torch.Tensor,
    chars: List[str] = LPR_CHARS,
    beam_width: int = 5,
    blank_idx: int = len(LPR_CHARS) - 1,
) -> List[Tuple[str, float]]:
    """
    Prefix Beam Search for CTC decoding.
    """
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)

    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    T = probs.shape[1]

    beams = {("", blank_idx): 1.0}

    for t in range(T):
        new_beams = {}
        for (prefix, last_char), p_prefix in beams.items():
            for c_idx in range(len(chars)):
                p = probs[c_idx, t]
                if p < 1e-4:
                    continue

                ch = chars[c_idx]
                if c_idx == blank_idx:
                    key = (prefix, blank_idx)
                    new_beams[key] = new_beams.get(key, 0.0) + p_prefix * p
                else:
                    if c_idx == last_char:
                        new_prefix = prefix
                    else:
                        new_prefix = prefix + ch
                    key = (new_prefix, c_idx)
                    new_beams[key] = new_beams.get(key, 0.0) + p_prefix * p

        sorted_beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width * 3]
        prefix_probs = {}
        for (prefix, _), p in sorted_beams:
            prefix_probs[prefix] = prefix_probs.get(prefix, 0.0) + p

        sorted_prefixes = sorted(prefix_probs.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        beams = {(pref, chars.index(pref[-1]) if pref else blank_idx): p for pref, p in sorted_prefixes}

    results = []
    for (prefix, _), p in sorted(beams.items(), key=lambda x: x[1], reverse=True):
        if prefix:
            results.append((prefix, min(1.0, float(p))))
    return results


def decode_lpr_logits(
    logits: torch.Tensor,
    chars: List[str] = LPR_CHARS,
    blank_idx: int = 0,
) -> Tuple[str, float]:
    """Greedy CTC argmax decoding."""
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    preds = torch.argmax(probs, dim=0).cpu().numpy()
    conf_scores = torch.max(probs, dim=0).values.cpu().numpy()

    decoded_chars = []
    confs = []
    prev = blank_idx
    for idx, c_idx in enumerate(preds):
        if c_idx != blank_idx and c_idx != prev:
            decoded_chars.append(chars[c_idx])
            confs.append(float(conf_scores[idx]))
        prev = c_idx

    text = "".join(decoded_chars)
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return text, round(avg_conf, 3)


def normalize_plate_format(text: str) -> str:
    """Disambiguate character string using default regional syntax validator."""
    corr, _, _, _ = default_syntax_validator.validate_and_correct(text)
    return corr


class PlateOCR:
    """
    Production ALPR / ANPR Engine.
    Supports PyTorch and ONNX Runtime execution with batched dynamic inputs.
    """

    def __init__(self, config: dict):
        self.cfg = config.get("ocr", {})
        self.cfg_paths = config.get("paths", {})
        self.languages: List[str] = self.cfg.get("languages", ["en"])
        use_gpu = self.cfg.get("gpu", False)
        try:
            if use_gpu and not torch.cuda.is_available():
                logger.warning("GPU requested for OCR but CUDA is unavailable. Falling back to CPU.")
                use_gpu = False
        except Exception:
            use_gpu = False
        self.use_gpu: bool = use_gpu
        self.device = torch.device("cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu")
        self.min_confidence: float = self.cfg.get("min_confidence", 0.15)
        self.syntax_validator = RegionalSyntaxValidator(default_jurisdiction=self.cfg.get("default_jurisdiction", "us-ca"))

        # ── ONNX Session Initialization ─────────────────────────────────────
        self.ort_session = None
        self.alpr_chars = LPR_CHARS
        models_dir = Path(self.cfg_paths.get("models_dir", "models"))
        onnx_candidates = [
            self.cfg_paths.get("lprnet_onnx", ""),
            str(models_dir / "lprnet.onnx"),
        ]
        for op in onnx_candidates:
            if op and Path(op).exists():
                self.ort_session = self._init_onnx_session(op)
                if self.ort_session is not None:
                    break

        # ── PyTorch LPRNet Fallback ──────────────────────────────────────────
        self.alpr_model: Optional[LPRNet] = None
        if self.ort_session is None:
            alpr_candidates = [
                self.cfg_paths.get("lprnet_weights", ""),
                str(models_dir / "lprnet.pt"),
            ]
            for w_path in alpr_candidates:
                if w_path and Path(w_path).exists():
                    try:
                        model = LPRNet(class_num=len(self.alpr_chars))
                        state_dict = torch.load(w_path, map_location=self.device, weights_only=True)
                        model.load_state_dict(state_dict)
                        model.to(self.device)
                        model.eval()
                        self.alpr_model = model
                        logger.info(f"Loaded dedicated ALPR sequence model (LPRNet) from {w_path}")
                        break
                    except Exception as e:
                        logger.warning(f"Could not load dedicated ALPR model {w_path}: {e}")

        # Preprocessing settings
        pre = self.cfg.get("preprocessing", {})
        self.do_grayscale: bool = pre.get("grayscale", True)
        self.do_clahe: bool = pre.get("clahe", True)
        self.do_adaptive_thresh: bool = pre.get("adaptive_threshold", False)
        self.do_morph: bool = pre.get("morph_cleanup", False)
        self.do_deskew: bool = pre.get("deskew", True)

        self.regex_patterns: List[str] = self.cfg.get("regex_patterns", [
            r"^[0-9A-Z]{5,8}$",
            r"^[0-9][A-Z]{3}[0-9]{3}$",
            r"^[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,3}$",
        ])
        self.strip_chars: str = self.cfg.get("strip_chars", r"""!@#$%^&*()_+-=[]{}|;':",./<>?""")

    def _init_onnx_session(self, onnx_path: str):
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
            logger.info(f"Loaded ONNX LPRNet from {onnx_path} (Providers: {session.get_providers()})")
            return session
        except Exception as e:
            logger.warning(f"Could not initialize ONNX session for {onnx_path}: {e}")
            return None

    def read(self, plate_crop_bgr: np.ndarray) -> Tuple[str, float]:
        """Returns (plate_text, confidence)."""
        if plate_crop_bgr is None or plate_crop_bgr.size == 0:
            return "", 0.0

        if self.ort_session is not None or self.alpr_model is not None:
            text, conf = self._read_lprnet(plate_crop_bgr)
            if text and len(text) >= 3:
                return text, conf

        # EasyOCR fallback
        h, w = plate_crop_bgr.shape[:2]
        if h < 100 or w < 260:
            scale = max(100.0 / max(h, 1), 260.0 / max(w, 1))
            plate_crop_bgr = cv2.resize(
                plate_crop_bgr,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_LANCZOS4,
            )

        processed = self._preprocess(plate_crop_bgr)
        raw_results = self._run_easyocr(processed)
        return self._postprocess(raw_results)

    @torch.no_grad()
    def _read_lprnet(self, plate_crop_bgr: np.ndarray) -> Tuple[str, float]:
        """Runs sequence recognition with regional syntax disambiguation."""
        try:
            h, w = plate_crop_bgr.shape[:2]
            if h < 8 or w < 16:
                return "", 0.0

            img = cv2.resize(plate_crop_bgr, (94, 24), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            img = (img - 127.5) * 0.0078125
            img = np.transpose(img, (2, 0, 1))

            if self.ort_session is not None:
                inp_name = self.ort_session.get_inputs()[0].name
                logits = self.ort_session.run(None, {inp_name: np.expand_dims(img, axis=0)})[0]
            else:
                tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
                logits = self.alpr_model(tensor)

            raw_cands = beam_search_ctc(logits, self.alpr_chars, beam_width=5)
            if not raw_cands:
                raw_text, conf = decode_lpr_logits(logits, self.alpr_chars)
                raw_cands = [(raw_text, conf)]

            rescored = self.syntax_validator.rescore_candidates(raw_cands, top_k=1)
            if rescored:
                best_text, best_conf, _, _ = rescored[0]
                return best_text, best_conf

            return "", 0.0
        except Exception as e:
            logger.debug(f"LPRNet inference error: {e}")
            return "", 0.0

    def read_candidates(self, plate_crop_bgr: np.ndarray, top_k: int = 5) -> List[dict]:
        """Returns ranked Top-K candidate dictionaries with syntax scoring."""
        if plate_crop_bgr is None or plate_crop_bgr.size == 0:
            return []

        if self.ort_session is not None or self.alpr_model is not None:
            try:
                h, w = plate_crop_bgr.shape[:2]
                if h < 8 or w < 16:
                    return []

                img = cv2.resize(plate_crop_bgr, (94, 24), interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
                img = (img - 127.5) * 0.0078125
                img = np.transpose(img, (2, 0, 1))

                if self.ort_session is not None:
                    inp_name = self.ort_session.get_inputs()[0].name
                    logits = self.ort_session.run(None, {inp_name: np.expand_dims(img, axis=0)})[0]
                else:
                    tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        logits = self.alpr_model(tensor)

                raw_cands = beam_search_ctc(logits, self.alpr_chars, beam_width=top_k)
                rescored = self.syntax_validator.rescore_candidates(raw_cands, top_k=top_k)

                return [{"text": txt, "confidence": conf} for txt, conf, _, _ in rescored]
            except Exception as e:
                logger.debug(f"Candidate decoding error: {e}")

        # EasyOCR fallback
        text, conf = self.read(plate_crop_bgr)
        if text and len(text) >= 3 and len(set(text)) > 2:
            return [{"text": text, "confidence": round(float(conf), 2)}]
        return []

    def read_batch_lprnet(self, crops_bgr: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Batched LPRNet inference [B, 3, 24, 94]."""
        if not crops_bgr:
            return []

        batch_imgs = []
        valid_indices = []
        for i, crop in enumerate(crops_bgr):
            if crop is not None and crop.size > 0:
                h, w = crop.shape[:2]
                if h >= 8 and w >= 16:
                    img = cv2.resize(crop, (94, 24), interpolation=cv2.INTER_CUBIC)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
                    img = (img - 127.5) * 0.0078125
                    img = np.transpose(img, (2, 0, 1))
                    batch_imgs.append(img)
                    valid_indices.append(i)

        results = [("", 0.0)] * len(crops_bgr)
        if not batch_imgs:
            return results

        batch_arr = np.stack(batch_imgs)

        if self.ort_session is not None:
            inp_name = self.ort_session.get_inputs()[0].name
            all_logits = self.ort_session.run(None, {inp_name: batch_arr})[0]
        elif self.alpr_model is not None:
            tensor = torch.from_numpy(batch_arr).to(self.device)
            with torch.no_grad():
                all_logits = self.alpr_model(tensor).cpu().numpy()
        else:
            return results

        for vi, src_idx in enumerate(valid_indices):
            logits_i = all_logits[vi:vi+1]
            raw_cands = beam_search_ctc(logits_i, self.alpr_chars, beam_width=5)
            rescored = self.syntax_validator.rescore_candidates(raw_cands, top_k=1)
            if rescored:
                results[src_idx] = (rescored[0][0], rescored[0][1])

        return results

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        if self.do_grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.bilateralFilter(img, d=5, sigmaColor=40, sigmaSpace=40)

        if self.do_clahe:
            clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(2, 2))
            img = clahe.apply(img)

        if self.do_adaptive_thresh:
            img = cv2.adaptiveThreshold(
                img, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=15,
                C=9,
            )

        if self.do_morph:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        if self.do_deskew:
            img = self._deskew(img)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img

    @staticmethod
    def _deskew(gray: np.ndarray) -> np.ndarray:
        try:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            coords = np.column_stack(np.where(binary > 0))
            if coords.shape[0] < 20:
                return gray
            coords_xy = coords[:, ::-1]
            rect = cv2.minAreaRect(coords_xy)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = 90 - angle
            else:
                angle = -angle

            if abs(angle) < 0.5 or abs(angle) > 25.0:
                return gray

            h, w = gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            return gray

    def _run_easyocr(self, img: np.ndarray) -> list:
        try:
            reader = self._get_easyocr_reader()
            if reader is None:
                return []
            return reader.readtext(img, detail=1, paragraph=False)
        except Exception as e:
            logger.debug(f"EasyOCR error: {e}")
            return []

    _easyocr_reader = None

    def _get_easyocr_reader(self):
        if PlateOCR._easyocr_reader is None:
            try:
                import easyocr
                PlateOCR._easyocr_reader = easyocr.Reader(
                    self.languages,
                    gpu=self.use_gpu,
                    verbose=False,
                )
            except Exception as e:
                logger.warning(f"EasyOCR init failed: {e}")
        return PlateOCR._easyocr_reader

    def _postprocess(self, raw_results: list) -> Tuple[str, float]:
        if not raw_results:
            return "", 0.0

        candidates = []
        for item in raw_results:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            text = str(item[1]).strip()
            conf = float(item[2]) if len(item) >= 3 else 0.5
            cleaned = self._clean_text(text)
            if not cleaned or len(cleaned) < 3 or len(set(cleaned)) <= 2:
                continue
            candidates.append((cleaned, conf))

        rescored = self.syntax_validator.rescore_candidates(candidates, top_k=1)
        if rescored:
            return rescored[0][0], rescored[0][1]

        return "", 0.0

    def _clean_text(self, text: str) -> str:
        return self.syntax_validator.clean_raw_string(text)

    def _matches_plate_pattern(self, text: str) -> bool:
        _, is_valid, _, _ = self.syntax_validator.validate_and_correct(text)
        return is_valid
