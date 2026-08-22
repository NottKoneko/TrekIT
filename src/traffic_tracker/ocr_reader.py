"""
ocr_reader.py
-------------
License plate OCR using EasyOCR with a multi-stage image preprocessing pipeline.

Pipeline:
  1. Grayscale conversion
  2. CLAHE (Contrast-Limited Adaptive Histogram Equalization)
  3. Adaptive thresholding
  4. Morphological cleanup (remove noise)
  5. Deskew correction
  6. EasyOCR inference
  7. Regex post-filtering + character cleanup
"""

import logging
import re
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Lazy-import EasyOCR to avoid slow startup when not needed
_easyocr_reader = None


def _get_reader(languages: List[str], use_gpu: bool):
    global _easyocr_reader
    if _easyocr_reader is None:
        import sys
        import os
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ["PYTHONIOENCODING"] = "utf-8"
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8")
            except Exception:
                pass
        if hasattr(sys.stderr, "reconfigure"):
            try:
                sys.stderr.reconfigure(encoding="utf-8")
            except Exception:
                pass
        import easyocr
        logger.info(f"Initialising EasyOCR (languages={languages}, gpu={use_gpu})…")
        _easyocr_reader = easyocr.Reader(languages, gpu=use_gpu, verbose=False)
    return _easyocr_reader


# Common US state plate header / footer words to reject
US_STATE_HEADERS = {
    "CALIFORNIA", "TEXAS", "FLORIDA", "NEWYORK", "ARIZONA",
    "NEVADA", "OREGON", "WASHINGTON", "DMV", "CA", "EXEMPT",
    "SESES", "STATE", "HONDA", "TOYOTA", "NISSAN", "FORD",
    "CHEVROLET", "CADILLAC", "DODGE", "HYUNDAI", "LEXUS",
}


def normalize_plate_format(text: str) -> str:
    """
    Applies California & standard US 7-character plate format disambiguation.
    Standard Format: 1 Digit - 3 Letters - 3 Digits (e.g. 9GAD429, 9JWM255, 8MZW276, 1ABC123)
    """
    if not text:
        return ""

    clean = re.sub(r"[^A-Z0-9]", "", text.upper())
    if len(clean) != 7:
        return clean

    DIGIT_MAP = {
        'O': '0', 'D': '0', 'Q': '0',
        'I': '1', 'L': '1', 'T': '1',
        'Z': '2',
        'A': '4',
        'S': '5',
        'G': '6', 'C': '6',
        'B': '8',
        'J': '9',
    }
    LETTER_MAP = {
        '0': 'O', '1': 'I', '2': 'Z',
        '4': 'A', '5': 'S', '6': 'G',
        '8': 'B', '9': 'J',
    }

    chars = list(clean)

    # Position 0 must be a digit
    if chars[0] in DIGIT_MAP:
        chars[0] = DIGIT_MAP[chars[0]]

    # Positions 1, 2, 3 must be letters
    for i in (1, 2, 3):
        if chars[i] in LETTER_MAP:
            chars[i] = LETTER_MAP[chars[i]]

    # Positions 4, 5, 6 must be digits
    for i in (4, 5, 6):
        if chars[i] in DIGIT_MAP:
            chars[i] = DIGIT_MAP[chars[i]]

    return "".join(chars)

def estimate_sharpness(img_bgr: np.ndarray) -> float:
    """
    Computes Laplacian variance to estimate image sharpness / motion blur.
    Sharp plate crops return > 50.0; heavily motion-blurred or defocused crops return < 20.0.
    """
    if img_bgr is None or img_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ── Dedicated ALPR Sequence Recognition Architecture (LPRNet) ─────────────
import torch
import torch.nn as nn
from pathlib import Path

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
    """
    Lightweight License Plate Recognition Network (LPRNet).
    Specialized end-to-end convolutional character sequence model for ALPR.
    """
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


def decode_lpr_logits(logits: torch.Tensor, chars_list: List[str] = LPR_CHARS) -> Tuple[str, float]:
    """
    Greedy CTC decoder for LPR sequence logits.
    Merges consecutive identical character predictions and strips blank tokens.
    """
    probs = torch.softmax(logits, dim=1)
    max_probs, preds = torch.max(probs, dim=1)

    preds_np = preds.cpu().numpy()[0]
    probs_np = max_probs.cpu().numpy()[0]

    blank_idx = len(chars_list) - 1
    decoded_chars = []
    char_confs = []

    prev_idx = -1
    for i, idx in enumerate(preds_np):
        if idx != blank_idx and idx != prev_idx:
            decoded_chars.append(chars_list[idx])
            char_confs.append(float(probs_np[i]))
        prev_idx = idx

    raw_text = "".join(decoded_chars)
    avg_conf = float(np.mean(char_confs)) if char_confs else 0.0
    return raw_text, avg_conf


def beam_search_ctc(
    logits: torch.Tensor,
    chars_list: List[str] = LPR_CHARS,
    beam_width: int = 5,
    blank_idx: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """
    Decodes LPRNet sequence logits using Prefix Beam Search.
    Returns Top-K ranked candidates with sequence probability scores.
    """
    if blank_idx is None:
        blank_idx = len(chars_list) - 1

    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()  # (C, T)
    classes, time_steps = probs.shape

    # Initialize beams: prefix -> (prob_blank, prob_non_blank)
    beams = {"": (1.0, 0.0)}

    for t in range(time_steps):
        next_beams = {}
        for prefix, (p_b, p_nb) in beams.items():
            p_total = p_b + p_nb

            # 1. Blank token transition
            p_b_next = p_total * float(probs[blank_idx, t])
            curr_b, curr_nb = next_beams.get(prefix, (0.0, 0.0))
            next_beams[prefix] = (curr_b + p_b_next, curr_nb)

            # 2. Non-blank token transitions
            for c in range(classes):
                if c == blank_idx:
                    continue
                char = chars_list[c]
                p_char = float(probs[c, t])
                new_prefix = prefix + char

                if len(prefix) > 0 and prefix[-1] == char:
                    # Repeating character without blank in-between
                    p_nb_next = p_b * p_char
                    curr_b, curr_nb = next_beams.get(new_prefix, (0.0, 0.0))
                    next_beams[new_prefix] = (curr_b, curr_nb + p_nb_next)

                    # Collapse with existing prefix
                    curr_b_same, curr_nb_same = next_beams.get(prefix, (0.0, 0.0))
                    next_beams[prefix] = (curr_b_same, curr_nb_same + p_nb * p_char)
                else:
                    curr_b, curr_nb = next_beams.get(new_prefix, (0.0, 0.0))
                    next_beams[new_prefix] = (curr_b, curr_nb + p_total * p_char)

        # Prune down to top beam_width prefixes
        sorted_prefixes = sorted(
            next_beams.items(),
            key=lambda item: item[1][0] + item[1][1],
            reverse=True,
        )[:beam_width]
        beams = dict(sorted_prefixes)

    results = []
    for prefix, (p_b, p_nb) in beams.items():
        if prefix:
            total_p = p_b + p_nb
            results.append((prefix, round(float(total_p), 3)))

    return sorted(results, key=lambda x: -x[1])


def unwarp_plate(crop_bgr: np.ndarray, target_size: Tuple[int, int] = (94, 24)) -> np.ndarray:
    """
    Isolates and rectifies rotated/skewed license plate contours to a frontal perspective plane.
    Uses minimum area bounding quadrilateral and perspective homography transformation.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr

    h, w = crop_bgr.shape[:2]
    if h < 10 or w < 20:
        return crop_bgr

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
    blurred = cv2.bilateralFilter(gray, 5, 40, 40)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return crop_bgr

    # Find valid plate-like contour
    valid_rects = []
    min_area = (h * w) * 0.12
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw == 0 or rh == 0:
            continue
        ar = max(rw, rh) / min(rw, rh)
        if 1.5 <= ar <= 6.5:
            valid_rects.append((area, rect))

    if not valid_rects:
        largest_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_cnt) < min_area:
            return crop_bgr
        rect = cv2.minAreaRect(largest_cnt)
    else:
        rect = max(valid_rects, key=lambda x: x[0])[1]

    box = cv2.boxPoints(rect)
    pts = np.array(box, dtype="float32")

    # Order points: top-left, top-right, bottom-right, bottom-left
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]       # top-left
    ordered[2] = pts[np.argmax(s)]       # bottom-right
    ordered[1] = pts[np.argmin(diff)]    # top-right
    ordered[3] = pts[np.argmax(diff)]    # bottom-left

    tw, th = target_size
    dst = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype="float32")
    try:
        M = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(crop_bgr, M, (tw, th), flags=cv2.INTER_CUBIC)
        return warped if warped is not None and warped.size > 0 else crop_bgr
    except Exception:
        return crop_bgr


class PlateOCR:
    """
    Reads text from license plate image crops using dedicated ALPR (LPRNet) or EasyOCR fallback.

    Usage:
        ocr = PlateOCR(config)
        text, confidence = ocr.read(plate_crop_bgr)
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

        # ── Load Dedicated ALPR sequence model if available ────────────────
        self.alpr_model: Optional[LPRNet] = None
        self.alpr_chars = LPR_CHARS
        alpr_candidates = [
            self.cfg_paths.get("lprnet_weights", ""),
            "models/lprnet.pt",
            "models/lprnet.onnx",
            "models/crnn_plate.pt",
            "models/crnn_plate.onnx",
        ]
        for w_path in alpr_candidates:
            if w_path and Path(w_path).exists():
                try:
                    if str(w_path).endswith(".pt"):
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

        pre = self.cfg.get("preprocessing", {})
        self.do_grayscale: bool = pre.get("grayscale", True)
        self.do_clahe: bool = pre.get("clahe", True)
        self.do_adaptive_thresh: bool = pre.get("adaptive_threshold", False)
        self.do_morph: bool = pre.get("morph_cleanup", False)
        self.do_deskew: bool = pre.get("deskew", True)

        self.regex_patterns: List[str] = self.cfg.get("regex_patterns", [
            r"^[0-9A-Z]{5,8}$",
            r"^[0-9][A-Z]{3}[0-9]{3}$",  # Standard California 7-character format (e.g. 9JWM255, 8MZW276)
            r"^[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,3}$",
        ])
        self.strip_chars: str = self.cfg.get(
            "strip_chars",
            r"""!@#$%^&*()_+-=[]{}|;':",./<>?"""
        )

    # ── Public API ──────────────────────────────────────────────────────

    def read(self, plate_crop_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Returns (plate_text, confidence).
        Returns ("", 0.0) when nothing valid is detected.
        """
        if plate_crop_bgr is None or plate_crop_bgr.size == 0:
            return "", 0.0

        # 1. If dedicated ALPR model is loaded, run high-speed sequence recognition
        if self.alpr_model is not None:
            text, conf = self._read_lprnet(plate_crop_bgr)
            if text and len(text) >= 3:
                return text, conf

        # 2. EasyOCR fallback with multi-stage preprocessing
        # Up-scale tiny crops so plate height is at least 100px for EasyOCR accuracy
        h, w = plate_crop_bgr.shape[:2]
        if h < 100 or w < 260:
            scale = max(100.0 / max(h, 1), 260.0 / max(w, 1))
            plate_crop_bgr = cv2.resize(
                plate_crop_bgr,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        processed = self._preprocess(plate_crop_bgr)
        raw_results = self._run_easyocr(processed)
        return self._postprocess(raw_results)

    @torch.no_grad()
    def _read_lprnet(self, plate_crop_bgr: np.ndarray) -> Tuple[str, float]:
        """Run dedicated LPRNet sequence model inference with beam search and unwarping."""
        try:
            # 1. 4-Point perspective unwarping
            unwarped = unwarp_plate(plate_crop_bgr, target_size=(94, 24))

            # 2. Trim top 14% (state name / "California" header) and bottom 10% (frame text)
            h, w = unwarped.shape[:2]
            clean_crop = unwarped
            if h >= 14 and w >= 24:
                top_cut = int(h * 0.14)
                bot_cut = max(top_cut + 8, int(h * 0.90))
                clean_crop = unwarped[top_cut:bot_cut, :]
                if clean_crop.size == 0:
                    clean_crop = unwarped

            # LPRNet expects (94, 24) input in RGB
            img = cv2.resize(clean_crop, (94, 24), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)
            img = (img - 127.5) * 0.0078125
            img = np.transpose(img, (2, 0, 1))
            tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

            logits = self.alpr_model(tensor)
            candidates = beam_search_ctc(logits, self.alpr_chars, beam_width=5)
            if not candidates:
                raw_text, conf = decode_lpr_logits(logits, self.alpr_chars)
                candidates = [(raw_text, conf)]

            best_cand, best_score = "", 0.0
            for raw_text, conf in candidates:
                cleaned = self._clean_text(raw_text)
                normalized = normalize_plate_format(cleaned) if len(cleaned) == 7 else cleaned
                is_valid = self._matches_plate_pattern(normalized)
                score = conf * (1.5 if is_valid else 1.0)
                if score > best_score:
                    best_score = score
                    best_cand = normalized

            return best_cand, round(min(best_score, 1.0), 3)
        except Exception as e:
            logger.debug(f"LPRNet inference error: {e}")
            return "", 0.0

    def read_candidates(self, plate_crop_bgr: np.ndarray, top_k: int = 5) -> List[dict]:
        """
        Returns ranked Top-K candidate dictionaries:
        [{"text": "9GAD429", "confidence": 0.98}, {"text": "9GAD129", "confidence": 0.74}]
        """
        if plate_crop_bgr is None or plate_crop_bgr.size == 0:
            return []

        if self.alpr_model is not None:
            try:
                unwarped = unwarp_plate(plate_crop_bgr, target_size=(94, 24))
                h, w = unwarped.shape[:2]
                clean_crop = unwarped
                if h >= 14 and w >= 24:
                    top_cut = int(h * 0.14)
                    bot_cut = max(top_cut + 8, int(h * 0.90))
                    clean_crop = unwarped[top_cut:bot_cut, :]
                    if clean_crop.size == 0:
                        clean_crop = unwarped

                img = cv2.resize(clean_crop, (94, 24), interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
                img = (img - 127.5) * 0.0078125
                img = np.transpose(img, (2, 0, 1))
                tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits = self.alpr_model(tensor)
                    candidates = beam_search_ctc(logits, self.alpr_chars, beam_width=top_k)

                out = []
                seen = set()
                for text, conf in candidates:
                    cleaned = self._clean_text(text)
                    norm = normalize_plate_format(cleaned) if len(cleaned) == 7 else cleaned
                    if norm and norm not in seen:
                        seen.add(norm)
                        out.append({"text": norm, "confidence": round(float(conf), 2)})
                return out[:top_k]
            except Exception as e:
                logger.debug(f"Candidate decoding error: {e}")

        # EasyOCR fallback
        text, conf = self.read(plate_crop_bgr)
        if text:
            return [{"text": text, "confidence": round(float(conf), 2)}]
        return []

    def read_all(
        self, plate_crop_bgr: np.ndarray
    ) -> List[Tuple[str, float]]:
        """
        Returns all detected text segments (unfiltered), sorted by confidence.
        Useful for debugging.
        """
        if plate_crop_bgr is None or plate_crop_bgr.size == 0:
            return []
        processed = self._preprocess(plate_crop_bgr)
        raw_results = self._run_easyocr(processed)
        out = []
        for item in raw_results:
            if isinstance(item, (list, tuple)):
                if len(item) >= 3:
                    out.append((str(item[1]).strip(), float(item[2])))
                elif len(item) == 2:
                    out.append((str(item[0]).strip(), float(item[1])))
            elif isinstance(item, str):
                out.append((item.strip(), 1.0))
        return sorted(out, key=lambda x: -x[1])

    # ── Preprocessing pipeline ──────────────────────────────────────────

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline to a BGR plate crop."""
        if self.do_grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Gentle bilateral filter removes noise without blurring thin stroke gradients
        img = cv2.bilateralFilter(img, d=5, sigmaColor=40, sigmaSpace=40)

        if self.do_clahe:
            # Clip limit 2.0 for contrast boost that preserves stroke gradients
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
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
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        if self.do_deskew:
            img = self._deskew(img)

        # Convert back to 3-channel for EasyOCR input compatibility
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img

    @staticmethod
    def _deskew(gray: np.ndarray) -> np.ndarray:
        """Correct skew using image moments on binarised foreground pixels."""
        try:
            # Otsu's threshold to isolate actual text characters
            _, binary = cv2.threshold(gray, 0, 255,
                                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # OpenCV points format is (x, y)
            coords = np.column_stack(np.where(binary > 0))
            if coords.shape[0] < 20:
                return gray
            coords_xy = coords[:, ::-1]
            rect = cv2.minAreaRect(coords_xy)
            angle = rect[-1]
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90
            if abs(angle) < 0.5 or abs(angle) > 45.0:
                return gray
            (h, w) = gray.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                gray, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            return rotated
        except Exception:
            return gray

    # ── EasyOCR inference ───────────────────────────────────────────────

    def _run_easyocr(self, img: np.ndarray) -> list:
        """Run EasyOCR with alphanumeric character allowlist."""
        try:
            reader = _get_reader(self.languages, self.use_gpu)
            # Enforce alphanumeric allowlist to prevent punctuation noise
            results = reader.readtext(
                img,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                paragraph=False,
                detail=1,
            )
            return results or []  # list of (bbox, text, confidence)
        except Exception as e:
            logger.warning(f"EasyOCR inference error: {e}")
            return []

    # ── Post-processing ──────────────────────────────────────────────────

    def _postprocess(self, raw_results: list) -> Tuple[str, float]:
        """
        Filters state headers, discards short noise (<5 chars),
        stitches left-to-right word chunks together, and disambiguates California 7-character format.
        """
        if not raw_results:
            return "", 0.0

        valid_items = []
        for item in raw_results:
            if not isinstance(item, (list, tuple)):
                continue
            if len(item) >= 3:
                bbox, text, conf = item[0], str(item[1]), float(item[2])
            elif len(item) == 2:
                bbox, text, conf = None, str(item[0]), float(item[1])
            else:
                continue

            if conf < self.min_confidence:
                continue

            cleaned = self._clean_text(text)
            if not cleaned or cleaned in US_STATE_HEADERS:
                continue

            # Get X-center coordinate to sort left-to-right
            cx = 0.0
            if bbox is not None and len(bbox) >= 4:
                try:
                    pts = np.array(bbox)
                    cx = float(np.mean(pts[:, 0]))
                except Exception:
                    cx = 0.0

            valid_items.append({"text": cleaned, "conf": conf, "cx": cx})

        if not valid_items:
            return "", 0.0

        # Sort fragments from left to right along the plate
        valid_items.sort(key=lambda it: it["cx"])

        # Merge fragments together (e.g. '9' + 'JWM' + '255' -> '9JWM255')
        merged_text = "".join(it["text"] for it in valid_items)
        avg_conf = sum(it["conf"] for it in valid_items) / max(len(valid_items), 1)

        # Apply California format normalization (e.g. 1 Digit - 3 Letters - 3 Digits)
        normalized_merged = normalize_plate_format(merged_text) if merged_text else ""

        # 1. Exact match with normalized plate format (e.g. 1 Digit - 3 Letters - 3 Digits)
        if normalized_merged and 5 <= len(normalized_merged) <= 8 and self._matches_plate_pattern(normalized_merged):
            return normalized_merged, round(avg_conf, 3)

        # 2. Exact match with merged raw text
        if merged_text and 5 <= len(merged_text) <= 8 and self._matches_plate_pattern(merged_text):
            return merged_text, round(avg_conf, 3)

        # 3. Fallback to individual candidate if it matches strict regex pattern
        for it in sorted(valid_items, key=lambda x: -x["conf"]):
            candidate = normalize_plate_format(it.get("text", "")) if it.get("text") else ""
            if candidate and 5 <= len(candidate) <= 8 and self._matches_plate_pattern(candidate):
                return candidate, round(it["conf"], 3)

        # 4. Relaxed fallback for partial / vanity / non-standard reads (3-8 alphanumeric characters)
        if merged_text and 3 <= len(merged_text) <= 8 and merged_text.isalnum():
            candidate = normalize_plate_format(merged_text) if len(merged_text) == 7 else merged_text
            return candidate, round(avg_conf * 0.85, 3)

        # 5. Fallback to highest confidence single alphanumeric fragment (>= 3 chars)
        for it in sorted(valid_items, key=lambda x: -x["conf"]):
            txt = it.get("text", "")
            if txt and 3 <= len(txt) <= 8 and txt.isalnum():
                return txt, round(it["conf"] * 0.80, 3)

        return "", 0.0

    def _clean_text(self, text: str) -> str:
        """Strip unwanted characters and normalise to uppercase."""
        text = text.upper().strip()
        text = text.translate(str.maketrans("", "", self.strip_chars))
        text = re.sub(r"\s+", "", text)   # remove internal whitespace
        return text

    def _matches_plate_pattern(self, text: str) -> bool:
        """Return True if text matches any configured plate regex pattern."""
        for pattern in self.regex_patterns:
            if re.match(pattern, text):
                return True
        return False
