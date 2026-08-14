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
    Standard Format: 1 Digit - 3 Letters - 3 Digits (e.g. 9JWM255, 8MZW276, 1ABC123)
    """
    if len(text) != 7:
        return text

    DIGIT_MAP = {'O': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'A': '4', 'S': '5', 'G': '6', 'B': '8'}
    LETTER_MAP = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}

    chars = list(text)

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


class PlateOCR:
    """
    Reads text from license plate image crops using EasyOCR.

    Usage:
        ocr = PlateOCR(config)
        text, confidence = ocr.read(plate_crop_bgr)
    """

    def __init__(self, config: dict):
        self.cfg = config.get("ocr", {})
        self.languages: List[str] = self.cfg.get("languages", ["en"])
        use_gpu = self.cfg.get("gpu", False)
        try:
            import torch
            if use_gpu and not torch.cuda.is_available():
                logger.warning("GPU requested for EasyOCR but CUDA is unavailable. Falling back to CPU.")
                use_gpu = False
        except Exception:
            use_gpu = False
        self.use_gpu: bool = use_gpu
        self.min_confidence: float = self.cfg.get("min_confidence", 0.15)

        pre = self.cfg.get("preprocessing", {})
        self.do_grayscale: bool = pre.get("grayscale", True)
        self.do_clahe: bool = pre.get("clahe", True)
        # NOTE: adaptive_threshold and morph_cleanup are intentionally OFF by
        # default. Binarising plate crops destroys the gradient information
        # that EasyOCR's CRAFT text-detector relies on, causing missed strokes
        # and merged character loops. Disable them unless you have a specific
        # reason (e.g. extremely high-contrast scanned plates).
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
        normalized_merged = normalize_plate_format(merged_text)

        # Check strictly for length between 5 and 8
        if 5 <= len(normalized_merged) <= 8 and self._matches_plate_pattern(normalized_merged):
            return normalized_merged, round(avg_conf, 3)

        if 5 <= len(merged_text) <= 8 and self._matches_plate_pattern(merged_text):
            return merged_text, round(avg_conf, 3)

        # Fallback: check if any individual candidate is valid (5-8 chars)
        for it in sorted(valid_items, key=lambda x: -x["conf"]):
            candidate = normalize_plate_format(it["text"])
            if 5 <= len(candidate) <= 8 and self._matches_plate_pattern(candidate):
                return candidate, round(it["conf"], 3)

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
