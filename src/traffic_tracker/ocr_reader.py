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
        self.use_gpu: bool = self.cfg.get("gpu", False)
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
            r"^[A-Z0-9]{2,12}$",
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

        # Up-scale tiny crops so plate height is at least 90px for EasyOCR accuracy
        h, w = plate_crop_bgr.shape[:2]
        if h < 90 or w < 220:
            scale = max(90.0 / max(h, 1), 220.0 / max(w, 1))
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
        for _, text, conf in raw_results:
            out.append((text.strip(), float(conf)))
        return sorted(out, key=lambda x: -x[1])

    # ── Preprocessing pipeline ──────────────────────────────────────────

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline to a BGR plate crop.

        Aggressive binarisation (adaptive threshold, morphological ops) is
        disabled by default because EasyOCR’s CRAFT detector expects
        natural grey-scale gradients, not binary black-and-white images.
        Keep grayscale + gentle CLAHE, then pass directly to EasyOCR.
        """
        if self.do_grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.do_clahe:
            # Clip limit 2.0 (was 3.0) for gentler contrast boost that
            # preserves stroke gradients EasyOCR relies on.
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

        # Convert back to 3-channel so EasyOCR is happy
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img

    @staticmethod
    def _deskew(gray: np.ndarray) -> np.ndarray:
        """Correct skew using image moments on binarised foreground pixels.

        The original implementation used ``np.where(gray > 0)`` which matched
        almost every pixel in any non-black grayscale image, making the
        bounding-box angle always 0°. This version applies Otsu’s threshold
        to isolate actual text/foreground pixels before computing the angle.
        """
        try:
            # Otsu's threshold to reliably isolate text foreground
            _, binary = cv2.threshold(gray, 0, 255,
                                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            coords = np.column_stack(np.where(binary > 0))
            if coords.shape[0] < 20:   # too few foreground pixels to trust
                return gray
            angle = cv2.minAreaRect(coords)[-1]
            # minAreaRect returns −θ for CW rotations; adjust to intuitive range
            if angle < -45:
                angle = 90 + angle
            if abs(angle) < 0.5:   # skip negligible corrections
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
        """Run EasyOCR and return raw results list."""
        try:
            reader = _get_reader(self.languages, self.use_gpu)
            results = reader.readtext(img, detail=1)
            return results  # list of (bbox, text, confidence)
        except Exception as e:
            logger.warning(f"EasyOCR inference error: {e}")
            return []

    # ── Post-processing ──────────────────────────────────────────────────

    def _postprocess(self, raw_results: list) -> Tuple[str, float]:
        """
        Merge detected text segments, clean up, and validate with regex.
        Returns best (text, confidence) or ("", 0.0) if nothing valid found.
        """
        candidates: List[Tuple[str, float]] = []

        for _, text, conf in raw_results:
            conf = float(conf)
            if conf < self.min_confidence:
                continue

            # Clean the text
            cleaned = self._clean_text(text)
            if not cleaned:
                continue

            # Validate against plate patterns
            if self._matches_plate_pattern(cleaned):
                candidates.append((cleaned, conf))

        if not candidates:
            # Return the highest-confidence segment regardless of pattern
            # (useful for international plates with unusual formats)
            for _, text, conf in sorted(raw_results, key=lambda x: -x[2]):
                cleaned = self._clean_text(text)
                if cleaned and float(conf) >= self.min_confidence:
                    return cleaned, float(conf)
            return "", 0.0

        # Return highest-confidence valid candidate
        candidates.sort(key=lambda x: -x[1])
        return candidates[0]

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
