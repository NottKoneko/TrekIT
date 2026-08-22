"""
utils.py
--------
Drawing, cropping, logging, and FPS utilities for Traffic Tracker AI.
"""

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Colour palette for bounding box overlays ──────────────────────────────
# Maps vehicle colour names → BGR drawing colours (keys match color_classes.json)
_COLOUR_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "Beige":   (180, 210, 230),
    "Black":   (60, 60, 60),
    "Blue":    (230, 120, 30),
    "Brown":   (42, 82, 130),
    "Gold":    (0, 185, 215),
    "Green":   (40, 180, 40),
    "Grey":    (160, 160, 160),
    "Gray":    (160, 160, 160),   # legacy alias — kept for safety
    "Orange":  (0, 140, 255),
    "Pink":    (180, 105, 255),
    "Purple":  (200, 50, 130),
    "Red":     (30, 30, 220),
    "Silver":  (200, 200, 200),
    "Tan":     (90, 160, 195),
    "White":   (240, 240, 240),
    "Yellow":  (0, 220, 220),
    "Unknown": (80, 80, 80),
}

_DEFAULT_BOX_COLOUR = (0, 200, 255)   # cyan fallback


# ── Data record for logging ────────────────────────────────────────────────
@dataclass
class VehicleRecord:
    """One logged vehicle sighting (one per stable track) conforming to enterprise schema."""
    track_id: int
    plate_text: str
    plate_conf: float
    color: str
    color_conf: float
    vehicle_type: str
    type_conf: float
    frame_first_seen: int
    frame_last_seen: int
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    snapshot_path: str = ""   # Optional path to saved crop image
    bbox: Optional[Tuple[int, int, int, int]] = None
    plate_bbox: Optional[Tuple[int, int, int, int]] = None
    orientation: str = "Rear"
    orientation_conf: float = 0.90
    heading_deg: Optional[float] = None
    plate_candidates: List[dict] = field(default_factory=list)
    jurisdiction: str = "us-ca"

    def to_enterprise_dict(self) -> dict:
        """Serializes record to enterprise commercial ANPR schema."""
        cands = self.plate_candidates if self.plate_candidates else (
            [{"text": self.plate_text, "confidence": round(float(self.plate_conf), 2)}]
            if self.plate_text else []
        )
        return {
            "vehicle": {
                "track_id": self.track_id,
                "bbox": list(self.bbox) if self.bbox else [],
                "type": self.vehicle_type,
                "type_confidence": round(float(self.type_conf), 2),
                "color": self.color,
                "color_confidence": round(float(self.color_conf), 2),
                "orientation": self.orientation,
                "orientation_confidence": round(float(self.orientation_conf), 2),
                "heading_deg": round(float(self.heading_deg), 1) if self.heading_deg is not None else None,
            },
            "plate": {
                "detected": bool(self.plate_text),
                "bbox": list(self.plate_bbox) if self.plate_bbox else [],
                "candidates": cands,
                "jurisdiction": self.jurisdiction,
            },
            "timestamp": self.timestamp,
        }


# ── Drawing helpers ────────────────────────────────────────────────────────

def draw_vehicle_overlay(
    frame: np.ndarray,
    track_id: int,
    bbox: Tuple[int, int, int, int],
    color_label: str,
    type_label: str,
    plate_text: str,
    color_conf: float = 0.0,
    type_conf: float = 0.0,
) -> np.ndarray:
    """
    Draw a styled bounding box + label badge on the frame.
    Returns the modified frame (in-place modification).
    """
    if frame is None or frame.size == 0:
        return frame

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame

    color_key = color_label.capitalize() if color_label else ""
    box_colour = _COLOUR_PALETTE.get(color_key, _COLOUR_PALETTE.get(color_label, _DEFAULT_BOX_COLOUR))

    # Main bounding box (2px thick)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_colour, 2)

    # Compose label string
    parts = [f"#{track_id}"]
    if color_label and color_label.lower() != "unknown":
        parts.append(color_label.capitalize())
    if type_label and type_label.lower() != "unknown":
        parts.append(type_label)
    if plate_text:
        parts.append(f"[{plate_text}]")
    label = "  ".join(parts)

    # Badge background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    pad = 4
    badge_x1 = x1
    badge_y1 = max(0, y1 - th - 2 * pad)
    badge_x2 = min(w, x1 + tw + 2 * pad)
    badge_y2 = y1

    # Badge background (solid — no frame.copy() overhead)
    cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), box_colour, -1)

    # Label text (dark for light colours, white for dark)
    brightness = sum(box_colour) / 3
    text_colour = (20, 20, 20) if brightness > 160 else (240, 240, 240)
    cv2.putText(
        frame, label,
        (badge_x1 + pad, badge_y2 - pad // 2),
        font, font_scale, text_colour, thickness, cv2.LINE_AA,
    )

    return frame


def draw_plate_overlay(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    plate_text: str,
) -> np.ndarray:
    """Draw a small green box around the plate region."""
    if frame is None or frame.size == 0:
        return frame

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 80), 1)
    if plate_text:
        cv2.putText(
            frame, plate_text,
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (0, 255, 80), 1, cv2.LINE_AA,
        )
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Render FPS counter in the top-right corner."""
    text = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.55, 1)
    x = frame.shape[1] - tw - 10
    cv2.putText(frame, text, (x, 22), font, 0.55, (0, 255, 180), 1, cv2.LINE_AA)
# ── Image Quality & Homography Utilities ───────────────────────────────────

def assess_image_quality(img_bgr: np.ndarray) -> Tuple[float, float, float]:
    """
    Computes image sharpness and exposure quality metrics.
    
    Returns:
        (laplacian_var, normalized_sharpness, luminance_balance)
        - laplacian_var: Q_sharpness = Var(nabla^2 I)
        - normalized_sharpness: Q_tilde in [0.0, 1.0]
        - luminance_balance: balance score in [0.0, 1.0] (1.0 = perfectly exposed)
    """
    if img_bgr is None or img_bgr.size == 0:
        return 0.0, 0.0, 0.0

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    normalized_sharpness = min(1.0, max(0.0, laplacian_var / 180.0))

    mean_lum = float(np.mean(gray))
    luminance_balance = max(0.0, 1.0 - abs(mean_lum - 128.0) / 128.0)

    return round(laplacian_var, 2), round(normalized_sharpness, 3), round(luminance_balance, 3)


def rectify_plate_quad(
    image: np.ndarray,
    corner_pts: np.ndarray,
    target_size: Tuple[int, int] = (94, 24),
) -> Optional[np.ndarray]:
    """
    Rectifies an oriented 4-point quadrilateral plate crop into a canonical frontal plane.
    
    Args:
        image: Full frame or ROI image (BGR).
        corner_pts: Array of shape (4, 2) with quadrilateral corner coordinates (x, y).
        target_size: (width, height) of rectified canonical output.
        
    Returns:
        Rectified warped plate image of shape (height, width, 3) or None.
    """
    if image is None or image.size == 0 or corner_pts is None or len(corner_pts) != 4:
        return None

    pts = np.array(corner_pts, dtype="float32")
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
        warped = cv2.warpPerspective(image, M, (tw, th), flags=cv2.INTER_LANCZOS4)
        return warped if warped is not None and warped.size > 0 else None
    except Exception:
        return None


def score_plate_keyframe(
    conf_det: float,
    crop_bgr: np.ndarray,
    frame_shape: Tuple[int, int],
    plate_bbox: Tuple[int, int, int, int],
) -> float:
    """
    Computes quality-weighted keyframe rank score:
    Q = 0.4 * conf_det + 0.4 * Q_tilde_sharpness + 0.2 * (area_plate / area_frame)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0

    _, q_sharp, _ = assess_image_quality(crop_bgr)
    
    hf, wf = frame_shape[:2]
    frame_area = max(1.0, float(hf * wf))
    
    px1, py1, px2, py2 = plate_bbox
    plate_area = max(0.0, float((px2 - px1) * (py2 - py1)))
    area_ratio = min(1.0, (plate_area / frame_area) * 100.0)  # Scale up ratio (typical plate is 0.5-2% of frame)

    q_score = 0.4 * float(conf_det) + 0.4 * q_sharp + 0.2 * area_ratio
    return round(float(q_score), 4)


# ── Crop utilities ─────────────────────────────────────────────────────────

def crop_bbox(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: int = 0,
    padding_ratio: float = 0.0,
) -> Optional[np.ndarray]:
    """Return a cropped region from frame, with optional absolute padding or relative padding_ratio."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    if padding_ratio > 0.0:
        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(bw * padding_ratio)
        pad_y = int(bh * padding_ratio)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
    elif padding > 0:
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
    else:
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def save_crop(
    crop: np.ndarray,
    directory: str,
    filename: str,
) -> str:
    """Save a crop image and return the saved path."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    path = str(Path(directory) / filename)
    cv2.imwrite(path, crop)
    return path


# ── FPS tracker ────────────────────────────────────────────────────────────

class FPSCounter:
    """Rolling-window FPS counter."""

    def __init__(self, window: int = 30):
        self.window = window
        self._times: List[float] = []

    def tick(self) -> float:
        """Call once per frame. Returns current FPS."""
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self.window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / max(elapsed, 1e-6)


# ── Log export ─────────────────────────────────────────────────────────────

def export_csv(records: List[VehicleRecord], path: str) -> str:
    """Export vehicle records to CSV. Returns saved path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return path

    fieldnames = list(asdict(records[0]).keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))

    logger.info(f"Exported {len(records)} records to {path}")
    return path


def export_json(records: List[VehicleRecord], path: str) -> str:
    """Export vehicle records to JSON. Returns saved path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2)

    logger.info(f"Exported {len(records)} records to {path}")
    return path


# ── Retention & Cleanup ───────────────────────────────────────────────────

def cleanup_old_snapshots(
    snapshot_dir: str,
    max_age_days: float = 7.0,
    max_size_mb: float = 2000.0,
) -> int:
    """
    Purge `.jpg` snapshots in `snapshot_dir` exceeding `max_age_days`
    or when total directory size exceeds `max_size_mb`.

    Returns number of deleted files.
    """
    sdir = Path(snapshot_dir)
    if not sdir.exists():
        return 0

    now = time.time()
    max_age_sec = max_age_days * 86400.0
    max_bytes = max_size_mb * 1024.0 * 1024.0

    files = sorted(
        [f for f in sdir.glob("*.jpg") if f.is_file()],
        key=lambda f: f.stat().st_mtime,
    )

    deleted_count = 0

    # 1. Purge by age
    remaining_files = []
    for f in files:
        try:
            mtime = f.stat().st_mtime
            if (now - mtime) > max_age_sec:
                f.unlink()
                deleted_count += 1
            else:
                remaining_files.append(f)
        except Exception as e:
            logger.warning(f"Failed to check/remove {f}: {e}")

    # 2. Purge by total size limit (oldest first)
    total_size = sum(f.stat().st_size for f in remaining_files if f.exists())
    if total_size > max_bytes:
        for f in remaining_files:
            if total_size <= max_bytes:
                break
            try:
                fsize = f.stat().st_size
                f.unlink()
                total_size -= fsize
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to remove oversized snapshot {f}: {e}")

    if deleted_count > 0:
        logger.info(f"Purged {deleted_count} old/oversized snapshot crops from {snapshot_dir}")

    return deleted_count

