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
    """One logged vehicle sighting (one per stable track)."""
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

    box_colour = _COLOUR_PALETTE.get(color_label, _DEFAULT_BOX_COLOUR)

    # Main bounding box (2px thick)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_colour, 2)

    # Compose label string
    parts = [f"#{track_id}"]
    if color_label and color_label != "Unknown":
        parts.append(color_label)
    if type_label and type_label != "Unknown":
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
    return frame


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

