"""
pipeline.py
-----------
Unified video / image processing pipeline for Traffic Tracker AI.

Orchestrates:
  1. VehicleDetector  — YOLO detection + ByteTrack
  2. VehicleClassifier — color + body-type (MobileNetV3 / HSV fallback)
  3. PlateOCR         — EasyOCR with preprocessing

Key features:
  - Temporal voting across frames (eliminates single-frame flicker)
  - Skip-frame classification (saves CPU on laptop)
  - Stable VehicleRecord logging per unique track ID
"""

import logging
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from logging.handlers import RotatingFileHandler

from .classifier import VehicleClassifier
from .detector import VehicleDetection, VehicleDetector
from .ocr_reader import PlateOCR
from .utils import (
    FPSCounter,
    VehicleRecord,
    cleanup_old_snapshots,
    crop_bbox,
    draw_fps,
    draw_plate_overlay,
    draw_vehicle_overlay,
    save_crop,
)

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML config file. Returns empty dict on failure."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"Config not found at {config_path}; using defaults.")
        return {}
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}


# ── Per-track temporal state ────────────────────────────────────────────

@dataclass
class _TrackState:
    """Accumulates per-frame predictions for one tracked vehicle using continuous EMA probability smoothing.

    Maintaining running probability distributions eliminates single-frame jitter
    (such as oscillating between Sedan / Convertible / SUV).
    """
    color_probs: Optional[np.ndarray] = None   # shape (15,), smoothed with EMA
    type_probs:  Optional[np.ndarray] = None   # shape (7,), smoothed with EMA
    plate_votes: defaultdict = field(default_factory=lambda: defaultdict(float))
    frame_first_seen: int = 0
    frame_last_seen: int = 0
    hit_count: int = 0                         # Number of frames this track has been detected
    ocr_attempts: int = 0
    best_crop: Optional[np.ndarray] = None   # highest-confidence frame crop
    _best_conf: float = 0.0                  # confidence of best_crop



# ── Sheet-metal crop helper ─────────────────────────────────────────────────

def _get_steel_crop(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    top_skip: float = 0.20,
    bottom_skip: float = 0.20,
    side_skip: float = 0.05,
) -> Optional[np.ndarray]:
    """Return the central sheet-metal band of a vehicle bounding box.

    Strips the top ``top_skip`` fraction (roof, windshield) and the bottom
    ``bottom_skip`` fraction (tyres, road surface) so the color classifier
    sees actual paint rather than black rubber or tinted glass.

    Args:
        frame: Full BGR frame.
        bbox: Vehicle bounding box (x1, y1, x2, y2).
        top_skip: Fraction of bbox height to remove from the top.
        bottom_skip: Fraction of bbox height to remove from the bottom.
        side_skip: Fraction of bbox width to remove from each side.

    Returns:
        Cropped numpy array or None if the resulting crop is empty.
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    bh = y2 - y1
    bw = x2 - x1
    if bh <= 20 or bw <= 20:
        return None

    sy1 = y1 + int(bh * top_skip)
    sy2 = y2 - int(bh * bottom_skip)
    sx1 = x1 + int(bw * side_skip)
    sx2 = x2 - int(bw * side_skip)

    if sy2 <= sy1 or sx2 <= sx1:
        return None
    return frame[sy1:sy2, sx1:sx2].copy()


# ── Main pipeline class ─────────────────────────────────────────────────────

class TrafficPipeline:
    """
    End-to-end traffic analysis pipeline.

    Typical usage (video):
        pipeline = TrafficPipeline()
        cap = cv2.VideoCapture("video.mp4")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated, results = pipeline.process_frame(frame)
            cv2.imshow("Traffic Tracker", annotated)

    Typical usage (single image):
        pipeline = TrafficPipeline()
        annotated, results = pipeline.process_image(image)
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self._setup_logging()

        track_cfg = self.config.get("tracking", {})
        clf_cfg = self.config.get("classification", {})

        self.vote_window: int = track_cfg.get("vote_window", 10)
        self.min_votes: int = track_cfg.get("min_votes", 3)
        self.min_hits: int = track_cfg.get("min_hits", 3)
        self.ema_alpha: float = track_cfg.get("ema_alpha", 0.25)
        self.classify_every_n: int = clf_cfg.get("classify_every_n_frames", 2)
        self.ocr_every_n: int = self.config.get("ocr", {}).get("ocr_every_n_frames", 1)

        self.detector = VehicleDetector(self.config)
        self.classifier = VehicleClassifier(self.config)
        self.ocr = PlateOCR(self.config)
        self.fps_counter = FPSCounter()

        # Track state: track_id → _TrackState
        self._tracks: Dict[int, _TrackState] = defaultdict(_TrackState)
        # Finalised records: track_id → VehicleRecord
        self._records: Dict[int, VehicleRecord] = {}
        # Last known plate detections per track (used when throttled)
        self._cached_plates: Dict[int, list] = {}

        self._frame_idx: int = 0
        self._snapshot_dir = str(
            Path(self.config.get("paths", {}).get("logs_dir", "logs")) / "snapshots"
        )

        # Trigger snapshot retention cleanup if enabled
        ret_cfg = self.config.get("logging", {}).get("retention", {})
        if ret_cfg.get("enabled", False):
            cleanup_old_snapshots(
                self._snapshot_dir,
                max_age_days=ret_cfg.get("max_age_days", 7.0),
                max_size_mb=ret_cfg.get("max_dir_size_mb", 2000.0),
            )

        logger.info("TrafficPipeline initialised.")

    # ── Public API ──────────────────────────────────────────────────────────

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, List[VehicleRecord]]:
        """
        Process a single BGR video frame.

        Returns:
            annotated_frame: Frame with overlaid bounding boxes and labels
            stable_records:  List of VehicleRecord for all currently-active tracks
        """
        self._frame_idx += 1
        annotated = frame.copy()

        # ── Detection + tracking ─────────────────────────────────────────
        vehicles: List[VehicleDetection] = self.detector.detect(frame)

        # ── Per-vehicle processing ───────────────────────────────────────
        active_ids = set()
        for vehicle in vehicles:
            tid = vehicle.track_id
            active_ids.add(tid)
            state = self._tracks[tid]
            state.hit_count += 1
            state.frame_last_seen = self._frame_idx
            if state.frame_first_seen == 0:
                state.frame_first_seen = self._frame_idx

            # Check if vehicle touches frame boundaries (edge truncation)
            hf, wf = frame.shape[:2]
            vx1, vy1, vx2, vy2 = vehicle.bbox
            is_edge = (vx1 <= 8 or vy1 <= 8 or vx2 >= wf - 8 or vy2 >= hf - 8)

            # 10% padding for full vehicle silhouette (captures roofline/spoiler/C-pillar)
            type_crop = crop_bbox(frame, vehicle.bbox, padding_ratio=0.10)

            # Central sheet-metal sub-crop for colour classification
            steel_crop = _get_steel_crop(frame, vehicle.bbox)

            # Use detector plates if fresh, else fall back to cached plates
            if vehicle.plates:
                self._cached_plates[tid] = vehicle.plates
            plates = self._cached_plates.get(tid, [])

            # ── Classification (Continuous EMA probability smoothing) ───────
            if self._frame_idx % self.classify_every_n == 0:
                if steel_crop is not None:
                    p_color = self.classifier.predict_color_probs(steel_crop)
                    alpha_c = self.ema_alpha if not is_edge else 0.05
                    if state.color_probs is None:
                        state.color_probs = p_color
                    else:
                        state.color_probs = (
                            alpha_c * p_color + (1.0 - alpha_c) * state.color_probs
                        )
                    c_conf = float(np.max(state.color_probs))
                    if c_conf > state._best_conf:
                        state.best_crop = steel_crop
                        state._best_conf = c_conf

                # Only update body type when the vehicle is not heavily cut off at screen borders
                if type_crop is not None and not is_edge:
                    p_type = self.classifier.predict_type_probs(type_crop)
                    if state.type_probs is None:
                        state.type_probs = p_type
                    else:
                        state.type_probs = (
                            self.ema_alpha * p_type + (1.0 - self.ema_alpha) * state.type_probs
                        )

            # ── OCR on plate sub-regions (runs until a confident plate is read) ─────
            best_plate_weight = (
                max(state.plate_votes.values()) if state.plate_votes else 0.0
            )
            already_has_plate = best_plate_weight >= 1.5

            if not already_has_plate and plates:
                ocr_stride = getattr(self, "ocr_every_n", 1)
                if self._frame_idx % max(1, ocr_stride) == 0:
                    for plate_det in plates:
                        plate_crop = crop_bbox(frame, plate_det.bbox)
                        if plate_crop is not None:
                            text, conf = self.ocr.read(plate_crop)
                            state.ocr_attempts += 1
                            if text and len(text) >= 2:
                                state.plate_votes[text] += conf ** 2
                                break

            # ── Draw plate box on annotated frame (highest-confidence candidate) ──
            if plates:
                best_plate_det = max(plates, key=lambda p: p.confidence)
                current_plate = (
                    max(state.plate_votes, key=state.plate_votes.get)
                    if state.plate_votes else ""
                )
                draw_plate_overlay(annotated, best_plate_det.bbox, current_plate)

            # ── Derive stable labels from EMA probability distributions ─────
            color_lbl, type_lbl, plate_text = self._get_stable_labels(state)

            # ── Draw vehicle overlay ─────────────────────────────────────
            draw_vehicle_overlay(
                annotated,
                track_id=tid,
                bbox=vehicle.bbox,
                color_label=color_lbl,
                type_label=type_lbl,
                plate_text=plate_text,
            )

            # ── Update record ─────────────────────────────────────────────
            self._update_record(tid, state, color_lbl, type_lbl, plate_text)

        # ── FPS counter ──────────────────────────────────────────────────
        fps = self.fps_counter.tick()
        draw_fps(annotated, fps)

        # ── Prune stale tracks ───────────────────────────────────────────
        max_age = self.config.get("tracking", {}).get("max_age", 60)
        stale = [
            tid for tid, s in self._tracks.items()
            if self._frame_idx - s.frame_last_seen > max_age
        ]
        for tid in stale:
            self._finalise_track(tid)
            del self._tracks[tid]

        stable_records = [
            r for tid, r in self._records.items()
            if tid in active_ids
        ]
        return annotated, stable_records

    def process_image(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, List[VehicleRecord]]:
        """
        Process a single still image (no tracking IDs, no temporal voting).

        Returns annotated image and list of per-vehicle results.
        """
        annotated = image.copy()
        vehicles = self.detector.detect_image(image)
        records: List[VehicleRecord] = []

        for i, vehicle in enumerate(vehicles):
            type_crop   = crop_bbox(image, vehicle.bbox, padding_ratio=0.10)
            steel_crop  = _get_steel_crop(image, vehicle.bbox)
            color_lbl, color_conf = ("Unknown", 0.0)
            type_lbl, type_conf   = ("Unknown", 0.0)
            plate_text = ""

            if steel_crop is not None:
                color_lbl, color_conf = self.classifier.predict_color(steel_crop)
            if type_crop is not None:
                type_lbl, type_conf = self.classifier.predict_type(type_crop)

            for plate_det in vehicle.plates:
                plate_crop = crop_bbox(image, plate_det.bbox)
                if plate_crop is not None:
                    text, conf = self.ocr.read(plate_crop)
                    if text and len(text) >= 2:
                        plate_text = text
                        break

            draw_vehicle_overlay(
                annotated,
                track_id=i,
                bbox=vehicle.bbox,
                color_label=color_lbl,
                type_label=type_lbl,
                plate_text=plate_text,
            )

            records.append(VehicleRecord(
                track_id=i,
                plate_text=plate_text,
                plate_conf=0.0,
                color=color_lbl,
                color_conf=color_conf,
                vehicle_type=type_lbl,
                type_conf=type_conf,
                frame_first_seen=0,
                frame_last_seen=0,
            ))

        return annotated, records

    @property
    def all_records(self) -> List[VehicleRecord]:
        """Return all VehicleRecord objects (active + finalised)."""
        return list(self._records.values())

    def reset(self):
        """Clear all track state and records (e.g. between videos)."""
        self._tracks.clear()
        self._records.clear()
        self._cached_plates.clear()
        self._frame_idx = 0
        if hasattr(self.detector.model, "predictor") and self.detector.model.predictor is not None:
            if hasattr(self.detector.model.predictor, "trackers"):
                self.detector.model.predictor.trackers = []
        logger.info("Pipeline state reset.")

    # ── Internal helpers ────────────────────────────────────────────────────

    def _get_stable_labels(
        self, state: _TrackState
    ) -> Tuple[str, str, str]:
        """Return best consensus labels from smoothed EMA probability distributions."""
        color_lbl = "Unknown"
        type_lbl  = "Unknown"
        plate_text = ""

        if state.color_probs is not None:
            c_idx = int(np.argmax(state.color_probs))
            if state.color_probs[c_idx] >= self.classifier.conf_cutoff:
                color_lbl = self.classifier.color_classes[c_idx]

        if state.type_probs is not None:
            t_idx = int(np.argmax(state.type_probs))
            if state.type_probs[t_idx] >= self.classifier.conf_cutoff:
                type_lbl = self.classifier.type_classes[t_idx]

        if state.plate_votes:
            plate_text = max(state.plate_votes, key=state.plate_votes.get)

        return color_lbl, type_lbl, plate_text

    def _update_record(
        self,
        tid: int,
        state: _TrackState,
        color_lbl: str,
        type_lbl: str,
        plate_text: str,
    ):
        """Create or update a VehicleRecord for this track only if confirmed."""
        if state.hit_count < self.min_hits:
            return

        color_conf = float(np.max(state.color_probs)) if state.color_probs is not None else 0.0
        type_conf  = float(np.max(state.type_probs))  if state.type_probs  is not None else 0.0
        plate_total = sum(state.plate_votes.values())
        plate_conf = (
            state.plate_votes[plate_text] / max(plate_total, 1e-9)
            if plate_text else 0.0
        )

        self._records[tid] = VehicleRecord(
            track_id=tid,
            plate_text=plate_text,
            plate_conf=round(plate_conf, 3),
            color=color_lbl,
            color_conf=round(color_conf, 3),
            vehicle_type=type_lbl,
            type_conf=round(type_conf, 3),
            frame_first_seen=state.frame_first_seen,
            frame_last_seen=state.frame_last_seen,
        )

    def _finalise_track(self, tid: int):
        """Save a snapshot of a stale track's best crop."""
        if tid not in self._records:
            return
        state = self._tracks.get(tid)
        if state and state.best_crop is not None:
            fname = f"track_{tid}_f{state.frame_first_seen}.jpg"
            path = save_crop(state.best_crop, self._snapshot_dir, fname)
            self._records[tid].snapshot_path = path

    def _setup_logging(self):
        """Configure Python logging based on config with RotatingFileHandler."""
        log_cfg = self.config.get("logging", {})
        level = getattr(logging, log_cfg.get("level", "INFO"), logging.INFO)
        handlers = [logging.StreamHandler()]
        if log_cfg.get("log_to_file", False):
            log_path = log_cfg.get("log_filename", "logs/tracker.log")
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=handlers,
        )
