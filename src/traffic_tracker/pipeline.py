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


# ── Per-track temporal state ────────────────────────────────────────────────

@dataclass
class _TrackState:
    """Accumulates per-frame predictions for one tracked vehicle."""
    color_votes: Counter = field(default_factory=Counter)
    type_votes: Counter = field(default_factory=Counter)
    plate_votes: Counter = field(default_factory=Counter)
    frame_first_seen: int = 0
    frame_last_seen: int = 0
    ocr_attempts: int = 0
    best_crop: Optional[np.ndarray] = None   # highest-confidence frame crop


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

    Or for a single image:
        annotated, results = pipeline.process_image(image)
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self._setup_logging()

        track_cfg = self.config.get("tracking", {})
        clf_cfg = self.config.get("classification", {})

        self.vote_window: int = track_cfg.get("vote_window", 10)
        self.min_votes: int = track_cfg.get("min_votes", 3)
        self.classify_every_n: int = clf_cfg.get("classify_every_n_frames", 3)
        self.ocr_every_n: int = self.config.get("ocr", {}).get("ocr_every_n_frames", 3)

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
            stable_records:  List of VehicleRecord for all currently-active
                             tracks that have enough temporal votes
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
            state.frame_last_seen = self._frame_idx
            if state.frame_first_seen == 0:
                state.frame_first_seen = self._frame_idx

            # Crop the vehicle region
            crop = crop_bbox(frame, vehicle.bbox, padding=4)

            # Use detector plates if fresh, else fall back to cached plates
            if vehicle.plates:
                self._cached_plates[tid] = vehicle.plates
            plates = self._cached_plates.get(tid, [])

            # ── Classification (skip-frame) ──────────────────────────────
            if (
                crop is not None
                and self._frame_idx % self.classify_every_n == 0
            ):
                color_lbl, color_conf = self.classifier.predict_color(crop)
                type_lbl, type_conf = self.classifier.predict_type(crop)

                state.color_votes[color_lbl] += 1
                state.type_votes[type_lbl] += 1

                if (
                    state.best_crop is None
                    or color_conf > getattr(state, "_best_conf", 0)
                ):
                    state.best_crop = crop
                    state._best_conf = color_conf

            # ── OCR on plate sub-regions (cached & skip-frame for speed) ───
            for plate_det in plates:
                # Check if we already have a stable plate for this track or reached max attempts
                ocr_attempts = getattr(state, "ocr_attempts", 0)
                already_has_plate = bool(
                    state.plate_votes
                    and state.plate_votes.most_common(1)[0][1] >= 3
                )

                if not already_has_plate and ocr_attempts < 6:
                    if self._frame_idx % getattr(self, "ocr_every_n", 3) == 0:
                        plate_crop = crop_bbox(frame, plate_det.bbox)
                        if plate_crop is not None:
                            text, conf = self.ocr.read(plate_crop)
                            state.ocr_attempts = ocr_attempts + 1
                            if text:
                                state.plate_votes[text] += 1

                # Draw plate box on annotated frame
                current_plate = state.plate_votes.most_common(1)[0][0] if state.plate_votes else ""
                draw_plate_overlay(annotated, plate_det.bbox, current_plate)

            # ── Derive stable labels from votes ──────────────────────────
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
        max_age = self.config.get("tracking", {}).get("max_age", 30)
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
            crop = crop_bbox(image, vehicle.bbox, padding=4)
            color_lbl, color_conf = ("Unknown", 0.0)
            type_lbl, type_conf = ("Unknown", 0.0)
            plate_text = ""

            if crop is not None:
                color_lbl, color_conf = self.classifier.predict_color(crop)
                type_lbl, type_conf = self.classifier.predict_type(crop)

            for plate_det in vehicle.plates:
                plate_crop = crop_bbox(image, plate_det.bbox)
                if plate_crop is not None:
                    text, conf = self.ocr.read(plate_crop)
                    if text:
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
        logger.info("Pipeline state reset.")

    # ── Internal helpers ────────────────────────────────────────────────────

    def _get_stable_labels(
        self, state: _TrackState
    ) -> Tuple[str, str, str]:
        """Return majority-vote labels if enough votes accumulated."""
        color_lbl = "Unknown"
        type_lbl = "Unknown"
        plate_text = ""

        total_color = sum(state.color_votes.values())
        if total_color >= self.min_votes and state.color_votes:
            color_lbl = state.color_votes.most_common(1)[0][0]

        total_type = sum(state.type_votes.values())
        if total_type >= self.min_votes and state.type_votes:
            type_lbl = state.type_votes.most_common(1)[0][0]

        if state.plate_votes:
            plate_text = state.plate_votes.most_common(1)[0][0]

        return color_lbl, type_lbl, plate_text

    def _update_record(
        self,
        tid: int,
        state: _TrackState,
        color_lbl: str,
        type_lbl: str,
        plate_text: str,
    ):
        """Create or update a VehicleRecord for this track."""
        color_conf = (
            state.color_votes[color_lbl] / max(sum(state.color_votes.values()), 1)
        )
        type_conf = (
            state.type_votes[type_lbl] / max(sum(state.type_votes.values()), 1)
        )
        plate_conf = (
            state.plate_votes[plate_text] / max(sum(state.plate_votes.values()), 1)
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
