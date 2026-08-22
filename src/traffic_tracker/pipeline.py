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

from .classifier import VehicleClassifier, get_body_crop, _hsv_color_fallback
from .detector import VehicleDetection, VehicleDetector
from .ocr_reader import PlateOCR, estimate_sharpness
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

    def __init__(self, config_path: str = "config.yaml", config: Optional[dict] = None):
        self.config = config if config is not None else load_config(config_path)
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
            bw = vx2 - vx1
            bh = vy2 - vy1
            is_edge = (vx1 <= 8 or vy1 <= 8 or vx2 >= wf - 8 or vy2 >= hf - 8)

            # Dynamic padding: 8% padding for large vehicles (>=60x45), 0% for small/distant vehicles (<60x45)
            pad = 0.08 if (bw >= 60 and bh >= 45) else 0.0
            type_crop = crop_bbox(frame, vehicle.bbox, padding_ratio=pad)

            # Use detector plates if fresh, else fall back to cached plates
            if vehicle.plates:
                self._cached_plates[tid] = vehicle.plates
            plates = self._cached_plates.get(tid, [])

            # ── Classification (Continuous EMA probability smoothing) ───────
            if self._frame_idx % self.classify_every_n == 0:
                if bw < 60 or bh < 45:
                    # Resolution Gate: assign default Sedan and neutral/HSV color without forcing MobileNet
                    if state.type_probs is None and "Sedan" in self.classifier.type_classes:
                        p_type = np.zeros(len(self.classifier.type_classes), dtype=np.float32)
                        p_type[self.classifier.type_classes.index("Sedan")] = 0.40
                        state.type_probs = p_type
                    if state.color_probs is None and type_crop is not None:
                        c_lbl, _, _ = _hsv_color_fallback(type_crop)
                        if c_lbl in self.classifier.color_classes:
                            p_color = np.zeros(len(self.classifier.color_classes), dtype=np.float32)
                            p_color[self.classifier.color_classes.index(c_lbl)] = 0.50
                            state.color_probs = p_color
                elif self.classifier.multitask_model is not None and type_crop is not None:
                    # Single forward pass for both color and type
                    p_color, p_type = self.classifier.predict_attributes_probs(type_crop)
                    alpha_c = self.ema_alpha if not is_edge else 0.05
                    if state.color_probs is None:
                        state.color_probs = p_color
                    else:
                        state.color_probs = alpha_c * p_color + (1.0 - alpha_c) * state.color_probs

                    if not is_edge:
                        if state.type_probs is None:
                            state.type_probs = p_type
                        else:
                            state.type_probs = self.ema_alpha * p_type + (1.0 - self.ema_alpha) * state.type_probs
                else:
                    if type_crop is not None:
                        p_color = self.classifier.predict_color_probs(type_crop)
                        alpha_c = self.ema_alpha if not is_edge else 0.05
                        if state.color_probs is None:
                            state.color_probs = p_color
                        else:
                            state.color_probs = (
                                alpha_c * p_color + (1.0 - alpha_c) * state.color_probs
                            )
                        c_conf = float(np.max(state.color_probs))
                        if c_conf > state._best_conf:
                            state.best_crop = type_crop
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
            top_plate_text = (
                max(state.plate_votes, key=state.plate_votes.get)
                if state.plate_votes else ""
            )
            best_plate_weight = state.plate_votes.get(top_plate_text, 0.0)
            
            # Only consider locked-in if weight >= 1.5 AND top candidate matches valid plate regex
            already_has_plate = (
                best_plate_weight >= 1.5 and self.ocr._matches_plate_pattern(top_plate_text)
            )

            if not already_has_plate and plates:
                ocr_stride = getattr(self, "ocr_every_n", 1)
                if self._frame_idx % max(1, ocr_stride) == 0:
                    for plate_det in plates:
                        px1, py1, px2, py2 = plate_det.bbox
                        pw = px2 - px1
                        ph = py2 - py1

                        # Resolution Gate: Allow distant plates (EasyOCR reader upscales small crops)
                        if pw < 14 or ph < 6:
                            continue

                        plate_crop = crop_bbox(frame, plate_det.bbox)
                        if plate_crop is not None:
                            # Sharpness Gate: Discard only severe blur; use sharpness for voting weight
                            sharpness = estimate_sharpness(plate_crop)
                            if sharpness < 3.0 and min(pw, ph) >= 25:
                                continue

                            text, conf = self.ocr.read(plate_crop)
                            state.ocr_attempts += 1
                            if text and len(text) >= 3:
                                # Bonus weight for matching standard plate regex patterns and sharp focus
                                is_valid_pattern = self.ocr._matches_plate_pattern(text)
                                mult = 2.0 if is_valid_pattern else (0.8 if len(text) >= 5 else 0.4)
                                sharpness_bonus = min(max(sharpness / 40.0, 0.4), 2.0)
                                state.plate_votes[text] += (conf ** 2) * mult * sharpness_bonus
                                # Do NOT break — evaluate all plate candidates in the frame

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
            vx1, vy1, vx2, vy2 = vehicle.bbox
            bw = vx2 - vx1
            bh = vy2 - vy1

            # Dynamic padding: 8% for >=60x45, 0% for <60x45 to prevent foliage bleed
            pad = 0.08 if (bw >= 60 and bh >= 45) else 0.0
            type_crop  = crop_bbox(image, vehicle.bbox, padding_ratio=pad)
            body_crop  = get_body_crop(image, vehicle.bbox) if (bw >= 60 and bh >= 45) else None
            color_lbl, color_conf = ("Unknown", 0.0)
            type_lbl, type_conf   = ("Unknown", 0.0)
            plate_text = ""
            plate_conf = 0.0
            best_plate_det = None

            # Resolution Gate: If < 60x45, bypass MobileNet and assign Sedan + HSV/neutral color fallback
            if bw < 60 or bh < 45:
                type_lbl, type_conf = "Sedan", 0.40
                if type_crop is not None:
                    c_lbl, c_conf, _ = _hsv_color_fallback(type_crop)
                    color_lbl, color_conf = (c_lbl if c_lbl != "Unknown" else "Grey"), max(c_conf, 0.40)
                else:
                    color_lbl, color_conf = "Grey", 0.40
            else:
                if type_crop is not None:
                    color_lbl, color_conf = self.classifier.predict_color(type_crop)
                    type_lbl, type_conf   = self.classifier.predict_type(type_crop)
                elif body_crop is not None:
                    color_lbl, color_conf = self.classifier.predict_color(body_crop)
                    type_lbl, type_conf   = self.classifier.predict_type(body_crop)

            best_text = ""
            best_score = 0.0
            best_plate_det = None

            for plate_det in vehicle.plates:
                # Gating: only run on confident plate boxes (>= 0.35 confidence and area >= 150px)
                if plate_det.conf < 0.35:
                    continue
                pw = plate_det.bbox[2] - plate_det.bbox[0]
                ph = plate_det.bbox[3] - plate_det.bbox[1]
                if pw * ph < 150:
                    continue

                plate_crop = crop_bbox(image, plate_det.bbox)
                if plate_crop is not None:
                    text, conf = self.ocr.read(plate_crop)
                    if text and len(text) >= 3 and len(set(text)) > 2:
                        is_valid = self.ocr._matches_plate_pattern(text)
                        score = conf * (2.5 if is_valid else (1.0 if len(text) >= 5 else 0.5))
                        if score > best_score:
                            best_score = score
                            best_text = text
                            best_plate_det = plate_det

            plate_text = best_text
            plate_conf = min(best_score, 1.0)

            # Draw license plate box overlay directly on the plate
            if best_plate_det is not None:
                draw_plate_overlay(annotated, best_plate_det.bbox, plate_text)

            draw_vehicle_overlay(
                annotated,
                track_id=i,
                bbox=vehicle.bbox,
                color_label=color_lbl,
                type_label=type_lbl,
                plate_text=plate_text,
            )

            plate_candidates = []
            if best_plate_det is not None:
                plate_crop = crop_bbox(image, best_plate_det.bbox)
                if plate_crop is not None:
                    plate_candidates = self.ocr.read_candidates(plate_crop, top_k=5)

            records.append(VehicleRecord(
                track_id=i,
                plate_text=plate_text,
                plate_conf=round(plate_conf, 3),
                color=color_lbl,
                color_conf=round(color_conf, 3),
                vehicle_type=type_lbl,
                type_conf=round(type_conf, 3),
                frame_first_seen=0,
                frame_last_seen=0,
                bbox=vehicle.bbox,
                plate_bbox=best_plate_det.bbox if best_plate_det else None,
                orientation="Rear" if best_plate_det else "Front",
                orientation_conf=0.92 if best_plate_det else 0.70,
                plate_candidates=plate_candidates,
                jurisdiction="us-ca",
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
        if hasattr(self, "detector") and self.detector is not None:
            self.detector.reset()
        logger.info("Pipeline state reset.")

    # ── Internal helpers ────────────────────────────────────────────────────

    def _get_stable_labels(
        self, state: _TrackState
    ) -> Tuple[str, str, str]:
        """Return best consensus labels from smoothed EMA probability distributions."""
        color_lbl = "Unknown"
        type_lbl  = "Unknown"
        plate_text = ""

        if state.color_probs is not None and len(state.color_probs) > 0:
            c_idx = int(np.argmax(state.color_probs))
            if 0 <= c_idx < len(self.classifier.color_classes) and state.color_probs[c_idx] >= self.classifier.conf_cutoff:
                color_lbl = self.classifier.color_classes[c_idx]

        if state.type_probs is not None and len(state.type_probs) > 0:
            t_idx = int(np.argmax(state.type_probs))
            if 0 <= t_idx < len(self.classifier.type_classes) and state.type_probs[t_idx] >= self.classifier.conf_cutoff:
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
        # Top-K plate candidates from temporal votes
        plate_cands = [
            {"text": txt, "confidence": round(float(vote / max(plate_total, 1e-9)), 2)}
            for txt, vote in sorted(state.plate_votes.items(), key=lambda x: -x[1])[:5]
        ]

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
            plate_candidates=plate_cands,
            jurisdiction="us-ca",
            orientation="Rear" if plate_text else "Front",
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


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Traffic Tracker AI Pipeline CLI")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or video file")
    parser.add_argument("--output", type=str, default=None, help="Path to save annotated output")
    parser.add_argument("--save-debug-crops", action="store_true", help="Save isolated vehicle and plate crops for inspection")
    parser.add_argument("--json", action="store_true", help="Output enterprise JSON schema")
    parser.add_argument("--conf", type=float, default=None, help="Detection confidence threshold override")
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    if args.conf is not None:
        cfg.setdefault("detection", {})["confidence_threshold"] = args.conf

    pipeline = TrafficPipeline(config=cfg)
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file {input_path} does not exist.")
        sys.exit(1)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if input_path.suffix.lower() in img_exts:
        img_bgr = cv2.imread(str(input_path))
        if img_bgr is None:
            print(f"Error reading image {input_path}")
            sys.exit(1)

        annotated, records = pipeline.process_image(img_bgr)

        if args.json:
            import json
            enterprise_data = [r.to_enterprise_dict() for r in records]
            print(json.dumps(enterprise_data, indent=2))
        else:
            print("\n" + "=" * 65)
            print(f"  Traffic Tracker AI — Detection Results ({len(records)} vehicles)")
            print("=" * 65)
            for idx, r in enumerate(records, 1):
                color = f"{r.color.capitalize()} ({r.color_conf:.0%})" if r.color and r.color.lower() != "unknown" else "Unknown"
                v_type = f"{r.vehicle_type} ({r.type_conf:.0%})" if r.vehicle_type and r.vehicle_type != "Unknown" else "Unknown"
                plate = f"{r.plate_text} ({r.plate_conf:.0%})" if r.plate_text else "None detected"
                print(f"Vehicle #{idx:02d} | Color: {color:<16} | Type: {v_type:<16} | Plate: {plate}")
            print("=" * 65 + "\n")

        if args.output:
            cv2.imwrite(args.output, annotated)
            print(f"Annotated image saved to: {args.output}")

        if args.save_debug_crops:
            debug_dir = Path("logs/debug_crops")
            debug_dir.mkdir(parents=True, exist_ok=True)
            v_detections = pipeline.detector.detect_image(img_bgr)
            for idx, v in enumerate(v_detections, 1):
                v_crop = crop_bbox(img_bgr, v.bbox)
                b_crop = get_body_crop(img_bgr, v.bbox)
                if v_crop is not None:
                    cv2.imwrite(str(debug_dir / f"vehicle_{idx:02d}_full.jpg"), v_crop)
                if b_crop is not None:
                    cv2.imwrite(str(debug_dir / f"vehicle_{idx:02d}_body.jpg"), b_crop)
                for p_idx, p in enumerate(v.plates, 1):
                    p_crop = crop_bbox(img_bgr, p.bbox)
                    if p_crop is not None:
                        cv2.imwrite(str(debug_dir / f"vehicle_{idx:02d}_plate_{p_idx}.jpg"), p_crop)
            print(f"Debug crops saved to: {debug_dir}")
