"""
pipeline.py
-----------
Unified, production-grade video and image processing pipeline for TrekIT ALPR AI.

Features:
  1. Multi-Threaded Asynchronous Pipeline (AsyncTrafficPipeline) with FrameQueue & WorkerPool
  2. Quality-Ranked Keyframe Selection:
       Q = 0.4 * conf_det + 0.4 * Q_tilde_sharpness + 0.2 * (area_plate / area_frame)
  3. Multi-Head Vehicle Intelligence (Color, Type, Orientation)
  4. Homography Perspective Rectification & Regional Syntax Disambiguation
  5. Enterprise ANPR JSON serialization & CSV logging
"""

import argparse
import json
import logging
import queue
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml
from logging.handlers import RotatingFileHandler

from .classifier import VehicleClassifier, get_body_crop, _hsv_color_fallback
from .detector import Detection, VehicleDetection, VehicleDetector
from .ocr_reader import PlateOCR, estimate_sharpness
from .syntax_validator import RegionalSyntaxValidator, default_syntax_validator
from .utils import (
    FPSCounter,
    VehicleRecord,
    assess_image_quality,
    cleanup_old_snapshots,
    crop_bbox,
    draw_fps,
    draw_plate_overlay,
    draw_vehicle_overlay,
    export_csv,
    export_json,
    rectify_plate_quad,
    save_crop,
    score_plate_keyframe,
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
    """
    Accumulates per-frame predictions for one tracked vehicle using
    continuous EMA probability smoothing and quality-ranked keyframe scoring.
    """
    color_probs: Optional[np.ndarray] = None          # shape (15,), smoothed with EMA
    type_probs:  Optional[np.ndarray] = None          # shape (7,), smoothed with EMA
    orientation_probs: Optional[np.ndarray] = None   # shape (3,), smoothed with EMA
    plate_votes: defaultdict = field(default_factory=lambda: defaultdict(float))
    keyframe_pool: List[dict] = field(default_factory=list)  # Top-K candidate keyframes
    frame_first_seen: int = 0
    frame_last_seen: int = 0
    hit_count: int = 0
    ocr_attempts: int = 0
    best_crop: Optional[np.ndarray] = None
    _best_conf: float = 0.0
    heading_deg: Optional[float] = None
    last_centroid: Optional[Tuple[float, float]] = None


# ── Synchronous Production Pipeline ────────────────────────────────────────

class TrafficPipeline:
    """
    Production-grade ANPR and Vehicle Intelligence Pipeline.
    """

    def __init__(self, config_path: str = "config.yaml", config: Optional[dict] = None):
        self.config = config if config is not None else load_config(config_path)
        self._setup_logging()

        track_cfg = self.config.get("tracking", {})
        clf_cfg = self.config.get("classification", {})
        ocr_cfg = self.config.get("ocr", {})

        self.vote_window: int = track_cfg.get("vote_window", 10)
        self.min_votes: int = track_cfg.get("min_votes", 3)
        self.min_hits: int = track_cfg.get("min_hits", 3)
        self.ema_alpha: float = track_cfg.get("ema_alpha", 0.25)
        self.classify_every_n: int = clf_cfg.get("classify_every_n_frames", 2)
        self.ocr_every_n: int = ocr_cfg.get("ocr_every_n_frames", 1)
        self.max_keyframes: int = ocr_cfg.get("max_keyframes_per_track", 3)

        self.detector = VehicleDetector(self.config)
        self.classifier = VehicleClassifier(self.config)
        self.ocr = PlateOCR(self.config)
        self.syntax_validator = RegionalSyntaxValidator(
            default_jurisdiction=ocr_cfg.get("default_jurisdiction", "us-ca")
        )
        self.fps_counter = FPSCounter()

        self._tracks: Dict[int, _TrackState] = defaultdict(_TrackState)
        self._records: Dict[int, VehicleRecord] = {}
        self._cached_plates: Dict[int, list] = {}

        self._frame_idx: int = 0
        self._snapshot_dir = str(
            Path(self.config.get("paths", {}).get("logs_dir", "logs")) / "snapshots"
        )

        ret_cfg = self.config.get("logging", {}).get("retention", {})
        if ret_cfg.get("enabled", False):
            cleanup_old_snapshots(
                self._snapshot_dir,
                max_age_days=ret_cfg.get("max_age_days", 7.0),
                max_size_mb=ret_cfg.get("max_dir_size_mb", 2000.0),
            )

        logger.info("TrafficPipeline initialised with Quality Keyframe Ranking & Multi-Head Attributes.")

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, List[VehicleRecord]]:
        """
        Process a single BGR video frame with keyframe quality gating.
        """
        self._frame_idx += 1
        fps = self.fps_counter.tick()

        if frame is None or frame.size == 0:
            return frame, list(self._records.values())

        annotated = frame.copy()
        draw_fps(annotated, fps)

        # 1. Run YOLOv8 detection + ByteTrack
        vehicles: List[VehicleDetection] = self.detector.detect(frame)
        active_track_ids = set()

        for vehicle in vehicles:
            tid = vehicle.track_id
            if tid < 0:
                continue
            active_track_ids.add(tid)

            state = self._tracks[tid]
            state.hit_count += 1
            if state.frame_first_seen == 0:
                state.frame_first_seen = self._frame_idx
            state.frame_last_seen = self._frame_idx

            # Calculate centroid trajectory heading
            vx1, vy1, vx2, vy2 = vehicle.bbox
            curr_centroid = ((vx1 + vx2) / 2.0, (vy1 + vy2) / 2.0)
            if state.last_centroid is not None:
                dx = curr_centroid[0] - state.last_centroid[0]
                dy = curr_centroid[1] - state.last_centroid[1]
                if abs(dx) + abs(dy) > 2.0:
                    state.heading_deg = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
            state.last_centroid = curr_centroid

            plates = vehicle.plates
            if plates:
                self._cached_plates[tid] = plates
            elif tid in self._cached_plates:
                plates = self._cached_plates[tid]

            # Dynamic padding & resolution gating
            bw, bh = vx2 - vx1, vy2 - vy1
            pad_ratio = 0.08 if (bw >= 60 and bh >= 45) else 0.0
            type_crop = crop_bbox(frame, vehicle.bbox, padding_ratio=pad_ratio)
            body_crop = get_body_crop(frame, vehicle.bbox)

            # Edge cut-off detection
            h_f, w_f = frame.shape[:2]
            is_edge = (vx1 <= 5 or vy1 <= 5 or vx2 >= w_f - 5 or vy2 >= h_f - 5)

            # Attribute classification
            if self._frame_idx % max(1, self.classify_every_n) == 0 or state.hit_count <= 2:
                if bw < 60 or bh < 45:
                    c_lbl, c_conf, _ = _hsv_color_fallback(type_crop) if type_crop is not None else ("Grey", 0.4, 0.0)
                    n_c, n_t, n_o = len(self.classifier.color_classes), len(self.type_classes), len(self.classifier.orientation_classes)
                    p_col = np.zeros(n_c, dtype=np.float32)
                    c_idx = self.classifier.color_classes.index(c_lbl) if c_lbl in self.classifier.color_classes else 0
                    p_col[c_idx] = max(c_conf, 0.40)
                    p_type = np.zeros(n_t, dtype=np.float32)
                    t_idx = self.type_classes.index("Sedan") if "Sedan" in self.type_classes else 0
                    p_type[t_idx] = 0.40
                    p_orient = np.array([0.8, 0.1, 0.1], dtype=np.float32)
                else:
                    target_crop = type_crop if type_crop is not None else body_crop
                    p_col, p_type, p_orient = self.classifier.predict_attributes_probs(target_crop)

                if state.color_probs is None:
                    state.color_probs = p_col
                    state.type_probs = p_type
                    state.orientation_probs = p_orient
                else:
                    state.color_probs = self.ema_alpha * p_col + (1.0 - self.ema_alpha) * state.color_probs
                    if not is_edge:
                        state.type_probs = self.ema_alpha * p_type + (1.0 - self.ema_alpha) * state.type_probs
                    state.orientation_probs = self.ema_alpha * p_orient + (1.0 - self.ema_alpha) * state.orientation_probs

                c_conf = float(np.max(state.color_probs))
                if c_conf > state._best_conf and type_crop is not None:
                    state.best_crop = type_crop
                    state._best_conf = c_conf

            # Quality-Weighted Keyframe Ranking & Plate Recognition
            top_plate_text = max(state.plate_votes, key=state.plate_votes.get) if state.plate_votes else ""
            best_plate_weight = state.plate_votes.get(top_plate_text, 0.0)
            already_locked = (best_plate_weight >= 1.8 and self.syntax_validator.validate_and_correct(top_plate_text)[1])

            if not already_locked and plates:
                for p_det in plates:
                    if p_det.confidence < 0.30:
                        continue
                    pw = p_det.bbox[2] - p_det.bbox[0]
                    ph = p_det.bbox[3] - p_det.bbox[1]
                    if pw < 14 or ph < 6:
                        continue

                    p_crop = crop_bbox(frame, p_det.bbox)
                    if p_crop is not None:
                        q_score = score_plate_keyframe(p_det.confidence, p_crop, frame.shape, p_det.bbox)
                        
                        # Maintain top-K keyframes
                        state.keyframe_pool.append({"q": q_score, "crop": p_crop, "det": p_det})
                        state.keyframe_pool.sort(key=lambda k: -k["q"])
                        state.keyframe_pool = state.keyframe_pool[:self.max_keyframes]

                        # Run recognition on top keyframes
                        text, conf = self.ocr.read(p_crop)
                        state.ocr_attempts += 1
                        if text and len(text) >= 3 and len(set(text)) > 2:
                            corr_text, is_valid, jur, weight = self.syntax_validator.validate_and_correct(text)
                            if corr_text:
                                state.plate_votes[corr_text] += (conf ** 2) * weight * (1.0 + q_score)

            # Overlays
            if plates:
                best_plate = max(plates, key=lambda p: p.confidence)
                curr_plate = max(state.plate_votes, key=state.plate_votes.get) if state.plate_votes else ""
                draw_plate_overlay(annotated, best_plate.bbox, curr_plate)

            color_lbl, type_lbl, plate_text, orient_lbl = self._get_stable_labels(state)
            draw_vehicle_overlay(
                annotated,
                track_id=tid,
                bbox=vehicle.bbox,
                color_label=color_lbl,
                type_label=type_lbl,
                plate_text=plate_text,
            )

            self._update_record(tid, state, color_lbl, type_lbl, plate_text, orient_lbl, vehicle.bbox, plates)

        # Stale track cleanup
        max_age = self.config.get("tracking", {}).get("max_age", 60)
        stale_tids = [
            t for t, s in self._tracks.items()
            if (self._frame_idx - s.frame_last_seen) > max_age and t not in active_track_ids
        ]
        for tid in stale_tids:
            self._finalise_track(tid)
            self._tracks.pop(tid, None)
            self._cached_plates.pop(tid, None)

        active_records = [self._records[tid] for tid in active_track_ids if tid in self._records]
        return annotated, active_records

    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[VehicleRecord]]:
        """
        Process a single still image.
        """
        if image is None or image.size == 0:
            return image, []

        annotated = image.copy()
        vehicles = self.detector.detect_image(image)
        records: List[VehicleRecord] = []

        for i, vehicle in enumerate(vehicles):
            vx1, vy1, vx2, vy2 = vehicle.bbox
            bw, bh = vx2 - vx1, vy2 - vy1
            pad_ratio = 0.08 if (bw >= 60 and bh >= 45) else 0.0
            type_crop = crop_bbox(image, vehicle.bbox, padding_ratio=pad_ratio)
            body_crop = get_body_crop(image, vehicle.bbox)

            if bw < 60 or bh < 45:
                type_lbl, type_conf = "Sedan", 0.40
                c_lbl, c_conf, _ = _hsv_color_fallback(type_crop) if type_crop is not None else ("Grey", 0.40, 0.0)
                color_lbl, color_conf = (c_lbl if c_lbl != "Unknown" else "Grey"), max(c_conf, 0.40)
                orient_lbl, orient_conf = "Rear", 0.70
            else:
                target_crop = type_crop if type_crop is not None else body_crop
                color_lbl, color_conf = self.classifier.predict_color(target_crop)
                type_lbl, type_conf = self.classifier.predict_type(target_crop)
                orient_lbl, orient_conf = self.classifier.predict_orientation(target_crop)

            best_text = ""
            best_score = 0.0
            best_plate_det = None
            plate_candidates = []

            for p_det in vehicle.plates:
                if p_det.confidence < 0.30:
                    continue
                pw = p_det.bbox[2] - p_det.bbox[0]
                ph = p_det.bbox[3] - p_det.bbox[1]
                if pw * ph < 120:
                    continue

                p_crop = crop_bbox(image, p_det.bbox)
                if p_crop is not None:
                    text, conf = self.ocr.read(p_crop)
                    if text and len(text) >= 3 and len(set(text)) > 2:
                        corr_text, is_valid, jur, weight = self.syntax_validator.validate_and_correct(text)
                        score = conf * (2.0 if is_valid else 1.0)
                        if score > best_score:
                            best_score = score
                            best_text = corr_text or text
                            best_plate_det = p_det
                            plate_candidates = self.ocr.read_candidates(p_crop, top_k=5)

            if best_plate_det is not None:
                draw_plate_overlay(annotated, best_plate_det.bbox, best_text)

            draw_vehicle_overlay(
                annotated,
                track_id=i,
                bbox=vehicle.bbox,
                color_label=color_lbl,
                type_label=type_lbl,
                plate_text=best_text,
            )

            records.append(VehicleRecord(
                track_id=i,
                plate_text=best_text,
                plate_conf=round(min(best_score, 1.0), 3),
                color=color_lbl,
                color_conf=round(color_conf, 3),
                vehicle_type=type_lbl,
                type_conf=round(type_conf, 3),
                frame_first_seen=0,
                frame_last_seen=0,
                bbox=vehicle.bbox,
                plate_bbox=best_plate_det.bbox if best_plate_det else None,
                orientation=orient_lbl,
                orientation_conf=orient_conf,
                plate_candidates=plate_candidates,
                jurisdiction="us-ca",
            ))

        return annotated, records

    @property
    def all_records(self) -> List[VehicleRecord]:
        return list(self._records.values())

    def reset(self):
        self._tracks.clear()
        self._records.clear()
        self._cached_plates.clear()
        self._frame_idx = 0
        if hasattr(self, "detector") and self.detector is not None:
            self.detector.reset()
        logger.info("Pipeline state reset.")

    def _get_stable_labels(self, state: _TrackState) -> Tuple[str, str, str, str]:
        color_lbl, type_lbl, orient_lbl = "Unknown", "Unknown", "Rear"
        plate_text = ""

        if state.color_probs is not None:
            c_idx = int(np.argmax(state.color_probs))
            if 0 <= c_idx < len(self.classifier.color_classes) and state.color_probs[c_idx] >= self.classifier.conf_cutoff:
                color_lbl = self.classifier.color_classes[c_idx]

        if state.type_probs is not None:
            t_idx = int(np.argmax(state.type_probs))
            if 0 <= t_idx < len(self.type_classes) and state.type_probs[t_idx] >= self.classifier.conf_cutoff:
                type_lbl = self.type_classes[t_idx]

        if state.orientation_probs is not None:
            o_idx = int(np.argmax(state.orientation_probs))
            if 0 <= o_idx < len(self.classifier.orientation_classes):
                orient_lbl = self.classifier.orientation_classes[o_idx]

        if state.plate_votes:
            plate_text = max(state.plate_votes, key=state.plate_votes.get)

        return color_lbl, type_lbl, plate_text, orient_lbl

    def _update_record(
        self,
        tid: int,
        state: _TrackState,
        color_lbl: str,
        type_lbl: str,
        plate_text: str,
        orient_lbl: str,
        bbox: Tuple[int, int, int, int],
        plates: list,
    ):
        if state.hit_count < self.min_hits:
            return

        color_conf = float(np.max(state.color_probs)) if state.color_probs is not None else 0.0
        type_conf = float(np.max(state.type_probs)) if state.type_probs is not None else 0.0
        orient_conf = float(np.max(state.orientation_probs)) if state.orientation_probs is not None else 0.85

        plate_total = sum(state.plate_votes.values())
        plate_conf = (state.plate_votes[plate_text] / max(plate_total, 1e-9)) if plate_text else 0.0

        plate_cands = [
            {"text": txt, "confidence": round(float(vote / max(plate_total, 1e-9)), 2)}
            for txt, vote in sorted(state.plate_votes.items(), key=lambda x: -x[1])[:5]
        ]
        best_pbbox = plates[0].bbox if plates else None

        self._records[tid] = VehicleRecord(
            track_id=tid,
            plate_text=plate_text,
            plate_conf=round(min(plate_conf, 1.0), 3),
            color=color_lbl,
            color_conf=round(color_conf, 3),
            vehicle_type=type_lbl,
            type_conf=round(type_conf, 3),
            frame_first_seen=state.frame_first_seen,
            frame_last_seen=state.frame_last_seen,
            bbox=bbox,
            plate_bbox=best_pbbox,
            orientation=orient_lbl,
            orientation_conf=round(orient_conf, 3),
            heading_deg=state.heading_deg,
            plate_candidates=plate_cands,
            jurisdiction="us-ca",
        )

    def _finalise_track(self, tid: int):
        if tid not in self._records:
            return
        state = self._tracks.get(tid)
        if state and state.best_crop is not None:
            fname = f"track_{tid}_f{state.frame_first_seen}.jpg"
            path = save_crop(state.best_crop, self._snapshot_dir, fname)
            self._records[tid].snapshot_path = path

    def _setup_logging(self):
        log_cfg = self.config.get("logging", {})
        log_file = log_cfg.get("file", "")
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            max_bytes = log_cfg.get("max_bytes", 10_485_760)
            backup_count = log_cfg.get("backup_count", 5)
            handler = RotatingFileHandler(
                str(log_path), maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            logging.getLogger().addHandler(handler)


# ── Asynchronous Multi-Threaded Decoupled Pipeline ─────────────────────────

class AsyncTrafficPipeline:
    """
    High-throughput asynchronous video pipeline.
    Decouples frame capture, YOLO tracking, and recognition workers using thread-safe queues.
    """

    def __init__(self, config_path: str = "config.yaml", config: Optional[dict] = None, queue_size: int = 30):
        self.pipeline = TrafficPipeline(config_path=config_path, config=config)
        self.queue_size = queue_size
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.result_queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._capture_thread = None
        self._process_thread = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start_stream(self, video_source: Union[str, int]):
        """Starts asynchronous capture and processing threads."""
        self._stop_event.clear()
        self._capture_thread = threading.Thread(target=self._capture_worker, args=(video_source,), daemon=True)
        self._process_thread = threading.Thread(target=self._process_worker, daemon=True)
        self._capture_thread.start()
        self._process_thread.start()

    def stream(self, video_source: Union[str, int]) -> Generator[Tuple[np.ndarray, List[VehicleRecord]], None, None]:
        """Generator yielding (annotated_frame, active_records) in real-time."""
        self.start_stream(video_source)
        while not self._stop_event.is_set():
            try:
                res = self.result_queue.get(timeout=0.1)
                if res is None:
                    break
                yield res
            except queue.Empty:
                if not self._capture_thread.is_alive() and self.frame_queue.empty():
                    break
                continue
        self.stop()

    def stop(self):
        self._stop_event.set()
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
        if self._process_thread and self._process_thread.is_alive():
            self._process_thread.join(timeout=1.0)

    def _capture_worker(self, source: Union[str, int]):
        cap = cv2.VideoCapture(source)
        try:
            while not self._stop_event.is_set() and cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                try:
                    self.frame_queue.put(frame, timeout=0.1)
                except queue.Full:
                    # Drop oldest frame to maintain live real-time latency
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except Exception:
                        pass
        finally:
            cap.release()
            self.frame_queue.put(None)

    def _process_worker(self):
        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
                if frame is None:
                    self.result_queue.put(None)
                    break
                annotated, records = self.pipeline.process_frame(frame)
                try:
                    self.result_queue.put((annotated, records), timeout=0.1)
                except queue.Full:
                    pass
            except queue.Empty:
                continue


# ── CLI Entrypoint ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TrekIT ALPR AI Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--input", default="", help="Input image or video path")
    parser.add_argument("--video", default="", help="Input video file / camera index")
    parser.add_argument("--output", default="", help="Output annotated file or directory")
    parser.add_argument("--conf", type=float, default=None, help="Vehicle detection confidence threshold")
    parser.add_argument("--json", action="store_true", help="Print enterprise JSON output")
    parser.add_argument("--async-mode", action="store_true", help="Run with asynchronous multi-threaded pipeline")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.conf is not None:
        cfg.setdefault("detection", {})["confidence_threshold"] = args.conf

    input_path = args.input or args.video
    if not input_path:
        print("Please provide --input <image/video>.")
        return

    pipeline = TrafficPipeline(config=cfg)
    p = Path(input_path)

    # Image inference
    if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        img = cv2.imread(str(p))
        if img is None:
            print(f"Failed to read image from {input_path}")
            return
        ann, recs = pipeline.process_image(img)
        if args.json:
            print(json.dumps([r.to_enterprise_dict() for r in recs], indent=2))
        else:
            for r in recs:
                plate_str = f"[{r.plate_text}] ({r.plate_conf:.0%})" if r.plate_text else "None"
                print(f"Vehicle #{r.track_id}: {r.color} {r.vehicle_type} ({r.orientation}) | Plate: {plate_str}")
        if args.output:
            cv2.imwrite(args.output, ann)
            print(f"Saved annotated image to {args.output}")

    # Video stream inference
    else:
        cap = cv2.VideoCapture(str(p))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            ann, recs = pipeline.process_frame(frame)
        cap.release()
        recs = pipeline.all_records
        if args.json:
            print(json.dumps([r.to_enterprise_dict() for r in recs], indent=2))
        else:
            print(f"Processed video. Found {len(recs)} unique vehicles.")


if __name__ == "__main__":
    main()
