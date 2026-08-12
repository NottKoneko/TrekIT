"""
detector.py
-----------
YOLOv8n vehicle & license plate detection with ByteTrack multi-object tracking.

Responsibilities:
  - Load YOLOv8n model (auto-downloads pretrained weights as fallback)
  - Detect vehicle bounding boxes in a frame
  - Detect license plate sub-regions within vehicle crops
  - Run ByteTrack to assign persistent tracking IDs across frames
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import requests
import torch

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detected object."""
    track_id: int               # ByteTrack persistent ID (-1 if not tracked)
    class_id: int
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2 (pixel coords)
    is_plate: bool = False


@dataclass
class VehicleDetection:
    """A vehicle with its associated license plate detections."""
    track_id: int
    bbox: Tuple[int, int, int, int]     # vehicle bbox
    confidence: float
    plates: List[Detection] = field(default_factory=list)


class VehicleDetector:
    """
    Wraps YOLOv8n for vehicle + license plate detection.

    If `weights_path` doesn't exist:
      1. Tries custom fine-tuned model from config
      2. Falls back to YOLOv8n pretrained on COCO

    The single model handles both vehicle detection (via COCO classes)
    and license plate detection (via custom class) if a fine-tuned model
    is provided. Otherwise runs COCO-only vehicle detection with a
    separate lightweight plate detector.
    """

    # COCO class IDs for vehicles
    VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    PLATE_CLASS_ID = 0    # class 0 in the custom plate detection model

    FALLBACK_URL = (
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
    )

    def __init__(
        self,
        config: dict,
        weights_path: Optional[str] = None,
        plate_weights_path: Optional[str] = None,
    ):
        """
        Args:
            config: Full config dict (from config.yaml)
            weights_path: Path to custom vehicle+plate YOLO model (.pt)
            plate_weights_path: Path to a dedicated plate detector model (.pt)
        """
        from ultralytics import YOLO

        self.cfg = config.get("detection", {})
        self.track_cfg = config.get("tracking", {})
        self.device = self.cfg.get("device", "cpu")
        self.conf_thresh = self.cfg.get("confidence_threshold", 0.45)
        self.iou_thresh = self.cfg.get("nms_iou_threshold", 0.45)
        self.input_size = self.cfg.get("input_size", 640)
        self.vehicle_class_ids = set(self.cfg.get("vehicle_class_ids", [2, 3, 5, 7]))
        self.plate_class_id = self.cfg.get("plate_class_id", 0)

        # ── Load vehicle detector (yolov8n.pt for COCO vehicle classes) ────
        coco_weights = self._resolve_weights("models/yolov8n_pretrained.pt")
        logger.info(f"Loading vehicle detector from: {coco_weights}")
        self.model = YOLO(coco_weights)
        self.model.to(self.device)

        # ── Load custom license plate detector (plate_detector.pt) ─────────
        custom_plate_path = weights_path or config.get("paths", {}).get("yolo_weights", "")
        self.plate_model = None
        if custom_plate_path and Path(custom_plate_path).exists():
            logger.info(f"Loading license plate detector from: {custom_plate_path}")
            self.plate_model = YOLO(custom_plate_path)
            self.plate_model.to(self.device)
        else:
            logger.info("Custom plate detector not found; using main model for plates.")

        self._frame_count = 0
        # track_id → last frame plate detection ran
        self._plate_last_checked: dict = {}
        self.plate_check_every: int = self.cfg.get("plate_check_every_n", 5)

    # ── Public API ──────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[VehicleDetection]:
        """
        Run detection + tracking on a single BGR frame.

        Returns:
            List of VehicleDetection objects, each containing the vehicle
            bbox/track_id and any license plates found within it.
        """
        self._frame_count += 1

        # ── Downscale oversized frames for YOLO (4K→1080p saves ~75% YOLO time) ──
        h, w = frame.shape[:2]
        scale = 1.0
        if max(h, w) > 1920:
            scale = 1920 / max(h, w)
            infer_frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                     interpolation=cv2.INTER_AREA)
        else:
            infer_frame = frame

        # Run tracker on (possibly downscaled) frame
        results = self.model.track(
            infer_frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.input_size,
            device=self.device,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )

        vehicles: List[VehicleDetection] = []
        if not results or results[0].boxes is None:
            return vehicles

        boxes = results[0].boxes
        for i, cls_id in enumerate(boxes.cls.cpu().numpy().astype(int)):
            if cls_id not in self.vehicle_class_ids:
                continue

            track_id = int(boxes.id[i].cpu()) if boxes.id is not None else -1
            conf = float(boxes.conf[i].cpu())
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

            # Scale bboxes back to original frame coordinates
            if scale != 1.0:
                inv = 1.0 / scale
                x1, y1, x2, y2 = (int(x1*inv), int(y1*inv),
                                   int(x2*inv), int(y2*inv))

            bbox = (int(x1), int(y1), int(x2), int(y2))

            vehicle = VehicleDetection(
                track_id=track_id,
                bbox=bbox,
                confidence=conf,
            )

            # ── Throttled plate detection: only run every N frames per track ──
            last = self._plate_last_checked.get(track_id, -9999)
            if self._frame_count - last >= self.plate_check_every:
                vehicle.plates = self._detect_plates_in_crop(frame, bbox)
                self._plate_last_checked[track_id] = self._frame_count
            # else: vehicle.plates stays empty — pipeline reuses cached plate from state

            vehicles.append(vehicle)

        return vehicles

    def detect_image(self, image: np.ndarray) -> List[VehicleDetection]:
        """Single-image detection without tracking (no track IDs assigned)."""
        results = self.model(
            image,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.input_size,
            device=self.device,
            verbose=False,
        )
        vehicles: List[VehicleDetection] = []
        if not results or results[0].boxes is None:
            return vehicles

        boxes = results[0].boxes
        for i, cls_id in enumerate(boxes.cls.cpu().numpy().astype(int)):
            if cls_id not in self.vehicle_class_ids:
                continue
            conf = float(boxes.conf[i].cpu())
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            bbox = (int(x1), int(y1), int(x2), int(y2))
            vehicle = VehicleDetection(track_id=-1, bbox=bbox, confidence=conf)
            vehicle.plates = self._detect_plates_in_crop(image, bbox)
            vehicles.append(vehicle)

        return vehicles

    # ── Internal helpers ────────────────────────────────────────────────

    def _detect_plates_in_crop(
        self, frame: np.ndarray, vehicle_bbox: Tuple[int, int, int, int]
    ) -> List[Detection]:
        """
        Detect license plates within the cropped vehicle region.
        Uses `self.plate_model` (trained YOLO) if available. If no plate detected,
        falls back to geometric bumper area heuristic.
        """
        x1, y1, x2, y2 = vehicle_bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        vh = y2 - y1
        vw = x2 - x1
        if vh <= 10 or vw <= 10:
            return []

        # ── 1. If custom trained plate model exists, use it ─────────────────
        if self.plate_model is not None:
            crop = frame[y1:y2, x1:x2]
            try:
                # Use 0.20 conf threshold for plate detection (plates are small/distant)
                plate_conf = min(self.conf_thresh, 0.20)
                results = self.plate_model(
                    crop,
                    conf=plate_conf,
                    iou=self.iou_thresh,
                    device=self.device,
                    verbose=False,
                )
                plates: List[Detection] = []
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for i, cls_id in enumerate(boxes.cls.cpu().numpy().astype(int)):
                        conf = float(boxes.conf[i].cpu())
                        px1, py1, px2, py2 = boxes.xyxy[i].cpu().numpy().astype(int)
                        abs_bbox = (x1 + px1, y1 + py1, x1 + px2, y1 + py2)
                        plates.append(
                            Detection(
                                track_id=-1,
                                class_id=int(cls_id),
                                label="license_plate",
                                confidence=conf,
                                bbox=abs_bbox,
                                is_plate=True,
                            )
                        )
                if plates:
                    return plates
            except Exception as e:
                logger.warning(f"Plate detection failed: {e}")

        # ── 2. Fallback bumper area heuristic if plate_model is missing or returned 0 detections ─────────
        px1 = x1 + int(vw * 0.20)
        px2 = x1 + int(vw * 0.80)
        py1 = y1 + int(vh * 0.55)
        py2 = y1 + int(vh * 0.95)

        return [
            Detection(
                track_id=-1,
                class_id=0,
                label="license_plate",
                confidence=0.5,
                bbox=(px1, py1, px2, py2),
                is_plate=True,
            )
        ]

    def _resolve_weights(self, weights_path: str) -> str:
        """
        Return a valid weights path, downloading YOLOv8n pretrained as fallback.
        """
        if weights_path and Path(weights_path).exists():
            return weights_path

        fallback = Path("models/yolov8n_pretrained.pt")
        if fallback.exists():
            logger.info("Using cached pretrained YOLOv8n weights.")
            return str(fallback)

        logger.warning(
            f"Custom weights not found at '{weights_path}'. "
            "Downloading pretrained YOLOv8n from Ultralytics..."
        )
        fallback.parent.mkdir(parents=True, exist_ok=True)
        try:
            resp = requests.get(self.FALLBACK_URL, timeout=120, stream=True)
            resp.raise_for_status()
            with open(fallback, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Downloaded pretrained weights to {fallback}")
        except Exception as e:
            logger.error(f"Could not download pretrained weights: {e}")
            # Let ultralytics handle it via its own auto-download
            return "yolov8n.pt"

        return str(fallback)
