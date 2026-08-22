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
        dev_str = self.cfg.get("device", "cpu")
        if dev_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested for detector but not available. Falling back to CPU.")
            dev_str = "cpu"
        self.device = dev_str
        self.use_fp16 = self.cfg.get("fp16", True) and (self.device == "cuda" or "cuda" in str(self.device))
        self.conf_thresh = self.cfg.get("confidence_threshold", 0.45)
        self.iou_thresh = self.cfg.get("nms_iou_threshold", 0.45)
        self.input_size = self.cfg.get("input_size", 640)
        self.vehicle_class_ids = set(self.cfg.get("vehicle_class_ids", [2, 3, 5, 7]))
        self.plate_class_id = self.cfg.get("plate_class_id", 0)

        # ── Load vehicle detector (yolo11 / yolov8 for COCO vehicle classes) ────
        coco_weights = weights_path or self._resolve_weights("models/yolo11s.pt", "models/yolo11n.pt", "models/yolov8n_pretrained.pt")
        logger.info(f"Loading vehicle detector from: {coco_weights} (FP16: {self.use_fp16})")
        self.model = YOLO(coco_weights)
        self.model.to(self.device)

        # ── Load custom license plate detector (plate_detector.pt) ─────────
        custom_plate_path = plate_weights_path or config.get("paths", {}).get("yolo_weights", "")
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
        self.plate_check_every: int = self.cfg.get("plate_check_every_n", 1)

        # Resolve custom tracker YAML (bytetrack_custom.yaml with 60-frame buffer)
        custom_tracker = Path("bytetrack_custom.yaml")
        if custom_tracker.exists():
            self.tracker_name = str(custom_tracker.resolve())
        else:
            self.tracker_name = self.track_cfg.get("tracker", "bytetrack.yaml")

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

        # Run tracker with low association floor (0.10) so ByteTrack can maintain
        # Kalman tracks across momentary confidence dips without creating new IDs.
        results = self.model.track(
            infer_frame,
            conf=0.10,
            iou=self.iou_thresh,
            imgsz=self.input_size,
            device=self.device,
            tracker=self.tracker_name,
            persist=True,
            verbose=False,
        )

        vehicles: List[VehicleDetection] = []
        if not results or results[0].boxes is None:
            return vehicles

        boxes = results[0].boxes
        cls_list = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
        conf_list = boxes.conf.cpu().numpy().astype(float) if boxes.conf is not None else []
        id_list = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
        xyxy_list = boxes.xyxy.cpu().numpy().astype(int) if boxes.xyxy is not None else []

        for i, cls_id in enumerate(cls_list):
            if cls_id not in self.vehicle_class_ids:
                continue

            conf = float(conf_list[i]) if i < len(conf_list) else 0.0
            # Filter output boxes with primary confidence threshold
            if conf < self.conf_thresh:
                continue

            track_id = int(id_list[i]) if (id_list is not None and i < len(id_list)) else -1
            if i >= len(xyxy_list):
                continue
            x1, y1, x2, y2 = xyxy_list[i]

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
            vehicles.append(vehicle)

        # ── Full-Scene Plate Detection (Scale-Matched) ────────────────────────
        # Plate models trained on street scenes (plates 2-5% of frame) expect
        # full-scene context rather than zoomed-in vehicle crops.
        all_scene_plates: List[Detection] = []
        if self.plate_model is not None and vehicles:
            all_scene_plates = self._detect_plates_full_frame(infer_frame, (h, w), scale)
            self._associate_plates_to_vehicles(vehicles, all_scene_plates)

        # ── Fallback plate detection for vehicles without matched plates ──────
        for vehicle in vehicles:
            if not vehicle.plates:
                last = self._plate_last_checked.get(vehicle.track_id, -9999)
                if self._frame_count - last >= self.plate_check_every:
                    vehicle.plates = self._detect_plates_in_crop(frame, vehicle.bbox)
                    self._plate_last_checked[vehicle.track_id] = self._frame_count

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

        h, w = image.shape[:2]
        boxes = results[0].boxes
        cls_list = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
        conf_list = boxes.conf.cpu().numpy().astype(float) if boxes.conf is not None else []
        xyxy_list = boxes.xyxy.cpu().numpy().astype(int) if boxes.xyxy is not None else []

        for i, cls_id in enumerate(cls_list):
            if cls_id not in self.vehicle_class_ids:
                continue
            conf = float(conf_list[i]) if i < len(conf_list) else 0.0
            if i >= len(xyxy_list):
                continue
            x1, y1, x2, y2 = xyxy_list[i]
            bbox = (int(x1), int(y1), int(x2), int(y2))
            vehicle = VehicleDetection(track_id=-1, bbox=bbox, confidence=conf)
            vehicles.append(vehicle)

        # Full-scene plate detection
        if self.plate_model is not None and vehicles:
            scene_plates = self._detect_plates_full_frame(image, (h, w), scale=1.0)
            self._associate_plates_to_vehicles(vehicles, scene_plates)

        # Morphological fallback for vehicles without detected plates
        for vehicle in vehicles:
            if not vehicle.plates:
                vehicle.plates = self._detect_plates_in_crop(image, vehicle.bbox)

        return vehicles

    def reset(self):
        """Reset internal frame counter and ByteTrack tracker state."""
        self._frame_count = 0
        self._plate_last_checked.clear()
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            if hasattr(self.model.predictor, "trackers"):
                try:
                    del self.model.predictor.trackers
                except Exception:
                    pass
            self.model.predictor = None

    # ── Internal helpers ────────────────────────────────────────────────

    def _detect_plates_full_frame(
        self, infer_frame: np.ndarray, orig_shape: Tuple[int, int], scale: float = 1.0
    ) -> List[Detection]:
        """
        Run plate detection on the full frame at native street-scene scale.
        Matches how plate models (e.g. YOLO on Kaggle plate dataset) are trained.
        """
        if self.plate_model is None:
            return []

        orig_h, orig_w = orig_shape
        all_plates: List[Detection] = []
        try:
            results = self.plate_model(
                infer_frame,
                conf=0.12,
                iou=self.iou_thresh,
                imgsz=self.input_size,
                device=self.device,
                verbose=False,
            )
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                cls_list = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
                conf_list = boxes.conf.cpu().numpy().astype(float) if boxes.conf is not None else []
                xyxy_list = boxes.xyxy.cpu().numpy().astype(int) if boxes.xyxy is not None else []

                for i, cls_id in enumerate(cls_list):
                    conf = float(conf_list[i]) if i < len(conf_list) else 0.0
                    if i >= len(xyxy_list):
                        continue
                    px1, py1, px2, py2 = xyxy_list[i]
                    if scale != 1.0:
                        inv = 1.0 / scale
                        px1, py1, px2, py2 = int(px1 * inv), int(py1 * inv), int(px2 * inv), int(py2 * inv)

                    # Add padding around plate box (15%) so character edges aren't clipped
                    pw = px2 - px1
                    ph = py2 - py1
                    pad_w = int(pw * 0.15)
                    pad_h = int(ph * 0.15)
                    px1 = max(0, px1 - pad_w)
                    py1 = max(0, py1 - pad_h)
                    px2 = min(orig_w, px2 + pad_w)
                    py2 = min(orig_h, py2 + pad_h)

                    all_plates.append(
                        Detection(
                            track_id=-1,
                            class_id=int(cls_id),
                            label="license_plate",
                            confidence=conf,
                            bbox=(px1, py1, px2, py2),
                            is_plate=True,
                        )
                    )
        except Exception as e:
            logger.warning(f"Full-scene plate detection error: {e}")

        return all_plates

    @staticmethod
    def _associate_plates_to_vehicles(
        vehicles: List[VehicleDetection], plates: List[Detection]
    ):
        """
        Assigns detected plates to their containing vehicle by checking if the plate center
        lies within the vehicle bounding box.
        """
        for plate in plates:
            px1, py1, px2, py2 = plate.bbox
            pcx = (px1 + px2) // 2
            pcy = (py1 + py2) // 2

            best_v = None
            best_area = float("inf")
            for v in vehicles:
                vx1, vy1, vx2, vy2 = v.bbox
                # Allow a 15px margin around vehicle bbox
                if (vx1 - 15 <= pcx <= vx2 + 15) and (vy1 - 15 <= pcy <= vy2 + 15):
                    v_area = (vx2 - vx1) * (vy2 - vy1)
                    if v_area < best_area:
                        best_area = v_area
                        best_v = v

            if best_v is not None:
                best_v.plates.append(plate)

    def _detect_plates_in_crop(
        self, frame: np.ndarray, vehicle_bbox: Tuple[int, int, int, int]
    ) -> List[Detection]:
        """
        Detect license plates within the cropped vehicle region.
        Uses `self.plate_model` with resolution upscaling and low confidence floor (0.15).
        Falls back to OpenCV ANPR morphological gradient contour locator.
        """
        x1, y1, x2, y2 = vehicle_bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        vh = y2 - y1
        vw = x2 - x1
        if vh <= 10 or vw <= 10:
            return []

        # ── 1. If custom trained plate model exists, run it ─────────────────
        if self.plate_model is not None:
            crop = frame[y1:y2, x1:x2]
            
            # Upscale small vehicle crops so license plate details aren't lost
            crop_scale = 1.0
            if vw < 320 or vh < 200:
                crop_scale = min(320.0 / max(vw, 1), 200.0 / max(vh, 1))
                crop_infer = cv2.resize(
                    crop,
                    (int(vw * crop_scale), int(vh * crop_scale)),
                    interpolation=cv2.INTER_CUBIC,
                )
            else:
                crop_infer = crop

            try:
                # Use conf=0.15 to catch small / shadowed / distant plates
                results = self.plate_model(
                    crop_infer,
                    conf=0.15,
                    iou=self.iou_thresh,
                    device=self.device,
                    verbose=False,
                )
                plates: List[Detection] = []
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    cls_list = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
                    conf_list = boxes.conf.cpu().numpy().astype(float) if boxes.conf is not None else []
                    xyxy_list = boxes.xyxy.cpu().numpy().astype(int) if boxes.xyxy is not None else []

                    for i, cls_id in enumerate(cls_list):
                        conf = float(conf_list[i]) if i < len(conf_list) else 0.0
                        if i >= len(xyxy_list):
                            continue
                        px1, py1, px2, py2 = xyxy_list[i]

                        # Scale back if upscaled
                        if crop_scale != 1.0:
                            inv = 1.0 / crop_scale
                            px1, py1, px2, py2 = int(px1 * inv), int(py1 * inv), int(px2 * inv), int(py2 * inv)

                        # Add 15% padding around plate box so character edges aren't clipped
                        pw = px2 - px1
                        ph = py2 - py1
                        pad_w = int(pw * 0.15)
                        pad_h = int(ph * 0.15)
                        px1 = max(0, px1 - pad_w)
                        py1 = max(0, py1 - pad_h)
                        px2 = min(vw, px2 + pad_w)
                        py2 = min(vh, py2 + pad_h)

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

        # ── 2. OpenCV ANPR Plate Contour Locator fallback ───────────────────
        crop = frame[y1:y2, x1:x2]
        try:
            ymin = int(vh * 0.35)
            crop_lower = crop[ymin:, :]
            lh, lw = crop_lower.shape[:2]

            if lh >= 15 and lw >= 30:
                gray = cv2.cvtColor(crop_lower, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                sobelx = cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=3)
                _, thresh = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
                closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

                contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                anpr_plates: List[Detection] = []

                for cnt in contours:
                    cx, cy, cw, ch = cv2.boundingRect(cnt)
                    aspect_ratio = cw / float(max(ch, 1))
                    area = cw * ch
                    mid_x = cx + cw / 2.0
                    abs_y = ymin + cy

                    # Reject contours at outer edges (headlights/taillights/mirrors) or too high up
                    if not (0.15 * vw <= mid_x <= 0.85 * vw):
                        continue
                    if abs_y < 0.35 * vh:
                        continue

                    if 1.5 <= aspect_ratio <= 5.8 and 80 <= area <= (lw * lh * 0.40):
                        abs_px1 = x1 + cx
                        abs_py1 = y1 + ymin + cy
                        abs_px2 = x1 + cx + cw
                        abs_py2 = y1 + ymin + cy + ch

                        pad_w = int(cw * 0.12)
                        pad_h = int(ch * 0.12)
                        abs_px1 = max(0, abs_px1 - pad_w)
                        abs_py1 = max(0, abs_py1 - pad_h)
                        abs_px2 = min(w, abs_px2 + pad_w)
                        abs_py2 = min(h, abs_py2 + pad_h)

                        anpr_plates.append(
                            Detection(
                                track_id=-1,
                                class_id=0,
                                label="license_plate",
                                confidence=0.75,
                                bbox=(abs_px1, abs_py1, abs_px2, abs_py2),
                                is_plate=True,
                            )
                        )
                if anpr_plates:
                    return anpr_plates[:2]
        except Exception as e:
            logger.debug(f"OpenCV ANPR plate locator failed: {e}")

        # ── 3. No plate detected — return empty list ────────────────────────
        return []

    def _resolve_weights(self, *candidates: str) -> str:
        """
        Return the first existing weights path from candidates, downloading fallback if none exist.
        """
        for c in candidates:
            if c and Path(c).exists():
                return c

        fallback = Path("models/yolov8n_pretrained.pt")
        if fallback.exists():
            logger.info("Using cached pretrained YOLO weights.")
            return str(fallback)

        fallback_url = self.cfg.get("yolo_fallback_url", self.FALLBACK_URL)
        logger.warning(
            f"Weights not found in candidates {candidates}. "
            "Downloading pretrained YOLO from Ultralytics..."
        )
        fallback.parent.mkdir(parents=True, exist_ok=True)
        try:
            resp = requests.get(fallback_url, timeout=120, stream=True)
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
