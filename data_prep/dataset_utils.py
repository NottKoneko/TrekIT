"""
dataset_utils.py
----------------
Utility functions for dataset preparation:
  - Stratified train/val split
  - YOLO annotation format conversion
  - Image validation and resizing
"""

import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Train / Val split ───────────────────────────────────────────────────────

def stratified_train_val_split(
    source_dir: str,
    output_dir: str,
    val_ratio: float = 0.20,
    seed: int = 42,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> Dict[str, int]:
    """
    Split a flat class-folder dataset (source_dir/<class_name>/*.jpg) into
    train/ and val/ splits, preserving class distribution.

    source_dir/
      Sedan/img001.jpg ...
      SUV/img002.jpg ...

    output_dir/
      train/Sedan/img001.jpg ...
      val/SUV/img002.jpg ...

    Returns a dict with split counts.
    """
    random.seed(seed)
    source = Path(source_dir)
    out = Path(output_dir)

    train_count = 0
    val_count = 0

    for class_dir in sorted(source.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        images = [
            p for p in class_dir.iterdir()
            if p.suffix.lower() in extensions
        ]
        random.shuffle(images)
        n_val = max(1, int(len(images) * val_ratio))
        val_imgs = images[:n_val]
        train_imgs = images[n_val:]

        for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
            dest = out / split / class_name
            dest.mkdir(parents=True, exist_ok=True)
            for img_path in imgs:
                shutil.copy2(img_path, dest / img_path.name)

        train_count += len(train_imgs)
        val_count += len(val_imgs)
        logger.info(
            f"  {class_name}: {len(train_imgs)} train / {len(val_imgs)} val"
        )

    logger.info(f"Split complete: {train_count} train, {val_count} val")
    return {"train": train_count, "val": val_count}


# ── YOLO annotation converter ──────────────────────────────────────────────

def convert_bbox_to_yolo(
    img_w: int,
    img_h: int,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
) -> Tuple[float, float, float, float]:
    """
    Convert absolute (x_min, y_min, x_max, y_max) pixel bbox
    to YOLO normalised (cx, cy, w, h) format (all 0–1).
    """
    cx = ((x_min + x_max) / 2) / img_w
    cy = ((y_min + y_max) / 2) / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)


def write_yolo_annotation(
    label_path: str,
    class_id: int,
    cx: float,
    cy: float,
    w: float,
    h: float,
) -> None:
    """Append one YOLO annotation line to a .txt label file."""
    with open(label_path, "a", encoding="utf-8") as f:
        f.write(f"{class_id} {cx} {cy} {w} {h}\n")


# ── Image validation & resizing ────────────────────────────────────────────

def validate_and_resize_images(
    directory: str,
    target_size: Optional[int] = None,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    remove_corrupt: bool = True,
) -> Dict[str, int]:
    """
    Walk a directory, validate each image, and optionally resize.

    Returns:
        {"valid": N, "corrupt": M, "resized": K}
    """
    stats = {"valid": 0, "corrupt": 0, "resized": 0}
    for img_path in Path(directory).rglob("*"):
        if img_path.suffix.lower() not in extensions:
            continue
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError("cv2 returned None")

            if target_size is not None:
                h, w = img.shape[:2]
                if max(h, w) > target_size:
                    scale = target_size / max(h, w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(img_path), img)
                    stats["resized"] += 1

            stats["valid"] += 1

        except Exception as e:
            logger.warning(f"Corrupt image {img_path}: {e}")
            stats["corrupt"] += 1
            if remove_corrupt:
                img_path.unlink(missing_ok=True)

    logger.info(
        f"Validation: {stats['valid']} valid, {stats['corrupt']} corrupt, "
        f"{stats['resized']} resized"
    )
    return stats


# ── Dataset YAML generator (for YOLO training) ──────────────────────────────

def write_yolo_dataset_yaml(
    output_path: str,
    train_dir: str,
    val_dir: str,
    class_names: List[str],
) -> str:
    """Write a YOLO dataset.yaml file for ultralytics training."""
    yaml_content = (
        f"path: .\n"
        f"train: {train_dir}\n"
        f"val: {val_dir}\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    logger.info(f"YOLO dataset YAML written to {output_path}")
    return output_path
