"""
veri776_parser.py
------------------
Parses the VeRi-776 surveillance vehicle dataset and trains the unified
Multi-Task VehicleAttributeNet (Color + Body Type) on CCTV traffic perspectives.

Usage:
  python data_prep/veri776_parser.py --data_dir data/veri776 --epochs 30 --batch_size 32
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Standard TrekIT classes
TREKIT_COLORS = [
    "beige", "black", "blue", "brown", "gold", "green", "grey",
    "orange", "pink", "purple", "red", "silver", "tan", "white", "yellow"
]

TREKIT_TYPES = [
    "Convertible", "Coupe", "Hatchback", "SUV", "Sedan", "Truck", "Van"
]

# VeRi-776 mapping to TrekIT standard classes
VERI_COLOR_MAP = {
    1: "yellow",
    2: "orange",
    3: "green",
    4: "grey",
    5: "red",
    6: "blue",
    7: "white",
    8: "gold",
    9: "brown",
    10: "black",
    "yellow": "yellow", "orange": "orange", "green": "green",
    "gray": "grey", "grey": "grey", "red": "red", "blue": "blue",
    "white": "white", "golden": "gold", "gold": "gold",
    "brown": "brown", "black": "black", "silver": "silver",
    "beige": "beige", "pink": "pink", "purple": "purple",
}

VERI_TYPE_MAP = {
    1: "Sedan",
    2: "SUV",
    3: "Van",
    4: "Hatchback",
    5: "SUV",        # Map MPV/Crossovers -> SUV rather than Van
    6: "Truck",      # Pickup -> Truck
    7: "Truck",      # Bus / Heavy -> Truck
    8: "Truck",      # Truck -> Truck
    "sedan": "Sedan", "suv": "SUV", "van": "Van",
    "hatchback": "Hatchback", "mpv": "SUV", "pickup": "Truck",
    "bus": "Truck", "truck": "Truck", "coupe": "Coupe",
    "convertible": "Convertible",
}


class VehicleAttributeNet(nn.Module):
    """
    Multi-Task MobileNetV3-Large network for joint vehicle Color + Type classification.
    """
    def __init__(self, n_colors: int = len(TREKIT_COLORS), n_types: int = len(TREKIT_TYPES), pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v3_large(weights=weights)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_feat = 960

        self.color_head = nn.Sequential(
            nn.Linear(in_feat, 256),
            nn.Hardswish(),
            nn.Dropout(0.25),
            nn.Linear(256, n_colors),
        )
        self.type_head = nn.Sequential(
            nn.Linear(in_feat, 256),
            nn.Hardswish(),
            nn.Dropout(0.25),
            nn.Linear(256, n_types),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x)
        feat = self.pool(feat)
        feat = torch.flatten(feat, 1)
        return self.color_head(feat), self.type_head(feat)


class VeRiAttributeDataset(Dataset):
    """
    Dataset pairing vehicle surveillance crops with (color_idx, type_idx).
    """
    def __init__(self, samples: List[Tuple[Path, int, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, c_idx, t_idx = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(c_idx, dtype=torch.long), torch.tensor(t_idx, dtype=torch.long)


def parse_veri776(veri_dir: Path) -> List[Tuple[Path, int, int]]:
    """
    Parses VeRi-776 annotations from XML files, list text files, or directory trees.
    Supports train_label.xml and test_label.xml with multi-byte XML declarations.
    """
    if (veri_dir / "VeRi").exists():
        veri_dir = veri_dir / "VeRi"

    samples = []
    xml_targets = [
        (veri_dir / "train_label.xml", veri_dir / "image_train"),
        (veri_dir / "test_label.xml", veri_dir / "image_test"),
    ]

    import re
    for xml_path, img_dir in xml_targets:
        if not img_dir.exists():
            img_dir = veri_dir

        if xml_path.exists():
            logger.info(f"Parsing VeRi-776 XML: {xml_path}")
            try:
                raw_bytes = xml_path.read_bytes()
                try:
                    raw_text = raw_bytes.decode("gb18030", errors="ignore")
                except Exception:
                    raw_text = raw_bytes.decode("utf-8", errors="ignore")

                clean_xml = re.sub(r"<\?xml[^>]*\?>", "", raw_text)
                root = ET.fromstring(clean_xml)

                for item in root.findall(".//Item"):
                    img_name = item.get("imageName")
                    c_id = int(item.get("colorID", 0))
                    t_id = int(item.get("typeID", 0))

                    c_name = VERI_COLOR_MAP.get(c_id, "grey")
                    t_name = VERI_TYPE_MAP.get(t_id, "Sedan")

                    c_idx = TREKIT_COLORS.index(c_name.lower()) if c_name.lower() in TREKIT_COLORS else 6
                    t_idx = TREKIT_TYPES.index(t_name) if t_name in TREKIT_TYPES else 4

                    p = img_dir / img_name
                    if p.exists():
                        samples.append((p, c_idx, t_idx))
            except Exception as e:
                logger.warning(f"Error reading {xml_path}: {e}")

    logger.info(f"Parsed {len(samples)} valid VeRi-776 multi-task samples.")
    return samples


def organize_veri_imagefolder(veri_dir: Path, output_dir: Path = Path("data/veri_organized")) -> Tuple[Path, Path]:
    """
    Organizes VeRi-776 images into ImageFolder-ready directory trees:
      - data/veri_organized/color/<color_name>/<img_name>.jpg
      - data/veri_organized/type/<type_name>/<img_name>.jpg
    """
    import shutil
    color_root = output_dir / "color"
    type_root = output_dir / "type"

    for c in TREKIT_COLORS:
        (color_root / c).mkdir(parents=True, exist_ok=True)
    for t in TREKIT_TYPES:
        (type_root / t).mkdir(parents=True, exist_ok=True)

    samples = parse_veri776(veri_dir)
    logger.info(f"Organizing {len(samples)} images into {output_dir}...")

    for path, c_idx, t_idx in tqdm(samples, desc="Organizing VeRi ImageFolders"):
        c_name = TREKIT_COLORS[c_idx]
        t_name = TREKIT_TYPES[t_idx]

        dst_c = color_root / c_name / path.name
        if not dst_c.exists():
            shutil.copyfile(path, dst_c)

        dst_t = type_root / t_name / path.name
        if not dst_t.exists():
            shutil.copyfile(path, dst_t)

    logger.info(f"Done! ImageFolders ready at:\n  Color: {color_root}\n  Type:  {type_root}")
    return color_root, type_root


def parse_western_cctv_crops(extra_dirs: List[Path]) -> List[Tuple[Path, int, int]]:
    """
    Parses Western CCTV surveillance crops from BDD100K, BoxCars116k, or MIO-TCD
    and maps them to standard TREKIT_COLORS and TREKIT_TYPES.
    """
    extra_samples = []
    type_to_idx = {t.lower(): i for i, t in enumerate(TREKIT_TYPES)}

    for edir in extra_dirs:
        if not edir.exists():
            continue
        for type_subdir in edir.iterdir():
            if type_subdir.is_dir():
                type_name = type_subdir.name.lower()
                matched_type_idx = None
                for t_name, t_idx in type_to_idx.items():
                    if t_name in type_name or type_name in t_name:
                        matched_type_idx = t_idx
                        break
                if matched_type_idx is None:
                    matched_type_idx = type_to_idx.get("sedan", 4)

                for img_p in list(type_subdir.glob("*.jpg")) + list(type_subdir.glob("*.png")):
                    c_idx = TREKIT_COLORS.index("grey") if "grey" in TREKIT_COLORS else 6
                    extra_samples.append((img_p, c_idx, matched_type_idx))

    if extra_samples:
        logger.info(f"Loaded {len(extra_samples)} Western surveillance crops from extra datasets.")
    return extra_samples


def train_vehicle_attribute_net(
    data_dir: str = "data/veri776",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_path: str = "models/vehicle_attributes.pt",
):
    """
    Trains the multi-task VehicleAttributeNet on combined VeRi-776 and Western CCTV surveillance crops.
    """
    veri_path = Path(data_dir)
    samples = parse_veri776(veri_path)

    # Ingest Western surveillance crops if available
    extra_dirs = [Path("data/bdd100k_subset"), Path("data/boxcars116k"), Path("data/miotcd_cctv_subset")]
    western_samples = parse_western_cctv_crops(extra_dirs)
    samples.extend(western_samples)

    if not samples:
        logger.warning(f"No samples found in {data_dir}.")
        return

    import random
    random.seed(42)
    random.shuffle(samples)
    split = int(0.85 * len(samples))
    train_samples, val_samples = samples[:split], samples[split:]

    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomPerspective(distortion_scale=0.30, p=0.6),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = VeRiAttributeDataset(train_samples, train_tf)
    val_ds = VeRiAttributeDataset(val_samples, val_tf)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = VehicleAttributeNet(len(TREKIT_COLORS), len(TREKIT_TYPES), pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting VehicleAttributeNet training for {epochs} epochs on {device}...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, c_correct, t_correct, total = 0.0, 0, 0, 0
        for imgs, c_labels, t_labels in tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}"):
            imgs, c_labels, t_labels = imgs.to(device), c_labels.to(device), t_labels.to(device)
            optimizer.zero_grad()
            c_preds, t_preds = model(imgs)
            loss = criterion(c_preds, c_labels) + 1.2 * criterion(t_preds, t_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            c_correct += (c_preds.argmax(1) == c_labels).sum().item()
            t_correct += (t_preds.argmax(1) == t_labels).sum().item()
            total += imgs.size(0)

        scheduler.step()
        train_c_acc = c_correct / total
        train_t_acc = t_correct / total

        model.eval()
        val_c_correct, val_t_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, c_labels, t_labels in val_dl:
                imgs, c_labels, t_labels = imgs.to(device), c_labels.to(device), t_labels.to(device)
                c_preds, t_preds = model(imgs)
                val_c_correct += (c_preds.argmax(1) == c_labels).sum().item()
                val_t_correct += (t_preds.argmax(1) == t_labels).sum().item()
                val_total += imgs.size(0)

        val_c_acc = val_c_correct / max(val_total, 1)
        val_t_acc = val_t_correct / max(val_total, 1)
        avg_acc = (val_c_acc + val_t_acc) / 2.0

        logger.info(
            f"Epoch {epoch:02d} | Train Loss: {total_loss/total:.3f} | "
            f"Color Acc: {val_c_acc:.1%} | Type Acc: {val_t_acc:.1%}"
        )

        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(model.state_dict(), out_file)
            logger.info(f"Saved new best model to {out_file} (Avg Acc: {best_acc:.1%})")

    logger.info(f"Training complete. Best model weights: {out_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/veri776")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", type=str, default="models/vehicle_attributes.pt")
    args = parser.parse_args()

    train_vehicle_attribute_net(args.data_dir, args.epochs, args.batch_size, output_path=args.output)
