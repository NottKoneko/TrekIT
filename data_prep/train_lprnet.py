"""
train_lprnet.py
----------------
Trains the Lightweight License Plate Recognition Network (LPRNet)
using PyTorch CTCLoss on cropped plate characters and synthetic generators.

Usage:
  python data_prep/train_lprnet.py --epochs 40 --batch_size 64
"""

import os
import sys
import random
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

LPR_CHARS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "-"
]
CHAR_DICT = {c: i for i, c in enumerate(LPR_CHARS)}


class SmallBasicBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
        self.shortcut = nn.Conv2d(ch_in, ch_out, kernel_size=1) if ch_in != ch_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.block(x)


class LPRNet(nn.Module):
    def __init__(self, class_num: int = len(LPR_CHARS), dropout_rate: float = 0.5):
        super().__init__()
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            SmallBasicBlock(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            SmallBasicBlock(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SmallBasicBlock(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(256, class_num, kernel_size=1, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)  # (N, class_num, 1, seq_len)
        return logits.squeeze(2)   # (N, class_num, seq_len)


class SyntheticPlateDataset(Dataset):
    """
    Generates synthetic US/California license plates dynamically for robust CTC sequence training.
    """
    def __init__(self, size: int = 10000):
        self.size = size
        self.chars = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate random 7-character CA plate (e.g. 8XYZ123) or 6-character plate
        if random.random() < 0.75:
            plate_str = f"{random.choice('123456789')}{''.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ', k=3))}{''.join(random.choices('0123456789', k=3))}"
        else:
            plate_str = "".join(random.choices(self.chars, k=random.randint(5, 7)))

        # Draw plate on image
        img = np.ones((24, 94, 3), dtype=np.uint8) * random.randint(200, 245)
        # Add slight noise/vignette
        noise = np.random.randint(-15, 15, (24, 94, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Draw characters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        x_offset = 6
        for ch in plate_str:
            cv2.putText(img, ch, (x_offset, 18), font, font_scale, (random.randint(10, 40), random.randint(10, 40), random.randint(10, 40)), thickness, cv2.LINE_AA)
            x_offset += 12 + random.randint(-1, 1)

        # Normalize to [-0.5, 0.5]
        img_f = img.astype(np.float32)
        img_f = (img_f - 127.5) * 0.0078125
        img_t = torch.from_numpy(np.transpose(img_f, (2, 0, 1)))

        target = [CHAR_DICT[c] for c in plate_str if c in CHAR_DICT]
        return img_t, torch.tensor(target, dtype=torch.long), len(target)


def collate_fn(batch):
    imgs, targets, lengths = zip(*batch)
    imgs = torch.stack(imgs, 0)
    flat_targets = torch.cat(targets)
    target_lengths = torch.tensor(lengths, dtype=torch.long)
    return imgs, flat_targets, target_lengths


def train_lprnet(
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    output_path: str = "models/lprnet.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = SyntheticPlateDataset(size=20000)
    val_ds = SyntheticPlateDataset(size=2000)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = LPRNet(class_num=len(LPR_CHARS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    blank_idx = len(LPR_CHARS) - 1
    ctc_loss = nn.CTCLoss(blank=blank_idx, reduction="mean", zero_infinity=True)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    logger.info(f"Starting LPRNet CTC training for {epochs} epochs on {device}...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_samples = 0.0, 0
        for imgs, targets, target_lengths in tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}"):
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()

            logits = model(imgs)  # (N, class_num, seq_len)
            log_probs = logits.permute(2, 0, 1).log_softmax(2)  # (seq_len, N, class_num)
            input_lengths = torch.full((imgs.size(0),), logits.size(2), dtype=torch.long, device=device)

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_samples += imgs.size(0)

        scheduler.step()
        avg_train_loss = total_loss / max(total_samples, 1)

        # Validation
        model.eval()
        val_loss, val_samples = 0.0, 0
        with torch.no_grad():
            for imgs, targets, target_lengths in val_dl:
                imgs, targets = imgs.to(device), targets.to(device)
                logits = model(imgs)
                log_probs = logits.permute(2, 0, 1).log_softmax(2)
                input_lengths = torch.full((imgs.size(0),), logits.size(2), dtype=torch.long, device=device)
                loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
                val_loss += loss.item() * imgs.size(0)
                val_samples += imgs.size(0)

        avg_val_loss = val_loss / max(val_samples, 1)
        logger.info(f"Epoch {epoch:02d} | Train CTCLoss: {avg_train_loss:.4f} | Val CTCLoss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), out_file)
            logger.info(f"Saved best LPRNet weights to {out_file}")

    logger.info(f"LPRNet Training Complete. Output: {out_file}")


if __name__ == "__main__":
    train_lprnet()
