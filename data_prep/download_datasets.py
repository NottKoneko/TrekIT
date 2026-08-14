"""
download_datasets.py
--------------------
Downloads all three datasets required for Traffic Tracker AI training:

  1. fareselmenshawii/license-plate-dataset  (via kagglehub)
  2. landrykezebou/vcor-vehicle-color-recognition-dataset (via kagglehub)
  3. tanganke/stanford_cars  (via HuggingFace streaming — saves ~1GB subset)

Usage:
    python data_prep/download_datasets.py

Requires:
    pip install kagglehub datasets huggingface-hub Pillow tqdm

Kaggle authentication:
    Place your kaggle.json in ~/.kaggle/  OR  set environment variables:
      KAGGLE_USERNAME and KAGGLE_KEY
"""

import os
import sys
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent / "data"
STANFORD_OUTPUT_DIR = BASE_DIR / "stanford_cars_1gb"
STANFORD_MAX_GB = 1.0         # Download cap in gigabytes
STANFORD_MAX_BYTES = int(STANFORD_MAX_GB * 1024 ** 3)
STANFORD_PER_CLASS_CAP = 15   # Max images per class to keep dataset balanced


# ── 1. Kaggle datasets ──────────────────────────────────────────────────────

def download_license_plate_dataset() -> str:
    """Download the Kaggle license plate detection dataset."""
    try:
        import kagglehub
    except ImportError:
        logger.error("kagglehub not installed. Run: pip install kagglehub")
        sys.exit(1)

    logger.info("Downloading license plate dataset from Kaggle…")
    path = kagglehub.dataset_download("fareselmenshawii/license-plate-dataset")
    logger.info(f"License plate dataset saved to: {path}")
    return path


def download_vcor_color_dataset() -> str:
    """Download the VCOR vehicle colour recognition dataset."""
    try:
        import kagglehub
    except ImportError:
        logger.error("kagglehub not installed. Run: pip install kagglehub")
        sys.exit(1)

    logger.info("Downloading VCOR vehicle colour dataset from Kaggle…")
    path = kagglehub.dataset_download(
        "landrykezebou/vcor-vehicle-color-recognition-dataset"
    )
    logger.info(f"VCOR dataset saved to: {path}")
    return path


# ── 2. Stanford Cars 1GB subset ─────────────────────────────────────────────

def download_stanford_cars_subset() -> str:
    """
    Stream tanganke/stanford_cars from HuggingFace and save a balanced
    class-capped subset that fits within ~1GB of disk space.

    Returns the output directory path.
    """
    try:
        from datasets import load_dataset
        from PIL import Image as PILImage
        from tqdm import tqdm
    except ImportError:
        logger.error(
            "Missing packages. Run: pip install datasets huggingface-hub Pillow tqdm"
        )
        sys.exit(1)

    STANFORD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Streaming Stanford Cars dataset → {STANFORD_OUTPUT_DIR} "
        f"(cap: {STANFORD_MAX_GB} GB, max {STANFORD_PER_CLASS_CAP} imgs/class)"
    )

    # Load as streaming dataset to avoid downloading the full 6GB
    ds = load_dataset(
        "tanganke/stanford_cars",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    class_counts: dict = {}
    total_bytes: int = 0
    saved_count: int = 0

    pbar = tqdm(desc="Saving Stanford Cars subset", unit="img")
    for sample in ds:
        label: int = sample["label"]
        img: PILImage.Image = sample["image"]

        # Per-class cap for balanced dataset
        class_counts[label] = class_counts.get(label, 0)
        if class_counts[label] >= STANFORD_PER_CLASS_CAP:
            continue

        # Save image
        class_dir = STANFORD_OUTPUT_DIR / f"class_{label:03d}"
        class_dir.mkdir(exist_ok=True)
        img_path = class_dir / f"{saved_count:06d}.jpg"

        # Convert to RGB (some HF images are RGBA or palette mode)
        img_rgb = img.convert("RGB")
        img_rgb.save(img_path, "JPEG", quality=90)

        file_size = img_path.stat().st_size
        total_bytes += file_size
        class_counts[label] += 1
        saved_count += 1
        pbar.update(1)
        pbar.set_postfix({
            "GB": f"{total_bytes / 1024**3:.3f}",
            "imgs": saved_count,
        })

        # Stop when we hit the size cap
        if total_bytes >= STANFORD_MAX_BYTES:
            logger.info(
                f"Reached {STANFORD_MAX_GB}GB cap — stopping stream. "
                f"Saved {saved_count} images across {len(class_counts)} classes."
            )
            break

    pbar.close()

    # Save metadata JSON alongside images
    meta = {
        "total_images": saved_count,
        "total_bytes": total_bytes,
        "total_gb": round(total_bytes / 1024 ** 3, 3),
        "classes_seen": len(class_counts),
        "per_class_cap": STANFORD_PER_CLASS_CAP,
    }
    with open(STANFORD_OUTPUT_DIR / "subset_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        f"Stanford Cars subset saved: {saved_count} images, "
        f"{meta['total_gb']} GB at {STANFORD_OUTPUT_DIR}"
    )
    return str(STANFORD_OUTPUT_DIR)


# ── 3. MIO-TCD Real CCTV Traffic Camera Dataset (HuggingFace / Kaggle) ─────
MIOTCD_OUTPUT_DIR = BASE_DIR / "miotcd_cctv_subset"

def download_miotcd_cctv_subset() -> str:
    """
    Download a subset of real CCTV traffic camera images from MIO-TCD for
    robust high-angle vehicle body type classification.
    """
    try:
        from datasets import load_dataset
        from PIL import Image as PILImage
        from tqdm import tqdm
    except ImportError:
        logger.warning("datasets not installed. Skipping MIO-TCD streaming.")
        return ""

    MIOTCD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checking MIO-TCD CCTV Traffic dataset → {MIOTCD_OUTPUT_DIR}...")
    try:
        ds = load_dataset("jonassj/MIO-TCD-classification", split="train", streaming=True)
        counts = {}
        max_per_class = 200
        saved = 0
        pbar = tqdm(desc="Saving MIO-TCD CCTV subset", unit="img")
        for sample in ds:
            lbl = str(sample.get("label", sample.get("class", "unknown")))
            img = sample.get("image")
            if img is None:
                continue
            counts[lbl] = counts.get(lbl, 0)
            if counts[lbl] >= max_per_class:
                continue
            d = MIOTCD_OUTPUT_DIR / lbl
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"{saved:06d}.jpg"
            img.convert("RGB").save(p, "JPEG", quality=90)
            counts[lbl] += 1
            saved += 1
            pbar.update(1)
            if saved >= max_per_class * 7:
                break
        pbar.close()
        logger.info(f"MIO-TCD CCTV subset ready: {saved} real-world surveillance crops.")
        return str(MIOTCD_OUTPUT_DIR)
    except Exception as e:
        logger.warning(f"Could not stream MIO-TCD (fallback to Stanford Cars): {e}")
        return ""


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("Traffic Tracker AI — Dataset Downloader")
    logger.info("=" * 60)

    results = {}

    try:
        results["license_plate"] = download_license_plate_dataset()
    except Exception as e:
        logger.error(f"License plate dataset failed: {e}")

    try:
        results["vcor_color"] = download_vcor_color_dataset()
    except Exception as e:
        logger.error(f"VCOR colour dataset failed: {e}")

    try:
        results["stanford_cars_subset"] = download_stanford_cars_subset()
    except Exception as e:
        logger.error(f"Stanford Cars subset failed: {e}")

    try:
        cctv_path = download_miotcd_cctv_subset()
        if cctv_path:
            results["miotcd_cctv_subset"] = cctv_path
    except Exception as e:
        logger.warning(f"MIO-TCD CCTV dataset download skipped: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("Download Summary:")
    for name, path in results.items():
        logger.info(f"  {name}: {path}")
    logger.info("=" * 60)
    logger.info(
        "Next step: run python data_prep/stanford_cars_mapper.py "
        "to map 196 labels → 7 body types."
    )


if __name__ == "__main__":
    main()
