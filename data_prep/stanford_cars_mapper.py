"""
stanford_cars_mapper.py
-----------------------
Maps Stanford Cars' 196 fine-grained make/model labels → 7 body type categories.

Stanford Cars labels look like: "BMW 3 Series Sedan 2012"
We extract the body type keyword from each label and reorganise the subset
images into body-type class folders ready for MobileNetV3 training.

Body type categories:
  Sedan | SUV | Truck | Coupe | Hatchback | Van | Convertible

Usage:
    python data_prep/stanford_cars_mapper.py

Expects:
    data/stanford_cars_1gb/class_000/ ... class_195/ folders
    (created by download_datasets.py)

Outputs:
    data/stanford_cars_typed/Sedan/ ... Convertible/  (training-ready)
    data/stanford_cars_typed/label_map.csv
"""

import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Body type keyword mapping ──────────────────────────────────────────────
# Order matters: more specific keywords checked before generic ones
BODY_TYPE_KEYWORDS: Dict[str, list] = {
    "Convertible": ["convertible", "cabriolet", "roadster", "spyder", "spider"],
    "Van":         ["van", "minivan", "cargo van", "transit"],
    "Truck":       ["truck", "pickup", "ram", "silverado", "f-150", "f150", "tacoma", "tundra"],
    "Hatchback":   ["hatchback", "hatch", "5-door", "5 door"],
    "SUV":         ["suv", "4wd", "crossover", "sport utility", "wagon",
                    "explorer", "tahoe", "suburban", "cr-v", "rav4", "highlander"],
    "Coupe":       ["coupe", "2-door", "2 door", "fastback"],
    "Sedan":       ["sedan", "saloon", "4-door", "4 door"],
}

OTHER_LABEL = "Sedan"  # Default for unmatched labels (most common car body type)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_DIR = BASE_DIR / "stanford_cars_1gb"
OUTPUT_DIR = BASE_DIR / "stanford_cars_typed"

# Stanford Cars 196 class names (index 0–195 corresponds to class_000–class_195)
# Source: Stanford Cars dataset official class list
STANFORD_196_CLASSES = [
    "AM General Hummer SUV 2000", "Acura Integra Type R 2001", "Acura RL Sedan 2012",
    "Acura TL Sedan 2012", "Acura TL Type-S 2008", "Acura TSX Sedan 2012",
    "Acura ZDX Hatchback 2012", "Aston Martin V8 Vantage Convertible 2012",
    "Aston Martin V8 Vantage Coupe 2012", "Aston Martin Virage Convertible 2012",
    "Aston Martin Virage Coupe 2012", "Audi RS 4 Convertible 2008",
    "Audi A5 Coupe 2012", "Audi TTS Coupe 2012", "Audi R8 Coupe 2012",
    "Audi V8 Sedan 1994", "Audi 100 Sedan 1994", "Audi 100 Wagon 1994",
    "Audi TT Hatchback 2011", "Audi S6 Sedan 2011", "Audi S5 Convertible 2012",
    "Audi S5 Coupe 2012", "Audi S4 Sedan 2012", "Audi S4 Sedan 2007",
    "Audi TT RS Coupe 2012", "BMW ActiveHybrid 5 Sedan 2012", "BMW 1 Series Convertible 2012",
    "BMW 1 Series Coupe 2012", "BMW 3 Series Sedan 2012", "BMW 3 Series Wagon 2012",
    "BMW 6 Series Convertible 2007", "BMW X5 SUV 2007", "BMW X6 SUV 2012",
    "BMW M3 Coupe 2012", "BMW M5 Sedan 2010", "BMW M6 Convertible 2010",
    "BMW X3 SUV 2012", "BMW Z4 Convertible 2012", "Bentley Continental GT Coupe 2012",
    "Bentley Continental GT Coupe 2007", "Bentley Continental Flying Spur Sedan 2007",
    "Bentley Mulsanne Sedan 2011", "Bentley Continental Supersports Conv. Convertible 2012",
    "Bugatti Veyron 16.4 Convertible 2009", "Bugatti Veyron 16.4 Coupe 2009",
    "Buick Enclave SUV 2012", "Buick Rainier SUV 2007", "Buick Regal GS 2012",
    "Buick Verano Sedan 2012", "Cadillac CTS-V Coupe 2012", "Cadillac Escalade EXT Crew Cab 2007",
    "Cadillac SRX SUV 2012", "Chevrolet Silverado 1500 Classic Extended Cab 2007",
    "Chevrolet Silverado 1500 Extended Cab 2012", "Chevrolet Silverado 1500 Hybrid Crew Cab 2012",
    "Chevrolet Silverado 1500 Regular Cab 2012", "Chevrolet Silverado 2500HD Regular Cab 2012",
    "Chevrolet Tahoe Hybrid SUV 2012", "Chevrolet TrailBlazer SS 2009",
    "Chevrolet Traverse SUV 2012", "Chevrolet Colorado Crew Cab 2012",
    "Chevrolet Camaro Convertible 2012", "Chevrolet Cobalt SS 2010",
    "Chevrolet Corvette Convertible 2012", "Chevrolet Corvette Ron Fellows Edition Z06 2007",
    "Chevrolet Corvette ZR1 2012", "Chevrolet Express Cargo Van 2007",
    "Chevrolet Express Van 2007", "Chevrolet HHR SS 2010",
    "Chevrolet Impala Sedan 2007", "Chevrolet Malibu Hybrid Sedan 2010",
    "Chevrolet Malibu Sedan 2007", "Chevrolet Monte Carlo Coupe 2007",
    "Chevrolet Avalanche Crew Cab 2012", "Chrysler 300 SRT-8 2010",
    "Chrysler Aspen SUV 2009", "Chrysler Crossfire Convertible 2008",
    "Chrysler PT Cruiser Convertible 2008", "Chrysler Town and Country Minivan 2012",
    "Chrysler 300 Sedan 2012", "Daewoo Nubira Wagon 2002",
    "Dodge Caliber Wagon 2012", "Dodge Caliber Wagon 2007",
    "Dodge Caravan Minivan 1997", "Dodge Challenger SRT8 2011",
    "Dodge Charger SRT-8 2009", "Dodge Charger Sedan 2012",
    "Dodge Dakota Club Cab 2007", "Dodge Dakota Crew Cab 2010",
    "Dodge Durango SUV 2012", "Dodge Durango SUV 2007",
    "Dodge Journey SUV 2012", "Dodge Magnum Wagon 2008",
    "Dodge Ram Pickup 3500 Crew Cab 2010", "Dodge Ram Pickup 3500 Quad Cab 2009",
    "Dodge Sprinter Cargo Van 2009", "Dodge Viper Convertible 2010",
    "Dodge Viper SRT-10 Coupe 2010", "Eagle Talon Hatchback 1998",
    "FIAT 500 Abarth 2012", "FIAT 500 Convertible 2012",
    "Ferrari 458 Italia Convertible 2012", "Ferrari 458 Italia Coupe 2012",
    "Ferrari California Convertible 2012", "Ferrari FF Coupe 2012",
    "Fisker Karma Sedan 2012", "Ford F-150 Regular Cab 2012",
    "Ford F-450 Super Duty Crew Cab 2012", "Ford Fiesta Sedan 2012",
    "Ford Focus Sedan 2007", "Ford Freestar Minivan 2007",
    "Ford GT Coupe 2006", "Ford Galaxy Minivan 2007",
    "Ford Mustang Convertible 2007", "Ford Mustang Convertible 1993",
    "Ford Ranger SuperCab 2011", "Ford F-150 Regular Cab 2007",
    "GMC Acadia SUV 2012", "GMC Canyon Extended Cab 2012",
    "GMC Savana Van 2012", "GMC Sierra 1500 Classic Extended Cab 2007",
    "GMC Sierra 1500 Extended Cab 2012", "GMC Sierra 1500 Hybrid Crew Cab 2012",
    "GMC Sierra 1500 Regular Cab 2012", "GMC Sierra 2500HD Regular Cab 2012",
    "GMC Terrain SUV 2012", "GMC Yukon Hybrid SUV 2012",
    "Geo Metro Convertible 1993", "HUMMER H2 SUT Crew Cab 2009",
    "HUMMER H3T Crew Cab 2010", "Honda Accord Coupe 2012",
    "Honda Accord Sedan 2012", "Honda Odyssey Minivan 2012",
    "Honda Odyssey Minivan 2007", "Hyundai Azera Sedan 2012",
    "Hyundai Elantra Sedan 2007", "Hyundai Elantra Touring Hatchback 2012",
    "Hyundai Genesis Sedan 2012", "Hyundai Santa Fe SUV 2012",
    "Hyundai Sonata Hybrid Sedan 2012", "Hyundai Sonata Sedan 2012",
    "Hyundai Tucson SUV 2012", "Hyundai Veloster Hatchback 2012",
    "Hyundai Veracruz SUV 2012", "Infiniti G Coupe IPL 2012",
    "Infiniti QX56 SUV 2011", "Isuzu Ascender SUV 2008",
    "Jaguar XK XKR 2012", "Jeep Compass SUV 2012",
    "Jeep Grand Cherokee SUV 2012", "Jeep Liberty SUV 2012",
    "Jeep Patriot SUV 2012", "Jeep Wrangler SUV 2012",
    "Lamborghini Aventador Coupe 2012", "Lamborghini Diablo Coupe 2001",
    "Lamborghini Gallardo LP 570-4 Superleggera 2012",
    "Lamborghini Reventon Coupe 2008", "Land Rover LR2 SUV 2012",
    "Land Rover Range Rover SUV 2012", "Lincoln Town Car Sedan 2011",
    "MINI Cooper Roadster Convertible 2012", "Maybach Landaulet Convertible 2012",
    "Mazda Tribute SUV 2011", "McLaren MP4-12C Coupe 2012",
    "Mercedes-Benz 300-Class Convertible 1993", "Mercedes-Benz C-Class Sedan 2012",
    "Mercedes-Benz E-Class Sedan 2012", "Mercedes-Benz S-Class Sedan 2012",
    "Mercedes-Benz SL-Class Coupe 2009", "Mercedes-Benz Sprinter Van 2012",
    "Mitsubishi Lancer Sedan 2012", "Nissan 240SX Coupe 1998",
    "Nissan Juke Hatchback 2012", "Nissan Leaf Hatchback 2012",
    "Nissan NV Passenger Van 2012", "Plymouth Neon Coupe 1999",
    "Porsche Panamera Sedan 2012", "Ram C/V Cargo Van Minivan 2012",
    "Rolls-Royce Ghost Sedan 2012", "Rolls-Royce Phantom Drophead Coupe Convertible 2012",
    "Rolls-Royce Phantom Sedan 2012", "Scion xD Hatchback 2012",
    "Spyker C8 Convertible 2009", "Spyker C8 Coupe 2009",
    "Suzuki Aerio Sedan 2007", "Suzuki Kizashi Sedan 2012",
    "Suzuki SX4 Hatchback 2012", "Suzuki SX4 Sedan 2012",
    "Tesla Model S Sedan 2012", "Toyota 4Runner SUV 2012",
    "Toyota Camry Sedan 2012", "Toyota Corolla Sedan 2012",
    "Toyota Sequoia SUV 2012", "Volkswagen Beetle Hatchback 2012",
    "Volkswagen Golf Hatchback 2012", "Volkswagen Golf Hatchback 1991",
    "Volvo 240 Sedan 1993", "Volvo C30 Hatchback 2012",
    "smart fortwo Convertible 2012",
]


def label_to_body_type(label: str) -> str:
    """
    Keyword-match a Stanford Cars class label to one of 7 body types.
    Returns the matched body type or OTHER_LABEL as default.
    """
    lower = label.lower()
    for body_type, keywords in BODY_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return body_type
    return OTHER_LABEL


def build_label_map() -> Dict[int, str]:
    """Build int class_id → body_type dict from the 196 class names."""
    label_map: Dict[int, str] = {}
    for i, class_name in enumerate(STANFORD_196_CLASSES):
        body_type = label_to_body_type(class_name)
        label_map[i] = body_type
    return label_map


def reorganise_images(label_map: Dict[int, str]) -> None:
    """
    Copy images from class_NNN folders → body-type named folders.
    Original images are kept intact.
    """
    if not INPUT_DIR.exists():
        logger.error(
            f"Input directory {INPUT_DIR} not found. "
            "Run download_datasets.py first."
        )
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create body-type subdirectories
    all_types = set(BODY_TYPE_KEYWORDS.keys()) | {OTHER_LABEL}
    for bt in all_types:
        (OUTPUT_DIR / bt).mkdir(exist_ok=True)

    copied = 0
    skipped = 0
    body_type_counts: Dict[str, int] = {bt: 0 for bt in all_types}

    for class_dir in sorted(INPUT_DIR.glob("class_*")):
        if not class_dir.is_dir():
            continue
        class_id_str = class_dir.name.replace("class_", "")
        try:
            class_id = int(class_id_str)
        except ValueError:
            continue

        body_type = label_map.get(class_id, OTHER_LABEL)
        dest_dir = OUTPUT_DIR / body_type

        for img_path in class_dir.glob("*.jpg"):
            dest_path = dest_dir / f"c{class_id:03d}_{img_path.name}"
            shutil.copy2(img_path, dest_path)
            copied += 1
            body_type_counts[body_type] += 1

    logger.info(f"Reorganised {copied} images into {OUTPUT_DIR}")
    logger.info("Distribution by body type:")
    for bt, count in sorted(body_type_counts.items()):
        logger.info(f"  {bt:<15}: {count}")


def export_label_csv(label_map: Dict[int, str]) -> None:
    """Save class_id → original_label → body_type mapping as CSV."""
    csv_path = OUTPUT_DIR / "label_map.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["class_id", "original_label", "body_type"]
        )
        writer.writeheader()
        for class_id, body_type in label_map.items():
            orig_label = (
                STANFORD_196_CLASSES[class_id]
                if class_id < len(STANFORD_196_CLASSES)
                else "Unknown"
            )
            writer.writerow({
                "class_id": class_id,
                "original_label": orig_label,
                "body_type": body_type,
            })
    logger.info(f"Label map saved to {csv_path}")


def main():
    logger.info("Building Stanford Cars 196 → 7 body-type label map…")
    label_map = build_label_map()

    # Print summary
    from collections import Counter
    counts = Counter(label_map.values())
    logger.info("Label mapping summary (Stanford Cars 196 classes → body types):")
    for bt, count in counts.most_common():
        logger.info(f"  {bt:<15}: {count} original classes")

    export_label_csv(label_map)
    reorganise_images(label_map)
    logger.info("Done. Training-ready images are in: data/stanford_cars_typed/")


if __name__ == "__main__":
    main()
