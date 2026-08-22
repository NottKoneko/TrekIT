# TrekIT 🚗💨

**TrekIT** is a production-grade, end-to-end computer vision and intelligent transportation system (ITS) designed for real-time vehicle tracking, multi-task attribute classification (color and body type), full-scene license plate detection, and automated number plate recognition (ALPR / ANPR).

Built with PyTorch, Ultralytics YOLO, OpenCV, EasyOCR, and Gradio, TrekIT runs seamlessly on surveillance CCTV feeds, dashcam footage, still images, and live webcams.

---

## 🌟 Key Features

### 1. 🔍 Full-Scene Vehicle & Plate Detection (YOLO + NMS Deduplication)
- **Scale-Matched Plate Detection**: Runs full-scene plate inference to match native street perspectives (where license plates occupy 2–5% of the frame) instead of passing distorted, tight vehicle crops.
- **NMS & Overlap Suppression**: Custom bounding box deduplication (`_suppress_duplicate_boxes`) merges overlapping sub-part detections to eliminate duplicate boxes on the same car.
- **Area Filtering**: Automatic filtering (`min_box_area`) to reject far-distance artifacts and background noise.
- **Morphological Fallback**: Gradient-based morphological plate localization for vehicles where direct YOLO inference misses the plate.

### 2. 🎨 Multi-Task Surveillance Vehicle Attributes (`VehicleAttributeNet`)
- **Trained on VeRi-776 Surveillance Data**: 49,325 real-world CCTV surveillance angles eliminating perspective domain shift.
- **Single-Pass Joint Inference**: Predicts both **15 Vehicle Colors** and **7 Body Types** in a single GPU forward pass (~2ms).
  - **Colors (15)**: `Beige`, `Black`, `Blue`, `Brown`, `Gold`, `Green`, `Grey`, `Orange`, `Pink`, `Purple`, `Red`, `Silver`, `Tan`, `White`, `Yellow`.
  - **Body Types (7)**: `Convertible`, `Coupe`, `Hatchback`, `SUV` *(includes Crossovers/MPVs)*, `Sedan`, `Truck`, `Van`.
- **Reflection & Shadow Immunity**: Physical HSV chromatic saturation analysis ($S \ge 110$) cross-checks neural predictions to prevent tinted panoramic glass roof reflections (e.g. foliage/sky) from overriding true paint color.
- **Zero-Shot HSV Fallback**: Geometric and mathematical HSV histogram clustering when neural weights are unavailable.

### 3. 🔤 Robust License Plate Recognition (ALPR / ANPR)
- **Dedicated LPRNet Sequence Model**: Sub-millisecond Connectionist Temporal Classification (CTC) sequence decoder for license plates.
- **Multi-Stage EasyOCR Engine**:
  - Cubic resolution upscaling for distant plates.
  - Bilateral noise suppression & CLAHE contrast boost.
  - Otsu-based deskew rotation correction.
  - Top state header (e.g. red script *"California"*) and frame border trimming.
  - Multi-candidate scoring with US / California 7-character format validation (`1 Digit - 3 Letters - 3 Digits`).

### 4. 🛰️ Multi-Object Tracking & Smoothing (ByteTrack)
- **60-Frame Lost-Track Buffer**: Retains vehicle identities through momentary occlusions, traffic lights, and overlapping trajectories.
- **Continuous EMA Probability Smoothing**: Exponential Moving Average probability distributions eliminate single-frame flickering for stable telemetry.

### 5. 💻 Interactive Gradio Web Dashboard
- **Video Analysis**: Real-time video processing with H.264 browser playback and snapshot captures.
- **Image Analysis**: Drag-and-drop single-image analysis with detailed attribute summaries.
- **Live Webcam**: Low-latency live webcam streaming with dynamic FPS counters.
- **Analytics & Export**: Telemetry logs, category breakdown charts, and one-click CSV/JSON exports.
- **Tailscale & LAN Support**: Ready for zero-config remote viewing across private networks.

---

## 🏗️ Architecture

```
                       [ Input Frame (Video / Image / Stream) ]
                                       │
             ┌─────────────────────────┴─────────────────────────┐
             ▼                                                   ▼
    [ Vehicle Detector ]                               [ Full-Scene Plate YOLO ]
  (YOLOv8 / YOLO11 + ByteTrack)                        (Native Scale Plate Detector)
             │                                                   │
             ▼                                                   ▼
    Vehicle Bounding Boxes                            Plate Bounding Boxes
             │                                                   │
             └─────────────────────────┬─────────────────────────┘
                                       ▼
                       [ Vehicle-Plate Association ]
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
          [ Full Vehicle Crop ]                  [ Plate Sub-Crop ]
                    │                                     │
                    ▼                                     ▼
        [ VehicleAttributeNet ]                  [ Multi-Candidate OCR ]
    (Joint 15 Colors + 7 Types)                  - Top Header Noise Trimming
    - CCTV Perspective Aligned                   - Format Pattern Scoring
    - Glass Roof Reflection Check                - LPRNet / EasyOCR Engine
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       ▼
                       [ EMA Probability Smoothing ]
                                       │
                                       ▼
                       [ Annotated Output + Telemetry ]
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/NottKoneko/TrekIT.git
cd TrekIT

# Create a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Web Dashboard

```bash
python app/dashboard.py
```
Open your browser at **http://127.0.0.1:7860** (or via your Tailscale IP on remote devices).

---

## 🛠️ CLI Diagnostics & Batch Processing

Run inference directly from the command line:

```powershell
# Analyze an image and display results
python -m src.traffic_tracker.pipeline --input test_car.jpg

# Save annotated output image
python -m src.traffic_tracker.pipeline --input test_car.jpg --output output.jpg

# Save isolated vehicle, body paint, and plate crops for inspection
python -m src.traffic_tracker.pipeline --input test_car.jpg --save-debug-crops
```
Debug crops are automatically saved to `logs/debug_crops/`.

---

## 🧠 Model Training

All training scripts and end-to-end notebooks are included in the repository:

### 1. Multi-Task `VehicleAttributeNet` (VeRi-776)
Parses the VeRi-776 surveillance dataset and trains the joint Color + Body Type network:
```bash
python data_prep/veri776_parser.py --data_dir data/veri776 --epochs 30 --batch_size 32 --output models/vehicle_attributes.pt
```

### 2. Dedicated `LPRNet` (CTC Loss Sequence Model)
Generates synthetic alphanumeric sequence variations and trains the 1ms ALPR model:
```bash
python data_prep/train_lprnet.py --epochs 40 --batch_size 64 --output models/lprnet.pt
```

### 3. Google Colab / Jupyter Notebook
Open [`notebooks/TrafficTrackerAI_Training.ipynb`](notebooks/TrafficTrackerAI_Training.ipynb) for a fully interactive training walkthrough with automatic Kaggle dataset downloads.

---

## 📁 Repository Structure

```
TrekIT/
├── app/
│   └── dashboard.py                  # Gradio Web Dashboard
├── config.yaml                       # Central Configuration (thresholds, paths, classes)
├── bytetrack_custom.yaml             # Custom ByteTrack config (60-frame buffer)
├── requirements.txt                  # Python dependencies
├── models/                           # Model weights & class JSON definitions
│   ├── color_classes.json
│   ├── type_classes.json
│   ├── plate_detector.pt             # Full-scene YOLO plate detector
│   ├── vehicle_attributes.pt         # VeRi-776 Multi-Task MobileNetV3
│   └── yolov8n_pretrained.pt         # Pretrained vehicle tracking model
├── notebooks/
│   └── TrafficTrackerAI_Training.ipynb # Complete training notebook
├── data_prep/                        # Dataset prep & training scripts
│   ├── dataset_utils.py              # Download & extraction utilities
│   ├── train_lprnet.py               # LPRNet CTC sequence trainer
│   └── veri776_parser.py             # VeRi-776 parser & VehicleAttributeNet trainer
└── src/
    └── traffic_tracker/              # Core tracking & inference package
        ├── __init__.py
        ├── detector.py               # YOLO vehicle/plate detector + NMS deduplication
        ├── classifier.py             # VehicleAttributeNet & HSV reflection cross-check
        ├── ocr_reader.py             # EasyOCR & LPRNet sequence reader with CLAHE/deskew
        ├── pipeline.py               # Multi-stage tracking, EMA smoothing, and CLI
        └── utils.py                  # Overlay rendering, bounding box, and logging tools
```

---

## ⚙️ Configuration Overview (`config.yaml`)

```yaml
detection:
  confidence_threshold: 0.20      # Detection sensitivity threshold
  nms_iou_threshold: 0.45         # NMS IoU threshold (merges duplicate boxes)
  min_box_area: 100               # Rejects distant noise artifacts
  input_size: 640                 # YOLO input resolution
  device: "cuda"                  # "cuda" or "cpu"
  fp16: true                      # Half-precision GPU acceleration

classification:
  confidence_cutoff: 0.35         # Minimum softmax probability
  classify_every_n_frames: 2      # Inference stride for video processing

tracking:
  max_age: 60                     # Frames to keep lost tracks alive
  ema_alpha: 0.25                 # Temporal probability smoothing momentum
```

---

## 📄 License & Acknowledgments

- **Frameworks**: PyTorch, Ultralytics YOLO, EasyOCR, OpenCV, Gradio.
- **Datasets**: VeRi-776, VCOR, Stanford Cars, Kaggle License Plate Dataset.
