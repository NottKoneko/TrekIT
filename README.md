# TrekIT

**TrekIT** is an end-to-end computer vision and intelligent transportation pipeline for real-time vehicle tracking, color classification, body type identification, and automated number plate recognition (ANPR / ALPR).

---

## Key Features

- **Full-Scene License Plate Detection & ANPR**:
  - Full-scene YOLO plate detection matching native street-scene scale (plates occupying 2–5% of frame).
  - OpenCV morphological ANPR gradient locator fallback.
  - Multi-stage EasyOCR pipeline featuring cubic resolution upscaling, bilateral noise suppression, CLAHE contrast enhancement, Otsu deskew correction, California/US 7-character format disambiguation (`1 Digit - 3 Letters - 3 Digits`), and robust partial/vanity read handling.
- **Vehicle Color Classification (15 Classes)**:
  - MobileNetV3 deep CNN trained on full vehicle crops (`Beige`, `Black`, `Blue`, `Brown`, `Gold`, `Green`, `Grey`, `Orange`, `Pink`, `Purple`, `Red`, `Silver`, `Tan`, `White`, `Yellow`).
  - Mathematical HSV histogram fallback for zero-shot color clustering on central body sheet metal.
- **Vehicle Body Type Classification (7 Classes)**:
  - MobileNetV3 classifier for `Convertible`, `Coupe`, `Hatchback`, `SUV`, `Sedan`, `Truck`, and `Van`.
  - Supports unified multi-task network (`VehicleAttributeNet`) or independent single-task heads.
- **Multi-Object Tracking (ByteTrack)**:
  - Persistent track IDs across frames with a 60-frame lost-track buffer (`bytetrack_custom.yaml`).
  - Continuous Exponential Moving Average (EMA) probability smoothing for robust attribute voting.
- **Interactive Gradio Dashboard**:
  - **Video Analysis**: Real-time batch video processing with H.264 browser playback and snapshot captures.
  - **Image Analysis**: Drag-and-drop vehicle detection and plate recognition.
  - **Live Webcam**: Low-latency live stream tracking with dynamic FPS overlay.
  - **Analytics & Export**: Interactive dataframes, category breakdown charts, and CSV/JSON export.

---

## System Architecture

```
                       [ Input Frame (Video / Image / Stream) ]
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
   [ Vehicle Detector ]                               [ Full-Scene Plate YOLO ]
 (YOLO11 / YOLOv8 + ByteTrack)                        (Native Scale Plate Detection)
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
         ┌─────────┴─────────┐                           ▼
         ▼                   ▼                   [ EasyOCR Pipeline ]
 [ MobileNetV3 Color ] [ MobileNetV3 Type ]    - Bilateral + CLAHE + Deskew
   (15 Colors + HSV)     (7 Body Types)        - Cubic Upscaling
         │                   │                 - CA / US Disambiguation
         └─────────┬─────────┘                           │
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
                      [ EMA Probability Smoothing ]
                                      │
                                      ▼
                      [ Annotated Output + Telemetry ]
```

---

## Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/NottKoneko/TrekIT.git
cd TrekIT

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Web Dashboard
```bash
python app/dashboard.py
```
Open your browser at **http://127.0.0.1:7860**.

---

## Model Training & Dataset Recommendations

To train or fine-tune models, open [`notebooks/TrafficTrackerAI_Training.ipynb`](notebooks/TrafficTrackerAI_Training.ipynb) in Jupyter Lab or Google Colab.

### Model Weight Directory (`models/`)
- `models/plate_detector.pt` (or `plate_detector.onnx`): YOLO license plate detector.
- `models/color_classifier.pt` (or `color_classifier.onnx`): MobileNetV3 vehicle color classifier.
- `models/type_classifier.pt` (or `type_classifier.onnx`): MobileNetV3 vehicle body type classifier.
- `models/vehicle_attributes.pt`: Unified multi-task attribute network.
- `models/color_classes.json` & `models/type_classes.json`: Target label definitions.

### Dataset Perspective & Architecture Notes

1. **License Plate Detection (Full Scene vs Crops)**:
   - Street-level plate detectors (e.g. Kaggle `license-plate-dataset`) are trained on full frames where plates occupy 2–5% of the total pixel area. TrekIT runs plate detection at full-scene scale to preserve receptive fields and anchor scales.
2. **Color Classification (Full Silhouette vs Sheet Metal)**:
   - Deep CNN classifiers (MobileNetV3 on VCOR) learn spatial vehicle contours (windows, grilles, lights, silhouette) alongside paint features. TrekIT feeds full vehicle crops to the CNN, reserving sheet-metal crops for mathematical HSV histogram clustering.
3. **Body Type Classification (Perspective Alignment)**:
   - Eye-level datasets (e.g. Stanford Cars) feature 3/4 front DSLR profiles. For elevated traffic cameras, dashcams, and CCTV feeds, datasets with elevated viewpoints (such as **Comprehensive Cars / CompCars** or **BoxCars116k**) provide optimal generalization.

---

## Project Structure

```
TrekIT/
├── app/
│   └── dashboard.py                  # Gradio web dashboard UI
├── config.yaml                       # Master configuration (paths, thresholds, classes)
├── bytetrack_custom.yaml             # Custom ByteTrack tracker config (60-frame buffer)
├── requirements.txt                  # Python dependencies
├── models/                           # Model weights & class JSON files
├── notebooks/
│   └── TrafficTrackerAI_Training.ipynb # End-to-end training & export notebook
├── data_prep/                        # Dataset ingestion & conversion scripts
│   ├── dataset_utils.py
│   ├── download_datasets.py
│   └── stanford_cars_mapper.py
└── src/
    └── traffic_tracker/              # Core tracking & inference package
        ├── __init__.py
        ├── detector.py               # YOLO vehicle & full-scene plate detector + ByteTrack
        ├── classifier.py             # MobileNetV3 color & body type classifier (15 colors, 7 types)
        ├── ocr_reader.py             # Preprocessing & EasyOCR plate reader with CA format normalization
        ├── pipeline.py               # Multi-stage tracking, EMA smoothing, and snapshot pipeline
        └── utils.py                  # Drawing, bounding box, and logging utilities
```

---

## License & Credits

Built with PyTorch, Ultralytics YOLO, EasyOCR, OpenCV, and Gradio.