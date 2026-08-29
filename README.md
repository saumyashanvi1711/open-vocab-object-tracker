# Real-Time Open-Vocabulary Object Tracking & Spatial Zone Analytics

An edge-optimized computer vision pipeline combining **YOLO-World** zero-shot object detection with **ByteTrack** multi-object tracking and real-time **Polygon Zone Analytics**. 

Built for high performance on Apple Silicon (M-series / A-series unified memory architectures) using PyTorch with **Metal Performance Shaders (MPS)** acceleration.

---

## Key Features

- **Open-Vocabulary Zero-Shot Detection:** Detect any arbitrary object using plain-text prompts (e.g., `"black coffee mug"`, `"cell phone"`, `"keys"`) without model retraining.
- **Persistent Multi-Object Tracking:** Integrated ByteTrack data association maintains stable track IDs across occlusion and temporal blinks.
- **Geometric Zone Analytics:** Uses OpenCV point-in-polygon containment math on object bottom-center anchor points to calculate entry events in virtual desktop zones.
- **Hardware Acceleration:** Native PyTorch `device='mps'` integration for ultra-fast frame processing on Apple Silicon GPUs without thermal throttling.

---

## Tech Stack

- **Language:** Python 3.11+
- **Detector:** Ultralytics YOLO-World (`yolov8s-worldv2`)
- **Tracker:** ByteTrack
- **Computer Vision:** OpenCV, Shapely, NumPy
- **Deep Learning Framework:** PyTorch (MPS acceleration)

---

## Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/saumyashanvi1711/open-vocab-object-tracker.git](https://github.com/YOUR-USERNAME/open-vocab-object-tracker.git)
   cd open-vocab-object-tracker

Create and Activate a Virtual Environment:

python3 -m venv venv
source venv/bin/activate

Install Dependencies:

pip install -r requirements.txt

Usage
1. Basic Open-Vocabulary Tracking
Run the core real-time tracking pipeline on your webcam feed:

python3 tracker.py

2. Spatial Polygon Zone Analytics
Run the spatial containment analytics script with custom polygon counting overlays:

python3 zone_analytics.py