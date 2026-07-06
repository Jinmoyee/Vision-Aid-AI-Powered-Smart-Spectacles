# Vision Aid — AI-Powered Smart Spectacles

**An Edge-AI Assistive Wearable for Real-Time Indoor Navigation via Custom YOLOv8 Optimization**

Vision Aid is a low-cost, offline, edge-computing wearable that helps visually impaired users navigate indoor spaces. It runs a custom-trained YOLOv8n model on a Raspberry Pi 4, fuses detections with ultrasonic proximity data, and delivers real-time spoken feedback — with no dependency on cloud APIs.

> 📄 Companion research paper (preprint): *An Edge-AI Assistive Wearable Architecture for Real-Time Indoor Navigation via Custom YOLOv8 Optimization*<br>
> **Authors:** Jinmoyee Thakuria, Ashmita Sarkar<br>
> **DOI:** https://doi.org/10.5281/zenodo.21223013

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Hardware Requirements](#hardware-requirements)
- [Wiring / GPIO Pinout](#wiring--gpio-pinout)
- [Software Setup](#software-setup)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Limitations & Future Work](#limitations--future-work)
- [Citation](#citation)
- [License](#license)

---

## Overview

Most assistive object-detection systems either rely on general-purpose datasets (poorly suited to cluttered indoor scenes) or on cloud inference (which introduces latency, connectivity dependence, and privacy risk). Vision Aid addresses this by:

- Running a **custom-optimized YOLOv8n model** entirely on-device on a Raspberry Pi 4.
- Using a **hybrid 9-class dataset** built from COCO 2017 and SUN RGB-D, curated specifically for indoor obstacles (doors, cabinets, trash bins, chairs, tables, sofas, beds, monitors, people).
- Combining vision detections with **ultrasonic distance sensing** (HC-SR04) to estimate proximity to the nearest object.
- Converting detections into natural spoken sentences via **offline text-to-speech** (`pyttsx3`), so the system works without an internet connection.

## Repository Structure

```
.
├── data.yaml                  # YOLO dataset configuration
├── train.py                   # Training script for the custom YOLOv8n model
│
├── graphs/                    # Training/validation performance plots
│   ├── BoxF1_curve.png
│   ├── BoxPR_curve.png
│   ├── BoxP_curve.png
│   ├── BoxR_curve.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── labels.jpg
│   └── results.png
│
├── Rasp PI/
│   └── piCode.py               # Deployment script that runs on the Raspberry Pi
│
├── result images/              # Sample training/validation batch visualizations
│   ├── train_batch*.jpg
│   └── val_batch*_labels.jpg / val_batch*_pred.jpg
│
└── weights/
    ├── best.pt                 # Best-performing trained model checkpoint
    └── last.pt                 # Final-epoch checkpoint
```

## Hardware Requirements

| Component | Details |
|---|---|
| Compute | Raspberry Pi 4 Model B (2 GB RAM or higher) |
| Camera | USB webcam (e.g., Logitech C920) or Pi Camera Module |
| Proximity Sensor | HC-SR04 ultrasonic distance sensor |
| Audio Output | Speaker or bone-conduction headphones (3.5mm / Bluetooth) |
| Power | 5V/3A portable power bank (for wearable use) |
| Mounting | Spectacles frame / headset enclosure |

## Wiring / GPIO Pinout

The HC-SR04 ultrasonic sensor connects to the Raspberry Pi's GPIO header as follows (BCM numbering, matching `piCode.py`):

| HC-SR04 Pin | Raspberry Pi GPIO (BCM) |
|---|---|
| VCC | 5V |
| TRIG | GPIO 23 |
| ECHO | GPIO 24 (via voltage divider — ECHO outputs 5V, Pi GPIO is 3.3V tolerant) |
| GND | GND |

> ⚠️ **Important:** The HC-SR04's ECHO pin outputs 5V logic. Use a voltage divider (e.g., 1kΩ + 2kΩ resistors) or a logic-level shifter between ECHO and GPIO 24 to avoid damaging the Raspberry Pi's GPIO pin.

The camera connects via USB (or the CSI ribbon cable if using a Pi Camera Module — requires minor code changes to switch from `cv2.VideoCapture(0)` to `picamera2`).

## Software Setup

### 1. Clone the repository

```bash
git clone https://github.com/Jinmoyee/Vision-Aid-AI-Powered-Smart-Spectacles.git
cd Vision-Aid-AI-Powered-Smart-Spectacles
```

### 2. Install dependencies (on the Raspberry Pi)

```bash
sudo apt update
sudo apt install -y python3-pip espeak libespeak1
pip3 install ultralytics opencv-python RPi.GPIO pyttsx3
```

> `pyttsx3` on Linux uses `espeak` as its backend, so `espeak` must be installed at the OS level.

### 3. Enable GPIO access

Ensure your user has permission to access GPIO (usually automatic on Raspberry Pi OS) and that the camera is enabled/detected:

```bash
ls /dev/video*
```

### 4. Place the trained model

Copy `weights/best.pt` into the same directory as `piCode.py` (or update `MODEL_PATH` in the script to point to its location).

## Usage

Run the deployment script directly on the Raspberry Pi:

```bash
cd "Rasp PI"
python3 piCode.py
```

**What it does:**

1. Opens the default camera (`cv2.VideoCapture(0)`) and grabs frames in a loop.
2. Runs each frame through the YOLOv8n model (`best.pt`) at a confidence threshold of `0.5`.
3. Counts detected objects per class (e.g., "2 chairs", "a door").
4. Pings the HC-SR04 ultrasonic sensor to estimate distance to the nearest obstacle.
5. Builds a natural-language sentence (e.g., *"a door and 2 chairs detected, nearest object about 1.2 meters away"*) and speaks it aloud via `pyttsx3`.
6. Repeats on a ~1.5 second loop until interrupted (`Ctrl+C`), then releases the camera and cleans up GPIO.

**Tuning parameters** (inside `piCode.py`):

| Variable | Purpose | Default |
|---|---|---|
| `CONF_THRESHOLD` | Minimum detection confidence | `0.5` |
| `tts_engine rate` | Speech speed | `165` wpm |
| loop `time.sleep()` | Delay between detection cycles | `1.5` s |
| ultrasonic timeout | Max wait for echo pulse | `0.04` s |

### Retraining the model

To retrain or fine-tune on your own data, edit `data.yaml` to point to your dataset paths and run:

```bash
python3 train.py
```

## Dataset

A custom hybrid dataset of **10,650 images / 28,652 annotated instances** was built by combining COCO 2017 (chairs, tables, sofas, beds, monitors, people) with SUN RGB-D (doors, cabinets, trash bins) to target indoor navigation hazards specifically.

| Class | Label | Train | Val | Source |
|---|---|---|---|---|
| 0 | Chair | 6,752 | 671 | COCO 2017 |
| 1 | Table | 3,224 | 325 | COCO 2017 |
| 2 | Door | 1,250 | 333 | SUN RGB-D |
| 3 | Sofa | 1,706 | 165 | COCO 2017 |
| 4 | Cabinet | 3,246 | 723 | SUN RGB-D |
| 5 | Bed | 1,311 | 130 | COCO 2017 |
| 6 | Monitor | 2,000 | 204 | COCO 2017 |
| 7 | Trash Bin | 1,344 | 348 | SUN RGB-D |
| 8 | Person | 3,150 | 350 | COCO 2017 |

See `graphs/labels.jpg` and `result images/` for visualizations of the dataset distribution and sample annotated batches.

### Dataset Sources

This project's custom hybrid dataset was built from two publicly available academic datasets. Due to their licensing terms, the raw source data and derived annotations are not redistributed in this repository. Please obtain them directly from the original providers:

- **COCO 2017** — https://cocodataset.org/#download (licensed CC BY 4.0; credit the COCO Consortium)
- **SUN RGB-D** — https://rgbd.cs.princeton.edu/ (non-commercial research/educational use only; cite Song, Lichtenberg & Xiao, *SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite*, CVPR 2015, plus the underlying NYU Depth v2, Berkeley B3DO, and SUN3D papers as required by the SUN RGB-D terms)

Only class-subset selection, re-annotation, and format conversion (COCO / SUN RGB-D → YOLO format) were performed for this project — no original source imagery is included in this repo.

## Model Performance

Trained for 100 epochs using SGD; validation box loss converged to ~1.01, classification loss to ~1.34.

**Overall mAP50: 0.577** across 9 classes.

| Class | mAP50 | Notes |
|---|---|---|
| Trash Bin | 0.817 | Strong performer — consistent geometry |
| Door | 0.765 | Strong performer — crisp structural outline |
| Monitor | 0.729 | Strong performer |
| Chair | 0.402 | Hurt by design variance & occlusion |
| Person | 0.189 | Domain mismatch: COCO's full-body outdoor framing vs. cropped, close-range indoor wearable perspective |

**Architecture comparison (YOLOv5n baseline vs. proposed YOLOv8n):**

| Metric | YOLOv5n | YOLOv8n | Δ |
|---|---|---|---|
| Precision | 72.0% | 87.0% | +15.0% |
| mAP50 | 0.91 | 0.94 | +0.03 |
| Model size | 27.0 MB | 25.0 MB | −7.4% |
| Parameters | 7.2M | 6.8M | −5.5% |

See `graphs/confusion_matrix_normalized.png`, `graphs/BoxPR_curve.png`, and `graphs/results.png` for full training diagnostics.

## Limitations & Future Work

- **Person detection** underperforms due to a domain gap between outdoor COCO framing and close-range wearable perspectives — planned fix: collect first-person indoor person crops for fine-tuning.
- **Chair detection** suffers under heavy occlusion (chairs tucked under tables) — planned fix: additional occlusion-augmented training samples.
- Current release uses a single ultrasonic sensor for proximity; future versions may add depth estimation to better handle structural occlusion and multi-object distance ranking.
- The current `piCode.py` runs detection, distance sensing, and speech synthesis sequentially in one loop; a multi-threaded pipeline (separating sensor polling from inference and speech) is planned to reduce end-to-end latency.

## Citation

If you use this work, please cite the preprint:

```bibtex
@article{thakuria2026visionaid,
  title   = {An Edge-AI Assistive Wearable Architecture for Real-Time Indoor Navigation via Custom YOLOv8 Optimization},
  author  = {Thakuria, Jinmoyee and Sarkar, Ashmita},
  journal = {Zenodo},
  year    = {2026},
  doi     = {10.5281/zenodo.21223013}
}
```

## License

This project is licensed under the **Apache License 2.0** — see the [`LICENSE`](./LICENSE) file for full terms.

Apache 2.0 permits anyone to use, modify, and distribute this code (including commercially), and includes an explicit patent grant from contributors to users. If you make significant modifications and redistribute them, you must note the changes you made. Note this covers the code in this repository only — it does **not** apply to the COCO 2017 / SUN RGB-D source datasets (see [Dataset Sources](#dataset-sources) above), which carry their own separate licenses.

---

*Vision Aid is developed by Jinmoyee Thakuria and Ashmita Sarkar as part of ongoing research into low-cost, offline assistive technology for underserved communities in Northeast India and similar resource-constrained regions.*
