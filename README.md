# 🧠 Smart Object & Scene Detection with Distance (Raspberry Pi 4)

This project implements a **vision-based assistive system** on **Raspberry Pi 4** that performs:

- 📷 Object Detection (COCO classes)
- 🌍 Scene Classification (Places365)
- 📏 Distance Measurement using Ultrasonic Sensor (HC-SR04)
- 📝 Natural language description of surroundings

The system captures an image using a **USB webcam**, detects objects and scene context using **PyTorch deep learning models**, measures distance using an ultrasonic sensor, and outputs a **human-readable sentence**.  
Audio output can be played through **connected headphones**.

This project is suitable for:
- Assistive technology for visually impaired users
- Smart surveillance
- Embedded AI systems
- Raspberry Pi AI projects

---

## 🚀 Features

- Faster R-CNN (ResNet50 FPN) for object detection
- ResNet50 (Places365) for scene classification
- Ultrasonic distance measurement (HC-SR04)
- Automatic download of scene labels
- Outputs result to terminal and `detection.txt`
- Works fully offline after setup

---

## 🧰 Hardware Requirements

| Component | Description |
|--------|-------------|
| Raspberry Pi 4 | 4GB RAM recommended |
| USB Webcam | Any UVC-compatible webcam |
| Ultrasonic Sensor | HC-SR04 |
| Headphones / Earphones | Connected via 3.5mm jack or USB |
| Jumper Wires | Male–Female |
| Breadboard | Optional |

---

## 🔌 Ultrasonic Sensor Wiring (HC-SR04)

| HC-SR04 Pin | Raspberry Pi GPIO |
|-----------|-------------------|
| VCC | 5V |
| GND | GND |
| TRIG | GPIO 23 |
| ECHO | GPIO 24 ⚠️ (Use voltage divider) |

⚠️ **IMPORTANT:**  
The Echo pin outputs **5V**. Use a **voltage divider (1kΩ + 2kΩ)** to reduce it to **3.3V** for Raspberry Pi safety.

---

## 🖥️ Software Requirements

- Raspberry Pi OS (64-bit recommended)
- Python 3.8+
- Internet (only for first-time model & label download)

---

## 📦 Required Python Libraries

Install system dependencies first:

```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv libatlas-base-dev espeak ffmpeg
```

## Install System Dependencies
sudo apt install -y \
python3-pip \
python3-opencv \
libatlas-base-dev \
ffmpeg \
espeak


## Upgrade pip
pip3 install --upgrade pip

## 📦 Python Package Installation

pip3 install \
torch \
torchvision \
pillow \
numpy \
opencv-python \
RPi.GPIO

## 📁 Project Structure

```plaintext
project/
├── data/combo/Final_Object_Detection_Scene_Classify.py
├── categories_places365.txt   # auto-downloaded
├── frame.jpg                  # captured image
├── detection.txt              # generated output
└── README.md


