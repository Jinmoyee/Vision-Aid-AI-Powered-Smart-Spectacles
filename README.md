# VISION AID: AI-Powered Smart Spectacle for the Blind

## 📌 Project Overview
VISION AID is an advanced AI-powered smart spectacle designed to assist visually impaired individuals. It leverages computer vision and object detection using YOLO (You Only Look Once) models, along with real-time distance measurement using an ultrasonic sensor. The system provides audio feedback via a buzzer and text-to-speech (TTS) to convey information about the user's surroundings, enhancing their ability to navigate safely and independently.

## 🧰 Hardware Components
This project is built using the following hardware components:

- **Raspberry Pi 4 Model B**: The central computing unit to process data and run object detection.
- **USB Camera**: Captures real-time video for object detection.
- **Ultrasonic Sensor (HC-SR04)**: Measures the distance to nearby objects.
- **Piezo Buzzer**: Provides audio alerts when objects are detected within a predefined distance.
- **Headphones (via audio jack)**: Outputs voice feedback for detected objects and distance.

## 🗂️ Software & Libraries
The project utilizes several essential Python libraries:

- **OpenCV**: For capturing and processing images.
- **Ultralytics YOLO**: For real-time object detection.
- **RPi.GPIO**: For controlling the Raspberry Pi's GPIO pins.
- **gTTS (Google Text-to-Speech)**: For voice output (when an internet connection is available).
- **Festival TTS**: For offline voice output (when no internet connection is available).
- **cvzone**: For easy drawing and annotation on image frames.

## 🎯 Key Features
1. **Object Detection**: Identifies objects like people, cars, and more using a custom-trained YOLO model.
2. **Distance Measurement**: Continuously measures the distance between the user and nearby objects using an ultrasonic sensor.
3. **Audio Feedback**: Describes the environment using TTS and provides distance alerts through a buzzer.
4. **Dual TTS System**: Uses Google TTS when the internet is available; falls back to Festival TTS for offline use.
5. **Real-Time Processing**: Captures and processes live video feed, providing immediate assistance.

## 📊 System Workflow
1. The USB camera captures a live video feed.
2. The YOLO model detects objects and identifies their class.
3. Detected objects are counted and announced using TTS.
4. The ultrasonic sensor measures the distance to the closest object.
5. If an object is within a predefined range (e.g., 30 cm), the buzzer alerts the user.

## 🛠️ Setup Instructions
### 1. Hardware Setup
- Connect the ultrasonic sensor to the Raspberry Pi GPIO pins:
  - **TRIG**: GPIO 18
  - **ECHO**: GPIO 15
- Connect the piezo buzzer to **GPIO 16**.
- Attach the USB camera to the Raspberry Pi.
- Ensure the Raspberry Pi is connected to a speaker or headphones via the audio jack.

### 2. Software Installation
1. Update the Raspberry Pi:
    ```bash
    sudo apt update && sudo apt upgrade -y
    ```

2. Install required libraries:
    ```bash
    sudo apt install python3-pip festival mpg321
    pip install opencv-python ultralytics cvzone RPi.GPIO requests gtts
    ```

3. Clone the project repository from GitHub:
    ```bash
    git clone https://github.com/yourusername/VISION-AID.git
    cd VISION-AID
    ```

4. Ensure the YOLO model (e.g., `best.pt`) is placed in the project directory.

### 3. Run the Program
Execute the Python script:

```bash
python3 object_detection.py
```

### 4. User Controls
- **Audio Output**: Provides object names and proximity.
- **Buzzer Alert**: Beeps when an object is closer than 30 cm.
- **Quit Program**: Press `q` on the keyboard.

## 🧪 Customization
- **Adjust Detection Sensitivity**: Modify the `THRESHOLD_DISTANCE_CM` value to change the buzzer's trigger range.
- **Change YOLO Model**: Replace the `best.pt` file with any compatible YOLOv8 model.
- **Modify Object Names**: Customize object categories by adjusting the YOLO model classes.

## 📌 Troubleshooting
1. **Camera Not Detected**: Ensure the USB camera is connected and enabled:
    ```bash
    ls /dev/video0
    ```
2. **No Sound Output**: Verify the speaker or headphones are connected and adjust volume:
    ```bash
    alsamixer
    ```
3. **Permission Issues**: Run the script with superuser privileges if needed:
    ```bash
    sudo python3 object_detection.py
    ```

## 📖 Future Enhancements
- Integrate GPS for navigation assistance.
- Add voice-command capabilities for better user interaction.
- Enhance object detection with multiple model support.

## 🤝 Contribution
Contributions are welcome! Feel free to fork the repository, create a feature branch, and submit a pull request.

## 📜 License
This project is licensed under the MIT License.

---

🚀 **Empowering Vision, Enabling Independence**
