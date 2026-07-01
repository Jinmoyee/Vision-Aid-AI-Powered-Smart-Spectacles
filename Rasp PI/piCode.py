import cv2
import time
from collections import Counter

import RPi.GPIO as GPIO
import pyttsx3
from ultralytics import YOLO

# ================== ULTRASONIC SETUP ==================
TRIG = 23
ECHO = 24

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)


def get_distance():
    GPIO.output(TRIG, False)
    time.sleep(0.1)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    timeout = time.time() + 0.04  # 40ms timeout to avoid infinite hang

    pulse_start = time.time()
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
        if pulse_start > timeout:
            return None

    pulse_end = time.time()
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()
        if pulse_end > timeout:
            return None

    pulse_duration = pulse_end - pulse_start
    distance_cm = pulse_duration * 17150
    distance_m = round(distance_cm / 100, 1)

    return distance_m


# ================== TEXT TO SPEECH SETUP ==================
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 165)   # speaking speed, tune to taste
tts_engine.setProperty("volume", 1.0)


def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()


# ================== best.pt SETUP ==================

MODEL_PATH = "best.pt"
CONF_THRESHOLD = 0.5

model = YOLO(MODEL_PATH)


def detect_objects(frame):
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
    detected = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            detected.append(name)

    return detected


# ================== SENTENCE FORMATTING ==================
def format_sentence(counts, distance):
    if not counts:
        return "No objects detected"

    parts = []
    for obj, cnt in counts.items():
        parts.append(f"a {obj}" if cnt == 1 else f"{cnt} {obj}s")

    objects_text = " and ".join(parts)

    if distance is not None:
        return f"{objects_text} detected, nearest object about {distance} meters away"
    return f"{objects_text} detected, distance unknown"


# ================== MAIN LOOP ==================
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        speak("Camera not detected")
        GPIO.cleanup()
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.5)
                continue

            objects = detect_objects(frame)
            counts = Counter(objects)

            try:
                distance = get_distance()
            except Exception:
                distance = None

            final_text = format_sentence(counts, distance)
            print(final_text)
            speak(final_text)

            time.sleep(1.5)  # adjust loop rate as needed

    except KeyboardInterrupt:
        print("Stopping Vision Aid...")

    finally:
        cap.release()
        GPIO.cleanup()


if __name__ == "__main__":
    main()