import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2
import numpy as np
import os
import urllib.request
import time
import RPi.GPIO as GPIO
from collections import Counter

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

    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance_cm = pulse_duration * 17150
    distance_m = round(distance_cm / 100, 1)

    return distance_m

# ================== SCENE LABELS ==================
scene_labels_file = "categories_places365.txt"
scene_labels_url = "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"

def download_scene_labels():
    urllib.request.urlretrieve(scene_labels_url, scene_labels_file)

if not os.path.exists(scene_labels_file):
    download_scene_labels()

with open(scene_labels_file, "r") as f:
    scene_labels = [line.strip().split(" ")[0].replace("/", " ") for line in f.readlines()]

# ================== COCO CLASSES ==================
COCO_CLASS_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe",
    "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "mirror", "dining table",
    "window", "desk", "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# ================== CAPTURE IMAGE ==================
image_path = "frame.jpg"
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

if ret:
    cv2.imwrite(image_path, frame)
else:
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.imwrite(image_path, frame)

cap.release()

# ================== LOAD MODELS ==================
object_model = models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
object_model.eval()

scene_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
scene_model.fc = torch.nn.Linear(scene_model.fc.in_features, 365)
scene_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

image = Image.open(image_path).convert("RGB")

# ================== OBJECT DETECTION ==================
def detect_objects(image):
    img_tensor = transforms.ToTensor()(image).unsqueeze(0)

    with torch.no_grad():
        preds = object_model(img_tensor)

    boxes, scores, labels = [], [], []
    detected = []

    for i, score in enumerate(preds[0]["scores"]):
        if score.item() > 0.7:
            cid = preds[0]["labels"][i].item()
            name = COCO_CLASS_NAMES[cid]
            detected.append(name)
            boxes.append(preds[0]["boxes"][i])
            scores.append(score.item())

    if boxes:
        keep = torch.ops.torchvision.nms(
            torch.stack(boxes),
            torch.tensor(scores),
            0.3
        )
        detected = [detected[i] for i in keep]

    return detected

# ================== SCENE CLASSIFICATION ==================
def classify_scene(image):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        out = scene_model(img_tensor)
    idx = out.argmax(dim=1).item()
    scene = scene_labels[idx].split()[-1].capitalize()
    return scene

# ================== FINAL SENTENCE ==================
def format_sentence(counts, scene, distance):
    parts = []
    for obj, cnt in counts.items():
        if cnt == 1:
            parts.append(f"a {obj}")
        else:
            parts.append(f"{cnt} {obj}")

    objects_text = " and ".join(parts)
    return f"{objects_text} detected near the {scene.lower()} with distance of {distance}m"

# ================== RUN ==================
objects = detect_objects(image)
scene = classify_scene(image)
counts = Counter(objects)

try:
    distance = get_distance()
except:
    distance = "unknown"

final_text = format_sentence(counts, scene, distance)

with open("detection.txt", "w") as f:
    f.write(final_text + "\n")

print(final_text)

GPIO.cleanup()
