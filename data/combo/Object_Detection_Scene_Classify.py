import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2
import numpy as np
import os
import urllib.request

# --- Step 1: Ensure Scene Labels File Exists ---
scene_labels_file = "categories_places365.txt"
scene_labels_url = "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"

def download_scene_labels():
    try:
        print(f"Downloading scene labels from: {scene_labels_url}")
        urllib.request.urlretrieve(scene_labels_url, scene_labels_file)
        print("Downloaded scene labels file successfully.")
    except Exception as e:
        print(f"Failed to download scene labels: {e}")

if not os.path.exists(scene_labels_file):
    print("Scene labels file not found! Downloading...")
    download_scene_labels()

# Load scene labels
if os.path.exists(scene_labels_file):
    with open(scene_labels_file, "r") as f:
        scene_labels = [line.strip().split(" ")[0].replace("/", " ") for line in f.readlines()]
else:
    print("Scene labels file still missing. Using default numbering.")
    scene_labels = [f"Scene_{i}" for i in range(365)]  # Default fallback

# --- Step 2: Use Correct COCO Class Labels ---
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

# --- Step 3: Capture Frame from Camera ---
image_path = "frame.jpg"
cap = cv2.VideoCapture(0)  # Use webcam
ret, frame = cap.read()

if ret:
    cv2.imwrite(image_path, frame)  # Save captured frame
    print("Frame captured and saved as frame.jpg")
else:
    print("Failed to capture frame, using blank image.")
    frame = np.zeros((224, 224, 3), dtype=np.uint8)  # Black fallback image
    cv2.imwrite(image_path, frame)

cap.release()

# --- Step 4: Load Object Detection and Scene Classification Models ---
object_model = models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
object_model.eval()

scene_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
scene_model.fc = torch.nn.Linear(scene_model.fc.in_features, 365)  # Adjust for 365 scene classes
scene_model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Step 5: Load the Image ---
try:
    image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print("Error: frame.jpg not found!")
    exit()

# --- Step 6: Object Detection with Non-Maximum Suppression (NMS) ---
def detect_objects(image):
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    with torch.no_grad():
        predictions = object_model(image_tensor)

    detected_objects = []
    boxes = []
    scores = []
    labels = []

    for i in range(len(predictions[0]["scores"])):
        score = predictions[0]["scores"][i].item()
        if score > 0.7:  # Confidence threshold
            class_id = predictions[0]["labels"][i].item()
            class_name = COCO_CLASS_NAMES[class_id] if class_id < len(COCO_CLASS_NAMES) else f"object_{class_id}"
            detected_objects.append((class_name, score))
            boxes.append(predictions[0]["boxes"][i])
            scores.append(score)
            labels.append(class_id)

    # Apply NMS to remove redundant detections
    if len(boxes) > 0:
        keep = torch.ops.torchvision.nms(torch.stack(boxes), torch.tensor(scores), 0.3)  # 30% IoU threshold
        detected_objects = [detected_objects[i] for i in keep]

    return [obj[0] for obj in detected_objects]  # Return only object names

# --- Step 7: Scene Classification ---
def classify_scene(image):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = scene_model(image_tensor)
        scene_class = output.argmax(dim=1).item()  # Get top class index

    # Ensure proper formatting
    if 0 <= scene_class < len(scene_labels):
        scene_label = scene_labels[scene_class]
        scene_label = scene_label.split()[-1]  # Take last meaningful word
        scene_label = scene_label.capitalize()  # Format properly
    else:
        scene_label = f"Scene_{scene_class}"  # Fallback

    return scene_label

# --- Step 8: Run Detection & Classification ---
detected_objects = detect_objects(image)
scene = classify_scene(image)

# --- Step 9: Save Results ---
with open("detection.txt", "w") as f:
    f.write(f"Objects: {', '.join(detected_objects)}\n")
    f.write(f"Background: {scene}\n")

print("Detection saved to detection.txt")
