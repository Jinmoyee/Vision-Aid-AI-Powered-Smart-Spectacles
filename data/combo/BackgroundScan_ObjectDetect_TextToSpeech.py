import cv2
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image
import requests
import time
import torchvision.models as models

# Load YOLOv8 for object detection
model = YOLO("yolov8n.pt")

# Load Places365 Model for Scene Classification
scene_model = models.resnet50(pretrained=False)
scene_model.fc = torch.nn.Linear(scene_model.fc.in_features, 365)  # 365 scene classes

# Load the checkpoint from URL
checkpoint_url = "http://places2.csail.mit.edu/models_places365/wideresnet18_places365.pth.tar"
checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url, map_location=torch.device('cpu'))

# Ensure compatibility of the state_dict
if "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

scene_model.load_state_dict(state_dict, strict=False)
scene_model.eval()

# Load Places365 Labels
label_url = "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"
labels = requests.get(label_url).text.splitlines()
places_labels = [line.split(" ")[0][3:] for line in labels]

# Define Preprocessing for Scene Classification
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Set up camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

frame_count = 0
scene_label = "Unknown"
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1
    object_counts = {}

    # **Step 1: Object Detection with YOLO**
    results = model(frame)
    detections = results[0].names
    for box in results[0].boxes:
        obj_name = detections[int(box.cls)]
        object_counts[obj_name] = object_counts.get(obj_name, 0) + 1

        # Draw bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, obj_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # **Step 2: Scene Classification (Every 10 Frames for Performance)**
    if frame_count % 10 == 0:  
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img_pil).unsqueeze(0)  # Preprocess image

        with torch.no_grad():
            scene_preds = scene_model(img_tensor)
            scene_label_idx = torch.argmax(scene_preds, dim=1).item()
            scene_label = places_labels[scene_label_idx]  # Get scene name

    # **Step 3: Display Results**
    cv2.putText(frame, f"Scene: {scene_label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    y_offset = 60
    for obj, count in object_counts.items():
        cv2.putText(frame, f"{count} {obj}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 20

    cv2.imshow("YOLO + Scene Recognition", frame)

    # **Exit on 'q' key**
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
