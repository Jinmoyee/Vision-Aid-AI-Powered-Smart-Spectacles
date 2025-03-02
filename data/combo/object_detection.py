import cv2
from ultralytics import YOLO

# Set up the camera with OpenCV
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# Load YOLOv8
model = YOLO("yolov8n.pt")

frame_count = 0
object_counts = {}
output_file = "detection_results.txt"

with open(output_file, "w") as file:
    while frame_count < 1:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_count += 1
        results = model(frame)
        detections = results[0].names
        for box in results[0].boxes:
            obj_name = detections[int(box.cls)]
            object_counts[obj_name] = object_counts.get(obj_name, 0) + 1
        
    # Write total counts of detected objects
    for obj, count in object_counts.items():
        file.write(f"{count} {obj}\n")
        print(f"{count} {obj}")

cap.release()
print(f"Detection results saved to {output_file}")
