"""
YOLOv8n Training Script — Combined Dataset (SUN RGB-D + COCO Indoor)
=====================================================================
Model  : YOLOv8n (nano)
Epochs : 100 (with early stopping, patience=30)
Classes: 9 (chair, table, door, sofa, cabinet, bed, monitor, trash_bin, person)

Early Stopping:
- patience=30 → stops if mAP doesn't improve for 30 consecutive epochs
- Automatically saves the best model (best.pt) regardless of stopping point

Usage:
    pip install ultralytics
    python train.py
"""

from ultralytics import YOLO

# ─── CONFIG ────────────────────────────────────────────────────────────────────

DATA_YAML = r"data.yaml path eg: ../data.yaml"
MODEL     = "yolov8n.pt"

EPOCHS    = 100
PATIENCE  = 30          # Stop if no mAP improvement for 30 epochs

IMG_SIZE  = 640
BATCH     = 16          
WORKERS   = 4
PROJECT   = r"project folder eg: ../runs"
RUN_NAME  = "combined_yolov8n"

# ─── TRAIN ─────────────────────────────────────────────────────────────────────

def main():
    model = YOLO(MODEL)

    results = model.train(
        data       = DATA_YAML,
        epochs     = EPOCHS,
        imgsz      = IMG_SIZE,
        batch      = BATCH,
        workers    = WORKERS,
        project    = PROJECT,
        name       = RUN_NAME,
        device     = 0,           
        patience   = PATIENCE,
        save       = True,
        plots      = True,        
        verbose    = True,
        optimizer  = "auto",      
        cos_lr     = True,      
        close_mosaic = 10,        
        amp        = True,       
    )

    print("\n─── TRAINING COMPLETE ─────────────────────────────")
    print(f"Best weights : {results.save_dir}/weights/best.pt")
    print(f"Last weights : {results.save_dir}/weights/last.pt")

    # ── Validation on best.pt ─────────────────────────────────────────────────
    print("\nRunning validation on best model ...")
    best_model = YOLO(f"{results.save_dir}/weights/best.pt")
    metrics = best_model.val(
        data    = DATA_YAML,
        imgsz   = IMG_SIZE,
        batch   = BATCH,
        device  = 0,
        verbose = True,
    )

    print(f"\n{'='*60}")
    print(f"Results")
    print(f"{'='*60}")
    print(f"mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"Precision    : {metrics.box.mp:.4f}")
    print(f"Recall       : {metrics.box.mr:.4f}")

    classes = ["chair", "table", "door", "sofa", "cabinet",
               "bed", "monitor", "trash_bin", "person"]

    print(f"\nPer-class AP@0.5:")
    for i, ap in enumerate(metrics.box.ap50):
        print(f"  {i:2d}  {classes[i]:<12}  {ap:.4f}")

    print(f"\nWeights      : {results.save_dir}/weights/best.pt")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()