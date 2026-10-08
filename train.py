"""Fine-tune a COCO-pretrained YOLOv8s on the port dataset.

Hyperparameters follow the proposal. Use --no-aug for the augmentation ablation.

    python train.py                         # full run (100 epochs, early stop after 20)
    python train.py --no-aug --name no_aug  # ablation
    python train.py --imgsz 960 --name img960
"""
import argparse

from ultralytics import YOLO

from config import DATA, RUNS, pick_device

# Augmentations listed in the proposal (Ultralytics names).
AUGMENT = dict(
    mosaic=1.0,      # mosaic
    scale=0.5,       # random scaling
    translate=0.1,   # random translation
    hsv_h=0.015,     # color jitter
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,      # horizontal flip
)


def train(name="baseline", epochs=100, imgsz=640, batch=16, augment=True, fraction=1.0, device=None):
    """Train and return the path to the best weights."""
    aug = AUGMENT if augment else {k: 0.0 for k in AUGMENT}
    model = YOLO("yolov8s.pt")  # pretrained on COCO
    model.train(
        data=str(DATA),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=3,
        patience=20,          # early stopping
        fraction=fraction,    # <1.0 only for quick tests
        device=device or pick_device(),
        project=str(RUNS),
        name=name,
        exist_ok=True,
        seed=42,
        **aug,
    )
    return RUNS / name / "weights" / "best.pt"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="baseline")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--no-aug", action="store_true", help="disable augmentation (ablation)")
    p.add_argument("--fraction", type=float, default=1.0, help="fraction of training images to use")
    p.add_argument("--device", default=None, help="e.g. 0, mps, cpu (auto if omitted)")
    a = p.parse_args()
    best = train(a.name, a.epochs, a.imgsz, a.batch, not a.no_aug, a.fraction, a.device)
    print(f"Best weights: {best}")
