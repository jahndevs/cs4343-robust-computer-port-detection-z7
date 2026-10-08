"""Evaluate trained weights on a dataset split (RQ1).

Prints and saves per-class precision, recall, F1 and AP@0.5, plus overall mAP@0.5.
Ultralytics also saves the confusion matrix and PR curves in the output folder.

    python evaluate.py runs/baseline/weights/best.pt
"""
import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

from config import DATA, RUNS, pick_device


def evaluate(weights, data=DATA, split="test", imgsz=640, name=None, plots=True, device=None):
    """Return (mAP@0.5, rows) where rows is a list of per-class dicts."""
    name = name or f"eval_{Path(weights).parent.parent.name}_{split}"
    metrics = YOLO(weights).val(
        data=str(data), split=split, imgsz=imgsz, batch=16, iou=0.7, conf=0.001,
        device=device or pick_device(), project=str(RUNS), name=name, exist_ok=True,
        plots=plots, verbose=False,
    )
    box = metrics.box
    rows = []
    for i, c in enumerate(box.ap_class_index):
        rows.append({
            "class": metrics.names[int(c)],
            "precision": round(float(box.p[i]), 4),
            "recall": round(float(box.r[i]), 4),
            "f1": round(float(box.f1[i]), 4),
            "ap50": round(float(box.ap50[i]), 4),
        })
    return float(box.map50), rows


def save_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("weights")
    p.add_argument("--split", default="test")
    p.add_argument("--imgsz", type=int, default=640)
    a = p.parse_args()

    map50, rows = evaluate(a.weights, split=a.split, imgsz=a.imgsz)
    out = RUNS / f"eval_{Path(a.weights).parent.parent.name}_{a.split}" / "per_class.csv"
    save_csv(rows, out)

    print(f"\n{'class':<12} {'P':>6} {'R':>6} {'F1':>6} {'AP50':>6}")
    for r in rows:
        print(f"{r['class']:<12} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} {r['ap50']:>6.3f}")
    print(f"\nmAP@0.5 = {map50:.3f}   (success target: >= 0.85, each class >= 0.70)")
    print(f"Saved: {out}")
