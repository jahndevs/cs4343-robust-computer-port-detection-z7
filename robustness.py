"""Robustness evaluation (RQ2).

Builds corrupted copies of the test set (4 corruptions x 3 severities), then reports
mAP@0.5 and per-class AP@0.5 for each one. Run once per model to compare
augmentation vs. no augmentation.

    python robustness.py runs/baseline/weights/best.pt
    python robustness.py runs/no_aug/weights/best.pt
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from config import CLASSES, DATASET, ROOT, RUNS
from evaluate import evaluate, save_csv

CORRUPTED = ROOT / "corrupted"

# Severity levels 1, 2, 3 for each corruption.
SEVERITY = {
    "gaussian_noise": [10, 25, 50],    # noise std (pixel values 0-255)
    "motion_blur": [5, 11, 21],        # kernel length in pixels
    "jpeg": [30, 15, 5],               # JPEG quality
    "brightness": [0.7, 0.5, 0.3],     # brightness multiplier
}


def corrupt(img, kind, level):
    """Apply one corruption to a BGR uint8 image. level is 1, 2 or 3."""
    s = SEVERITY[kind][level - 1]
    if kind == "gaussian_noise":
        rng = np.random.default_rng(0)
        out = img.astype(np.float32) + rng.normal(0, s, img.shape)
    elif kind == "motion_blur":
        k = np.zeros((s, s), np.float32)
        k[s // 2, :] = 1.0 / s  # horizontal streak
        out = cv2.filter2D(img, -1, k)
    elif kind == "jpeg":
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, s])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    elif kind == "brightness":
        out = img.astype(np.float32) * s
    else:
        raise ValueError(kind)
    return np.clip(out, 0, 255).astype(np.uint8)


def build_corrupted_set(kind, level, src=DATASET, dst_root=CORRUPTED):
    """Write a corrupted copy of the test split and return its data yaml path."""
    dst = dst_root / f"{kind}_{level}"
    yaml_path = dst / "data.yaml"
    if yaml_path.exists():
        return yaml_path  # already built
    (dst / "images" / "test").mkdir(parents=True, exist_ok=True)
    shutil.copytree(src / "labels" / "test", dst / "labels" / "test", dirs_exist_ok=True)
    for f in sorted((src / "images" / "test").iterdir()):
        img = cv2.imread(str(f))
        if img is not None:
            cv2.imwrite(str(dst / "images" / "test" / f.name), corrupt(img, kind, level))
    names = ", ".join(f'"{c}"' for c in CLASSES)
    yaml_path.write_text(
        f"train: images/test\nval: images/test\ntest: images/test\nnc: {len(CLASSES)}\nnames: [{names}]\n"
    )
    return yaml_path


def run(weights, imgsz=640, kinds=tuple(SEVERITY)):
    """Evaluate on the clean test set and every corrupted copy; save a CSV."""
    model_name = Path(weights).parent.parent.name
    results = []
    clean_map, clean_rows = evaluate(weights, name=f"robust_{model_name}/clean", imgsz=imgsz, plots=False)
    results.append({"corruption": "clean", "severity": 0, "map50": round(clean_map, 4),
                    **{r["class"]: r["ap50"] for r in clean_rows}})
    for kind in kinds:
        for level in (1, 2, 3):
            data = build_corrupted_set(kind, level)
            m, rows = evaluate(weights, data=data, name=f"robust_{model_name}/{kind}_{level}",
                               imgsz=imgsz, plots=False)
            results.append({"corruption": kind, "severity": level, "map50": round(m, 4),
                            **{r["class"]: r["ap50"] for r in rows}})
            print(f"{kind:<15} sev={level}  mAP@0.5={m:.3f}")
    out = RUNS / f"robust_{model_name}" / "robustness.csv"
    save_csv(results, out)
    print(f"Saved: {out}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("weights")
    p.add_argument("--imgsz", type=int, default=640)
    a = p.parse_args()
    run(a.weights, a.imgsz)
