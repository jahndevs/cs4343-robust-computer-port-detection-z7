"""Quick checks for the data, the corruptions, and the train -> evaluate pipeline.

    pytest -q              # everything (smoke test takes ~1-2 min)
    pytest -q -m "not slow"
"""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASSES, DATASET  # noqa: E402
from robustness import SEVERITY, corrupt  # noqa: E402

SPLITS = ["train", "val", "test"]


# ---------- data ----------

@pytest.mark.parametrize("split", SPLITS)
def test_every_image_has_a_valid_label_file(split):
    images = sorted(p.stem for p in (DATASET / "images" / split).iterdir())
    labels = sorted(p.stem for p in (DATASET / "labels" / split).iterdir())
    assert images == labels

    for f in (DATASET / "labels" / split).iterdir():
        for line in f.read_text().splitlines():
            cls, *box = line.split()
            assert 0 <= int(cls) < len(CLASSES), f"{f.name}: bad class {cls}"
            assert all(0.0 <= float(v) <= 1.0 for v in box), f"{f.name}: box out of range"


def test_no_source_image_appears_in_two_splits():
    """Roboflow copies share the prefix before '.rf.'; they must stay in one split."""
    seen = {}
    for split in SPLITS:
        for p in (DATASET / "images" / split).iterdir():
            src = re.split(r"\.rf\.", p.name)[0]
            assert seen.setdefault(src, split) == split, f"{src} in {seen[src]} and {split}"


# ---------- corruptions ----------

@pytest.mark.parametrize("kind", list(SEVERITY))
def test_corruption_keeps_shape_and_gets_worse_with_severity(kind):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    diffs = []
    for level in (1, 2, 3):
        out = corrupt(img, kind, level)
        assert out.shape == img.shape and out.dtype == np.uint8
        diffs.append(np.abs(out.astype(int) - img).mean())
    assert 0 < diffs[0] < diffs[1] < diffs[2], diffs


# ---------- end-to-end ----------

@pytest.mark.slow
def test_smoke_train_and_evaluate():
    """1 epoch on 5% of the data: only checks the pipeline runs, not accuracy."""
    from evaluate import evaluate
    from train import train

    best = train(name="smoke_test", epochs=1, imgsz=320, batch=8, fraction=0.05)
    assert best.exists()

    map50, rows = evaluate(best, split="val", imgsz=320, name="smoke_test_eval", plots=False)
    assert 0.0 <= map50 <= 1.0
    assert {r["class"] for r in rows} <= set(CLASSES)
