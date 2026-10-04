"""Posture accuracy of a trained detector on the human-labelled splits.

  python eval_posture.py [dataset_dir] [weights]

Reports, per split, a confusion matrix over {lying, standing, no-calf} plus
per-class precision / recall / F1, using the same conf threshold as config.json.
Box quality (mAP) comes from `yolo val`; this script is about the label a frame
ends up with, which is what the daily hours are built from.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CLASSES = ["lying", "standing"]
NONE = "no-calf"


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def gt_for(label_path: Path, w: int, h: int):
    lines = [l for l in label_path.read_text().splitlines() if l.strip()]
    if not lines:
        return NONE, None
    parts = lines[0].split()
    c, nums = int(parts[0]), [float(v) for v in parts[1:]]
    if len(nums) == 4:                      # class cx cy w h
        cx, cy, bw, bh = nums
        x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
    else:                                   # polygon: class x1 y1 x2 y2 ...
        xs, ys = nums[0::2], nums[1::2]     # Roboflow's smart-polygon tool
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return CLASSES[c], [x1 * w, y1 * h, x2 * w, y2 * h]


def main(ds="../dataset_v2", weights=None):
    import cv2
    from ultralytics import YOLO

    cfg = json.loads(Path("config.json").read_text())
    weights = weights or cfg["detector"]["model"]
    conf = cfg["detector"]["conf"]
    model = YOLO(weights)
    print(f"weights={weights}  conf={conf}  dataset={ds}\n")

    labels = [NONE] + CLASSES
    for split in ("valid", "test"):
        imgs = sorted(Path(f"{ds}/{split}/images").glob("*.jpg"))
        if not imgs:
            continue
        m = {g: {p: 0 for p in labels} for g in labels}
        ious = []
        for i in range(0, len(imgs), 32):
            chunk = imgs[i:i + 32]
            preds = model.predict([str(p) for p in chunk], conf=conf, verbose=False)
            for p, res in zip(chunk, preds):
                h, w = cv2.imread(str(p)).shape[:2]
                g, gbox = gt_for(Path(f"{ds}/{split}/labels/{p.stem}.txt"), w, h)
                if len(res.boxes):
                    j = int(res.boxes.conf.argmax())
                    pred = res.names[int(res.boxes.cls[j])]
                    pbox = [float(v) for v in res.boxes.xyxy[j].tolist()]
                    if gbox:
                        ious.append(iou(gbox, pbox))
                else:
                    pred = NONE
                m[g][pred] += 1

        n = sum(sum(r.values()) for r in m.values())
        correct = sum(m[k][k] for k in labels)
        print(f"--- {split}  (n={n})")
        print(f"{'':>10}" + "".join(f"{p:>10}" for p in labels) + "   <- predicted")
        for g in labels:
            print(f"{g:>10}" + "".join(f"{m[g][p]:>10}" for p in labels))
        print(f"\naccuracy {correct}/{n} = {correct / n:.3f}"
              f"   mean IoU {sum(ious) / len(ious):.3f} (n={len(ious)})")
        for c in labels:
            tp = m[c][c]
            fp = sum(m[g][c] for g in labels if g != c)
            fn = sum(m[c][p] for p in labels if p != c)
            prec = tp / (tp + fp) if tp + fp else float("nan")
            rec = tp / (tp + fn) if tp + fn else float("nan")
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
            print(f"  {c:>10}  support {tp + fn:3d}   precision {prec:.3f}  recall {rec:.3f}  F1 {f1:.3f}")
        print()


if __name__ == "__main__":
    main(*sys.argv[1:])
