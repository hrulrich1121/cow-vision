"""Detect the calf in each crop and label its posture.

Detection: Ultralytics YOLO (COCO weights out of the box, your Roboflow-trained
weights once you have them -- just point detector.model at the .pt file).

Posture baseline ("geometry"): the calf's bounding box is taller than it is wide
when it stands and flatter when it lies down. Thresholds live in config.json and
can be re-tuned from real data with `cowvision calibrate`.
"""
from __future__ import annotations

import csv
from pathlib import Path

DET_FIELDS = [
    "frame_path", "channel", "side", "date", "timestamp", "video", "offset_s",
    "detected", "conf", "x1", "y1", "x2", "y2", "box_w", "box_h",
    "aspect", "height_frac", "length_px", "length_cm", "posture_raw", "posture",
]

LYING, STANDING, UNKNOWN = "lying", "standing", "unknown"


def load_model(cfg: dict):
    from ultralytics import YOLO  # imported lazily so the rest of the CLI works without it

    return YOLO(cfg["detector"]["model"])


def classify_geometry(box_w: float, box_h: float, crop_h: float, p: dict) -> str:
    if box_h <= 0:
        return UNKNOWN
    aspect = box_w / box_h
    height_frac = box_h / crop_h if crop_h else 0.0
    if aspect >= p["lying_aspect_min"]:
        return LYING
    if aspect <= p["standing_aspect_max"] and height_frac >= p["standing_height_frac_min"]:
        return STANDING
    # middle ground: let height decide
    return STANDING if height_frac >= p["standing_height_frac_min"] else LYING


def _smooth(labels: list[str], window: int) -> list[str]:
    """Majority vote over a centred window - kills single-frame flicker."""
    if window <= 1 or len(labels) < window:
        return labels[:]
    half = window // 2
    out = []
    for i in range(len(labels)):
        chunk = [l for l in labels[max(0, i - half):i + half + 1] if l != UNKNOWN]
        if not chunk:
            out.append(UNKNOWN)
            continue
        out.append(max(set(chunk), key=chunk.count))
    return out


def run(index_csv: Path, out_csv: Path, cfg: dict, batch: int = 32,
        progress_every: int = 500) -> Path:
    from .config import scale_for

    rows = list(csv.DictReader(index_csv.open()))
    if not rows:
        raise SystemExit(f"no frames listed in {index_csv}")

    model = load_model(cfg)
    d = cfg["detector"]
    p = cfg["posture"]

    results_rows: list[dict] = []
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        preds = model.predict(
            [r["frame_path"] for r in chunk],
            conf=d["conf"], imgsz=d["imgsz"], classes=d["classes"] or None,
            device=d["device"] or None, verbose=False,
        )
        for r, res in zip(chunk, preds):
            crop_h = float(r["crop_h"])
            best = None
            if len(res.boxes):
                i = int(res.boxes.conf.argmax())
                xyxy = [float(x) for x in res.boxes.xyxy[i].tolist()]
                cls_name = str(res.names[int(res.boxes.cls[i])])
                best = (float(res.boxes.conf[i]), xyxy, cls_name)

            row = {k: r[k] for k in
                   ("frame_path", "channel", "side", "date", "timestamp", "video", "offset_s")}
            if best is None:
                row.update(detected=0, conf="", x1="", y1="", x2="", y2="",
                           box_w="", box_h="", aspect="", height_frac="",
                           length_px="", length_cm="", posture_raw=UNKNOWN)
            else:
                conf, (x1, y1, x2, y2), cls_name = best
                bw, bh = x2 - x1, y2 - y1
                cm_per_px = scale_for(cfg, r["channel"], r["side"])
                length_px = max(bw, bh)
                if p["method"] == "detector_class":
                    posture_raw = p["class_map"].get(cls_name, UNKNOWN)
                else:
                    posture_raw = classify_geometry(bw, bh, crop_h, p)
                row.update(
                    detected=1, conf=round(conf, 3),
                    x1=round(x1, 1), y1=round(y1, 1), x2=round(x2, 1), y2=round(y2, 1),
                    box_w=round(bw, 1), box_h=round(bh, 1),
                    aspect=round(bw / bh, 3) if bh else "",
                    height_frac=round(bh / crop_h, 3) if crop_h else "",
                    length_px=round(length_px, 1),
                    length_cm=round(length_px * cm_per_px, 1) if cm_per_px else "",
                    posture_raw=posture_raw,
                )
            results_rows.append(row)
        if progress_every and (start // batch) % max(1, progress_every // batch) == 0:
            print(f"  {min(start + batch, len(rows))}/{len(rows)} crops", flush=True)

    # smooth per (channel, side), in time order
    by_track: dict[tuple[str, str], list[dict]] = {}
    for row in results_rows:
        by_track.setdefault((row["channel"], row["side"]), []).append(row)
    for track in by_track.values():
        track.sort(key=lambda r: r["timestamp"])
        smoothed = _smooth([r["posture_raw"] for r in track], int(p["smooth_window"]))
        for r, s in zip(track, smoothed):
            r["posture"] = s

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DET_FIELDS)
        w.writeheader()
        w.writerows(sorted(results_rows, key=lambda r: (r["channel"], r["side"], r["timestamp"])))
    return out_csv


def calibrate(det_csv: Path) -> str:
    """Print the aspect / height distribution so thresholds can be picked from data."""
    import statistics

    vals = [(float(r["aspect"]), float(r["height_frac"]))
            for r in csv.DictReader(det_csv.open()) if r["detected"] == "1" and r["aspect"]]
    if not vals:
        return "No detections to calibrate on."
    aspects = sorted(v[0] for v in vals)
    heights = sorted(v[1] for v in vals)

    def q(xs, f):
        return round(xs[min(len(xs) - 1, int(f * len(xs)))], 3)

    lines = [f"detections: {len(vals)}", "",
             "aspect (box_w/box_h)  p05 %s  p25 %s  median %s  p75 %s  p95 %s"
             % (q(aspects, .05), q(aspects, .25), q(aspects, .5), q(aspects, .75), q(aspects, .95)),
             "height_frac           p05 %s  p25 %s  median %s  p75 %s  p95 %s"
             % (q(heights, .05), q(heights, .25), q(heights, .5), q(heights, .75), q(heights, .95)),
             "", f"mean aspect {round(statistics.mean(aspects), 3)}", "",
             "If the distribution is bimodal, put lying_aspect_min / standing_aspect_max",
             "either side of the valley between the two modes."]
    return "\n".join(lines)
