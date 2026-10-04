"""Pick frames to hand-label in Roboflow, enriched for the rare behaviours.

A uniform sample of a calf's day is >90% lying in an otherwise empty pen, so
eating and drinking barely appear in it - the v1/v2 label sets contain 581 lying
frames and no feeding at all. This module biases the sample instead:

  * extract only the feeding windows (`cowvision extract --between`),
  * group what comes back by side, posture and whether the calf is at the feed
    buckets, then fill a quota from each group,
  * drop near-duplicates first, since at 1 frame / 20 s a calf that has not moved
    produces dozens of identical crops.

The output is a flat directory of uniquely named JPEGs plus a manifest mapping
each one back to its frames_index.csv row, which is what Roboflow wants.
"""
from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

MANIFEST_FIELDS = [
    "frame_path", "channel", "side", "date", "timestamp", "video", "offset_s",
    "crop_w", "crop_h", "posture", "at_bucket", "group", "upload_name",
]

# Fraction of the bucket zone the calf's box has to cover before the frame is
# called "at the buckets". Only used to *choose* frames to label - the label
# itself is always a human's.
AT_BUCKET_MIN_COVER = 0.25


def upload_name(r: dict) -> str:
    """Flat, unique, and still readable: CH1_left_20260728_043708_000001.jpg."""
    p = Path(r["frame_path"])
    return f"{r['channel']}_{r['side']}_{r['date'].replace('-', '')}_{p.parent.name}_{p.name}"


def bucket_cover(r: dict, zone: list[float] | None) -> float:
    """How much of the feed-bucket zone the detection box covers, 0..1."""
    if not zone or r.get("detected") != "1" or not r.get("x1"):
        return 0.0
    w, h = float(r["crop_w"]), float(r["crop_h"])
    zx1, zy1, zx2, zy2 = zone[0] * w, zone[1] * h, zone[2] * w, zone[3] * h
    x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
    ox = max(0.0, min(x2, zx2) - max(x1, zx1))
    oy = max(0.0, min(y2, zy2) - max(y1, zy1))
    area = (zx2 - zx1) * (zy2 - zy1)
    return (ox * oy / area) if area else 0.0


def _phash(path: Path, size: int = 16):
    """Tiny grey thumbnail as a tuple - cheap near-duplicate key."""
    import cv2

    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    small = cv2.resize(im, (size, size)).astype("float32")
    return small


def _too_similar(a, b, thresh: float) -> bool:
    if a is None or b is None:
        return False
    return float(abs(a - b).mean()) < thresh


def select(index_csv: Path, out_dir: Path, det_csv: Path | None = None,
           zones: dict | None = None, target: int = 600,
           dup_thresh: float = 6.0, seed: int = 0) -> tuple[Path, dict]:
    """Choose ~`target` frames, copy them into out_dir, write manifest.csv.

    `det_csv` is the current model's output over the same frames. It is optional
    - without it every frame lands in one group and the selection is a plain
    deduplicated sample - but with it the rare "standing at the buckets" frames,
    the ones that actually show eating and drinking, get their own quota.
    """
    rows = list(csv.DictReader(Path(index_csv).open()))
    det = {}
    if det_csv and Path(det_csv).exists():
        det = {r["frame_path"]: r for r in csv.DictReader(Path(det_csv).open())}

    zones = zones or {}
    for r in rows:
        d = det.get(r["frame_path"], {})
        r["posture"] = d.get("posture", "unlabelled")
        zone = (zones.get(r["channel"]) or zones.get("default") or {}).get(r["side"])
        cover = bucket_cover({**r, **d}, zone)
        r["at_bucket"] = "1" if cover >= AT_BUCKET_MIN_COVER else "0"
        r["group"] = f"{r['side']}/{r['posture']}/{'bucket' if r['at_bucket'] == '1' else 'pen'}"

    # ---- drop near-duplicates, within a segment and side ------------------
    rows.sort(key=lambda r: (r["side"], r["video"], r["timestamp"]))
    kept, prev_key, prev_hash = [], None, None
    for r in rows:
        key = (r["side"], r["video"])
        h = _phash(Path(r["frame_path"]))
        if key == prev_key and _too_similar(h, prev_hash, dup_thresh):
            continue
        kept.append(r)
        prev_key, prev_hash = key, h

    # ---- quota per group --------------------------------------------------
    groups: dict[str, list[dict]] = {}
    for r in kept:
        groups.setdefault(r["group"], []).append(r)

    # Equal shares, then hand back whatever the small groups cannot use so the
    # rare feeding groups are taken whole rather than sampled down.
    quota = {g: target // len(groups) for g in groups} if groups else {}
    spare = target - sum(min(len(groups[g]), quota[g]) for g in groups)
    hungry = [g for g in groups if len(groups[g]) > quota[g]]
    for g in hungry:
        quota[g] += spare // len(hungry) if hungry else 0

    rng = random.Random(seed)
    chosen: list[dict] = []
    for g, rs in sorted(groups.items()):
        n = min(len(rs), quota[g])
        # spread the pick over time rather than clumping it in one segment
        rs = sorted(rs, key=lambda r: r["timestamp"])
        if n >= len(rs):
            chosen.extend(rs)
        else:
            step = len(rs) / n
            chosen.extend(rs[int(i * step)] for i in range(n))
    rng.shuffle(chosen)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in chosen:
            r["upload_name"] = upload_name(r)
            dest = out_dir / r["upload_name"]
            if not dest.exists():
                shutil.copy2(r["frame_path"], dest)
            w.writerow(r)

    stats = {
        "frames_in": len(rows),
        "after_dedup": len(kept),
        "selected": len(chosen),
        "per_group": {g: sum(1 for c in chosen if c["group"] == g) for g in sorted(groups)},
        "available_per_group": {g: len(rs) for g, rs in sorted(groups.items())},
    }
    return manifest, stats
