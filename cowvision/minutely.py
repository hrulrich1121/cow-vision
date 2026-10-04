"""Per-minute activity, for human validation against the source video.

One row per calf per minute: the majority posture over that minute's sampled
frames, plus the frame counts behind it so a disputed minute can be judged on
how strong the evidence was. `agreement` is the winning label's share of the
classified frames -- 1.0 means every frame agreed, ~0.5 means the minute was a
transition (the calf got up or lay down partway through).
"""
from __future__ import annotations

import csv
from pathlib import Path

ACTIVITIES = ("lying", "standing", "eating", "drinking")

FIELDS = [
    "channel", "side", "calf_id", "timestamp", "activity", "posture",
    "frames", "lying_frames", "standing_frames", "eating_frames", "drinking_frames",
    "undetected_frames", "agreement", "video",
]


def minutely(det_csvs: list[Path], out_csv: Path) -> tuple[Path, int]:
    buckets: dict[tuple[str, str, str], dict] = {}

    for det in det_csvs:
        with Path(det).open() as f:
            for r in csv.DictReader(f):
                minute = r["timestamp"][:16] + ":00"  # YYYY-MM-DD HH:MM:00
                key = (r["channel"], r["side"], minute)
                b = buckets.setdefault(key, {
                    **{a: 0 for a in ACTIVITIES},
                    "frames": 0, "undetected": 0, "lying_posture": 0, "videos": {},
                })
                b["frames"] += 1
                if r["detected"] != "1":
                    b["undetected"] += 1
                # pre-4-class detections.csv has no `activity` column
                act = r.get("activity") or r.get("posture") or ""
                if act in ACTIVITIES:
                    b[act] += 1
                if r.get("posture") == "lying":
                    b["lying_posture"] += 1
                b["videos"][r["video"]] = b["videos"].get(r["video"], 0) + 1

    rows = []
    for (channel, side, minute), b in sorted(buckets.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
        classified = sum(b[a] for a in ACTIVITIES)
        if not classified:
            activity, agreement = "unknown", ""
        else:
            activity = max(ACTIVITIES, key=lambda a: b[a])
            agreement = round(b[activity] / classified, 3)
        # posture is the coarse roll-up: a calf eating or drinking is standing
        posture = "unknown" if not classified else (
            "lying" if b["lying_posture"] * 2 >= classified else "standing")
        rows.append({
            "channel": channel,
            "side": side,
            "calf_id": f"{channel}_{side}",
            "timestamp": minute,
            "activity": activity,
            "posture": posture,
            "frames": b["frames"],
            "lying_frames": b["lying"],
            "standing_frames": b["standing"],
            "eating_frames": b["eating"],
            "drinking_frames": b["drinking"],
            "undetected_frames": b["undetected"],
            "agreement": agreement,
            "video": max(b["videos"], key=b["videos"].get),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return out_csv, len(rows)


def find_detections(root: Path) -> list[Path]:
    """detections.csv directly in `root`, or one per date subdirectory."""
    direct = root / "detections.csv"
    return [direct] if direct.exists() else sorted(root.glob("*/detections.csv"))
