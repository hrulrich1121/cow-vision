"""Goal 3: daily lying / standing time per calf."""
from __future__ import annotations

import csv
from pathlib import Path

SUMMARY_FIELDS = [
    "date", "channel", "side", "calf_id",
    "frames", "frames_with_calf", "detection_rate",
    "observed_hours", "lying_hours", "standing_hours", "unknown_hours",
    "lying_pct_of_observed", "standing_pct_of_observed",
    "mean_length_px", "mean_length_cm",
]


def summarize(det_csv: Path, out_csv: Path, sample_fps: float) -> Path:
    seconds_per_frame = 1.0 / float(sample_fps)
    buckets: dict[tuple[str, str, str], dict] = {}

    for r in csv.DictReader(det_csv.open()):
        key = (r["date"], r["channel"], r["side"])
        b = buckets.setdefault(key, {
            "frames": 0, "det": 0, "lying": 0, "standing": 0, "unknown": 0,
            "len_px": [], "len_cm": [],
        })
        b["frames"] += 1
        if r["detected"] == "1":
            b["det"] += 1
            if r["length_px"]:
                b["len_px"].append(float(r["length_px"]))
            if r["length_cm"]:
                b["len_cm"].append(float(r["length_cm"]))
        b[r["posture"] if r["posture"] in ("lying", "standing") else "unknown"] += 1

    rows = []
    for (date, channel, side), b in sorted(buckets.items()):
        h = lambda n: round(n * seconds_per_frame / 3600, 3)  # noqa: E731
        observed = b["lying"] + b["standing"]
        mean = lambda xs: round(sum(xs) / len(xs), 1) if xs else ""  # noqa: E731
        rows.append({
            "date": date,
            "channel": channel,
            "side": side,
            "calf_id": f"{channel}_{side}",
            "frames": b["frames"],
            "frames_with_calf": b["det"],
            "detection_rate": round(b["det"] / b["frames"], 3) if b["frames"] else 0,
            "observed_hours": h(observed),
            "lying_hours": h(b["lying"]),
            "standing_hours": h(b["standing"]),
            "unknown_hours": h(b["unknown"]),
            "lying_pct_of_observed": round(100 * b["lying"] / observed, 1) if observed else "",
            "standing_pct_of_observed": round(100 * b["standing"] / observed, 1) if observed else "",
            "mean_length_px": mean(b["len_px"]),
            "mean_length_cm": mean(b["len_cm"]),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return out_csv


def bouts(det_csv: Path, out_csv: Path, sample_fps: float, min_bout_s: float = 60.0) -> Path:
    """Continuous lying / standing bouts - useful for welfare metrics later."""
    seconds_per_frame = 1.0 / float(sample_fps)
    tracks: dict[tuple[str, str], list[dict]] = {}
    for r in csv.DictReader(det_csv.open()):
        tracks.setdefault((r["channel"], r["side"]), []).append(r)

    rows = []
    for (channel, side), rs in sorted(tracks.items()):
        rs.sort(key=lambda r: r["timestamp"])
        cur, start, n = None, None, 0
        for r in rs + [None]:
            label = r["posture"] if r else None
            if label != cur:
                if cur in ("lying", "standing"):
                    dur = n * seconds_per_frame
                    if dur >= min_bout_s:
                        rows.append({
                            "channel": channel, "side": side, "calf_id": f"{channel}_{side}",
                            "posture": cur, "start": start,
                            "duration_min": round(dur / 60, 2),
                        })
                cur, start, n = label, (r["timestamp"] if r else None), 0
            n += 1

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["channel", "side", "calf_id", "posture", "start", "duration_min"])
        w.writeheader()
        w.writerows(rows)
    return out_csv
