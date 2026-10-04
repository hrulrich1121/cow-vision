"""Goal 3: turn detections.csv into daily numbers.

summarize()    -> one row per calf per day (hours lying/standing, detection rate).
bouts()        -> continuous stretches of one activity, split on recording gaps.
bout_summary() -> per calf per day: bout counts and minutes for each activity.

The bout functions take either detections.csv (per frame, `posture`) or
minute_activity.csv (per minute, `activity`), so daily bout counts can be
rebuilt from the small CSV without going back to the frames.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

SUMMARY_FIELDS = [
    "date", "channel", "side", "calf_id",
    "frames", "frames_with_calf", "detection_rate",
    "observed_hours", "lying_hours", "standing_hours", "unknown_hours",
    "eating_hours", "drinking_hours",
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
            "eating": 0, "drinking": 0, "len_px": [], "len_cm": [],
        })
        b["frames"] += 1
        if r["detected"] == "1":
            b["det"] += 1
            if r["length_px"]:
                b["len_px"].append(float(r["length_px"]))
            if r["length_cm"]:
                b["len_cm"].append(float(r["length_cm"]))
        b[r["posture"] if r["posture"] in ("lying", "standing") else "unknown"] += 1
        # eating/drinking are a finer split of the standing frames, counted
        # separately so the lying/standing totals above stay unchanged
        act = r.get("activity") or ""
        if act in ("eating", "drinking"):
            b[act] += 1

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
            "eating_hours": h(b["eating"]),
            "drinking_hours": h(b["drinking"]),
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


# ---------------------------------------------------------------------------
# bouts: continuous stretches of one activity, per calf per day
# ---------------------------------------------------------------------------

BOUT_FIELDS = [
    "date", "channel", "side", "calf_id", "activity",
    "start", "end", "duration_min", "samples",
]

BOUT_SUMMARY_FIELDS = [
    "date", "channel", "side", "calf_id",
    "lying_bouts", "standing_bouts", "eating_bouts", "drinking_bouts",
    "lying_min", "standing_min", "eating_min", "drinking_min",
    "mean_lying_bout_min", "mean_standing_bout_min",
    "mean_eating_bout_min", "mean_drinking_bout_min",
    "observed_hours",
]

ACTIVITIES = ("lying", "standing", "eating", "drinking")

# A run shorter than this is treated as flicker and absorbed into its neighbours
# rather than counted as a bout. 300 s for lying/standing comes from the
# sensitivity sweep in docs/bout-threshold.md: below it the counts are dominated
# by single-sample flips, above it real bouts start being swallowed. Visits to
# the bucket or the bottle are legitimately short, so they get a lower floor.
# All four are overridable under "bouts" in config.json.
DEFAULT_MIN_BOUT_S = {"lying": 300.0, "standing": 300.0, "eating": 30.0, "drinking": 30.0}

# How long the detector may lose the calf before a bout is treated as over. Short
# undetected patches are the calf being briefly obscured, not a change of
# behaviour, so they are bridged; a longer one ends the bout honestly rather than
# assuming the calf held still through it. Separate from the floors above
# because it is a statement about the detector, not about the behaviour.
DEFAULT_MAX_UNKNOWN_S = 120.0


def _read_activity_rows(path: Path) -> tuple[list[dict], float]:
    """Read detections.csv or minute_activity.csv into common (calf, time, activity) rows.

    detections.csv carries the per-frame label in `posture`; minute_activity.csv
    carries the per-minute majority in `activity`. Returns the rows plus the
    nominal spacing between samples in seconds, inferred from the data.
    """
    rows = []
    with Path(path).open() as f:
        for r in csv.DictReader(f):
            label = r.get("activity") or r.get("posture") or ""
            rows.append({
                "channel": r["channel"], "side": r["side"],
                "timestamp": r["timestamp"],
                "activity": label if label in ACTIVITIES else "unknown",
            })
    return rows, _infer_step_s(rows)


def _infer_step_s(rows: list[dict]) -> float:
    """Median gap between consecutive samples of one calf - 1 s or 60 s in practice."""
    if len(rows) < 3:
        return 1.0
    key = (rows[0]["channel"], rows[0]["side"])
    ts = sorted(_parse(r["timestamp"]) for r in rows
                if (r["channel"], r["side"]) == key)[:2000]
    gaps = sorted((b - a).total_seconds() for a, b in zip(ts, ts[1:]) if b > a)
    return gaps[len(gaps) // 2] if gaps else 1.0


def _parse(ts: str) -> datetime:
    return datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S")


def _merge_short_runs(labels: list[str], min_samples: dict[str, int]) -> list[str]:
    """Absorb sub-threshold runs into their neighbours.

    Discarding a short run instead would leave the two stretches either side of
    it as separate bouts, so one flickering minute would still add a bout. Here
    the shortest offending run is repeatedly given to its longer neighbour until
    everything left clears its floor - the usual rule in lying-behaviour work,
    where brief interruptions are not counted as ending a bout.
    """
    runs: list[list] = []                       # [label, count]
    for l in labels:
        if runs and runs[-1][0] == l:
            runs[-1][1] += 1
        else:
            runs.append([l, 1])

    while len(runs) > 1:
        short = [i for i, (l, n) in enumerate(runs)
                 if l in min_samples and n < min_samples[l]]
        if not short:
            break
        i = min(short, key=lambda i: runs[i][1])
        prev, nxt = (runs[i - 1] if i else None), (runs[i + 1] if i + 1 < len(runs) else None)
        if prev and nxt:
            target = prev if prev[1] >= nxt[1] else nxt
        else:
            target = prev or nxt
        target[1] += runs[i][1]
        runs.pop(i)
        # the absorbing run may now touch an identical label - rejoin them
        j = runs.index(target)
        for k in (j, j - 1):
            if 0 <= k < len(runs) - 1 and runs[k][0] == runs[k + 1][0]:
                runs[k][1] += runs[k + 1][1]
                runs.pop(k + 1)

    out = []
    for l, n in runs:
        out.extend([l] * n)
    return out


def bouts(src_csv: Path, out_csv: Path, sample_fps: float | None = None,
          min_bout_s: float | dict | None = None, max_gap_s: float | None = None,
          max_unknown_s: float | None = None) -> Path:
    """Continuous activity bouts, one row each.

    A bout ends when the label changes *or* when the recording gaps - two
    segments either side of a missing hour are two bouts, not one. `sample_fps`
    is accepted for backwards compatibility but the spacing is read from the
    timestamps, which is what actually governs a bout's duration.
    """
    rows, step_s = _read_activity_rows(src_csv)
    if sample_fps:
        step_s = min(step_s, 1.0 / float(sample_fps)) or step_s
    gap_limit = max_gap_s if max_gap_s is not None else step_s * 3

    floors = dict(DEFAULT_MIN_BOUT_S)
    if isinstance(min_bout_s, dict):
        floors.update(min_bout_s)
    elif min_bout_s is not None:
        floors = {a: float(min_bout_s) for a in ACTIVITIES}

    tracks: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        tracks.setdefault((r["channel"], r["side"]), []).append(r)

    out = []
    for (channel, side), rs in sorted(tracks.items()):
        rs.sort(key=lambda r: r["timestamp"])
        run: list[dict] = []

        def flush():
            if not run:
                return
            act = run[0]["activity"]
            if act not in ACTIVITIES:
                return
            # the last sample also represents `step_s` of time, hence the +1
            dur = len(run) * step_s
            if dur < floors.get(act, 60.0):
                return
            start = _parse(run[0]["timestamp"])
            out.append({
                "date": run[0]["timestamp"][:10],
                "channel": channel, "side": side, "calf_id": f"{channel}_{side}",
                "activity": act,
                "start": run[0]["timestamp"],
                "end": (start + timedelta(seconds=dur)).strftime("%Y-%m-%d %H:%M:%S"),
                "duration_min": round(dur / 60, 2),
                "samples": len(run),
            })

        # flicker removal happens per contiguous recording block, before any
        # bout is cut, so a merge never bridges a gap in the footage
        blocks: list[list[dict]] = [[]]
        for r in rs:
            if blocks[-1] and (_parse(r["timestamp"])
                               - _parse(blocks[-1][-1]["timestamp"])).total_seconds() > gap_limit:
                blocks.append([])
            blocks[-1].append(r)

        min_samples = {a: max(1, int(round(s / step_s))) for a, s in floors.items()}
        # a brief run of undetected frames should not end a bout either. The
        # "+ 1" makes max_unknown_s inclusive: a patch of exactly that length is
        # still bridged, which is how the setting reads.
        unknown_s = DEFAULT_MAX_UNKNOWN_S if max_unknown_s is None else max_unknown_s
        min_samples["unknown"] = int(unknown_s / step_s) + 1
        for block in blocks:
            if not block:
                continue
            merged = _merge_short_runs([r["activity"] for r in block], min_samples)
            for r, label in zip(block, merged):
                if run and label != run[-1]["activity"]:
                    flush()
                    run = []
                run.append({**r, "activity": label})
            flush()
            run = []

    out.sort(key=lambda r: (r["calf_id"], r["start"]))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_csv).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BOUT_FIELDS)
        w.writeheader()
        w.writerows(out)
    return out_csv


def bout_summary(bouts_csv: Path, out_csv: Path, minutes_csv: Path | None = None) -> Path:
    """One row per calf per day: bout counts and total minutes for each activity.

    This is the researcher's deliverable - `*_bouts` are the daily frequencies
    and `*_min` the daily durations. Totals come from the bouts, so time in
    discarded sub-threshold fragments is not counted.
    """
    agg: dict[tuple[str, str, str], dict] = {}
    for r in csv.DictReader(Path(bouts_csv).open()):
        key = (r["date"], r["channel"], r["side"])
        a = agg.setdefault(key, {act: [0, 0.0] for act in ACTIVITIES})
        if r["activity"] in a:
            a[r["activity"]][0] += 1
            a[r["activity"]][1] += float(r["duration_min"])

    observed: dict[tuple[str, str, str], float] = {}
    if minutes_csv and Path(minutes_csv).exists():
        for r in csv.DictReader(Path(minutes_csv).open()):
            if r.get("activity") in ACTIVITIES:
                key = (r["timestamp"][:10], r["channel"], r["side"])
                observed[key] = observed.get(key, 0.0) + 1.0

    rows = []
    for (date, channel, side), a in sorted(agg.items()):
        row = {"date": date, "channel": channel, "side": side,
               "calf_id": f"{channel}_{side}"}
        for act in ACTIVITIES:
            n, mins = a[act]
            row[f"{act}_bouts"] = n
            row[f"{act}_min"] = round(mins, 1)
            row[f"mean_{act}_bout_min"] = round(mins / n, 2) if n else ""
        row["observed_hours"] = round(observed.get((date, channel, side), 0) / 60, 2) or ""
        rows.append(row)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_csv).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BOUT_SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return out_csv


def combine(summary_csvs: list[Path], out_csv: Path) -> Path:
    """Merge per-date daily_summary.csv files into one, one row per calf per day.

    A segment that runs over midnight puts a handful of frames in the next
    date, so the same (date, calf) can appear in two runs' summaries; those
    rows are added together here rather than left as duplicates.
    """
    merged: dict[tuple[str, str, str], dict] = {}
    for path in summary_csvs:
        for r in csv.DictReader(Path(path).open()):
            key = (r["date"], r["channel"], r["side"])
            m = merged.setdefault(key, {
                "frames": 0, "frames_with_calf": 0,
                "lying_hours": 0.0, "standing_hours": 0.0, "unknown_hours": 0.0,
                "eating_hours": 0.0, "drinking_hours": 0.0,
                "len_px_weighted": 0.0, "len_cm_weighted": 0.0, "len_cm_frames": 0,
            })
            det = int(r["frames_with_calf"])
            m["frames"] += int(r["frames"])
            m["frames_with_calf"] += det
            for k in ("lying_hours", "standing_hours", "unknown_hours",
                      "eating_hours", "drinking_hours"):
                m[k] += float(r.get(k) or 0)
            if r["mean_length_px"]:
                m["len_px_weighted"] += float(r["mean_length_px"]) * det
            if r["mean_length_cm"]:
                m["len_cm_weighted"] += float(r["mean_length_cm"]) * det
                m["len_cm_frames"] += det

    rows = []
    for (date, channel, side), m in sorted(merged.items()):
        observed = m["lying_hours"] + m["standing_hours"]
        det = m["frames_with_calf"]
        rows.append({
            "date": date, "channel": channel, "side": side,
            "calf_id": f"{channel}_{side}",
            "frames": m["frames"], "frames_with_calf": det,
            "detection_rate": round(det / m["frames"], 3) if m["frames"] else 0,
            "observed_hours": round(observed, 3),
            "lying_hours": round(m["lying_hours"], 3),
            "standing_hours": round(m["standing_hours"], 3),
            "unknown_hours": round(m["unknown_hours"], 3),
            "eating_hours": round(m["eating_hours"], 3),
            "drinking_hours": round(m["drinking_hours"], 3),
            "lying_pct_of_observed": round(100 * m["lying_hours"] / observed, 1) if observed else "",
            "standing_pct_of_observed": round(100 * m["standing_hours"] / observed, 1) if observed else "",
            "mean_length_px": round(m["len_px_weighted"] / det, 1) if det else "",
            "mean_length_cm": round(m["len_cm_weighted"] / m["len_cm_frames"], 1) if m["len_cm_frames"] else "",
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return out_csv
