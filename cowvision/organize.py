"""Goal 1: organize the videos by date."""
from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

from .naming import VideoInfo, scan_videos

MANIFEST_FIELDS = [
    "channel", "date", "start", "end", "nominal_duration_s",
    "size_bytes", "usable", "path",
]


def build_manifest(video_root: Path, out_dir: Path) -> tuple[list[VideoInfo], Path]:
    videos, skipped = scan_videos(video_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.csv"

    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for v in videos:
            size = v.path.stat().st_size
            w.writerow({
                "channel": v.channel,
                "date": v.date,
                "start": v.start.isoformat(sep=" "),
                "end": v.end.isoformat(sep=" "),
                "nominal_duration_s": int(v.nominal_duration_s),
                "size_bytes": size,
                "usable": "yes" if size > 0 else "no (zero bytes)",
                "path": str(v.path),
            })

    if skipped:
        (out_dir / "unparsed_files.txt").write_text(
            "\n".join(str(p) for p in skipped) + "\n"
        )
    return videos, manifest


def organize_by_date(videos: list[VideoInfo], dest_root: Path, mode: str = "link") -> int:
    """Create dest_root/YYYY-MM-DD/CHn/<file> as links (default), copies or moves."""
    n = 0
    for v in videos:
        target_dir = dest_root / v.date / v.channel
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / v.path.name
        if target.exists():
            continue
        if mode == "move":
            shutil.move(str(v.path), target)
        elif mode == "copy":
            shutil.copy2(v.path, target)
        else:  # link: hardlink, fall back to symlink, then copy
            try:
                os.link(v.path, target)
            except OSError:
                try:
                    target.symlink_to(v.path.resolve())
                except OSError:
                    shutil.copy2(v.path, target)
        n += 1
    return n


def coverage_report(videos: list[VideoInfo], out_dir: Path) -> Path:
    """Per date/channel: how many hours of video exist and where the gaps are."""
    out = out_dir / "coverage.csv"
    rows = []
    by_key: dict[tuple[str, str], list[VideoInfo]] = {}
    for v in videos:
        by_key.setdefault((v.date, v.channel), []).append(v)

    for (date, channel), vids in sorted(by_key.items()):
        vids.sort(key=lambda v: v.start)
        usable = [v for v in vids if v.path.stat().st_size > 0]
        covered = sum(v.nominal_duration_s for v in usable)
        gaps = []
        for a, b in zip(vids, vids[1:]):
            gap = (b.start - a.end).total_seconds()
            if abs(gap) > 60:
                gaps.append(f"{a.end:%H:%M:%S}->{b.start:%H:%M:%S} ({int(gap)}s)")
        rows.append({
            "date": date,
            "channel": channel,
            "files": len(vids),
            "zero_byte_files": len(vids) - len(usable),
            "first_start": min(v.start for v in vids).strftime("%H:%M:%S"),
            "last_end": max(v.end for v in vids).strftime("%H:%M:%S"),
            "covered_hours": round(covered / 3600, 2),
            "gaps": "; ".join(gaps),
        })

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["date"])
        w.writeheader()
        w.writerows(rows)
    return out
