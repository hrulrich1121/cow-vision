"""Parse camera filenames like CH1_20260729103855-20260729124739.mp4"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

# CH<n>_<start>-<end>.<ext>   timestamps = YYYYMMDDHHMMSS
PATTERN = re.compile(
    r"^(?P<channel>CH\d+)[_-](?P<start>\d{14})-(?P<end>\d{14})$", re.IGNORECASE
)
TS_FMT = "%Y%m%d%H%M%S"
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".dav", ".ts"}


@dataclass
class VideoInfo:
    path: Path
    channel: str
    start: datetime
    end: datetime

    @property
    def date(self) -> str:
        return self.start.strftime("%Y-%m-%d")

    @property
    def nominal_duration_s(self) -> float:
        return (self.end - self.start).total_seconds()

    def timestamp_at(self, offset_s: float) -> datetime:
        return self.start + timedelta(seconds=offset_s)


def parse_video_name(path: Path) -> VideoInfo | None:
    """Return VideoInfo, or None if the name doesn't match the camera convention."""
    if path.suffix.lower() not in VIDEO_EXTS:
        return None
    m = PATTERN.match(path.stem)
    if not m:
        return None
    try:
        start = datetime.strptime(m.group("start"), TS_FMT)
        end = datetime.strptime(m.group("end"), TS_FMT)
    except ValueError:
        return None
    return VideoInfo(
        path=path, channel=m.group("channel").upper(), start=start, end=end
    )


def scan_videos(root: Path) -> tuple[list[VideoInfo], list[Path]]:
    """Walk `root` and split video files into parsed and unparsed."""
    parsed: list[VideoInfo] = []
    skipped: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
            continue
        info = parse_video_name(p)
        if info is None:
            skipped.append(p)
        else:
            parsed.append(info)
    parsed.sort(key=lambda v: (v.channel, v.start))
    return parsed, skipped
