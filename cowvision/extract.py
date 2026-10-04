"""Goal 2: sample frames at N fps and crop each frame into left / right calf images.

Two backends:
  ffmpeg  - one pass per video, crops both pens straight out of the filter graph (fast)
  opencv  - pure python fallback when ffmpeg is not on PATH
"""
from __future__ import annotations

import csv
import shutil
import subprocess
from datetime import datetime, time as dtime, timedelta
from pathlib import Path

from .config import crops_for
from .naming import VideoInfo

SIDES = ("left", "right")
INDEX_FIELDS = [
    "frame_path", "channel", "side", "date", "timestamp",
    "video", "offset_s", "crop_w", "crop_h",
]


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def probe(video: Path) -> dict:
    """Real resolution / fps / duration from ffprobe, falling back to OpenCV."""
    if shutil.which("ffprobe"):
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
            "-of", "default=nw=1:nk=0", str(video),
        ]
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
        info: dict = {}
        for line in out.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                info[k.strip()] = v.strip()
        num, _, den = info.get("avg_frame_rate", "0/1").partition("/")
        fps = float(num) / float(den) if den and float(den) else 0.0
        return {
            "width": int(info.get("width", 0)),
            "height": int(info.get("height", 0)),
            "fps": round(fps, 3),
            "duration_s": float(info.get("duration", 0) or 0),
        }
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": round(fps, 3),
            "duration_s": round(n / fps, 1) if fps else 0.0,
        }
    finally:
        cap.release()


def _px_box(frac: list[float], w: int, h: int) -> tuple[int, int, int, int]:
    """Fractional [x1,y1,x2,y2] -> integer (x, y, w, h), clamped and even-sized."""
    x1 = max(0, min(w - 2, int(round(frac[0] * w))))
    y1 = max(0, min(h - 2, int(round(frac[1] * h))))
    x2 = max(x1 + 2, min(w, int(round(frac[2] * w))))
    y2 = max(y1 + 2, min(h, int(round(frac[3] * h))))
    return x1, y1, (x2 - x1) // 2 * 2, (y2 - y1) // 2 * 2


def frame_paths(out_root: Path, v: VideoInfo, side: str) -> Path:
    # One folder per segment: a day has ~12 segments and each restarts at 000001.jpg,
    # so sharing a folder would overwrite every segment but the last.
    return out_root / v.date / v.channel / side / v.start.strftime("%H%M%S")


def _stamp(v: VideoInfo, offset_s: float) -> datetime:
    return v.timestamp_at(offset_s)


def parse_windows(spec: str | None) -> list[tuple[dtime, dtime]]:
    """'06:30-08:00,18:30-20:00' -> clock windows to keep frames from.

    Used to pull a labelling set from the feeding windows only: a uniform sample
    over a day is >90% idle pen, so eating and drinking barely appear in it.
    """
    if not spec:
        return []
    out = []
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        if not b:
            raise ValueError(f"window needs START-END, got {part!r}")
        out.append(tuple(dtime(*map(int, t.split(":"))) for t in (a, b)))
    return out


def overlaps_windows(v: VideoInfo, windows: list[tuple[dtime, dtime]]) -> bool:
    """Could any frame of this segment fall inside a window?

    Decoding is the whole cost of extraction, so a segment that cannot contribute
    a single frame is skipped without being opened. Walked a minute at a time:
    segments are ~2 h, so this is cheap and avoids the date-wrap reasoning that
    comparing clock ranges directly would need.
    """
    if not windows:
        return True
    t = v.start
    while t <= v.end:
        if _in_windows(t, windows):
            return True
        t += timedelta(minutes=1)
    return _in_windows(v.end, windows)


def _in_windows(ts: datetime, windows: list[tuple[dtime, dtime]]) -> bool:
    if not windows:
        return True
    t = ts.time()
    # a window that wraps past midnight (22:00-02:00) is two ranges
    return any(a <= t < b if a <= b else (t >= a or t < b) for a, b in windows)


def extract_video(
    v: VideoInfo, out_root: Path, cfg: dict, index_writer: csv.DictWriter,
    limit_frames: int | None = None, backend: str = "auto",
    windows: list[tuple[dtime, dtime]] | None = None,
) -> int:
    meta = probe(v.path)
    w, h = meta["width"], meta["height"]
    if not w or not h:
        raise RuntimeError(f"cannot read video geometry: {v.path}")

    boxes = {s: _px_box(crops_for(cfg, v.channel)[s], w, h) for s in SIDES}
    for s in SIDES:
        frame_paths(out_root, v, s).mkdir(parents=True, exist_ok=True)

    # ffmpeg's filter graph writes frames contiguously, so it cannot skip the
    # gaps a clock window leaves; OpenCV handles those, and the server has no
    # ffmpeg anyway.
    use_ffmpeg = (backend == "ffmpeg" or (backend == "auto" and have_ffmpeg())) and not windows
    if use_ffmpeg:
        written = _extract_ffmpeg(v, out_root, cfg, boxes, limit_frames)
    else:
        written = _extract_opencv(v, out_root, cfg, boxes, limit_frames, windows or [], v)

    for n_frame, offset in written:
        for s in SIDES:
            bw, bh = boxes[s][2], boxes[s][3]
            p = frame_paths(out_root, v, s) / f"{n_frame:06d}.jpg"
            if not p.exists():
                continue
            ts = _stamp(v, offset)
            index_writer.writerow({
                "frame_path": str(p),
                "channel": v.channel,
                "side": s,
                "date": ts.strftime("%Y-%m-%d"),
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "video": v.path.name,
                "offset_s": round(offset, 3),
                "crop_w": bw,
                "crop_h": bh,
            })
    return len(written)


def _extract_ffmpeg(v, out_root, cfg, boxes, limit_frames) -> int:
    fps = cfg["sample_fps"]
    q = max(2, min(31, round(31 - (cfg["jpeg_quality"] / 100) * 29)))
    filt = [f"[0:v]fps={fps},split=2[a][b]"]
    maps: list[str] = []
    for tag, side in (("a", "left"), ("b", "right")):
        x, y, bw, bh = boxes[side]
        filt.append(f"[{tag}]crop={bw}:{bh}:{x}:{y}[{side}]")
        maps += ["-map", f"[{side}]", "-q:v", str(q)]
        if limit_frames:
            maps += ["-frames:v", str(limit_frames)]
        maps.append(str(frame_paths(out_root, v, side) / "%06d.jpg"))

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(v.path), "-filter_complex", ";".join(filt), *maps]
    subprocess.run(cmd, check=True)
    n = len(list(frame_paths(out_root, v, "left").glob("*.jpg")))
    step = 1.0 / float(fps)
    return [(i + 1, i * step) for i in range(n)]


def _extract_opencv(v, out_root, cfg, boxes, limit_frames, windows=(), vinfo=None) -> list:
    import cv2

    cap = cv2.VideoCapture(str(v.path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(src_fps / float(cfg["sample_fps"]))))
    qual = [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg["jpeg_quality"])]

    written: list[tuple[int, float]] = []
    idx = 0
    try:
        while True:
            if not cap.grab():
                break
            if idx % stride == 0:
                offset = idx / src_fps
                # outside a requested clock window the frame is skipped without
                # being decoded, which is most of the cost
                if windows and vinfo is not None and not _in_windows(
                        _stamp(vinfo, offset), windows):
                    idx += 1
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    break
                n_frame = len(written) + 1
                for side in SIDES:
                    x, y, bw, bh = boxes[side]
                    crop = frame[y:y + bh, x:x + bw]
                    out = frame_paths(out_root, v, side) / f"{n_frame:06d}.jpg"
                    cv2.imwrite(str(out), crop, qual)
                written.append((n_frame, offset))
                if limit_frames and len(written) >= limit_frames:
                    break
            idx += 1
    finally:
        cap.release()
    return written


def extract_all(videos, out_root: Path, cfg: dict, index_csv: Path,
                limit_frames=None, backend="auto", windows=None) -> int:
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with index_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        for v in videos:
            if v.path.stat().st_size == 0:
                print(f"  skip (zero bytes): {v.path.name}")
                continue
            if windows and not overlaps_windows(v, windows):
                continue
            print(f"  extracting {v.path.name} ...", flush=True)
            n = extract_video(v, out_root, cfg, w, limit_frames, backend, windows)
            print(f"    {n} sampled frames x 2 sides")
            total += n
    return total
