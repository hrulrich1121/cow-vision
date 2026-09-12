"""Goal 2: sample frames at N fps and crop each frame into left / right calf images.

Two backends:
  ffmpeg  - one pass per video, crops both pens straight out of the filter graph (fast)
  opencv  - pure python fallback when ffmpeg is not on PATH
"""
from __future__ import annotations

import csv
import shutil
import subprocess
from datetime import datetime
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
    return out_root / v.date / v.channel / side


def _stamp(v: VideoInfo, offset_s: float) -> datetime:
    return v.timestamp_at(offset_s)


def extract_video(
    v: VideoInfo, out_root: Path, cfg: dict, index_writer: csv.DictWriter,
    limit_frames: int | None = None, backend: str = "auto",
) -> int:
    meta = probe(v.path)
    w, h = meta["width"], meta["height"]
    if not w or not h:
        raise RuntimeError(f"cannot read video geometry: {v.path}")

    boxes = {s: _px_box(crops_for(cfg, v.channel)[s], w, h) for s in SIDES}
    for s in SIDES:
        frame_paths(out_root, v, s).mkdir(parents=True, exist_ok=True)

    use_ffmpeg = backend == "ffmpeg" or (backend == "auto" and have_ffmpeg())
    if use_ffmpeg:
        n = _extract_ffmpeg(v, out_root, cfg, boxes, limit_frames)
    else:
        n = _extract_opencv(v, out_root, cfg, boxes, limit_frames)

    step = 1.0 / float(cfg["sample_fps"])
    for i in range(n):
        offset = i * step
        for s in SIDES:
            bw, bh = boxes[s][2], boxes[s][3]
            p = frame_paths(out_root, v, s) / f"{i + 1:06d}.jpg"
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
    return n


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
    return len(list(frame_paths(out_root, v, "left").glob("*.jpg")))


def _extract_opencv(v, out_root, cfg, boxes, limit_frames) -> int:
    import cv2

    cap = cv2.VideoCapture(str(v.path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(src_fps / float(cfg["sample_fps"]))))
    qual = [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg["jpeg_quality"])]

    written = 0
    idx = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                written += 1
                for side in SIDES:
                    x, y, bw, bh = boxes[side]
                    crop = frame[y:y + bh, x:x + bw]
                    out = frame_paths(out_root, v, side) / f"{written:06d}.jpg"
                    cv2.imwrite(str(out), crop, qual)
                if limit_frames and written >= limit_frames:
                    break
            idx += 1
    finally:
        cap.release()
    return written


def extract_all(videos, out_root: Path, cfg: dict, index_csv: Path,
                limit_frames=None, backend="auto") -> int:
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with index_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        for v in videos:
            if v.path.stat().st_size == 0:
                print(f"  skip (zero bytes): {v.path.name}")
                continue
            print(f"  extracting {v.path.name} ...", flush=True)
            n = extract_video(v, out_root, cfg, w, limit_frames, backend)
            print(f"    {n} sampled frames x 2 sides")
            total += n
    return total
