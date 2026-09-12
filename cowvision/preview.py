"""Draw the configured pen boxes on a full frame so the split can be eyeballed."""
from __future__ import annotations

from pathlib import Path

from .config import crops_for
from .extract import SIDES, _px_box, probe
from .naming import VideoInfo

COLORS = {"left": (0, 200, 255), "right": (255, 120, 0)}  # BGR


def preview_video(v: VideoInfo, cfg: dict, out_dir: Path, at_seconds: list[float]) -> list[Path]:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = probe(v.path)
    w, h = meta["width"], meta["height"]
    boxes = {s: _px_box(crops_for(cfg, v.channel)[s], w, h) for s in SIDES}

    cap = cv2.VideoCapture(str(v.path))
    written: list[Path] = []
    try:
        for t in at_seconds:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            for side in SIDES:
                x, y, bw, bh = boxes[side]
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), COLORS[side], 3)
                cv2.putText(frame, side.upper(), (x + 12, y + 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS[side], 3)
            ts = v.timestamp_at(t).strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, f"{v.channel}  {ts}", (12, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            p = out_dir / f"{v.channel}_{v.timestamp_at(t):%Y%m%dT%H%M%S}_preview.jpg"
            cv2.imwrite(str(p), frame)
            written.append(p)
    finally:
        cap.release()
    return written
