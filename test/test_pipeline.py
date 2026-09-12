"""Offline end-to-end test: synthetic video + stub detector.

Needs no GPU, no network and no ultralytics install. Run from the repo root:

    PYTHONPATH=. python test/test_pipeline.py

Expected: CH9 left = 100% standing, CH9 right = 50% lying / 50% standing.
"""
from __future__ import annotations

import csv
import shutil
import sys
import types
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
VID_DIR = ROOT / "VIDEOS"
OUT = ROOT / "output"
CFG = ROOT / "config.json"


def make_video() -> Path:
    """Left pen: tall box (standing) all the way through.
    Right pen: flat box (lying) for the first 30 s, tall box after."""
    VID_DIR.mkdir(parents=True, exist_ok=True)
    path = VID_DIR / "CH9_20260729103855-20260729103955.mp4"
    W, H, FPS, SEC = 1280, 720, 10, 60
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(FPS * SEC):
        f = np.full((H, W, 3), 40, np.uint8)
        cv2.line(f, (W // 2, 0), (W // 2, H), (90, 90, 90), 6)
        cv2.rectangle(f, (200, 250), (380, 650), (200, 200, 210), -1)
        if i < FPS * SEC // 2:
            cv2.rectangle(f, (760, 480), (1150, 640), (200, 200, 210), -1)
        else:
            cv2.rectangle(f, (860, 250), (1040, 650), (200, 200, 210), -1)
        vw.write(f)
    vw.release()
    return path


# ---- stub detector standing in for ultralytics.YOLO -------------------------
class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy = [np.array(xyxy)]
        self.conf = np.array([conf])

    def __len__(self):
        return len(self.conf)


class _Empty:
    xyxy: list = []
    conf = np.array([])

    def __len__(self):
        return 0


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeYOLO:
    def __init__(self, *a, **k):
        pass

    def predict(self, paths, **k):
        out = []
        for p in paths:
            img = cv2.imread(str(p), 0)
            _, th = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
            cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cs = [c for c in cs if cv2.contourArea(c) > 500]
            if not cs:
                out.append(_Result(_Empty()))
                continue
            x, y, w, h = cv2.boundingRect(max(cs, key=cv2.contourArea))
            out.append(_Result(_Boxes([x, y, x + w, y + h], 0.9)))
        return out


def main() -> int:
    sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=FakeYOLO)

    for p in (OUT,):
        shutil.rmtree(p, ignore_errors=True)
    make_video()

    from cowvision import aggregate, detect
    from cowvision.cli import main as cli

    common = ["--out", str(OUT), "--config", str(CFG)]
    cli(common + ["init-config", "--force"])
    cli(common + ["organize", "--videos", str(VID_DIR), "--by-date"])
    cli(common + ["extract", "--videos", str(VID_DIR)])

    from cowvision.config import load_config
    cfg = load_config(CFG)
    detect.run(OUT / "frames_index.csv", OUT / "detections.csv", cfg)
    aggregate.summarize(OUT / "detections.csv", OUT / "daily_summary.csv", cfg["sample_fps"])
    aggregate.bouts(OUT / "detections.csv", OUT / "bouts.csv", cfg["sample_fps"], min_bout_s=5)

    rows = {r["calf_id"]: r for r in csv.DictReader((OUT / "daily_summary.csv").open())}
    print("\n" + (OUT / "daily_summary.csv").read_text())

    ok = True
    checks = [
        ("CH9_left standing%", float(rows["CH9_left"]["standing_pct_of_observed"]), 100.0),
        ("CH9_right lying%", float(rows["CH9_right"]["lying_pct_of_observed"]), 50.0),
        ("CH9_right standing%", float(rows["CH9_right"]["standing_pct_of_observed"]), 50.0),
    ]
    for name, got, want in checks:
        good = abs(got - want) <= 5
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {name}: {got} (expected ~{want})")
    for cid in ("CH9_left", "CH9_right"):
        got = float(rows[cid]["detection_rate"])
        ok &= got == 1.0
        print(f"{'PASS' if got == 1.0 else 'FAIL'}  {cid} detection_rate: {got}")

    print("\nALL PASS" if ok else "\nFAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
