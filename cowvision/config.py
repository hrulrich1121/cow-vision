"""Configuration loading. Plain JSON so the tool has no extra dependencies."""
from __future__ import annotations

import copy
import json
from pathlib import Path

DEFAULT_CONFIG = {
    # ---- sampling -------------------------------------------------------
    "sample_fps": 1.0,          # frames analysed per second of video
    "jpeg_quality": 85,
    "save_crops": True,         # goal 2 wants the cropped images on disk

    # ---- pen geometry ---------------------------------------------------
    # Boxes are FRACTIONS of the frame: [x1, y1, x2, y2] in 0..1.
    # "default" applies to any channel without its own entry.
    # Adjust after looking at the output of `cowvision preview`.
    "crops": {
        "default": {
            "left":  [0.00, 0.00, 0.50, 1.00],
            "right": [0.50, 0.00, 1.00, 1.00]
        }
    },

    # ---- detection ------------------------------------------------------
    "detector": {
        "model": "yolov8n.pt",   # swap for your Roboflow-trained weights later
        "conf": 0.30,
        "imgsz": 640,
        "device": "",            # "" = auto, "cpu", "0" for first GPU
        # COCO ids that can plausibly be a calf: cow=19, horse=17, sheep=18, dog=16
        "classes": [19, 17, 18, 16],
        "max_per_crop": 1        # one calf per pen
    },

    # ---- posture --------------------------------------------------------
    # Baseline rule on the detection box inside its crop:
    #   aspect = width / height ; height_frac = height / crop_height
    # A standing calf is taller and narrower; a lying calf is flatter.
    # Re-tune with `cowvision calibrate` once you have real detections.
    "posture": {
        "method": "geometry",            # "geometry" | "classifier"
        "lying_aspect_min": 1.45,        # aspect >= this  -> lying
        "standing_aspect_max": 1.15,     # aspect <= this  -> standing
        "standing_height_frac_min": 0.35,
        "smooth_window": 5,              # median filter over N sampled frames
        "classifier_weights": ""         # path to a posture classifier, when trained
    },

    # ---- pixel -> real world -------------------------------------------
    # Fill in once you have a reference object of known length in each pen.
    # length_cm = box_length_px * cm_per_px
    "scale": {
        "default": {"left": None, "right": None}
    }
}


def load_config(path: Path | None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path and Path(path).exists():
        user = json.loads(Path(path).read_text())
        _deep_update(cfg, user)
    return cfg


def _deep_update(base: dict, new: dict) -> dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def write_default_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(DEFAULT_CONFIG, indent=2))


def crops_for(cfg: dict, channel: str) -> dict:
    return cfg["crops"].get(channel, cfg["crops"]["default"])


def scale_for(cfg: dict, channel: str, side: str):
    entry = cfg.get("scale", {}).get(channel) or cfg.get("scale", {}).get("default", {})
    return entry.get(side)
