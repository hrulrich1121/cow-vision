"""Pipeline config: sampling, crop boxes, detector, posture, scale.
Loads from config.json, falling back to the defaults below.
"""
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
    "posture": {
        "method": "geometry",            # "geometry" | "detector_class" (see below)
        # detector class name -> posture. Eating and drinking are things a calf
        # does while standing, so they roll up into "standing" here and the
        # lying/standing totals stay comparable with the earlier runs.
        "class_map": {
            "lying": "lying",
            "standing": "standing",
            "eating": "standing",
            "drinking": "standing"
        },
        # detector class name -> activity, the finer label the bout counts and
        # the eating/drinking minutes are built from. A class missing here is
        # carried through under its own name.
        "activity_map": {
            "lying": "lying",
            "standing": "standing",
            "eating": "eating",
            "drinking": "drinking"
        },
        "lying_aspect_min": 1.45,        # aspect >= this  -> lying
        "standing_aspect_max": 1.15,     # aspect <= this  -> standing
        "standing_height_frac_min": 0.35,
        "smooth_window": 5,              # median filter over N sampled frames
        "classifier_weights": ""         # unused for now -- reserved for later
    },

    # ---- feed zones -------------------------------------------------------
    # Where the feed/water buckets sit inside each *crop*, as fractions
    # [x1, y1, x2, y2]. Only used to pick frames worth hand-labelling: a calf
    # whose box covers this zone is probably at the buckets. Never a label in
    # its own right - a calf can stand in front of the buckets without eating,
    # which is exactly why the behaviour is learnt rather than inferred here.
    "zones": {
        "default": {"left": None, "right": None}
    },

    # ---- bouts -----------------------------------------------------------
    # Minimum length of a run of one activity, in seconds. Shorter runs are
    # absorbed into their neighbours (see aggregate._merge_short_runs), so these
    # are flicker filters as much as bout definitions - raising one lowers the
    # daily bout count and lengthens the mean bout.
    "bouts": {
        "min_bout_s": {
            "lying": 300, "standing": 300, "eating": 30, "drinking": 30
        },
        "max_unknown_s": 120, # an undetected patch up to this long is bridged
                              # rather than ending the bout
        "max_gap_s": None     # null = 3x the sample spacing; bouts never bridge a
                              # longer break, so a missing segment splits a bout
    },

    # ---- pixel -> real world ---------------------------------------------
    # length_cm = box_length_px * cm_per_px. Unused for now
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
