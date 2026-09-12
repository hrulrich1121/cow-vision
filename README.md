# cowvision — dairy calf video pipeline

Covers the three goals from the research team's email:

1. **Organize the videos by date** → `cowvision organize`
2. **Crop each frame into the left and right calf, ~1 frame/second** → `cowvision extract`
3. **Daily lying and standing time** → `cowvision detect` + `cowvision summarize`

Everything is driven by the camera filename convention
`CH1_20260729103855-20260729124739.mp4` = channel, start timestamp, end timestamp.
That is where every frame's real wall-clock time comes from, so file names must not be
renamed.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`ffmpeg` on PATH is strongly recommended — extraction is several times faster and seeks
correctly. Without it the pipeline falls back to OpenCV automatically.

## Quick start

```bash
cowvision() { python -m cowvision "$@"; }

python -m cowvision init-config                                   # writes config.json
python -m cowvision organize --videos ./VIDEOS --by-date          # goal 1
python -m cowvision preview  --videos ./VIDEOS                    # check the L/R split
python -m cowvision extract  --videos ./VIDEOS                    # goal 2
python -m cowvision detect                                        # YOLO + posture
python -m cowvision summarize                                     # goal 3
```

Everything lands under `output/` (override with `--out`).

### Step 0 — check the split before you extract 100k frames

```bash
python -m cowvision preview --videos ./VIDEOS --at 5,1800,5400
```

This writes full frames with the configured pen boxes drawn on them, one per channel.
The default is a straight 50/50 vertical split. If the divider is not in the middle, or
if the pens do not fill the frame top to bottom, edit `crops` in `config.json` — boxes
are **fractions** of the frame, `[x1, y1, x2, y2]`:

```json
"crops": {
  "default": { "left": [0.00, 0.00, 0.50, 1.00], "right": [0.50, 0.00, 1.00, 1.00] },
  "CH1":     { "left": [0.02, 0.10, 0.47, 0.95], "right": [0.53, 0.10, 0.98, 0.95] }
}
```

Per-channel entries override `default`. Re-run `preview` until the boxes look right.

### Step 1 — organize

Writes `output/manifest.csv` (one row per video: channel, date, start, end, duration,
size, **usable**) and `output/coverage.csv` (hours of video per date/channel and the
gaps between files). `--by-date` also builds `output/by-date/YYYY-MM-DD/CHn/` using
hard links, so it costs no extra disk. `--mode copy|move` if you'd rather.

Zero-byte files are flagged and skipped rather than crashing the run — one of the four
sample videos (`CH1_20260729170509-...`) is currently 0 bytes.

### Step 2 — extract + crop

```bash
python -m cowvision extract --videos ./VIDEOS                # all of them
python -m cowvision extract --videos ./VIDEOS --channel CH1 --date 2026-07-29
python -m cowvision extract --videos ./VIDEOS --limit-frames 60   # smoke test
```

Output layout:

```
output/frames/2026-07-29/CH1/left/000001.jpg
output/frames/2026-07-29/CH1/right/000001.jpg
output/frames_index.csv      # frame_path, channel, side, date, timestamp, offset_s, crop size
```

`frames_index.csv` is the contract between stages — the wall-clock timestamp of every
crop is computed there, so detection and aggregation never have to parse filenames again.
Sampling rate is `sample_fps` in config (1.0 = one frame per second).

Rough size: one 2-hour video at 1 fps ≈ 7,200 frames × 2 sides ≈ 14,400 JPEGs
(~1–2 GB at quality 85). Lower `jpeg_quality`, or `sample_fps` to 0.5, if disk is tight.

### Step 3 — detect and label posture

```bash
python -m cowvision detect
```

Runs Ultralytics YOLO on every crop, keeps the highest-confidence animal box, and writes
`output/detections.csv` with the box, its aspect ratio, its height as a fraction of the
crop, and a posture label.

Out of the box it uses COCO `yolov8n.pt` (`cow` / `horse` / `sheep` / `dog` classes —
calves in a pen get picked up by several of these). When you have Roboflow-trained
weights, point `detector.model` at the `.pt` file and narrow `detector.classes`.

**Posture baseline (geometry).** A standing calf's box is taller than it is wide; a lying
calf's is flatter. `posture.lying_aspect_min` / `standing_aspect_max` /
`standing_height_frac_min` control the split, and `smooth_window` majority-votes over N
consecutive samples so a single bad frame can't create a fake posture change.

Tune the thresholds from real data instead of guessing:

```bash
python -m cowvision calibrate
```

It prints the percentile spread of aspect and height across your detections. A camera
looking at a pen from the side gives a clearly bimodal aspect distribution — put the two
thresholds either side of the valley.

This baseline is deliberately the **first** posture pass, not the final one. Once a few
thousand crops are labeled in Roboflow, train a 2-class (standing / lying) classifier and
swap it in behind `posture.method = "classifier"`; the rest of the pipeline is unchanged.

### Step 4 — daily lying / standing time

```bash
python -m cowvision summarize
```

`output/daily_summary.csv`, one row per calf per day:

| column | meaning |
|---|---|
| `calf_id` | `CH1_left` / `CH1_right` |
| `frames`, `frames_with_calf`, `detection_rate` | how much of the day the calf was actually visible — read this before trusting the hours |
| `observed_hours` | video time where a posture was assigned |
| `lying_hours`, `standing_hours`, `unknown_hours` | the answer to goal 3 |
| `lying_pct_of_observed` | normalizes for missing video, so partial days are comparable |
| `mean_length_px`, `mean_length_cm` | body length in pixels; cm only once `scale` is filled in |

`output/bouts.csv` additionally gives continuous lying/standing bouts ≥ 1 min — bout
count and mean bout length are the usual welfare metrics and they come free from the
per-second labels.

Hours are computed as `frames × (1 / sample_fps)`, so they're only as complete as the
video: check `coverage.csv` before reporting a daily total.

## Pixels → centimeters

`detections.csv` records `length_px` (the long side of the box) for every frame. To turn
that into a real measurement, put a known-length reference in each pen — a meter stick, a
tape on the back wall at calf depth, or the pen divider width measured once — read its
length in pixels off a preview frame, and set

```json
"scale": { "CH1": { "left": 0.42, "right": 0.44 } }   // cm per pixel
```

`length_cm` then fills in automatically. Caveat worth raising with the research team: a
single scale factor is only valid at one distance from the camera, so a calf at the back
of the pen will measure short. If body length matters for the study, the honest fix is
either a fixed-position measurement zone (only measure when the calf is in a marked strip
of the pen) or a camera calibration with a checkerboard — worth deciding before you
collect a season of numbers.

## Running on the lab server

The pipeline is pure Python and has no interactive parts, so it runs fine over SSH:

```bash
scp -r cow-vision helmut@10.125.119.178:~/            # VPN up first
ssh helmut@10.125.119.178
cd ~/cow-vision && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# smoke test on one video before committing to a full run
python -m cowvision --out ~/cowout organize --videos /path/to/videos
python -m cowvision --out ~/cowout preview  --videos /path/to/videos
python -m cowvision --out ~/cowout extract  --videos /path/to/videos --limit-frames 60
python -m cowvision --out ~/cowout detect && python -m cowvision --out ~/cowout summarize

# full run, detached so it survives the SSH session dropping
nohup python -m cowvision --out ~/cowout all --videos /path/to/videos > run.log 2>&1 &
tail -f run.log
```

If the server has a GPU, set `detector.device` to `"0"` in config.json — detection is the
only slow stage and it's ~20× faster on a GPU.

## Layout

```
cowvision/naming.py      filename -> channel + real timestamps
cowvision/organize.py    manifest, coverage, by-date tree        (goal 1)
cowvision/extract.py     1 fps sampling + left/right crop        (goal 2)
cowvision/preview.py     draws the pen boxes on a full frame
cowvision/detect.py      YOLO + geometry posture + calibrate
cowvision/aggregate.py   daily summary + bouts                   (goal 3)
cowvision/cli.py         command line
test/test_pipeline.py    offline end-to-end test (fake detector, no GPU/network needed)
```

## Test

```bash
PYTHONPATH=. python test/test_pipeline.py
```

Builds a synthetic two-pen video (left calf standing throughout, right calf lying for the
first half), runs extract → detect → summarize with a stub detector, and should report
left = 100% standing, right = 50/50.
