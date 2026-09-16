# cow-vision

Pipeline for the calf pen videos: organizes footage by date, samples frames,
crops each frame into left/right pen, runs YOLO detection, and classifies
posture (standing/lying) into a daily summary.

Video files must keep the camera naming convention
`CH1_20260729103855-20260729124739.mp4` (channel, start, end timestamps) --
that's where every frame's real timestamp comes from.

## Install

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`ffmpeg` on PATH speeds up extraction; falls back to OpenCV automatically if
it's not installed.

## Usage

```bash
python -m cowvision init-config                          # writes config.json
python -m cowvision organize --videos ./VIDEOS --by-date
python -m cowvision preview  --videos ./VIDEOS            # check the L/R crop boxes
python -m cowvision extract  --videos ./VIDEOS
python -m cowvision detect
python -m cowvision summarize
```

Or run everything in one go:

```bash
python -m cowvision all --videos ./VIDEOS
```

Output goes to `output/` by default (`--out <dir>` to change it). Key files:

- `manifest.csv` / `coverage.csv` -- what video exists, any gaps
- `frames_index.csv` -- every sampled crop with its timestamp
- `detections.csv` -- per-frame detection + posture
- `daily_summary.csv` / `bouts.csv` -- the actual results

## Config

`config.json` holds the crop boxes, detector settings, and posture
thresholds. Run `preview` first and adjust `crops` if the left/right split
doesn't line up with the pens.

## Running the full corpus on the lab server

```bash
scp -r cow-vision helmut@10.125.119.178:~/        # VPN up first
ssh helmut@10.125.119.178
cd ~/cow-vision && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

nohup ./run_all.sh ~/Desktop ~/run > run_all.log 2>&1 &
tail -f run_all.log
```

`run_all.sh` processes one date at a time and skips any date that already has
a `daily_summary.csv`, so it's safe to kill and restart.

## Test

```bash
PYTHONPATH=. python test/test_pipeline.py
```

Runs the pipeline end-to-end on a synthetic video with a stub detector --
no GPU or network needed.
