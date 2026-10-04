# Server runbook: building the eating/drinking label set

The lab server is password-only over the VPN, so these are run by hand, one at a
time, rather than scripted from the laptop. All of them are **bash on the
server** unless marked otherwise.

    ssh helmut@10.125.119.178

Videos are in `~/Desktop`, the repo is `~/cow-vision`, the full run is in `~/run`.
The server has no ffmpeg, so every extract below uses the OpenCV backend.

## 1. Get the new code onto the server

Laptop, PowerShell, from `cow-vision`:

    git add -A; git commit -m "eating/drinking plumbing, bouts, label-set selection"; git push

Server:

    cd ~/cow-vision && git pull && . .venv/bin/activate

Check the new subcommands are there:

    python -m cowvision --help

`bouts` and `labelset` should both be listed.

Note that `cowvision/minutely.py` had never been committed - whatever copy is on
the server got there by hand. The push above is the first time it enters git, so
after this pull the server and the laptop are finally running the same code.

## 2. Pick the six days

Six days spread across the study, so the model does not learn one day's
lighting. Linking rather than copying — these are ~1 GB each.

    mkdir -p ~/feedset/videos
    for d in 20260729 20260802 20260806 20260810 20260814 20260818; do
      ln -sf ~/Desktop/CH1_${d}*.mp4 ~/feedset/videos/
    done
    ls ~/feedset/videos | wc -l

Expect roughly 60-70 files (about 12 segments a day). Most will be skipped in
the next step without being opened.

## 3. Extract the feeding windows only

**Use `config_label.json`, not `config.json`**, for every step in this section.
It is the same config except `sample_fps` is 0.05 — one frame per 20 s instead
of one per second. At 1 fps this would produce ~65,000 frames, twenty times more
than anyone is going to label.

    python -m cowvision --config config_label.json --out ~/feedset extract \
      --videos ~/feedset/videos \
      --between 06:30-08:00,18:30-20:00 \
      --backend opencv

`--between` does two things: a segment that cannot contribute a single frame is
skipped without being decoded (that is most of them), and within the segments
that remain, only frames whose wall-clock time falls in a window are written.
Expect roughly 4 segments a day to survive, so ~24 of the ~70 files, giving
~3,200 frames per side and ~6,500 in total. Budget an hour or two of CPU.

Check it landed before going on:

    wc -l ~/feedset/frames_index.csv
    head -3 ~/feedset/frames_index.csv

~6,500 data rows, and the timestamps in column 5 should all fall inside one of
the two windows.

## 4. Run the current model over them

This does not label the new classes — v2 cannot — but it gives each frame a
posture and a box, which is what the selection step needs to find the frames
where the calf is at the buckets.

    python -m cowvision --config config_label.json --out ~/feedset detect

## 5. Choose the frames to label

    python -m cowvision --config config_label.json --out ~/feedset labelset \
      --upload ~/feedset/upload --target 600

It prints how many frames survived near-duplicate removal and how many it took
from each `side/posture/where` group. The groups to look at are
`*/standing/bucket`: those are where eating and drinking live, and they are
taken whole rather than sampled down. If they come to less than ~50 frames
between them, raise `--target` or widen the windows before labelling rather than
after.

## 6. Pull the frames back and label them

Laptop, PowerShell:

    scp -r helmut@10.125.119.178:~/feedset/upload ..\sample\feedset_upload

Then upload that folder to Roboflow (project `helmut-ulrich/cow-detection-wgeny`)
and label it to the rules in `docs/labelling-eating-drinking.md`. Re-review the
178 existing `standing` frames in the same pass — some of them are eating. Leave
the 581 `lying` frames alone.

## 7. Train and re-run

Export from Roboflow as YOLOv8 into `../dataset_v3/` with
`names: ['lying','standing','eating','drinking']`, train from
`models/calf_posture_v2.pt` rather than from scratch, and check the
`standing` vs `eating` corner of the confusion matrix specifically.

`*.pt` is gitignored, so `git pull` will not carry the new weights to the
server - copy them up by hand (laptop, PowerShell):

    scp models\calf_posture_v3.pt helmut@10.125.119.178:~/cow-vision/models/

Then point `detector.model` at them in `config.json`, commit and push that, pull
on the server, and re-run the corpus:

    nohup ./run_all.sh ~/Desktop ~/run2 > run_all2.log 2>&1 &

`run_all.sh` now finishes with `combine`, `minutely` and `bouts`, so the
eating/drinking minutes and bouts come out without further steps. Pull back:

    scp helmut@10.125.119.178:'~/run2/{daily_summary_all,minute_activity,bouts_all,bout_summary}.csv' ..\results\

### Worth doing first

Helmut now has sudo and GPU access on that box. The previous full run was
CPU-only and took about a day. Installing ffmpeg and a CUDA build of torch there
would cut a re-run to roughly an hour, and there is a re-run coming as soon as
the 4-class model exists — so it is worth doing before step 7 rather than after.

## Re-running bouts on their own

The bout counts do not need a re-run of anything. Against the existing results:

    python -m cowvision --out ~/run bouts --run ~/run

That reads the per-date `detections.csv` files, so it resolves interruptions
shorter than a minute, which the per-minute file cannot. Changing
`bouts.min_bout_s` in `config.json` and re-running takes seconds — see
`docs/bout-threshold.md` for what that number does to the answer.
