#!/usr/bin/env bash
# Full corpus run on the lab server, one date at a time so a crash loses at most a day.
#   nohup ./run_all.sh ~/Desktop ~/run > run_all.log 2>&1 &
# Re-running skips dates that already have a daily_summary.csv.
# Each day's crops (~4 GB) are deleted once its detections.csv + summary exist, because
# the whole corpus at 1 fps is ~90 GB and the server has less free space than that.
# Set KEEP_FRAMES=1 to keep them.
set -u
VIDEOS=${1:-~/Desktop}
OUT=${2:-~/run}
KEEP_FRAMES=${KEEP_FRAMES:-0}
cd "$(dirname "$0")"
. .venv/bin/activate

mkdir -p "$OUT"
python -m cowvision --out "$OUT" organize --videos "$VIDEOS"

dates=$(tail -n +2 "$OUT/manifest.csv" | cut -d, -f2 | sort -u)
for d in $dates; do
  if [ -f "$OUT/$d/daily_summary.csv" ]; then
    echo "== $d already done, skipping"; continue
  fi
  echo "== $d  $(date)"
  python -m cowvision --out "$OUT/$d" extract --videos "$VIDEOS" --date "$d" --backend opencv \
    && python -m cowvision --out "$OUT/$d" detect \
    && python -m cowvision --out "$OUT/$d" summarize \
    || { echo "!! $d failed"; continue; }
  if [ "$KEEP_FRAMES" != "1" ]; then
    rm -rf "$OUT/$d/frames"
  fi
  df -h "$OUT" | tail -1
done

# one combined summary across dates
{ head -n1 "$(ls "$OUT"/*/daily_summary.csv | head -n1)"; for f in "$OUT"/*/daily_summary.csv; do tail -n +2 "$f"; done; } > "$OUT/daily_summary_all.csv"
{ head -n1 "$(ls "$OUT"/*/bouts.csv | head -n1)"; for f in "$OUT"/*/bouts.csv; do tail -n +2 "$f"; done; } > "$OUT/bouts_all.csv"
echo "== all done $(date)"
