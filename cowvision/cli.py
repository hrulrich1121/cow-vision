"""cowvision - dairy calf video pipeline.

  cowvision init-config
  cowvision organize   --videos <dir> [--out <dir>] [--by-date] [--mode link|copy|move]
  cowvision preview    --videos <dir> [--at 0,600,3600]
  cowvision extract    --videos <dir> [--limit-frames N]
  cowvision detect
  cowvision calibrate
  cowvision summarize
  cowvision all        --videos <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import aggregate, detect, extract, organize, preview
from .config import load_config, write_default_config
from .naming import scan_videos


def _paths(args):
    out = Path(args.out)
    return {
        "out": out,
        "frames": out / "frames",
        "index": out / "frames_index.csv",
        "det": out / "detections.csv",
        "summary": out / "daily_summary.csv",
        "bouts": out / "bouts.csv",
        "preview": out / "preview",
    }


def cmd_init_config(args):
    p = Path(args.config)
    if p.exists() and not args.force:
        print(f"{p} already exists (use --force to overwrite)")
        return
    write_default_config(p)
    print(f"wrote {p}")


def cmd_organize(args):
    cfg = load_config(Path(args.config))
    P = _paths(args)
    videos, manifest = organize.build_manifest(Path(args.videos), P["out"])
    if not videos:
        print("No videos matching CH<n>_<start>-<end>.<ext> were found.")
        return
    print(f"{len(videos)} videos -> {manifest}")
    cov = organize.coverage_report(videos, P["out"])
    print(f"coverage -> {cov}")
    if args.by_date:
        n = organize.organize_by_date(videos, P["out"] / "by-date", args.mode)
        print(f"{n} videos placed under {P['out'] / 'by-date'} ({args.mode})")
    for v in videos:
        if v.path.stat().st_size == 0:
            print(f"  ! zero-byte file, cannot be processed: {v.path.name}")
    return cfg


def cmd_preview(args):
    cfg = load_config(Path(args.config))
    P = _paths(args)
    videos, _ = scan_videos(Path(args.videos))
    videos = [v for v in videos if v.path.stat().st_size > 0]
    seen: set[str] = set()
    at = [float(x) for x in args.at.split(",")]
    for v in videos:
        if v.channel in seen:
            continue
        seen.add(v.channel)
        written = preview.preview_video(v, cfg, P["preview"], at)
        for p in written:
            print(f"  {p}")
    print(f"\nCheck these images, then edit 'crops' in {args.config} if the boxes are off.")


def cmd_extract(args):
    cfg = load_config(Path(args.config))
    P = _paths(args)
    videos, _ = scan_videos(Path(args.videos))
    if args.channel:
        videos = [v for v in videos if v.channel.upper() == args.channel.upper()]
    if args.date:
        videos = [v for v in videos if v.date == args.date]
    if not videos:
        print("no videos selected")
        return
    n = extract.extract_all(videos, P["frames"], cfg, P["index"],
                            args.limit_frames, args.backend)
    print(f"{n} sampled frames -> {P['frames']}\nindex -> {P['index']}")


def cmd_detect(args):
    cfg = load_config(Path(args.config))
    P = _paths(args)
    out = detect.run(P["index"], P["det"], cfg)
    print(f"detections -> {out}")


def cmd_calibrate(args):
    P = _paths(args)
    print(detect.calibrate(P["det"]))


def cmd_summarize(args):
    cfg = load_config(Path(args.config))
    P = _paths(args)
    s = aggregate.summarize(P["det"], P["summary"], cfg["sample_fps"])
    b = aggregate.bouts(P["det"], P["bouts"], cfg["sample_fps"])
    print(f"daily summary -> {s}\nbouts -> {b}\n")
    print(Path(s).read_text())


def cmd_all(args):
    cmd_organize(args)
    cmd_extract(args)
    cmd_detect(args)
    cmd_summarize(args)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cowvision", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--out", default="output", help="where manifests, frames and CSVs go")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init-config"); p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init_config)

    p = sub.add_parser("organize")
    p.add_argument("--videos", required=True)
    p.add_argument("--by-date", action="store_true")
    p.add_argument("--mode", default="link", choices=["link", "copy", "move"])
    p.set_defaults(func=cmd_organize)

    p = sub.add_parser("preview")
    p.add_argument("--videos", required=True)
    p.add_argument("--at", default="5,1800,5400", help="seconds into each video")
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("extract")
    p.add_argument("--videos", required=True)
    p.add_argument("--channel"); p.add_argument("--date")
    p.add_argument("--limit-frames", type=int)
    p.add_argument("--backend", default="auto", choices=["auto", "ffmpeg", "opencv"])
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("detect"); p.set_defaults(func=cmd_detect)
    p = sub.add_parser("calibrate"); p.set_defaults(func=cmd_calibrate)
    p = sub.add_parser("summarize"); p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("all")
    p.add_argument("--videos", required=True)
    p.add_argument("--by-date", action="store_true")
    p.add_argument("--mode", default="link", choices=["link", "copy", "move"])
    p.add_argument("--channel"); p.add_argument("--date")
    p.add_argument("--limit-frames", type=int)
    p.add_argument("--backend", default="auto", choices=["auto", "ffmpeg", "opencv"])
    p.set_defaults(func=cmd_all)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
