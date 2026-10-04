"""cowvision - dairy calf video pipeline.

  cowvision init-config
  cowvision organize   --videos <dir> [--out <dir>] [--by-date] [--mode link|copy|move]
  cowvision preview    --videos <dir> [--at 0,600,3600]
  cowvision extract    --videos <dir> [--limit-frames N] [--between 06:30-08:00]
  cowvision detect
  cowvision calibrate
  cowvision summarize
  cowvision minutely   [--run <dir>]   per-minute activity CSV, for validation
  cowvision bouts      [--run <dir>]   bouts.csv + bout_summary.csv (counts per day)
  cowvision labelset   --out <dir> --upload <dir> [--target N]   frames to hand-label
  cowvision combine    [--run <dir>]   merge per-date summaries into one CSV
  cowvision all        --videos <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import aggregate, detect, extract, labelset, minutely, organize, preview
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
        "bout_summary": out / "bout_summary.csv",
        "minutes": out / "minute_activity.csv",
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
    windows = extract.parse_windows(getattr(args, "between", None))
    if windows:
        print("  keeping frames inside " + ", ".join(
            f"{a.strftime('%H:%M')}-{b.strftime('%H:%M')}" for a, b in windows))
    n = extract.extract_all(videos, P["frames"], cfg, P["index"],
                            args.limit_frames, args.backend, windows)
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


def _run_root(args) -> Path:
    return Path(args.run) if args.run else Path(args.out)


def cmd_minutely(args):
    root = _run_root(args)
    dets = minutely.find_detections(root)
    if not dets:
        print(f"no detections.csv under {root}")
        return
    out = Path(args.output) if args.output else root / "minute_activity.csv"
    out, n = minutely.minutely(dets, out)
    print(f"{len(dets)} detections.csv -> {n} minute rows")
    print(out)


def cmd_bouts(args):
    """Bouts for a whole run: per-date detections.csv, or one minute_activity.csv.

    The per-minute file is enough to count bouts and is small enough to work on
    off the server, so it is accepted as a source in its own right.
    """
    cfg = load_config(Path(args.config))
    bc = cfg.get("bouts", {})
    root = _run_root(args)
    src = Path(args.source) if args.source else None
    if src is None:
        dets = minutely.find_detections(root)
        if len(dets) == 1:
            src = dets[0]
        elif dets:
            src = root / "detections_all.csv"
            _concat(dets, src)
        elif (root / "minute_activity.csv").exists():
            src = root / "minute_activity.csv"
        else:
            print(f"no detections.csv or minute_activity.csv under {root}")
            return
    out = Path(args.output) if args.output else root / "bouts.csv"
    b = aggregate.bouts(src, out, None, bc.get("min_bout_s"), bc.get("max_gap_s"),
                        bc.get("max_unknown_s"))
    summary = out.with_name("bout_summary.csv")
    aggregate.bout_summary(b, summary, src if src.name == "minute_activity.csv" else None)
    print(f"source {src}")
    print(f"bouts -> {b}")
    print(f"per-day counts -> {summary}")


def _concat(srcs: list[Path], dest: Path) -> Path:
    """Join per-date CSVs, keeping one header - the bout finder needs one stream.

    Streamed rather than collected: a full corpus run is several million rows,
    and `bouts()` sorts each calf's track itself, so sorting here would only
    double the memory for nothing.
    """
    import csv as _csv

    w = None
    with dest.open("w", newline="") as out:
        for s in srcs:
            with Path(s).open() as f:
                r = _csv.DictReader(f)
                if w is None:
                    w = _csv.DictWriter(out, fieldnames=r.fieldnames)
                    w.writeheader()
                w.writerows(r)
    return dest


def cmd_labelset(args):
    cfg = load_config(Path(args.config))
    P = _paths(args)
    manifest, stats = labelset.select(
        P["index"], Path(args.upload),
        P["det"] if P["det"].exists() else None,
        cfg.get("zones"), args.target, args.dup_thresh, args.seed)
    print(f"{stats['frames_in']} frames -> {stats['after_dedup']} after near-duplicate "
          f"removal -> {stats['selected']} selected")
    print()
    print("  group (side/posture/where)        picked / available")
    for g, avail in stats["available_per_group"].items():
        print(f"  {g:34s} {stats['per_group'][g]:5d} / {avail}")
    print()
    print(f"upload {Path(args.upload)} to Roboflow; manifest -> {manifest}")


def cmd_combine(args):
    root = _run_root(args)
    files = sorted(root.glob("*/daily_summary.csv"))
    if not files:
        print(f"no per-date daily_summary.csv under {root}")
        return
    out = Path(args.output) if args.output else root / "daily_summary_all.csv"
    print(f"{len(files)} files -> {aggregate.combine(files, out)}")


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
    p.add_argument("--between", help="clock windows to keep, e.g. 06:30-08:00,18:30-20:00")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("detect"); p.set_defaults(func=cmd_detect)
    p = sub.add_parser("calibrate"); p.set_defaults(func=cmd_calibrate)
    p = sub.add_parser("summarize"); p.set_defaults(func=cmd_summarize)

    for name, fn in (("minutely", cmd_minutely), ("combine", cmd_combine),
                     ("bouts", cmd_bouts)):
        p = sub.add_parser(name)
        p.add_argument("--run", help="run directory holding per-date results (default --out)")
        p.add_argument("--output", help="output csv path")
        if name == "bouts":
            p.add_argument("--source", help="detections.csv or minute_activity.csv to read")
        p.set_defaults(func=fn)

    p = sub.add_parser("labelset")
    p.add_argument("--upload", required=True, help="directory to fill with frames to label")
    p.add_argument("--target", type=int, default=600, help="roughly how many frames")
    p.add_argument("--dup-thresh", type=float, default=6.0,
                   help="mean grey difference below which two frames count as duplicates")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_labelset)

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
