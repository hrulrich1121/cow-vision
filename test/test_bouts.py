"""Unit tests for the bout logic and the clock-window extraction filter.

    PYTHONPATH=. python test/test_bouts.py

The bout counts the researcher gets are entirely a product of the merge rule and
the minimum-bout threshold, so each of those gets a case here with a hand-worked
expected answer.
"""
from __future__ import annotations

import csv
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from cowvision import aggregate
from cowvision.extract import _in_windows, parse_windows

OUT = Path(tempfile.mkdtemp(prefix="cvbouts"))
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, got, want) -> None:
    RESULTS.append((name, got == want, f"got {got!r}, want {want!r}"))


def write_minutes(path: Path, seq: str, start="2026-07-29 00:00:00", step_min=1):
    """seq is one letter per sample: l=lying, s=standing, e=eating, d=drinking,
    u=unknown, and '.' leaves a gap in the recording."""
    code = {"l": "lying", "s": "standing", "e": "eating", "d": "drinking", "u": "unknown"}
    t0 = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["channel", "side", "timestamp", "activity"])
        w.writeheader()
        for i, c in enumerate(seq):
            if c == ".":
                continue
            w.writerow({"channel": "CH1", "side": "left", "activity": code[c],
                        "timestamp": (t0 + timedelta(minutes=i * step_min))
                                     .strftime("%Y-%m-%d %H:%M:%S")})
    return path


def bouts_of(seq: str, **kw) -> list[dict]:
    src = write_minutes(OUT / "m.csv", seq)
    aggregate.bouts(src, OUT / "b.csv", None, kw.get("min_bout_s"), kw.get("max_gap_s"))
    return list(csv.DictReader((OUT / "b.csv").open()))


def main() -> int:
    floor = {"lying": 300, "standing": 300, "eating": 30, "drinking": 30}  # 5 min

    # 10 lying, 10 standing, 10 lying -> three bouts, nothing to merge
    b = bouts_of("l" * 10 + "s" * 10 + "l" * 10, min_bout_s=floor)
    check("plain alternation -> 3 bouts", len(b), 3)
    check("  first is lying", b[0]["activity"], "lying")
    check("  10 samples = 10 min", b[0]["duration_min"], "10.0")

    # a single standing minute inside a long lie is flicker: it must be absorbed,
    # leaving ONE lying bout rather than two separated by a 1-min standing bout
    b = bouts_of("l" * 10 + "s" + "l" * 10, min_bout_s=floor)
    check("1-min flip absorbed -> 1 bout", len(b), 1)
    check("  spans the whole run", b[0]["duration_min"], "21.0")

    # the same flip survives when the floor is low enough to admit it
    b = bouts_of("l" * 10 + "s" + "l" * 10, min_bout_s={k: 60 for k in floor})
    check("same flip kept at a 1-min floor", len(b), 3)

    # a brief undetected patch should not end a bout either
    b = bouts_of("l" * 10 + "u" * 2 + "l" * 10, min_bout_s=floor)
    check("short unknown patch absorbed", len(b), 1)

    # but a real break in the recording always splits, however short
    b = bouts_of("l" * 10 + "." * 10 + "l" * 10, min_bout_s=floor)
    check("recording gap splits the bout", len(b), 2)

    # eating has its own lower floor, so a 2-minute visit survives
    b = bouts_of("l" * 10 + "e" * 2 + "l" * 10, min_bout_s=floor)
    check("2-min eating visit kept", [r["activity"] for r in b],
          ["lying", "eating", "lying"])

    # per-day counts
    src = write_minutes(OUT / "m.csv", ("l" * 30 + "s" * 10) * 5)
    aggregate.bouts(src, OUT / "b.csv", None, floor)
    aggregate.bout_summary(OUT / "b.csv", OUT / "s.csv")
    row = next(csv.DictReader((OUT / "s.csv").open()))
    check("bout_summary lying_bouts", row["lying_bouts"], "5")
    check("bout_summary standing_bouts", row["standing_bouts"], "5")
    check("bout_summary lying_min", row["lying_min"], "150.0")
    check("bout_summary mean bout", row["mean_lying_bout_min"], "30.0")

    # ---- clock windows ----------------------------------------------------
    w = parse_windows("06:30-08:00,18:30-20:00")
    at = lambda s: datetime.strptime("2026-07-29 " + s, "%Y-%m-%d %H:%M:%S")  # noqa: E731
    check("06:29 outside", _in_windows(at("06:29:59"), w), False)
    check("06:30 inside (start is inclusive)", _in_windows(at("06:30:00"), w), True)
    check("08:00 outside (end is exclusive)", _in_windows(at("08:00:00"), w), False)
    check("19:15 inside", _in_windows(at("19:15:00"), w), True)
    check("midnight-wrapping window", 
          [_in_windows(at(t), parse_windows("22:00-02:00"))
           for t in ("23:00:00", "01:00:00", "12:00:00")], [True, True, False])
    check("no windows = keep everything", _in_windows(at("12:00:00"), []), True)

    for name, good, detail in RESULTS:
        print(f"{'PASS' if good else 'FAIL'}  {name}" + ("" if good else f"  -- {detail}"))
    bad = sum(1 for _, g, _ in RESULTS if not g)
    print(f"\n{len(RESULTS) - bad}/{len(RESULTS)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
