"""Figures for the results presentation, built from daily_summary_all.csv.

  python report/make_figures.py [summary_csv] [minute_csv]

Partial days (< 20 observed hours) are drawn on the coverage chart but kept out
of the trend, because a 6-hour day and a 24-hour day are not comparable.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

OUT = Path(__file__).parent / "figures"
FULL_DAY_H = 20.0

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
SERIES = {"left": "#2a78d6", "right": "#eb6834"}  # categorical slots 1 and 2
POSTURE = {"lying": "#2a78d6", "standing": "#eda100"}
LABEL = {"left": "CH1 left calf", "right": "CH1 right calf"}


def style(ax, title, ylabel, subtitle=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=13, fontweight="bold", loc="left",
                 pad=20 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK2, fontsize=9.5, va="bottom")
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def load(summary_csv):
    merged = defaultdict(lambda: {"frames": 0, "det": 0, "lying": 0.0, "standing": 0.0})
    for r in csv.DictReader(Path(summary_csv).open()):
        m = merged[(r["date"], r["side"])]
        m["frames"] += int(r["frames"])
        m["det"] += int(r["frames_with_calf"])
        m["lying"] += float(r["lying_hours"])
        m["standing"] += float(r["standing_hours"])
    days = {}
    for (date, side), m in merged.items():
        obs = m["lying"] + m["standing"]
        days[(date, side)] = {
            "observed": obs,
            "lying": m["lying"],
            "standing": m["standing"],
            "lying_pct": 100 * m["lying"] / obs if obs else None,
            "det_rate": 100 * m["det"] / m["frames"] if m["frames"] else 0,
        }
    return days


def short(d):
    return datetime.strptime(d, "%Y-%m-%d").strftime("%b %d")


def fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    b = sxy / sxx
    r = sxy / (sxx * syy) ** 0.5
    return b, my - b * mx, r


def fig_trend(days, dates, partial_dates):
    """Lying % per day. Real date axis, so missing days leave a visible gap."""
    fig, ax = plt.subplots(figsize=(11, 5.4), dpi=200, facecolor=SURFACE)
    day0 = datetime.strptime(dates[0], "%Y-%m-%d")
    xof = lambda d: (datetime.strptime(d, "%Y-%m-%d") - day0).days  # noqa: E731
    span = xof(dates[-1])
    fits = {}
    for side in ("left", "right"):
        xs = [xof(d) for d in dates]
        ys = [days[(d, side)]["lying_pct"] for d in dates]
        # break the line across dates with no full-day data
        px, py = [], []
        for x, y in zip(xs, ys):
            if px and x - px[-1] > 1:
                px.append(None)
                py.append(None)
            px.append(x)
            py.append(y)
        b, a, r = fit(xs, ys)
        fits[side] = (b, r)
        ax.plot([0, span], [a, a + b * span], color=SERIES[side], linewidth=1.2,
                linestyle=(0, (5, 4)), alpha=0.55, zorder=2)
        ax.plot(px, py, color=SERIES[side], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    for d in partial_dates:
        for side in ("left", "right"):
            if (d, side) in days and days[(d, side)]["lying_pct"] is not None:
                ax.plot(xof(d), days[(d, side)]["lying_pct"], marker="o", markersize=4,
                        color=MUTED, zorder=3)
    # de-collided end labels
    ends = sorted((days[(dates[-1], s)]["lying_pct"], s) for s in ("left", "right"))
    offsets = {ends[0][1]: -9, ends[1][1]: 9}
    for side in ("left", "right"):
        ax.annotate(LABEL[side], (span, days[(dates[-1], side)]["lying_pct"]),
                    xytext=(12, offsets[side]), textcoords="offset points",
                    color=INK, fontsize=10, va="center", fontweight="bold")
    left_b, left_r = fits["left"]
    right_b, right_r = fits["right"]
    style(ax, "Daily lying time swings widely; only the right calf drifts downward",
          "% of observed time spent lying",
          f"Dashed = least-squares fit. Left {left_b:+.2f} %/day (r={left_r:+.2f}, no trend), "
          f"right {right_b:+.2f} %/day (r={right_r:+.2f}). Grey = partial day.")
    ax.set_ylim(40, 95)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ticks = [xof(d) for d in dates]
    ax.set_xticks(ticks)
    ax.set_xticklabels([short(d) for d in dates], rotation=45, ha="right", fontsize=8.5)
    ax.set_xlim(-0.8, span + 4.2)
    fig.tight_layout()
    fig.savefig(OUT / "trend_lying_pct.png", facecolor=SURFACE)
    plt.close(fig)


def fig_hours(days, dates):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), dpi=200, facecolor=SURFACE, sharex=True)
    xs = list(range(len(dates)))
    for ax, side in zip(axes, ("left", "right")):
        ly = [days[(d, side)]["lying"] for d in dates]
        st = [days[(d, side)]["standing"] for d in dates]
        ax.bar(xs, ly, width=0.62, color=POSTURE["lying"], label="lying", zorder=3)
        ax.bar(xs, st, width=0.62, bottom=[v + 0.12 for v in ly], color=POSTURE["standing"],
               label="standing", zorder=3)
        style(ax, LABEL[side], "hours per day")
        ax.set_ylim(0, 26)
        ax.yaxis.set_major_locator(MultipleLocator(6))
    axes[0].legend(frameon=False, ncol=2, fontsize=9, labelcolor=INK2,
                   loc="lower right", bbox_to_anchor=(1, 1.01))
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([short(d) for d in dates], rotation=45, ha="right")
    fig.suptitle("Daily lying and standing hours", color=INK, fontsize=13,
                 fontweight="bold", x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(OUT / "daily_hours.png", facecolor=SURFACE)
    plt.close(fig)


def fig_coverage(days, all_dates):
    fig, ax = plt.subplots(figsize=(11, 3.8), dpi=200, facecolor=SURFACE)
    xs = list(range(len(all_dates)))
    obs = [max(days.get((d, "left"), {}).get("observed", 0),
               days.get((d, "right"), {}).get("observed", 0)) for d in all_dates]
    colors = [SERIES["left"] if o >= FULL_DAY_H else MUTED for o in obs]
    ax.bar(xs, obs, width=0.62, color=colors, zorder=3)
    style(ax, "Recorded coverage per day", "observed hours",
          "Grey = partial day, left out of the trend. Aug 20-26 and Aug 28-29 have no footage at all.")
    ax.axhline(24, color=GRID, linewidth=1, zorder=2)
    ax.set_ylim(0, 26)
    ax.yaxis.set_major_locator(MultipleLocator(6))
    ax.set_xticks(xs)
    ax.set_xticklabels([short(d) for d in all_dates], rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "coverage.png", facecolor=SURFACE)
    plt.close(fig)


def fig_diurnal(minute_csv):
    """% of minutes lying, by hour of day, pooled over every day."""
    buckets = defaultdict(lambda: [0, 0])
    for r in csv.DictReader(Path(minute_csv).open()):
        if r["activity"] not in ("lying", "standing"):
            continue
        hour = int(r["timestamp"][11:13])
        b = buckets[(r["side"], hour)]
        b[0] += 1
        b[1] += r["activity"] == "lying"
    if not buckets:
        return
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=200, facecolor=SURFACE)
    hours = list(range(24))
    ax.axvspan(-0.5, 5, color=GRID, alpha=0.6, zorder=1)
    ax.axvspan(21, 23.5, color=GRID, alpha=0.6, zorder=1)
    for side in ("left", "right"):
        ys = [100 * buckets[(side, h)][1] / buckets[(side, h)][0] if buckets[(side, h)][0] else None
              for h in hours]
        ax.plot(hours, ys, color=SERIES[side], linewidth=2, marker="o", markersize=4,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        ax.annotate(LABEL[side], (hours[-1], ys[-1]), xytext=(10, 0), textcoords="offset points",
                    color=INK, fontsize=10, va="center", fontweight="bold")
    style(ax, "Both calves drop sharply twice a day, around 07:00 and 19:00",
          "% of minutes lying",
          "Every minute of all 25 dates pooled by hour. Shaded = 21:00-05:00.")
    ax.text(2.4, 97, "night", color=MUTED, fontsize=9, va="top")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.set_xlim(-0.5, 27.5)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    fig.tight_layout()
    fig.savefig(OUT / "diurnal.png", facecolor=SURFACE)
    plt.close(fig)


def main(summary="daily_summary_all.csv", minute=None):
    OUT.mkdir(parents=True, exist_ok=True)
    days = load(summary)
    all_dates = sorted({d for d, _ in days})
    full = [d for d in all_dates
            if all(days.get((d, s), {}).get("observed", 0) >= FULL_DAY_H for s in ("left", "right"))]
    partial = [d for d in all_dates if d not in full]
    fig_trend(days, full, partial)
    fig_hours(days, full)
    fig_coverage(days, all_dates)
    if minute and Path(minute).exists():
        fig_diurnal(minute)
    print(f"{len(all_dates)} dates, {len(full)} full days -> {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print("  ", p.name)


if __name__ == "__main__":
    main(*sys.argv[1:])
