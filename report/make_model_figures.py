"""Model figures for the presentation: why COCO failed, training curves, confusion matrix.

  python report/make_model_figures.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

OUT = Path(__file__).parent / "figures"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"
GREEN = "#1baf7a"

SAMPLE = Path("../sample/roboflow_upload")


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


def fig_coco_vs_ours():
    """Same two crops, COCO weights vs the trained model."""
    from ultralytics import YOLO

    picks = [
        ("CH1_left_20260728_152014_000001.jpg", "left pen"),
        ("CH1_right_20260728_152014_000128.jpg", "right pen"),
    ]
    picks = [(n, t) for n, t in picks if (SAMPLE / n).exists()]
    if len(picks) < 2:
        cands = sorted(SAMPLE.glob("CH1_left_20260728_152014_*.jpg"))[:1] + \
                sorted(SAMPLE.glob("CH1_right_20260728_152014_*.jpg"))[:1]
        picks = [(p.name, f"{p.name.split('_')[1]} pen") for p in cands]

    coco, ours = YOLO("yolov8n.pt"), YOLO("models/calf_posture_v2.pt")
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 8.0), dpi=200, facecolor=SURFACE)
    for col, (name, title) in enumerate(picks):
        img = cv2.imread(str(SAMPLE / name))
        for row, (model, tag, color) in enumerate(
                ((coco, "COCO yolov8n (off the shelf)", RED),
                 (ours, "our trained model", GREEN))):
            res = model.predict(str(SAMPLE / name), conf=0.10, verbose=False)[0]
            im = img.copy()
            labels = []
            order = sorted(range(len(res.boxes)), key=lambda i: -float(res.boxes.conf[i]))[:3]
            for i in order:
                x1, y1, x2, y2 = (int(v) for v in res.boxes.xyxy[i].tolist())
                cv2.rectangle(im, (x1, y1), (x2, y2), color[::-1] if False else (60, 60, 230)
                              if row == 0 else (120, 175, 27), 2)
                labels.append(f"{res.names[int(res.boxes.cls[i])]} {float(res.boxes.conf[i]):.2f}")
            ax = axes[row][col]
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID)
            ax.set_title(f"{title} - {tag}", color=INK, fontsize=9.5, loc="left", pad=6)
            ax.set_xlabel(", ".join(labels) if labels else "nothing found",
                          color=color, fontsize=9.5, labelpad=6, fontweight="bold")
    fig.suptitle("Off-the-shelf COCO weights do not see a calf from above",
                 color=INK, fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "coco_vs_ours.png", facecolor=SURFACE)
    plt.close(fig)


def fig_training():
    rows = list(csv.DictReader(open("runs/runs/v2_full/results.csv")))
    key = {k.strip(): k for k in rows[0]}
    ep = [int(r[key["epoch"]]) for r in rows]
    series = [
        ("metrics/precision(B)", "precision", BLUE),
        ("metrics/recall(B)", "recall", ORANGE),
        ("metrics/mAP50(B)", "mAP@50", GREEN),
        ("metrics/mAP50-95(B)", "mAP@50-95", "#4a3aa7"),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 4.8), dpi=200, facecolor=SURFACE)
    finals = []
    for col, label, color in series:
        ys = [float(r[key[col]]) for r in rows]
        ax.plot(ep, ys, color=color, linewidth=2, zorder=3)
        finals.append((ys[-1], label, color))
    # stack the end labels so near-identical scores do not overprint
    finals.sort(reverse=True)
    for k, (v, label, color) in enumerate(finals):
        crowded = [f for f in finals if abs(f[0] - v) < 0.04]
        dy = 0 if len(crowded) == 1 else 15 - 15 * crowded.index((v, label, color))
        ax.annotate(f"{label}  {v:.3f}", (ep[-1], v), xytext=(12, dy),
                    textcoords="offset points", color=INK, fontsize=10, va="center",
                    fontweight="bold")
    style(ax, "Training converges in under 20 epochs", "score on the validation set",
          "YOLOv8n, 704 training images, 80 epochs, ~2 min on one RTX 5070.")
    ax.set_xlabel("epoch", color=INK2, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, ep[-1] + 26)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", facecolor=SURFACE)
    plt.close(fig)


def fig_confusion(matrix, labels, title, subtitle, fname):
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=200, facecolor=SURFACE)
    n = len(labels)
    total = sum(sum(r) for r in matrix)
    for i in range(n):
        for j in range(n):
            v = matrix[i][j]
            good = i == j
            ax.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1,
                                       facecolor=(GREEN if good else RED) if v else "#f4f3f0",
                                       alpha=0.14 + 0.5 * (v / total if total else 0),
                                       edgecolor=SURFACE, linewidth=3))
            ax.text(j + 0.5, n - 0.5 - i, str(v), ha="center", va="center",
                    color=INK, fontsize=17, fontweight="bold" if v else "normal")
    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.set_xticks([j + 0.5 for j in range(n)]); ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([n - 0.5 - i for i in range(n)]); ax.set_yticklabels(labels, fontsize=10)
    ax.tick_params(colors=INK2, length=0)
    ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
    ax.set_xlabel("what the model said", color=INK2, fontsize=10, labelpad=10)
    ax.set_ylabel("what the human labelled", color=INK2, fontsize=10, labelpad=10)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.suptitle(title, color=INK, fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.905, subtitle, color=INK2, fontsize=9.5, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / fname, facecolor=SURFACE)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_training()
    fig_coco_vs_ours()
    # numbers from eval_posture.py on dataset_v2 (valid + test pooled)
    fig_confusion(
        [[7, 1, 1], [0, 49, 0], [0, 0, 27]],
        ["no calf", "lying", "standing"],
        "Not one lying/standing mix-up on 85 held-out frames",
        "Validation + test splits, human-labelled, never seen in training. conf 0.25.",
        "confusion.png",
    )
    for p in sorted(OUT.glob("*.png")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
