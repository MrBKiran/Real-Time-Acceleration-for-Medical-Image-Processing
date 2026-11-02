"""Plotting + video helpers."""
from __future__ import annotations

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def show_panels(images, titles, path, extent=None, cmap="gray", overlay=None, figsize_per=(3.2, 5.2)):
    n = len(images)
    fig, ax = plt.subplots(1, n, figsize=(figsize_per[0] * n, figsize_per[1]))
    for a, im, t in zip(np.atleast_1d(ax), images, titles):
        a.imshow(im, cmap=cmap, vmin=0, vmax=1, extent=extent, aspect="equal")
        if overlay is not None:
            a.contour(np.linspace(extent[0], extent[1], overlay.shape[1]) if extent else np.arange(overlay.shape[1]),
                      np.linspace(extent[3], extent[2], overlay.shape[0]) if extent else np.arange(overlay.shape[0]),
                      overlay, levels=[0.5], colors="r", linewidths=0.8)
        a.set_title(t, fontsize=9)
        a.set_xlabel("x (mm)" if extent else "px")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pipeline_diagram(path, stage_ms=None):
    """Draw the operator-flow diagram (optionally annotated with measured ms)."""
    stages = [("RF source\n(ADC / dataset /\nsimulator)", "#dfe8f5"),
              ("Beamform\nDAS plane-wave\ncompounding\n(CUDA via PyTorch)", "#d5ecd4"),
              ("Envelope +\nlog compress", "#d5ecd4"),
              ("Enhance\nmedian / bilateral /\nCLAHE", "#d5ecd4"),
              ("Inference\nTinyUNet lesion\nsegmentation", "#fde7c8"),
              ("Sink\noverlay / video /\nHoloviz", "#ecdcf0")]
    keys = ["rf_source", "beamform", None, "enhance", "inference", "sink"]
    fig, ax = plt.subplots(figsize=(13, 2.6))
    ax.axis("off")
    w, g = 1.7, 0.45
    for i, ((txt, col), k) in enumerate(zip(stages, keys)):
        x = i * (w + g)
        ax.add_patch(plt.Rectangle((x, 0), w, 1.6, fc=col, ec="#444", lw=1.2))
        ax.text(x + w / 2, 0.95, txt, ha="center", va="center", fontsize=8.5)
        if stage_ms and k in stage_ms:
            ax.text(x + w / 2, 0.15, f"{stage_ms[k]:.1f} ms", ha="center", fontsize=9, weight="bold", color="#a02020")
        if i < len(stages) - 1:
            ax.annotate("", (x + w + g, 0.8), (x + w, 0.8), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set_xlim(-0.1, len(stages) * (w + g)); ax.set_ylim(-0.1, 1.8)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _tile(img, size, color=False):
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    u8 = cv2.resize(u8, size, interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)


def write_demo_video(frames, path, fps=10, scale=3, text=None):
    """frames: list of dict(bmode_raw, bmode, prob). Side-by-side: raw | enhanced | enhanced+AI."""
    H, W = frames[0]["bmode"].shape
    size = (W * scale, H * scale)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size[0] * 3, size[1] + 28))
    for f in frames:
        a = _tile(f["bmode_raw"], size)
        b = _tile(f["bmode"], size)
        c = _tile(f["bmode"], size)
        if "prob" in f:
            m = cv2.resize((f["prob"] > 0.5).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            tint = c.copy(); tint[m > 0] = (0.5 * tint[m > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
            c = tint
            cv2.drawContours(c, cnts, -1, (0, 255, 255), 1)
        row = np.hstack([a, b, c])
        bar = np.zeros((28, row.shape[1], 3), np.uint8)
        for i, t in enumerate(["Beamformed B-mode", "Enhanced", "AI lesion segmentation"]):
            cv2.putText(bar, t, (i * size[0] + 8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if text:
            cv2.putText(row, text(f), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        vw.write(np.vstack([row, bar]))
    vw.release()
