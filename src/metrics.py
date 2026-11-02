"""Image-quality and timing metrics."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


def cnr(img, inside, outside):
    a, b = img[inside], img[outside]
    return float(abs(a.mean() - b.mean()) / np.sqrt(a.var() + b.var() + 1e-12))


def gcnr(img, inside, outside, bins=64):
    """Generalised CNR (Rodriguez-Molares 2020): 1 - histogram overlap, in [0,1]."""
    lo, hi = float(img.min()), float(img.max())
    ha, _ = np.histogram(img[inside], bins, (lo, hi), density=True)
    hb, _ = np.histogram(img[outside], bins, (lo, hi), density=True)
    w = (hi - lo) / bins
    return float(1 - np.minimum(ha, hb).sum() * w)


def speckle_snr(env, region):
    """mean/std of the linear envelope; ~1.91 for fully developed speckle."""
    r = env[region]
    return float(r.mean() / (r.std() + 1e-12))


def ring_mask(mask, width=6):
    from scipy.ndimage import binary_dilation
    m = mask.astype(bool)
    return binary_dilation(m, iterations=width + 3) & ~binary_dilation(m, iterations=3)


def dice(pred, gt, eps=1e-7):
    p, g = pred.astype(bool), gt.astype(bool)
    return float((2 * (p & g).sum() + eps) / (p.sum() + g.sum() + eps))


def iou(pred, gt, eps=1e-7):
    p, g = pred.astype(bool), gt.astype(bool)
    return float(((p & g).sum() + eps) / ((p | g).sum() + eps))


@dataclass
class LatencyStats:
    samples: dict = field(default_factory=dict)

    def add(self, name, seconds):
        self.samples.setdefault(name, []).append(seconds)

    def summary(self, warmup=2):
        out = {}
        for k, v in self.samples.items():
            a = np.asarray(v[warmup:] if len(v) > warmup + 2 else v) * 1e3
            out[k] = dict(mean_ms=float(a.mean()), p50_ms=float(np.percentile(a, 50)),
                          p95_ms=float(np.percentile(a, 95)), max_ms=float(a.max()), n=int(a.size))
        return out


class Timer:
    def __init__(self, sync=None):
        self.sync = sync

    def __enter__(self):
        if self.sync:
            self.sync()
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        if self.sync:
            self.sync()
        self.dt = time.perf_counter() - self.t
