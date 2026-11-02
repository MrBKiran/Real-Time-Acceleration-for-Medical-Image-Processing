"""Speckle reduction / contrast enhancement - the open-source equivalents of
medfilt2, wiener2, imbilatfilt and adapthisteq.

Every filter takes and returns a float image in [0, 1].  ``*_t`` variants run on
torch tensors (GPU capable) and are what the real-time pipeline uses; the
``*_cpu`` variants use SciPy/OpenCV and serve as reference implementations.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import wiener as _sp_wiener
from scipy.ndimage import median_filter

# ----------------------------- CPU reference ---------------------------------

def median_cpu(img, k=3):
    return median_filter(img, size=k, mode="nearest")


def wiener_cpu(img, k=5):
    return np.clip(_sp_wiener(img.astype(np.float64), (k, k)), 0, 1).astype(np.float32)


def bilateral_cpu(img, k=5, sigma_s=2.0, sigma_r=0.1):
    return cv2.bilateralFilter(img.astype(np.float32), k, sigma_r, sigma_s)


def clahe_cpu(img, clip=2.0, tiles=(4, 4)):
    u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    out = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles).apply(u8)
    return out.astype(np.float32) / 255.0

# ----------------------------- torch (GPU capable) ---------------------------

def _windows(x, k):
    """(1,1,H,W) -> (H*W, k*k) sliding windows with reflect padding."""
    p = k // 2
    xp = F.pad(x, (p, p, p, p), mode="replicate")
    return F.unfold(xp, k).squeeze(0).T


def median_t(img: torch.Tensor, k=3):
    H, W = img.shape
    return _windows(img[None, None], k).median(dim=1).values.view(H, W)


def wiener_t(img: torch.Tensor, k=5):
    x = img[None, None]
    p = k // 2
    mean = F.avg_pool2d(F.pad(x, (p,) * 4, mode="replicate"), k, 1)
    var = (F.avg_pool2d(F.pad(x * x, (p,) * 4, mode="replicate"), k, 1) - mean ** 2).clamp_min(0)
    noise = var.mean()
    out = mean + (var - noise).clamp_min(0) / var.clamp_min(noise) * (x - mean)
    return out[0, 0].clamp(0, 1)


def bilateral_t(img: torch.Tensor, k=5, sigma_s=2.0, sigma_r=0.1):
    H, W = img.shape
    win = _windows(img[None, None], k)                         # (HW, k*k)
    r = (torch.arange(k, device=img.device) - k // 2).float()
    ys, xs = torch.meshgrid(r, r, indexing="ij")
    ws = torch.exp(-(ys ** 2 + xs ** 2) / (2 * sigma_s ** 2)).reshape(1, -1)
    center = img.reshape(-1, 1)
    wr = torch.exp(-((win - center) ** 2) / (2 * sigma_r ** 2))
    w = ws * wr
    return ((w * win).sum(1) / w.sum(1)).view(H, W)

# ----------------------------- composite -------------------------------------

class Enhancer:
    """Configurable chain, e.g. ``Enhancer(["median", "bilateral", "clahe"])`` (default: median + bilateral)."""

    STEPS_T = {"median": median_t, "wiener": wiener_t, "bilateral": bilateral_t}

    def __init__(self, steps=("median", "bilateral"), **kw):
        self.steps = list(steps)
        self.kw = kw

    def __call__(self, img):
        is_t = isinstance(img, torch.Tensor)
        x = img if is_t else torch.as_tensor(np.asarray(img, np.float32))
        for s in self.steps:
            if s == "clahe":
                out = clahe_cpu(x.detach().cpu().numpy(), **self.kw.get("clahe", {}))
                x = torch.as_tensor(out, device=x.device)
            else:
                x = self.STEPS_T[s](x, **self.kw.get(s, {}))
        return x if is_t else x.cpu().numpy()
