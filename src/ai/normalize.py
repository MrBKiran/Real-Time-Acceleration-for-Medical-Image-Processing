import numpy as np
import torch


def normalize_bmode(img, level=0.7):
    """Per-frame level normalisation: shift so the median grey level is ``level``.

    Makes the segmenter invariant to global gain / dynamic-range offsets (which differ
    between simulators, scanners and the choice of log-compression reference).
    Works on torch tensors and numpy arrays.
    """
    if isinstance(img, torch.Tensor):
        return (img - img.median() + level).clamp(0, 1)
    return np.clip(img - np.median(img) + level, 0, 1).astype(np.float32)
