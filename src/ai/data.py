"""Datasets for the segmentation model: synthetic phantoms and (optionally) BUSI."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.simulation.ultrasound import fast_bmode_phantom


class SyntheticLesions(Dataset):
    """Infinite-ish stream of labelled speckle phantoms (deterministic per index)."""

    def __init__(self, n=2000, shape=(256, 128), seed=0, augment=True, enhance_prob=0.7):
        # The deployed pipeline feeds the model *enhanced* frames, so most training
        # frames go through the same enhancement chain (train/inference match).
        self.n, self.shape, self.seed, self.augment, self.enhance_prob = n, shape, seed, augment, enhance_prob
        self._enh = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        img, mask, _ = fast_bmode_phantom(*self.shape, n_lesions=int(rng.integers(0, 3)), rng=rng)
        if rng.random() < self.enhance_prob:
            if self._enh is None:
                from src.enhancement.filters import Enhancer
                self._enh = Enhancer()
            img = self._enh(torch.from_numpy(img)).numpy().astype(np.float32)
        if self.augment and rng.random() < 0.5:
            img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
        return torch.from_numpy(img)[None], torch.from_numpy(mask.astype(np.float32))[None]


class BUSI(Dataset):
    """Breast Ultrasound Images (Al-Dhabyani 2020) - ``Dataset_BUSI_with_GT``.

    Layout: <root>/{benign,malignant,normal}/<name>.png and <name>_mask.png
    (several ``_mask_N.png`` are merged).  Images are resized to ``shape``.
    """

    def __init__(self, root, shape=(256, 128), classes=("benign", "malignant", "normal")):
        self.shape = shape
        self.items = []
        for cls in classes:
            for p in sorted((Path(root) / cls).glob("*.png")):
                if "_mask" in p.name:
                    continue
                masks = sorted(p.parent.glob(p.stem + "_mask*.png"))
                self.items.append((p, masks, cls))
        if not self.items:
            raise FileNotFoundError(f"no BUSI images under {root}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, masks, cls = self.items[i]
        h, w = self.shape
        img = cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), (w, h), interpolation=cv2.INTER_AREA)
        m = np.zeros((h, w), np.float32)
        for mp in masks:
            mm = cv2.resize(cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE), (w, h), interpolation=cv2.INTER_NEAREST)
            m = np.maximum(m, (mm > 127).astype(np.float32))
        return torch.from_numpy(img.astype(np.float32) / 255.0)[None], torch.from_numpy(m)[None]
