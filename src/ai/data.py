"""Datasets for the segmentation model: synthetic phantoms and (optionally) BUSI."""
from __future__ import annotations

import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.ai.normalize import normalize_bmode
from src.simulation.ultrasound import fast_bmode_phantom


class SyntheticLesions(Dataset):
    """Infinite-ish stream of labelled speckle phantoms (deterministic per index)."""

    def __init__(self, n=2000, shape=(256, 128), seed=0, augment=True, enhance_prob=0.7, return_label=False):
        # The deployed pipeline feeds the model *enhanced* frames, so most training
        # frames go through the same enhancement chain (train/inference match).
        self.n, self.shape, self.seed, self.augment, self.enhance_prob = n, shape, seed, augment, enhance_prob
        self._enh = None
        self.return_label = return_label      # label -1 = "no class" (ignored by the classification loss)

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
        img = normalize_bmode(img)
        if self.augment and rng.random() < 0.5:
            img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
        out = (torch.from_numpy(img)[None], torch.from_numpy(mask.astype(np.float32))[None])
        return out + (torch.tensor(-1),) if self.return_label else out


class BUSI(Dataset):
    """Breast Ultrasound Images (Al-Dhabyani 2020) - ``Dataset_BUSI_with_GT``.

    Layout: <root>/{benign,malignant,normal}/<name>.png with masks <name>_mask.png
    (extra lesions: <name>_mask_1.png ...; merged).  'normal' images get empty masks.
    Images are resized to ``shape`` (default 256x256; the model is fully
    convolutional, so it can also run on the pipeline's 256x128 frames).

    augment=True: random horizontal flip, 256x128 output (random crop or squeezed
    whole image - the pipeline's frame size), mild gamma/brightness jitter.
    """

    CLASSES = ("benign", "malignant", "normal")

    def __init__(self, root, shape=(256, 256), classes=CLASSES,
                 augment=False, crop_w=128, indices=None, return_label=False, crop_prob=0.7):
        self.shape, self.augment, self.crop_w = shape, augment, crop_w
        self.return_label, self.crop_prob = return_label, crop_prob
        self.classes = tuple(classes)
        items = []
        for cls in classes:
            for p in sorted((Path(root) / cls).glob("*.png")):
                if "_mask" in p.name:
                    continue
                masks = sorted(p.parent.glob(glob.escape(p.stem) + "_mask*.png"))
                items.append((p, masks, cls))
        if not items:
            raise FileNotFoundError(f"no BUSI images under {root} (expected benign/malignant/normal sub-folders)")
        self.items = items if indices is None else [items[i] for i in indices]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, masks, cls = self.items[i]
        h, w = self.shape
        img = cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), (w, h), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        m = np.zeros((h, w), np.float32)
        for mp in masks:
            mm = cv2.resize(cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE), (w, h), interpolation=cv2.INTER_NEAREST)
            m = np.maximum(m, (mm > 127).astype(np.float32))
        if self.augment:
            rng = np.random.default_rng()
            if rng.random() < 0.5:
                img, m = img[:, ::-1], m[:, ::-1]
            if self.crop_w and w > self.crop_w:               # always output (h, crop_w) so batches collate
                if rng.random() < self.crop_prob:
                    x0 = int(rng.integers(0, w - self.crop_w + 1))
                    img, m = img[:, x0:x0 + self.crop_w], m[:, x0:x0 + self.crop_w]
                else:                                           # whole image squeezed to the pipeline aspect
                    img = cv2.resize(np.ascontiguousarray(img), (self.crop_w, h), interpolation=cv2.INTER_AREA)
                    m = (cv2.resize(np.ascontiguousarray(m), (self.crop_w, h), interpolation=cv2.INTER_AREA) > 0.5).astype(np.float32)
            img = np.clip(img ** rng.uniform(0.8, 1.25) + rng.uniform(-0.08, 0.08), 0, 1)
        img = normalize_bmode(np.ascontiguousarray(img, dtype=np.float32))   # same per-frame normalisation as inference
        out = (torch.from_numpy(img)[None], torch.from_numpy(np.ascontiguousarray(m))[None])
        return out + (torch.tensor(self.classes.index(cls)),) if self.return_label else out

    def labels(self):
        return [self.classes.index(c) for _, _, c in self.items]
