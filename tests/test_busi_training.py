import cv2
import numpy as np
import torch

from src.ai.data import BUSI
from src.ai.train import main, split_busi


def _fake_busi(root, n=12):
    rng = np.random.default_rng(0)
    for cls in ("benign", "malignant", "normal"):
        (root / cls).mkdir(parents=True)
        for i in range(1, n + 1):
            img = (rng.random((300, 400)) * 120 + 40).astype(np.uint8)
            name = f"{cls} ({i})"
            cv2.imwrite(str(root / cls / f"{name}.png"), img)
            mask = np.zeros_like(img)
            if cls != "normal":
                cv2.circle(mask, (200, 150), 40, 255, -1)
                cv2.imwrite(str(root / cls / f"{name}_mask.png"), mask)
                if i == 1:   # second lesion mask file, merged by the loader
                    m2 = np.zeros_like(img); cv2.circle(m2, (60, 60), 20, 255, -1)
                    cv2.imwrite(str(root / cls / f"{name}_mask_1.png"), m2)


def test_busi_loader_layout_masks_and_shapes(tmp_path):
    _fake_busi(tmp_path)
    ds = BUSI(tmp_path)
    assert len(ds) == 36                                  # masks are not counted as images
    x, y = ds[0]
    assert x.shape == (1, 256, 256) and y.shape == (1, 256, 256)
    assert y.sum() > 0 and y.max() == 1                    # both mask files merged
    normal = [i for i, it in enumerate(ds.items) if it[2] == "normal"][0]
    assert ds[normal][1].sum() == 0                        # normal images have empty masks
    aug = BUSI(tmp_path, augment=True)
    assert all(aug[i][0].shape == (1, 256, 128) for i in range(8))


def test_split_is_disjoint_and_train_all_runs(tmp_path):
    _fake_busi(tmp_path)
    tr, va = split_busi(tmp_path, 0.2, 0)
    assert not ({p for p, _, _ in tr.items} & {p for p, _, _ in va.items})
    out = tmp_path / "m.pt"
    main(["--data", "all", "--busi-root", str(tmp_path), "--n", "16", "--epochs", "1",
          "--batch", "4", "--workers", "0", "--out", str(out)])
    ck = torch.load(out, weights_only=False)
    assert ck["data"] == "all" and "busi_dice" in ck["history"][0] and "synthetic_dice" in ck["history"][0]
