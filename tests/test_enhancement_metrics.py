import numpy as np
import torch

from src.enhancement import filters as F
from src.metrics import cnr, dice, gcnr, ring_mask
from src.simulation.ultrasound import fast_bmode_phantom


def _img():
    rng = np.random.default_rng(1)
    img, mask, env = fast_bmode_phantom(128, 96, n_lesions=1, rng=rng)
    return img, mask.astype(bool)


def test_torch_filters_match_cpu_references():
    img, _ = _img()
    t = torch.from_numpy(img)
    assert np.allclose(F.median_t(t, 3).numpy(), F.median_cpu(img, 3), atol=1e-6)
    assert np.abs(F.wiener_t(t, 5).numpy() - F.wiener_cpu(img, 5)).mean() < 0.02
    assert np.abs(F.bilateral_t(t, 5, 2.0, 0.1).numpy() - F.bilateral_cpu(img, 5, 2.0, 0.1)).mean() < 0.03


def test_speckle_filters_reduce_variance_and_keep_range():
    img, mask = _img()
    bg = ~mask
    t = torch.from_numpy(img)
    for f in (F.median_t, F.wiener_t, F.bilateral_t):
        out = f(t).numpy()
        assert out.min() >= -1e-5 and out.max() <= 1 + 1e-5
        assert out[bg].std() < img[bg].std()


def test_enhancer_chain_runs_on_numpy_and_torch():
    img, _ = _img()
    e = F.Enhancer(["median", "bilateral", "clahe"])
    assert e(img).shape == img.shape and isinstance(e(img), np.ndarray)
    assert isinstance(e(torch.from_numpy(img)), torch.Tensor)


def test_metrics_basic():
    a = np.zeros((20, 20)); a[:, :10] = 1
    ins, out = np.zeros_like(a, bool), np.zeros_like(a, bool)
    ins[:, :10], out[:, 10:] = True, True
    assert gcnr(a, ins, out) == 1.0
    assert dice(ins, ins) == 1.0 and dice(ins, out) < 0.01
    assert ring_mask(ins, 2).sum() > 0
    rng = np.random.default_rng(0)
    n = rng.normal(0, 1, (50, 50)); n2 = n + 5
    assert cnr(np.where(ins[:, :1].repeat(50, 1)[:50, :50] if False else np.arange(50)[:, None] < 25, n2, n),
               np.arange(50)[:, None].repeat(50, 1) < 25, np.arange(50)[:, None].repeat(50, 1) >= 25) > 3
