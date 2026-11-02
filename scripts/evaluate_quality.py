"""Before/after image-quality comparison of the enhancement filters on the simulated
sequence (ground-truth lesion masks known).

    python -m scripts.evaluate_quality
"""
import json
from pathlib import Path

import numpy as np
import torch

from scripts.common import LESIONS, lesion_mask, load_npz, make_das
from src.enhancement.filters import Enhancer
from src.metrics import cnr, gcnr, ring_mask, speckle_snr
from src.signal.torch_ops import log_compress
from src.visualization import show_panels

d = load_npz("data/sim_sequence.npz")
das, X, Z = make_das(d, 128, 256, "cpu")
chains = {"none": [], "median": ["median"], "wiener": ["wiener"], "bilateral": ["bilateral"],
          "median+bilateral": ["median", "bilateral"], "median+bilateral+clahe": ["median", "bilateral", "clahe"]}
n = len(d["rf"])
acc = {k: {l["name"]: [] for l in LESIONS} for k in chains}
snr = {k: [] for k in chains}
edge = {k: [] for k in chains}
first = {}
for fi in range(n):
    env = das.envelope(d["rf"][fi])
    b = log_compress(env)
    shift = float(d["step"]) * fi
    bg = ~lesion_mask(X, Z, shift).astype(bool)
    for k, steps in chains.items():
        out = Enhancer(steps)(b) if steps else b
        o = out.numpy()
        for li, l in enumerate(LESIONS):
            m = lesion_mask(X, Z, shift, [li]).astype(bool)
            acc[k][l["name"]].append((cnr(o, m, ring_mask(m, 6)), gcnr(o, m, ring_mask(m, 6))))
        snr[k].append(speckle_snr(o, bg))   # on display (log) image: lower spread = smoother
        # edge sharpness: mean gradient magnitude on lesion boundary of the cyst
        m0 = lesion_mask(X, Z, shift, [0]).astype(bool)
        gy, gx = np.gradient(o)
        g = np.hypot(gx, gy)
        from scipy.ndimage import binary_dilation, binary_erosion
        bnd = binary_dilation(m0, iterations=1) & ~binary_erosion(m0, iterations=1)
        edge[k].append(float(g[bnd].mean()))
        if fi == 0:
            first[k] = o
res = {}
for k in chains:
    res[k] = {"cnr": {ln: float(np.mean([v[0] for v in vals])) for ln, vals in acc[k].items()},
              "gcnr": {ln: float(np.mean([v[1] for v in vals])) for ln, vals in acc[k].items()},
              "bg_mean_over_std": float(np.mean(snr[k])), "cyst_edge_gradient": float(np.mean(edge[k]))}
Path("docs/results").mkdir(parents=True, exist_ok=True)
Path("docs/results/quality.json").write_text(json.dumps(res, indent=1))
for k, v in res.items():
    print(f"{k:26s} CNR cyst {v['cnr']['anechoic cyst']:.2f} hypo {v['cnr']['hypoechoic']:.2f} hyper {v['cnr']['hyperechoic']:.2f} | "
          f"gCNR cyst {v['gcnr']['anechoic cyst']:.2f} | bg mean/std {v['bg_mean_over_std']:.2f} | edge {v['cyst_edge_gradient']:.3f}")
ext = (-12, 12, 50, 5)
ks = list(chains)
show_panels([first[k] for k in ks], ks, "docs/results/enhancement_comparison.png", extent=ext, figsize_per=(2.6, 4.6))
