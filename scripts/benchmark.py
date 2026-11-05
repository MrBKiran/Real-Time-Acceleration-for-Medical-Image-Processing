"""Latency / throughput benchmarks for every stage.

    python -m scripts.benchmark --out docs/results/benchmark.json
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from scripts.common import grid, simulate_sequence
from src.ai.model import ARCHS, build_model, n_params
from src.beamforming.compound import coherent_compound
from src.beamforming.geometry import element_positions
from src.beamforming.torch_das import TorchDAS, get_device
from src.enhancement import filters as F
from src.metrics import Timer

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="docs/results/benchmark.json")
ap.add_argument("--device", default="auto")
ap.add_argument("--reps", type=int, default=15)
a = ap.parse_args()
dev = get_device(a.device)
sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else None


def bench(fn, reps=a.reps, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        with Timer(sync) as t:
            fn()
        ts.append(t.dt * 1e3)
    return dict(mean_ms=float(np.mean(ts)), p95_ms=float(np.percentile(ts, 95)), min_ms=float(np.min(ts)))


res = dict(device=str(dev), torch=torch.__version__, cpu=platform.processor() or platform.machine(),
           threads=torch.get_num_threads(), cuda=torch.cuda.get_device_name(0) if dev.type == "cuda" else None)
ex = element_positions()
fs, c = 31.25e6, 1540.0

# --- beamforming: NumPy reference vs torch, by angles and grid size ---
bf = {}
rng = np.random.default_rng(0)
for na in (1, 5, 11):
    ang = np.linspace(-10, 10, na) if na > 1 else np.array([0.0])
    rf = rng.normal(size=(na, 128, 2400)).astype(np.float32)
    for (nz, nx) in ((256, 128), (512, 256)):
        X, Z = grid(nx, nz)
        das = TorchDAS(ex, X, Z, ang, fs, c, 0.0, device=a.device)
        r = bench(lambda: das(rf))
        key = f"torch_{na}ang_{nz}x{nx}"
        bf[key] = r
        if (nz, nx) == (256, 128) and na <= 5:
            n = bench(lambda: coherent_compound(rf, X, Z, ex, ang, fs, c, 0.0), reps=2, warm=1)
            bf[f"numpy_ref_{na}ang_{nz}x{nx}"] = n
            bf[f"speedup_{na}ang_{nz}x{nx}"] = n["mean_ms"] / r["mean_ms"]
        print(key, r)
res["beamforming"] = bf

# --- enhancement ---
img_t = torch.rand(256, 128, device=dev)
img_n = img_t.cpu().numpy()
enh = {}
for name, tf, cf in (("median3", lambda: F.median_t(img_t, 3), lambda: F.median_cpu(img_n, 3)),
                     ("wiener5", lambda: F.wiener_t(img_t, 5), lambda: F.wiener_cpu(img_n, 5)),
                     ("bilateral5", lambda: F.bilateral_t(img_t, 5), lambda: F.bilateral_cpu(img_n, 5))):
    enh[name] = dict(torch=bench(tf), cpu_ref=bench(cf))
enh["clahe_cpu"] = bench(lambda: F.clahe_cpu(img_n))
enh["chain_median_bilateral"] = bench(lambda: F.Enhancer()(img_t))
enh["chain_median_bilateral_clahe"] = bench(lambda: F.Enhancer(["median", "bilateral", "clahe"])(img_t))
res["enhancement"] = enh

# --- inference (every architecture) ---
inf = {}
x = torch.rand(1, 1, 256, 128, device=dev)
xb = torch.rand(8, 1, 256, 128, device=dev)
for arch in ARCHS:
    m = build_model(arch).to(dev).eval()
    with torch.no_grad():
        r = {"params": n_params(m), "batch1": bench(lambda: m(x))}
        r8 = bench(lambda: m(xb))
        r["batch8"] = r8
        r["throughput_fps_batch8"] = 8 / (r8["mean_ms"] / 1e3)
        if dev.type == "cuda":
            mh = build_model(arch).to(dev).eval().half()
            r["batch1_fp16"] = bench(lambda: mh(x.half()))
    inf[arch] = r
res["inference"] = inf
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
