"""CNN vs transformer comparison (brief: "multi-model comparison").

Both models are trained with the same recipe (src.ai.train, synthetic domain-randomised
phantoms, same epochs/samples/seed); this script compares them on
  * in-distribution validation (held-out synthetic phantoms, enhanced),
  * cross-simulator generalisation (beamformed channel-data sequence, known masks),
  * size and per-frame latency on the current device.

    python -m scripts.compare_models --ckpts models/tinyunet.pt models/tinysegformer.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.evaluate_ai import evaluate_cross_simulator
from src.ai.data import SyntheticLesions
from src.ai.infer import LesionDetector
from src.ai.model import n_params
from src.ai.train import evaluate
from src.metrics import Timer

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", default=["models/tinyunet.pt", "models/tinysegformer.pt"])
ap.add_argument("--device", default="auto")
ap.add_argument("--out", default="docs/results/model_comparison.json")
a = ap.parse_args()

rows = {}
val = DataLoader(SyntheticLesions(187, seed=2, augment=False, enhance_prob=1.0), 16)
for ck in a.ckpts:
    det = LesionDetector(ck, a.device)
    m, dev = det.model, det.device
    st = torch.load(ck, map_location="cpu", weights_only=False)
    sync = torch.cuda.synchronize if dev.type == "cuda" else None
    x = torch.rand(1, 1, 256, 128, device=dev)
    with torch.no_grad():
        for _ in range(5):
            m(x)
        ts = []
        for _ in range(30):
            with Timer(sync) as t:
                m(x)
            ts.append(t.dt * 1e3)
    cross = evaluate_cross_simulator(ck)
    rows[det.arch] = dict(
        checkpoint=ck, params=n_params(m), epochs_trained=len(st.get("history", [])), best_epoch=st.get("best_epoch"),
        in_distribution_dice=evaluate(m, val, dev)["dice"],
        cross_simulator_dice_enhanced=cross["enhanced"]["dice"], cross_simulator_dice_raw=cross["raw"]["dice"],
        lesion_recall_enhanced=cross["lesion_recall_enhanced"], detections_per_frame=cross["mean_detections_per_frame"],
        latency_ms_batch1=float(np.mean(ts)), latency_p95_ms=float(np.percentile(ts, 95)), device=str(dev))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(rows, indent=1))
hdr = f"{'model':12s} {'params':>8s} {'in-dist Dice':>12s} {'cross-sim Dice':>15s} {'det/frame':>9s} {'ms/frame':>9s}"
print(hdr)
for k, r in rows.items():
    print(f"{k:12s} {r['params']:8,d} {r['in_distribution_dice']:12.3f} {r['cross_simulator_dice_enhanced']:15.3f} "
          f"{r['detections_per_frame']:9.2f} {r['latency_ms_batch1']:9.2f}")
