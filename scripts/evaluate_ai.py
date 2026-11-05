"""Dice/IoU of the AI stage against known lesion masks on the simulated *channel-data*
sequence (a different simulator from the one used to train the model).

    python -m scripts.evaluate_ai [--ckpt models/tinyunet.pt] [--out docs/results/ai_on_simulated_channel_data.json]
"""
import argparse
import json

import numpy as np

from scripts.common import lesion_mask, load_npz, make_das
from src.ai.infer import LesionDetector
from src.enhancement.filters import Enhancer
from src.metrics import dice, iou
from src.signal.torch_ops import log_compress

LESION_NAMES = ("anechoic cyst", "hypoechoic", "hyperechoic")


def evaluate_cross_simulator(ckpt, data="data/sim_sequence.npz", device="cpu"):
    d = load_npz(data)
    das, X, Z = make_das(d, 128, 256, device)
    det, enh = LesionDetector(ckpt, device), Enhancer()
    res = {"raw": [], "enhanced": []}
    per_lesion = {k: [] for k in LESION_NAMES}
    ndet = []
    for fi in range(len(d["rf"])):
        b = log_compress(das.envelope(d["rf"][fi]))
        shift = float(d["step"]) * fi
        gt = lesion_mask(X, Z, shift)
        e = enh(b)
        for k, img in (("raw", b), ("enhanced", e)):
            p = det(img).cpu().numpy() > 0.5
            res[k].append((dice(p, gt), iou(p, gt)))
        pe = det(e)
        p = pe.cpu().numpy() > 0.5
        for li, name in enumerate(LESION_NAMES):
            m = lesion_mask(X, Z, shift, [li]).astype(bool)
            per_lesion[name].append(float((p & m).sum() / m.sum()))     # fraction of lesion covered (recall)
        ndet.append(len(det.detections(pe)))
    out = {k: dict(dice=float(np.mean([a for a, _ in v])), iou=float(np.mean([b for _, b in v]))) for k, v in res.items()}
    out["lesion_recall_enhanced"] = {k: float(np.mean(v)) for k, v in per_lesion.items()}
    out["mean_detections_per_frame"] = float(np.mean(ndet))
    out["true_lesions_per_frame"] = 3
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/tinyunet.pt")
    ap.add_argument("--data", default="data/sim_sequence.npz")
    ap.add_argument("--out", default="docs/results/ai_on_simulated_channel_data.json")
    a = ap.parse_args()
    out = evaluate_cross_simulator(a.ckpt, a.data)
    print(json.dumps(out, indent=1))
    json.dump(out, open(a.out, "w"), indent=1)
