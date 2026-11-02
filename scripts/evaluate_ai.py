"""Dice/IoU of the AI stage against known lesion masks on the simulated *channel-data*
sequence (a different simulator from the one used to train the model).

    python -m scripts.evaluate_ai
"""
import json

import numpy as np

from scripts.common import lesion_mask, load_npz, make_das
from src.ai.infer import LesionDetector
from src.enhancement.filters import Enhancer
from src.metrics import dice, iou
from src.signal.torch_ops import log_compress

d = load_npz("data/sim_sequence.npz")
das, X, Z = make_das(d, 128, 256, "cpu")
det, enh = LesionDetector("models/tinyunet.pt", "cpu"), Enhancer()
res = {"raw": [], "enhanced": []}
per_lesion = {"anechoic cyst": [], "hypoechoic": [], "hyperechoic": []}
ndet = []
for fi in range(len(d["rf"])):
    b = log_compress(das.envelope(d["rf"][fi]))
    shift = float(d["step"]) * fi
    gt = lesion_mask(X, Z, shift)
    e = enh(b)
    for k, img in (("raw", b), ("enhanced", e)):
        p = det(img).numpy() > 0.5
        res[k].append((dice(p, gt), iou(p, gt)))
    p = det(e).numpy() > 0.5
    for li, name in enumerate(per_lesion):
        m = lesion_mask(X, Z, shift, [li]).astype(bool)
        per_lesion[name].append(float((p & m).sum() / m.sum()))     # fraction of lesion covered (recall)
    ndet.append(len(det.detections(det(e))))
out = {k: dict(dice=float(np.mean([a for a, _ in v])), iou=float(np.mean([b for _, b in v]))) for k, v in res.items()}
out["lesion_recall_enhanced"] = {k: float(np.mean(v)) for k, v in per_lesion.items()}
out["mean_detections_per_frame"] = float(np.mean(ndet))
out["true_lesions_per_frame"] = 3
print(json.dumps(out, indent=1))
json.dump(out, open("docs/results/ai_on_simulated_channel_data.json", "w"), indent=1)
