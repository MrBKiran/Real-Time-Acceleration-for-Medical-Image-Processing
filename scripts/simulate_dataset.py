"""Create a simulated plane-wave RF sequence (stand-in for ADC / dataset output).

    python -m scripts.simulate_dataset --out data/sim_sequence.npz --frames 12
"""
import argparse
from pathlib import Path

import numpy as np

from scripts.common import simulate_sequence

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/sim_sequence.npz")
ap.add_argument("--frames", type=int, default=12)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
d = simulate_sequence(a.frames, seed=a.seed)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(a.out, **{k: np.asarray(v) for k, v in d.items()})
print("saved", a.out, d["rf"].shape, d["rf"].dtype)
