"""Run the end-to-end streaming pipeline, save metrics JSON + demo video + figures.

    python -m scripts.run_pipeline --data data/sim_sequence.npz --ckpt models/tinyunet.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.common import lesion_mask, load_npz, make_das
from src.ai.infer import LesionDetector
from src.beamforming.torch_das import get_device
from src.enhancement.filters import Enhancer
from src.metrics import cnr, gcnr, ring_mask
from src.pipeline.operators import BeamformOp, CollectSink, EnhanceOp, InferOp, MotionOp, RFSource
from src.pipeline.runtime import Application
from src.visualization import pipeline_diagram, show_panels, write_demo_video

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/sim_sequence.npz")
ap.add_argument("--ckpt", default="models/tinyunet.pt")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--fps", type=float, default=None, help="throttle source to emulate ADC frame rate")
ap.add_argument("--device", default="auto")
ap.add_argument("--nx", type=int, default=128)
ap.add_argument("--nz", type=int, default=256)
ap.add_argument("--n-angles", type=int, default=None, help="use an evenly spaced subset of the transmit angles")
ap.add_argument("--no-motion", action="store_true", help="disable the optical-flow motion stage")
ap.add_argument("--out", default="docs/results")
a = ap.parse_args()

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
d = load_npz(a.data)
if a.n_angles:
    sel = np.round(np.linspace(0, len(d['angles']) - 1, a.n_angles)).astype(int)
    d['rf'], d['angles'] = d['rf'][:, sel], d['angles'][sel]
    if np.ndim(d['t0']) > 0:                      # per-angle t0 (e.g. from --t0 auto)
        d['t0'] = d['t0'][sel]
dev = get_device(a.device)
sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else None
das, X, Z = make_das(d, a.nx, a.nz, a.device)
sink = CollectSink()
dx_mm, dz_mm = (X[0, 1] - X[0, 0]) * 1e3, (Z[1, 0] - Z[0, 0]) * 1e3
ops = [RFSource(list(d["rf"]), a.frames, a.fps), BeamformOp(das)]
if not a.no_motion:
    ops.append(MotionOp(dx_mm, dz_mm))
ops += [EnhanceOp(Enhancer()), InferOp(LesionDetector(a.ckpt, a.device)), sink]

serial = Application(sync).add_flow(*ops).run_serial(min(a.frames, 30))
sink.frames.clear()
piped = Application(sync).add_flow(*ops).run(a.frames)

# ---- quality on frame 0 of the simulated sequence (ground truth known) ----
res = dict(device=str(dev), grid=[a.nz, a.nx], n_angles=int(len(d["angles"])), serial=serial, pipelined=piped)
if float(d["step"]) > 0:   # simulated sequence only: lesion masks are known
    fr = sink.frames[0]
    shift = float(d["step"]) * fr["idx"]
    m = lesion_mask(X, Z, shift, which=[0]).astype(bool)      # anechoic cyst
    ring = ring_mask(m, 6)
    res["quality_frame0"] = {
        "raw": dict(cnr=cnr(fr["bmode_raw"], m, ring), gcnr=gcnr(fr["bmode_raw"], m, ring)),
        "enhanced": dict(cnr=cnr(fr["bmode"], m, ring), gcnr=gcnr(fr["bmode"], m, ring)),
    }
    if not a.no_motion:
        # source loops over the recorded frames: skip the jump back at each wrap-around
        n_rec = len(d["rf"])
        lat = np.array([f["motion_mm"][0] for f in sink.frames if f["idx"] % n_rec != 0])
        ax = np.array([f["motion_mm"][1] for f in sink.frames if f["idx"] % n_rec != 0])
        true = float(d["step"]) * 1e3
        res["motion"] = dict(true_lateral_mm_per_frame=true, est_lateral_mean=float(lat.mean()), est_lateral_std=float(lat.std()),
                             est_axial_mean=float(ax.mean()), abs_error_mm=float(np.abs(lat - true).mean()))
(out / "pipeline_results.json").write_text(json.dumps(res, indent=1))
stage = {k: v["mean_ms"] for k, v in serial["latency"].items()}
pipeline_diagram(out / "pipeline_diagram.png", stage)
ext = (d["x_lim"][0] * 1e3, d["x_lim"][1] * 1e3, d["z_lim"][1] * 1e3, d["z_lim"][0] * 1e3)
f0 = sink.frames[0]
show_panels([f0["bmode_raw"], f0["bmode"], f0["prob"]], ["Beamformed", "Enhanced", "AI probability"],
            out / "frame0_panels.png", extent=ext)
write_demo_video(sink.frames, out / "demo.mp4", fps=10,
                 text=lambda f: f"frame {f['idx']}" + (f"  motion {f['motion_mm'][0]:+.2f} mm" if f.get("motion_mm") else "")
                 + (("  " + max(f["class_prob"], key=f["class_prob"].get) + f" {max(f['class_prob'].values()):.2f}") if f.get("class_prob") else ""))
print(json.dumps({k: res[k] for k in ("device", "grid", "n_angles")}))
print("per-stage latency (serial, mean / p95 ms):")
for k, v in serial["latency"].items():
    print(f"  {k:12s} {v['mean_ms']:8.2f} / {v['p95_ms']:8.2f}")
print("serial  end-to-end mean/p95 ms:", serial["latency"]["end_to_end"]["mean_ms"], serial["latency"]["end_to_end"]["p95_ms"])
print("pipelined throughput fps:", piped["fps"])
print(res.get("quality_frame0"))
if "motion" in res:
    print("motion:", {k: round(v, 4) for k, v in res["motion"].items()})
