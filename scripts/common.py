"""Shared helpers for the command-line scripts."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src.beamforming.geometry import element_positions
from src.beamforming.grid import create_grid
from src.beamforming.torch_das import TorchDAS
from src.simulation.ultrasound import default_t0, make_scatterers, simulate_plane_wave_rf

FS, FC, C = 31.25e6, 7.813e6, 1540.0
N_EL, PITCH = 128, 195.3e-6
X_LIM, Z_LIM = (-0.012, 0.012), (0.005, 0.050)

# tissue-mimicking lesions used by the simulated sequence (metres)
LESIONS = [
    dict(name="anechoic cyst", cx=-0.0045, cz=0.022, rx=0.0040, rz=0.0035, gain=0.0),
    dict(name="hypoechoic",    cx=0.0050, cz=0.016, rx=0.0035, rz=0.0030, gain=0.12),
    dict(name="hyperechoic",   cx=0.0020, cz=0.037, rx=0.0035, rz=0.0035, gain=3.0),
]


def grid(nx=128, nz=256):
    return create_grid(*X_LIM, *Z_LIM, nx=nx, nz=nz)


def lesion_mask(X, Z, shift=0.0, which=None):
    m = np.zeros(X.shape, np.uint8)
    for i, l in enumerate(LESIONS):
        if which is not None and i not in which:
            continue
        cx = l["cx"] + shift
        wrap = X_LIM[1] - X_LIM[0]
        cx = (cx - X_LIM[0]) % wrap + X_LIM[0]
        m |= (((X - cx) / l["rx"]) ** 2 + ((Z - l["cz"]) / l["rz"]) ** 2 < 1).astype(np.uint8)
    return m


def simulate_sequence(n_frames=12, angles_deg=(-8, -4, 0, 4, 8), step_mm=0.15,
                      density=40.0, seed=0, noise_db=-30.0, n_samples=2400, log=print):
    """Simulated moving-tissue acquisition: returns dict ready for np.savez."""
    ex = element_positions(N_EL, PITCH)
    x, z, a = make_scatterers(X_LIM, Z_LIM, density, LESIONS, seed=seed)
    t0 = default_t0(ex, angles_deg, C)
    w = X_LIM[1] - X_LIM[0]
    frames = []
    t = time.time()
    for k in range(n_frames):
        xs = ((x - X_LIM[0] + k * step_mm * 1e-3) % w + X_LIM[0]).astype(np.float32)
        rf, _ = simulate_plane_wave_rf(xs, z, a, ex, angles_deg, FS, FC, C, n_samples, t0,
                                       noise_db=noise_db, seed=seed + k)
        frames.append(rf)
        log(f"  simulated frame {k+1}/{n_frames} ({time.time()-t:.0f}s)")
    return dict(rf=np.stack(frames), angles=np.asarray(angles_deg, float), t0=t0, fs=FS, fc=FC, c=C,
                pitch=PITCH, n_el=N_EL, step=step_mm * 1e-3, x_lim=X_LIM, z_lim=Z_LIM)


def load_npz(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def make_das(meta, nx=128, nz=256, device="auto", **kw):
    ex = element_positions(int(meta["n_el"]), float(meta["pitch"]))
    X, Z = create_grid(*meta["x_lim"], *meta["z_lim"], nx=nx, nz=nz)
    return TorchDAS(ex, X, Z, meta["angles"], float(meta["fs"]), float(meta["c"]),
                    float(meta["t0"]), device=device, **kw), X, Z


def build_components(npz_path, ckpt, device="auto"):
    """Used by the Holoscan entry point: returns (frames, [bf, motion, enh, infer, sink])."""
    from src.ai.infer import LesionDetector
    from src.enhancement.filters import Enhancer
    from src.pipeline.operators import BeamformOp, CollectSink, EnhanceOp, InferOp, MotionOp
    d = load_npz(npz_path)
    das, X, Z = make_das(d, device=device)
    motion = MotionOp((X[0, 1] - X[0, 0]) * 1e3, (Z[1, 0] - Z[0, 0]) * 1e3)
    return list(d["rf"]), [BeamformOp(das), motion, EnhanceOp(Enhancer()), InferOp(LesionDetector(ckpt, device)), CollectSink()]
