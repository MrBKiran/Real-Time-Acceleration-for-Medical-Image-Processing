"""Convert Zenodo 7986407 (75-angle plane-wave, MATLAB v7.3 .mat) files into the
npz layout used by the pipeline, selecting a sparse subset of angles for speed.

    python -m scripts.convert_zenodo_mat --glob "7986407/CIRS073_RUMC/**/USDATA_*.mat" \
        --header 7986407/CIRS040GSE/CIRS040GSE/USHEADER_*.mat --n-angles 5 --out data/zenodo.npz

NOTE: not exercised against the real files in the development container (the
network policy blocks zenodo.org).  It follows the layout in your exploration
notebook: USDATA = (75, 1, 128, 2944).  ``--t0`` is the time of sample 0 relative
to the transmit reference - check your header for an initial-delay field.
"""
import argparse
import glob

import numpy as np
from scipy.io import loadmat

from src.io.loader import load_rf
from src.io.metadata import parse_metadata

ap = argparse.ArgumentParser()
ap.add_argument("--glob", required=True)
ap.add_argument("--header", required=True)
ap.add_argument("--n-angles", type=int, default=5)
ap.add_argument("--t0", type=float, default=0.0)
ap.add_argument("--max-files", type=int, default=64)
ap.add_argument("--out", default="data/zenodo.npz")
ap.add_argument("--x-lim", type=float, nargs=2, default=(-0.015, 0.015))
ap.add_argument("--z-lim", type=float, nargs=2, default=(0.005, 0.060))
a = ap.parse_args()

meta = parse_metadata(loadmat(glob.glob(a.header)[0]))
angles = np.asarray(meta["angles"], float)
angles = np.rad2deg(angles) if np.abs(angles).max() < 1.0 else angles   # header may store radians
sel = np.round(np.linspace(0, len(angles) - 1, a.n_angles)).astype(int)
files = sorted(glob.glob(a.glob, recursive=True))[: a.max_files]
frames = [load_rf(f)[sel, 0].astype(np.float32) for f in files]
np.savez_compressed(a.out, rf=np.stack(frames), angles=angles[sel], t0=a.t0, fs=meta["fs"], fc=meta["fc"],
                    c=meta["c"], pitch=meta["pitch"], n_el=frames[0].shape[1], step=0.0,
                    x_lim=a.x_lim, z_lim=a.z_lim)
print("saved", a.out, np.stack(frames).shape, "angles", angles[sel], "from", len(files), "files")
