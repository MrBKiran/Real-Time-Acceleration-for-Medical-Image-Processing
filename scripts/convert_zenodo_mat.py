"""Convert Zenodo 7986407 (75-angle plane-wave, MATLAB v7.3 .mat) files into the
npz layout used by the pipeline, selecting a sparse subset of angles for speed.

    python -m scripts.convert_zenodo_mat --glob "7986407/CIRS073_RUMC/**/USDATA_*.mat" \
        --header "7986407/CIRS040GSE/CIRS040GSE/USHEADER_*.mat" --n-angles 5 --t0 auto --out data/zenodo.npz

``--t0`` is the time of RF sample 0 relative to the transmit reference.  A wrong
value shifts and defocuses the image (3.2 us error -> 2.5 mm deeper, -6 dB).
``--t0 auto`` (recommended when the header does not state it) estimates t0 *and*
the transmit timing convention by angular-coherence autofocus on the first file
(src/beamforming/autofocus.py) and stores a per-angle t0.

Layout assumed (from the exploration notebook): USDATA read by h5py as
(n_angles, 1, n_channels, n_samples); USHEADER a MATLAB struct with fields
system, transducer, signalType, c, fc, fs, pitch, xmitAngles.  Validated on
synthetic files written in exactly that layout (tests/test_convert_zenodo.py),
not yet on the real Zenodo files.
"""
import argparse
import glob
import json

import numpy as np
from scipy.io import loadmat

from src.beamforming.autofocus import CONVENTIONS, estimate_t0, per_angle_t0
from src.beamforming.geometry import element_positions
from src.io.loader import load_rf
from src.io.metadata import parse_metadata

ap = argparse.ArgumentParser()
ap.add_argument("--glob", required=True)
ap.add_argument("--header", required=True)
ap.add_argument("--n-angles", type=int, default=5)
ap.add_argument("--t0", default="auto", help="seconds, or 'auto' for autofocus estimation")
ap.add_argument("--tx-ref", choices=CONVENTIONS + ("auto",), default="auto",
                help="transmit timing convention ('auto' = pick by autofocus; with a numeric --t0, 'auto' means center)")
ap.add_argument("--est-angles", type=int, default=9, help="angles used for the t0 estimate")
ap.add_argument("--max-files", type=int, default=64)
ap.add_argument("--out", default="data/zenodo.npz")
ap.add_argument("--x-lim", type=float, nargs=2, default=(-0.015, 0.015))
ap.add_argument("--z-lim", type=float, nargs=2, default=(0.005, 0.060))
a = ap.parse_args()

headers = glob.glob(a.header)
if not headers:
    raise SystemExit(f"no header file matches {a.header!r}")
meta = parse_metadata(loadmat(headers[0]))
angles = np.asarray(meta["angles"], float).ravel()
if np.abs(angles).max() < 1.0:                       # header stores radians
    angles = np.rad2deg(angles)
files = sorted(glob.glob(a.glob, recursive=True))[: a.max_files]
if not files:
    raise SystemExit(f"no RF file matches {a.glob!r}")
first = load_rf(files[0])
if first.ndim == 4:
    first = first[:, 0]
if first.shape[0] != len(angles):
    raise SystemExit(f"RF has {first.shape[0]} angles but the header lists {len(angles)}; "
                     f"check the array orientation (got shape {first.shape})")
n_el = first.shape[1]
ex = element_positions(n_el, meta["pitch"])
sel = np.round(np.linspace(0, len(angles) - 1, a.n_angles)).astype(int)

info = {"header": meta | {"angles": angles.tolist()}, "files": len(files)}
if str(a.t0).lower() == "auto":
    est_sel = np.round(np.linspace(0, len(angles) - 1, min(a.est_angles, len(angles)))).astype(int)
    convs = CONVENTIONS if a.tx_ref == "auto" else (a.tx_ref,)
    zl = (max(a.z_lim[0], 0.008), min(a.z_lim[1], 0.045))
    print(f"estimating t0 by autofocus on {files[0]} ({len(est_sel)} angles, conventions {convs}) ...")
    est = estimate_t0(first[est_sel].astype(np.float32), ex, angles[est_sel], meta["fs"], meta["c"], meta["fc"],
                      x_lim=(-0.01, 0.01), z_lim=zl, conventions=convs)
    t0_sys, conv = est["t0"], est["convention"]
    info["t0_estimate"] = {k: v for k, v in est.items() if k != "scans"} | {
        "coherence_by_convention": {k: v["coherence"] for k, v in est["scans"].items()}}
    print(f"  -> t0 = {t0_sys * 1e6:+.3f} us, convention = {conv}, coherence = {est['coherence']:.3f} "
          f"(others: {info['t0_estimate']['coherence_by_convention']})")
else:
    t0_sys, conv = float(a.t0), ("center" if a.tx_ref == "auto" else a.tx_ref)
t0 = per_angle_t0(t0_sys, ex, angles[sel], meta["c"], conv)

frames = []
for f in files:
    r = load_rf(f)
    frames.append((r[:, 0] if r.ndim == 4 else r)[sel].astype(np.float32))
rf = np.stack(frames)
np.savez_compressed(a.out, rf=rf, angles=angles[sel], t0=t0, t0_system=t0_sys, t0_convention=conv,
                    fs=meta["fs"], fc=meta["fc"], c=meta["c"], pitch=meta["pitch"], n_el=n_el, step=0.0,
                    x_lim=a.x_lim, z_lim=a.z_lim)
with open(str(a.out).rsplit(".", 1)[0] + "_conversion.json", "w") as fh:
    json.dump(info | {"t0_system": t0_sys, "t0_convention": conv, "angles_used": angles[sel].tolist()}, fh, indent=1, default=str)
print("saved", a.out, rf.shape, "angles", np.round(angles[sel], 2), "from", len(files), "files")
