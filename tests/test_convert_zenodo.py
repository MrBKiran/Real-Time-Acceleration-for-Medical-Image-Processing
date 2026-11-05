"""End-to-end check of scripts/convert_zenodo_mat.py on synthetic files written in
the Zenodo 7986407 layout (MATLAB v7.3 USDATA + MATLAB v5 USHEADER struct)."""
import json
import subprocess
import sys

import h5py
import numpy as np
from scipy.io import savemat

from scripts.common import load_npz, make_das
from src.beamforming.autofocus import per_angle_t0
from src.beamforming.geometry import element_positions
from src.simulation.ultrasound import make_scatterers, simulate_plane_wave_rf

FS, FC, C, PITCH = 31.25e6, 7.813e6, 1540.0, 195.3e-6
WIRES = [(0.0, 0.015), (-0.004, 0.025), (0.005, 0.035)]
T0_SYS = 0.6e-6                                  # system t0, first-element timing convention


def _write_fake_zenodo(root, n_files=2):
    ex = element_positions(128, PITCH)
    angles = np.linspace(-16, 16, 15)
    t0a = per_angle_t0(T0_SYS, ex, angles, C, "first_element")
    for k in range(n_files):
        x, z, a = make_scatterers((-0.012, 0.012), (0.008, 0.045), 5,
                                  points=[(wx, wz, 6.0) for wx, wz in WIRES], seed=k)
        rf = np.concatenate([simulate_plane_wave_rf(x, z, a, ex, angles[i:i + 1], FS, FC, C, 1800, t0a[i],
                                                    noise_db=-35, seed=10 * k + i)[0] for i in range(len(angles))])
        counts = np.round(rf / np.abs(rf).max() * 8000).astype(np.int16)        # ADC-like int16
        (root / "phantom").mkdir(exist_ok=True)
        with h5py.File(root / "phantom" / f"USDATA_2021062411360327{k}.mat", "w") as f:
            f.create_dataset("USDATA", data=counts[:, None])                  # (angles, 1, ch, samples)
    savemat(root / "USHEADER_20220330104510.mat", {"USHEADER": {
        "system": "Verasonics", "transducer": "L11-4v", "signalType": "RF", "c": C, "fc": FC, "fs": FS,
        "pitch": PITCH, "xmitAngles": np.deg2rad(angles)[None]}})              # radians, like many headers


def test_convert_auto_t0_then_beamform_and_run_pipeline(tmp_path):
    _write_fake_zenodo(tmp_path)
    out = tmp_path / "zen.npz"
    r = subprocess.run([sys.executable, "-m", "scripts.convert_zenodo_mat",
                        "--glob", str(tmp_path / "phantom" / "USDATA_*.mat"),
                        "--header", str(tmp_path / "USHEADER_*.mat"),
                        "--n-angles", "5", "--est-angles", "7", "--t0", "auto",
                        "--z-lim", "0.005", "0.045", "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    d = load_npz(out)
    assert d["rf"].shape == (2, 5, 128, 1800)
    all_angles = np.linspace(-16, 16, 15)
    expected = all_angles[np.round(np.linspace(0, 14, 5)).astype(int)]
    assert np.allclose(d["angles"], expected)                               # radians converted to degrees
    assert str(d["t0_convention"]) == "first_element"
    assert abs(float(d["t0_system"]) - T0_SYS) < 0.3e-6
    info = json.loads((tmp_path / "zen_conversion.json").read_text())
    coh = info["t0_estimate"]["coherence_by_convention"]
    assert coh["first_element"] > coh["center"]

    das, X, Z = make_das(d, 128, 256, "cpu")                                 # per-angle t0 from the file
    env = das.envelope(d["rf"][0]).numpy()
    for wx, wz in WIRES:
        win = (abs(X - wx) < 1.5e-3) & (abs(Z - wz) < 1.5e-3)
        iz, ix = np.unravel_index(np.where(win, env, 0).argmax(), env.shape)
        assert abs(X[iz, ix] - wx) < 0.3e-3 and abs(Z[iz, ix] - wz) < 0.3e-3

    res = subprocess.run([sys.executable, "-m", "scripts.run_pipeline", "--data", str(out), "--frames", "4",
                          "--device", "cpu", "--out", str(tmp_path / "res")], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    pr = json.loads((tmp_path / "res" / "pipeline_results.json").read_text())
    assert pr["n_angles"] == 5 and "quality_frame0" not in pr                # no ground-truth masks on real data
