"""Estimate the acquisition time offset ``t0`` (time of RF sample 0) by autofocus.

Why: a wrong t0 shifts and defocuses the whole image (on simulated data a
3.2 us error moves targets 2.5 mm deeper and halves their amplitude), and real
datasets often do not state t0 or the transmit timing convention.

How: images formed from different plane-wave angles only add coherently when
the delays - and therefore t0 - are right.  We maximise the angular coherence

    C(t0) = sum |sum_a I_a|^2 / (N_angles * sum_a sum |I_a|^2)

over t0 (coarse grid, then fine grid + parabolic refinement), for each candidate
transmit-timing convention, and keep the best:

* ``center``        - t = 0 when the plane wave crosses the array centre (x = 0)
                      (the convention of this repo's simulator / beamformer)
* ``first_element`` - t = 0 when the first element fires, common on research
                      scanners; equivalent to a per-angle t0 shifted by
                      (aperture/2)|sin(angle)|/c.
"""
from __future__ import annotations

import numpy as np
import torch

from src.beamforming.geometry import transmit_delay_plane_wave
from src.beamforming.torch_das import TorchDAS

CONVENTIONS = ("center", "first_element")


def per_angle_t0(t0_sys, element_x, angles_deg, c=1540.0, convention="center"):
    """Per-angle t0 for the beamformer, given the system t0 and timing convention."""
    angles = np.asarray(angles_deg, float)
    if convention == "center":
        return np.full(angles.shape, float(t0_sys))
    if convention == "first_element":
        ex = np.asarray(element_x, float)
        first = np.array([transmit_delay_plane_wave(ex, np.zeros_like(ex), a, c).min() for a in angles])
        return float(t0_sys) + first
    raise ValueError(f"unknown convention {convention!r}; use one of {CONVENTIONS}")


def angular_coherence(das, rf):
    I = das.angle_images(rf)
    return float((I.sum(0).abs() ** 2).sum() / (I.shape[0] * (I.abs() ** 2).sum()).clamp_min(1e-30))


def _scan(rf, ex, X, Z, angles, fs, c, fc, conv, ts, device):
    return np.array([angular_coherence(
        TorchDAS(ex, X, Z, angles, fs, c, per_angle_t0(t, ex, angles, c, conv), device=device,
                 fc=fc, lut_budget_bytes=0), rf) for t in ts])


def estimate_t0(rf, element_x, angles_deg, fs, c=1540.0, fc=7.813e6, x_lim=(-0.01, 0.01), z_lim=(0.01, 0.045),
                search=(-10e-6, 10e-6), conventions=CONVENTIONS, coarse=81, fine=41, grid=(96, 64), device="cpu"):
    """Returns dict(t0, convention, coherence, t0_per_angle, scans).

    rf: (n_angles, nch, ns) for one acquisition (>= 2 angles; more angles = sharper peak).
    """
    rf = np.asarray(rf, np.float32)
    if rf.ndim == 4:
        rf = rf[:, 0]
    angles = np.asarray(angles_deg, float)
    if len(angles) < 2:
        raise ValueError("t0 autofocus needs at least two transmit angles")
    ex = np.asarray(element_x, float)
    X, Z = np.meshgrid(np.linspace(*x_lim, grid[1]), np.linspace(*z_lim, grid[0]))
    rf_t = torch.as_tensor(rf)
    best, scans = None, {}
    for conv in conventions:
        ts = np.linspace(*search, coarse)
        cs = _scan(rf_t, ex, X, Z, angles, fs, c, fc, conv, ts, device)
        step = ts[1] - ts[0]
        tf = np.linspace(ts[cs.argmax()] - 1.5 * step, ts[cs.argmax()] + 1.5 * step, fine)
        cf = _scan(rf_t, ex, X, Z, angles, fs, c, fc, conv, tf, device)
        k = int(cf.argmax())
        t_hat = tf[k]
        if 0 < k < len(tf) - 1:                                   # parabolic peak refinement
            y0, y1, y2 = cf[k - 1], cf[k], cf[k + 1]
            den = y0 - 2 * y1 + y2
            if den < 0:
                t_hat = tf[k] + 0.5 * (y0 - y2) / den * (tf[1] - tf[0])
        coh = float(cf.max())
        scans[conv] = dict(t_coarse=ts.tolist(), coherence_coarse=cs.tolist(), t0=float(t_hat), coherence=coh)
        if best is None or coh > best["coherence"]:
            best = dict(t0=float(t_hat), convention=conv, coherence=coh)
    best["t0_per_angle"] = per_angle_t0(best["t0"], ex, angles, c, best["convention"]).tolist()
    best["scans"] = scans
    return best
