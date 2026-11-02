"""GPU-ready delay-and-sum plane-wave compounding (PyTorch).

Open-source stand-in for "MATLAB function -> GPU Coder -> CUDA".  The expensive,
frame-independent geometry (receive delays, f-number apodisation, interpolation
indices) is precomputed once into look-up tables, so each frame is only a
gather + multiply-accumulate - the same structure a hand-written CUDA kernel
would have.  Runs unchanged on CPU or CUDA.
"""
from __future__ import annotations

import numpy as np
import torch

from src.beamforming.geometry import transmit_delay_plane_wave


def get_device(prefer="auto"):
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(prefer)


class TorchDAS:
    def __init__(self, element_x, X, Z, angles_deg, fs, c=1540.0, t0=0.0,
                 f_number=1.5, min_aperture_elems=16, device="auto",
                 lut_budget_bytes=1.5e9):
        self.device = get_device(device)
        self.fs, self.c, self.t0 = float(fs), float(c), float(t0)
        self.angles = np.asarray(angles_deg, float)
        self.nz, self.nx = X.shape
        ex = np.asarray(element_x, np.float64)
        self.nch = ex.size
        dev = self.device
        Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
        Zt = torch.as_tensor(Z, dtype=torch.float32, device=dev)
        ext = torch.as_tensor(ex, dtype=torch.float32, device=dev)

        dxe = Xt[..., None] - ext                                   # (nz,nx,nch)
        rx_samples = torch.sqrt(dxe ** 2 + Zt[..., None] ** 2) / c * fs
        half = torch.clamp(Zt / (2 * f_number), min=min_aperture_elems * float(ex[1] - ex[0]))[..., None]
        inside = dxe.abs() < half
        w = torch.where(inside, 0.5 * (1 + torch.cos(np.pi * dxe / half)), torch.zeros_like(dxe))
        self.inv_wsum = 1.0 / w.sum(-1).clamp_min(1e-6)             # (nz,nx)
        self.w = w
        self.rx_samples = rx_samples
        self.ch_off = (torch.arange(self.nch, device=dev) * 0)      # placeholder, set per ns
        self.tx_samples = [
            ((torch.as_tensor(transmit_delay_plane_wave(X, Z, a, c), dtype=torch.float32, device=dev) - t0) * fs)
            for a in self.angles
        ]
        self._lut = None
        self._ns = None
        est = len(self.angles) * self.nz * self.nx * self.nch * 8
        self._use_lut = est <= lut_budget_bytes

    # -- LUT (built lazily once the number of samples is known) -----------------
    def _build_lut(self, ns):
        self._ns = ns
        self.ch_off = (torch.arange(self.nch, device=self.device) * ns)
        if not self._use_lut:
            return
        lut = []
        for tx in self.tx_samples:
            s = tx[..., None] + self.rx_samples
            i0 = torch.floor(s)
            frac = s - i0
            i0 = i0.long()
            valid = (i0 >= 0) & (i0 < ns - 1)
            wv = self.w * valid
            flat = torch.where(valid, i0 + self.ch_off, torch.zeros_like(i0)).to(torch.int32)
            lut.append((flat, frac, wv))
        self._lut = lut

    @torch.no_grad()
    def __call__(self, rf):
        """rf: (n_angles, nch, ns) array/tensor -> beamformed RF image (nz, nx)."""
        rf = torch.as_tensor(rf, dtype=torch.float32, device=self.device)
        if rf.ndim == 4:
            rf = rf[:, 0]
        na, nch, ns = rf.shape
        assert na == len(self.angles) and nch == self.nch, "rf shape does not match geometry"
        if self._ns != ns:
            self._build_lut(ns)
        out = torch.zeros(self.nz, self.nx, device=self.device)
        for ia in range(na):
            flat_rf = rf[ia].reshape(-1)
            if self._lut is not None:
                idx, frac, wv = self._lut[ia]
                idx = idx.long()
            else:
                s = self.tx_samples[ia][..., None] + self.rx_samples
                i0 = torch.floor(s)
                frac = s - i0
                i0 = i0.long()
                valid = (i0 >= 0) & (i0 < ns - 1)
                wv = self.w * valid
                idx = torch.where(valid, i0 + self.ch_off, torch.zeros_like(i0))
            s0 = flat_rf[idx]
            s1 = flat_rf[idx + 1]
            out += ((1 - frac) * s0 + frac * s1).mul_(wv).sum(-1) * self.inv_wsum
        return out / na

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
