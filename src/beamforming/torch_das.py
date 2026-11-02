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
    """mode="iq" (default): demodulate channel data to baseband, beamform complex
    samples with phase correction -> complex image; |.| is the envelope.  This
    allows a coarse (cheap) image grid and needs no Hilbert transform.
    mode="rf": classic real RF beamforming (needs dz << lambda/2).
    """

    def __init__(self, element_x, X, Z, angles_deg, fs, c=1540.0, t0=0.0,
                 f_number=1.5, min_aperture_elems=16, device="auto",
                 lut_budget_bytes=1.5e9, mode="iq", fc=7.813e6, lp_taps=33):
        assert mode in ("iq", "rf")
        self.mode, self.fc = mode, float(fc)
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
        self.rx_samples = torch.sqrt(dxe ** 2 + Zt[..., None] ** 2) / c * fs
        half = torch.clamp(Zt / (2 * f_number), min=min_aperture_elems * float(ex[1] - ex[0]))[..., None]
        inside = dxe.abs() < half
        self.w = torch.where(inside, 0.5 * (1 + torch.cos(np.pi * dxe / half)), torch.zeros_like(dxe))
        self.inv_wsum = 1.0 / self.w.sum(-1).clamp_min(1e-6)         # (nz,nx)
        self.tx_samples = [
            ((torch.as_tensor(transmit_delay_plane_wave(X, Z, a, c), dtype=torch.float32, device=dev) - t0) * fs)
            for a in self.angles
        ]
        if mode == "iq":
            from scipy.signal import firwin
            h = firwin(lp_taps, 0.6 * fc, fs=fs, window="hamming")
            self._lp = torch.as_tensor(h, dtype=torch.float32, device=dev).view(1, 1, -1)
        self._lut = None
        self._ns = None
        est = len(self.angles) * self.nz * self.nx * self.nch * 16
        self._use_lut = est <= lut_budget_bytes

    def _prep(self, ia, ns):
        """Per-angle gather indices, interpolation fraction and complex/real weights."""
        s = self.tx_samples[ia][..., None] + self.rx_samples
        i0 = torch.floor(s)
        frac = s - i0
        i0 = i0.long()
        valid = (i0 >= 0) & (i0 < ns - 1)
        idx = torch.where(valid, i0 + self.ch_off, torch.zeros_like(i0))
        wv = self.w * valid
        if self.mode == "iq":
            cycles = self.fc * (s.double() / self.fs + self.t0)       # absolute time -> carrier phase
            ph = 2 * np.pi * (cycles - torch.floor(cycles))
            wv = torch.polar(wv, ph.float())                          # w * exp(+j 2 pi fc tau)
        return idx.to(torch.int32), frac, wv

    def _build_lut(self, ns):
        self._ns = ns
        self.ch_off = torch.arange(self.nch, device=self.device) * ns
        self._lut = [self._prep(ia, ns) for ia in range(len(self.angles))] if self._use_lut else None

    def _demod(self, rf):
        """(A,nch,ns) real -> complex baseband with absolute-time reference."""
        ns = rf.shape[-1]
        t = torch.arange(ns, device=self.device, dtype=torch.float64) / self.fs + self.t0
        cyc = self.fc * t
        ph = (2 * np.pi * (cyc - torch.floor(cyc))).float()
        x = rf.reshape(-1, 1, ns)
        pad = self._lp.shape[-1] // 2
        re = torch.nn.functional.conv1d(torch.nn.functional.pad(x * torch.cos(ph), (pad, pad), mode="replicate"), self._lp)
        im = torch.nn.functional.conv1d(torch.nn.functional.pad(-x * torch.sin(ph), (pad, pad), mode="replicate"), self._lp)
        return (2 * torch.complex(re, im)).reshape(rf.shape)

    @torch.no_grad()
    def __call__(self, rf):
        """rf: (n_angles, nch, ns) -> RF image (mode rf, real) or complex baseband image (mode iq)."""
        rf = torch.as_tensor(rf, dtype=torch.float32, device=self.device)
        if rf.ndim == 4:
            rf = rf[:, 0]
        na, nch, ns = rf.shape
        assert na == len(self.angles) and nch == self.nch, "rf shape does not match geometry"
        if self._ns != ns:
            self._build_lut(ns)
        data = self._demod(rf) if self.mode == "iq" else rf
        out = torch.zeros(self.nz, self.nx, device=self.device, dtype=data.dtype)
        for ia in range(na):
            flat = data[ia].reshape(-1)
            idx, frac, wv = self._lut[ia] if self._lut is not None else self._prep(ia, ns)
            idx = idx.long()
            samp = (1 - frac) * flat[idx] + frac * flat[idx + 1]
            out += (samp * wv).sum(-1) * self.inv_wsum
        return out / na

    def envelope(self, rf):
        """Envelope image regardless of mode (Hilbert along z for rf mode)."""
        from src.signal.torch_ops import envelope
        img = self(rf)
        return img.abs() if self.mode == "iq" else envelope(img, 0)

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
