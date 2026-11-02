"""Ultrasound simulators used when the public datasets are not available.

Two levels of fidelity are provided:

* :func:`simulate_plane_wave_rf` - point-scatterer channel-data simulator
  (linear interpolation of echo arrival times, Gaussian-modulated pulse).
  Produces ``(n_angles, n_channels, n_samples)`` RF data that goes through the
  *real* beamforming code, so it validates geometry and timing end to end.
* :func:`fast_bmode_phantom` - a fast fully-developed-speckle model that makes
  an envelope image plus ground-truth lesion mask directly.  Used to generate
  thousands of labelled training frames for the AI stage.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

from src.beamforming.geometry import transmit_delay_plane_wave


def gaussian_pulse(fs, fc, bandwidth_frac=0.6):
    """Gaussian-modulated sinusoid (pulse-echo), peak at the centre sample."""
    sigma_t = 1.0 / (np.pi * bandwidth_frac * fc)  # -6 dB fractional bandwidth
    half = int(np.ceil(4 * sigma_t * fs))
    t = np.arange(-half, half + 1) / fs
    return (np.exp(-0.5 * (t / sigma_t) ** 2) * np.cos(2 * np.pi * fc * t)).astype(np.float32)


def default_t0(element_x, angles_deg, c=1540.0, pad=1.0e-6):
    """Time (s) of sample 0 so that the earliest wavefront is inside the record."""
    xe = np.asarray(element_x)
    earliest = min(
        transmit_delay_plane_wave(xe, np.zeros_like(xe), a, c).min() for a in angles_deg
    )
    return float(earliest - pad)


def simulate_plane_wave_rf(
    scat_x, scat_z, scat_amp, element_x, angles_deg,
    fs=31.25e6, fc=7.813e6, c=1540.0, n_samples=2944, t0=None,
    bandwidth_frac=0.6, chunk=20000, noise_db=None, seed=0,
):
    """Simulate unfocused plane-wave RF channel data from point scatterers."""
    scat_x = np.asarray(scat_x, np.float32)
    scat_z = np.asarray(scat_z, np.float32)
    scat_amp = np.asarray(scat_amp, np.float32)
    ex = np.asarray(element_x, np.float32)
    nch = ex.size
    if t0 is None:
        t0 = default_t0(ex, angles_deg, c)
    rf = np.zeros((len(angles_deg), nch, n_samples), np.float32)
    pulse = gaussian_pulse(fs, fc, bandwidth_frac)
    for ia, ang in enumerate(angles_deg):
        acc = np.zeros(nch * n_samples, np.float64)
        for s in range(0, scat_x.size, chunk):
            x = scat_x[s:s + chunk, None]
            z = scat_z[s:s + chunk, None]
            a = scat_amp[s:s + chunk, None]
            tau = transmit_delay_plane_wave(x, z, ang, c) + np.sqrt((x - ex) ** 2 + z ** 2) / c
            # element obliquity (cosine directivity) on receive
            obliq = z / np.sqrt((x - ex) ** 2 + z ** 2)
            samp = (tau - t0) * fs
            i0 = np.floor(samp).astype(np.int64)
            w = (samp - i0).astype(np.float32)
            amp = a * obliq
            ch = np.broadcast_to(np.arange(nch), i0.shape) * n_samples
            for off, wt in ((0, 1 - w), (1, w)):
                idx = i0 + off
                ok = (idx >= 0) & (idx < n_samples)
                acc += np.bincount((ch + idx)[ok], (amp * wt)[ok], minlength=acc.size)
        rf[ia] = fftconvolve(acc.reshape(nch, n_samples), pulse[None], mode="same", axes=1)
    if noise_db is not None:
        rng = np.random.default_rng(seed)
        rf += rng.normal(0, rf.std() * 10 ** (noise_db / 20), rf.shape).astype(np.float32)
    return rf, float(t0)


def make_scatterers(x_lim, z_lim, density_per_mm2=40.0, lesions=(), points=(), seed=0):
    """Random speckle scatterers + optional lesions (ellipses) / bright points.

    lesions: iterable of dict(cx, cz, rx, rz, gain) in metres; gain<1 hypoechoic,
    gain>1 hyperechoic, 0 = anechoic cyst.  points: iterable of (x, z, amp).
    """
    rng = np.random.default_rng(seed)
    area_mm2 = (x_lim[1] - x_lim[0]) * (z_lim[1] - z_lim[0]) * 1e6
    n = int(area_mm2 * density_per_mm2)
    x = rng.uniform(*x_lim, n).astype(np.float32)
    z = rng.uniform(*z_lim, n).astype(np.float32)
    a = rng.normal(0, 1, n).astype(np.float32)
    for l in lesions:
        inside = ((x - l["cx"]) / l["rx"]) ** 2 + ((z - l["cz"]) / l["rz"]) ** 2 < 1
        a[inside] *= l["gain"]
    if len(points):
        px, pz, pa = (np.array(v, np.float32) for v in zip(*points))
        x, z, a = np.r_[x, px], np.r_[z, pz], np.r_[a, pa]
    return x, z, a


def fast_bmode_phantom(nz=256, nx=128, n_lesions=None, rng=None,
                       speckle_sigma=(1.1, 1.5), allow_hyper=True):
    """Fast speckle phantom.  Returns (log-compressed image in [0,1], mask uint8, env).

    Fully developed speckle = |complex Gaussian noise (x) PSF|.  Lesions modulate
    the underlying scatterer amplitude, giving realistic CNR behaviour.
    """
    rng = rng or np.random.default_rng()
    amp = np.ones((nz, nx), np.float32)
    mask = np.zeros((nz, nx), np.uint8)
    zz, xx = np.mgrid[0:nz, 0:nx]
    n = rng.integers(0, 3) if n_lesions is None else n_lesions
    for _ in range(n):
        cz, cx = rng.uniform(0.2 * nz, 0.85 * nz), rng.uniform(0.2 * nx, 0.8 * nx)
        rz, rx = rng.uniform(0.05 * nz, 0.14 * nz), rng.uniform(0.07 * nx, 0.2 * nx)
        ang = rng.uniform(0, np.pi)
        dz, dx = zz - cz, xx - cx
        u = (dx * np.cos(ang) + dz * np.sin(ang)) / rx
        v = (-dx * np.sin(ang) + dz * np.cos(ang)) / rz
        inside = u ** 2 + v ** 2 < 1
        gain = rng.choice([0.0, 0.05, 0.15, 3.0] if allow_hyper else [0.0, 0.05, 0.15])
        amp[inside] = max(gain, 0.01)
        mask[inside] = 1
    # slow attenuation / TGC residual
    amp *= np.exp(-np.linspace(0, rng.uniform(0.0, 0.6), nz))[:, None]
    noise = rng.normal(size=(2, nz, nx)).astype(np.float32) * amp
    re = gaussian_filter(noise[0], speckle_sigma)
    im = gaussian_filter(noise[1], speckle_sigma)
    env = np.hypot(re, im)
    from src.signal.envelope import log_compress
    return log_compress(env, 60).astype(np.float32), mask, env.astype(np.float32)
