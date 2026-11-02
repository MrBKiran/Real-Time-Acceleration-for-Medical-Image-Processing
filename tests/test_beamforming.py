import numpy as np
import torch

from scripts.common import grid
from src.beamforming.compound import coherent_compound
from src.beamforming.geometry import element_positions
from src.beamforming.torch_das import TorchDAS
from src.signal.torch_ops import envelope, log_compress
from src.simulation.ultrasound import default_t0, make_scatterers, simulate_plane_wave_rf

FS, FC, C = 31.25e6, 7.813e6, 1540.0
EX = element_positions()


def _point_image(angles, pts, nx=96, nz=160, mode="iq"):
    x, z, a = zip(*pts)
    t0 = default_t0(EX, angles, C)
    rf, _ = simulate_plane_wave_rf(np.array(x), np.array(z), np.array(a), EX, angles, FS, FC, C, 2200, t0)
    X, Zg = grid(nx, nz)
    das = TorchDAS(EX, X, Zg, angles, FS, C, t0, device="cpu", mode=mode, fc=FC)
    env = das.envelope(rf).numpy()
    return env, X, Zg, das, rf, t0


def test_point_target_is_localised():
    pts = [(0.002, 0.025, 1.0)]
    env, X, Z, *_ = _point_image(np.array([-6.0, 0.0, 6.0]), pts)
    iz, ix = np.unravel_index(env.argmax(), env.shape)
    assert abs(X[iz, ix] - 0.002) < 0.4e-3
    assert abs(Z[iz, ix] - 0.025) < 0.4e-3


def test_two_targets_resolved_and_angle_sign_convention():
    pts = [(-0.003, 0.02, 1.0), (0.004, 0.032, 1.0)]
    env, X, Z, *_ = _point_image(np.array([-8.0, 0.0, 8.0]), pts, nx=120, nz=900)  # fine grid: PSF is ~0.12 mm in z
    for px, pz, _ in pts:  # peak near each target
        win = (abs(X - px) < 1e-3) & (abs(Z - pz) < 1e-3)
        assert env[win].max() > 0.8 * env.max()


def test_torch_matches_numpy_reference():
    angles = np.array([-5.0, 0.0, 5.0])
    pts = [(0.0, 0.02, 1.0), (0.004, 0.03, 0.7)]
    env, X, Z, das, rf, t0 = _point_image(angles, pts, nx=48, nz=400, mode="rf")  # fine dz: RF mode needs it
    ref = coherent_compound(rf, X, Z, EX, angles, fs=FS, c=C, t0=t0)
    got = das(rf).numpy()  # rf mode -> real
    # references differ only in apodisation (static Hann vs dynamic Hann) -> compare envelopes' peaks
    e_ref = np.abs(__import__("scipy.signal").signal.hilbert(ref, axis=0))
    e_got = envelope(torch.tensor(got), 0).numpy()
    assert np.unravel_index(e_ref.argmax(), e_ref.shape) == np.unravel_index(e_got.argmax(), e_got.shape)
    c = np.corrcoef(e_ref.ravel(), e_got.ravel())[0, 1]
    assert c > 0.9


def test_lut_and_on_the_fly_paths_agree():
    angles = np.array([-4.0, 4.0])
    env, X, Z, das, rf, t0 = _point_image(angles, [(0.0, 0.02, 1.0)], nx=32, nz=48)
    das2 = TorchDAS(EX, X, Z, angles, FS, C, t0, device="cpu", lut_budget_bytes=0, fc=FC)
    a, b = das(rf), das2(rf)
    assert (a - b).abs().max() < 1e-4 * a.abs().max()


def test_envelope_of_sinusoid_is_flat():
    t = torch.arange(1024)
    s = torch.cos(2 * np.pi * 0.125 * t)[:, None]
    e = envelope(s, 0)
    assert torch.allclose(e[100:900], torch.ones(800, 1), atol=1e-3)
    lc = log_compress(e)
    assert lc.max() <= 1 and lc.min() >= 0


def test_iq_peak_position_and_amplitude_on_target_centred_grid():
    angles = np.array([-6.0, 0.0, 6.0])
    pts = [(0.0, 0.02, 1.0), (0.004, 0.03, 0.7)]
    t0 = default_t0(EX, angles, C)
    rf, _ = simulate_plane_wave_rf(*zip(*pts), EX, angles, FS, FC, C, 2200, t0)
    peaks = {}
    for mode in ("iq", "rf"):
        for px, pz, pa in pts:
            X, Z = np.meshgrid(np.linspace(px - 1e-3, px + 1e-3, 41), np.linspace(pz - 0.5e-3, pz + 0.5e-3, 101))
            e = TorchDAS(EX, X, Z, angles, FS, C, t0, device="cpu", mode=mode, fc=FC).envelope(rf).numpy()
            iz, ix = np.unravel_index(e.argmax(), e.shape)
            assert abs(X[iz, ix] - px) < 50e-6 and abs(Z[iz, ix] - pz) < 30e-6
            peaks[mode, pa] = e.max()
    assert abs(peaks["iq", 0.7] / peaks["iq", 1.0] - 0.7) < 0.08       # amplitude is linear in scatterer strength
    # linear interpolation of the 4-samples/cycle carrier makes rf mode read low (documented reason for IQ)
    assert peaks["iq", 1.0] > peaks["rf", 1.0]


def test_iq_is_stable_on_coarse_grid_where_rf_aliases():
    """dz = 0.18 mm > lambda/2: IQ must still give a clean, centred point response."""
    pts = [(0.001, 0.025, 1.0)]
    env, X, Z, *_ = _point_image(np.array([-6.0, 0.0, 6.0]), pts, nx=96, nz=256)
    iz, ix = np.unravel_index(env.argmax(), env.shape)
    assert abs(Z[iz, ix] - 0.025) < 0.3e-3 and abs(X[iz, ix] - 0.001) < 0.3e-3
    far = (abs(Z - 0.025) > 2e-3) | (abs(X - 0.001) > 2e-3)
    assert env[far].max() < 0.25 * env.max()
