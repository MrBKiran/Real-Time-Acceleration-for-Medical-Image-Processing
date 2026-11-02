"""Envelope detection + log compression on the torch device (GPU friendly)."""
import torch


def envelope(rf_img: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """|analytic signal| along ``dim`` via FFT Hilbert transform."""
    n = rf_img.shape[dim]
    spec = torch.fft.fft(rf_img, dim=dim)
    h = torch.zeros(n, device=rf_img.device)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    shape = [1] * rf_img.ndim
    shape[dim] = n
    return torch.fft.ifft(spec * h.view(shape), dim=dim).abs()


def log_compress(env: torch.Tensor, dynamic_range: float = 60.0, ref: float | None = None) -> torch.Tensor:
    ref = env.max().clamp_min(1e-12) if ref is None else ref
    db = 20 * torch.log10(env / ref + 1e-12)
    return (db.clamp(-dynamic_range, 0) + dynamic_range) / dynamic_range
