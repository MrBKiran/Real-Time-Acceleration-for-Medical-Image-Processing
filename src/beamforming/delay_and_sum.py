
import numpy as np

from src.config import C, FS, F_NUMBER
from src.beamforming.geometry import transmit_delay_plane_wave

def transmit_delay(X, Z, angle_deg, c=C):
    return transmit_delay_plane_wave(X, Z, angle_deg, c)


def receive_delay(X, Z, element_x, c=C):
    return np.sqrt(
        (X[..., None]-element_x)**2 +
        Z[..., None]**2
    )/c


def beamform_single_angle(rf, X, Z, element_x, angle_deg, fs=FS, c=C, t0=0.0):
    """NumPy reference (slow, easy to read). The fast path is TorchDAS."""

    tx = transmit_delay(X, Z, angle_deg, c)

    rx = receive_delay(X, Z, element_x, c)

    total_delay = tx[..., None] + rx

    sample = (total_delay - t0) * fs

    idx0 = np.floor(sample).astype(np.int32)
    idx1 = idx0 + 1

    weight = sample - idx0

    nz, nx = X.shape
    n_samples = rf.shape[1]

    image = np.zeros((nz, nx), dtype=np.float32)
    
    weights = np.hanning(rf.shape[0]).astype(np.float32)
    
    min_aperture = 16 * (element_x[1] - element_x[0])   # 16 elements

    aperture_width = np.maximum(
        Z / (2 * F_NUMBER),
        min_aperture
    )

    for ch in range(rf.shape[0]):
        
        valid = (idx0[:, :, ch] >= 0) & (idx1[:, :, ch] < n_samples)

        aperture = np.abs(X - element_x[ch]) < aperture_width

        valid &= aperture

        tmp = np.zeros((nz, nx), dtype=np.float32)

        s0 = rf[ch, idx0[:, :, ch][valid]]
        s1 = rf[ch, idx1[:, :, ch][valid]]

        tmp[valid] = (
            (1 - weight[:, :, ch][valid]) * s0
            + weight[:, :, ch][valid] * s1
            )

        image += weights[ch] * tmp

    image /= np.sum(weights)

    return image