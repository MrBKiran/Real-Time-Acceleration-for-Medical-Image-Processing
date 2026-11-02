import numpy as np

from src.beamforming.delay_and_sum import beamform_single_angle


def coherent_compound(rf, X, Z, element_x, angles, fs=None, c=None, t0=0.0):
    """
    Coherently compound multiple plane-wave transmissions.

    Parameters
    ----------
    rf : ndarray
        Shape (n_angles, 1, n_channels, n_samples)

    X, Z : ndarray

    element_x : ndarray

    angles : ndarray
        Steering angles in degrees

    Returns
    -------
    compound : ndarray
        Beamformed RF image
    """

    from src.config import FS, C
    fs = FS if fs is None else fs
    c = C if c is None else c
    nz, nx = X.shape

    compound = np.zeros((nz, nx), dtype=np.float32)

    for i, angle in enumerate(angles):

        rf_angle = rf[i, 0] if rf.ndim == 4 else rf[i]

        image = beamform_single_angle(
            rf_angle,
            X,
            Z,
            element_x,
            angle, fs=fs, c=c, t0=t0
        )

        compound += image

    compound /= len(angles)

    return compound