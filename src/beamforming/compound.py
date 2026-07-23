import numpy as np

from src.beamforming.delay_and_sum import beamform_single_angle


def coherent_compound(rf, X, Z, element_x, angles):
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

    nz, nx = X.shape

    compound = np.zeros((nz, nx), dtype=np.float32)

    for i, angle in enumerate(angles):

        rf_angle = rf[i, 0, :, :]

        image = beamform_single_angle(
            rf_angle,
            X,
            Z,
            element_x,
            angle
        )

        compound += image

    compound /= len(angles)

    return compound