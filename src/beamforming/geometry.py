import numpy as np

from src.config import PITCH, N_ELEMENTS


def element_positions(n_elements=N_ELEMENTS, pitch=PITCH):
    """x-position (m) of each transducer element, centred on x = 0."""
    return (np.arange(n_elements) - (n_elements - 1) / 2) * pitch


def transmit_delay_plane_wave(X, Z, angle_deg, c=1540.0):
    """Plane-wave transmit time (s) to reach (X, Z).  Single source of truth for
    the sign convention shared by the simulator and every beamformer."""
    theta = np.deg2rad(-angle_deg)
    return (X * np.sin(theta) + Z * np.cos(theta)) / c
