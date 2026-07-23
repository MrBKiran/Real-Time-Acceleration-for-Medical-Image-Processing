import numpy as np

from src.config import PITCH, N_ELEMENTS


def element_positions():
    """
    Returns the x-position of each transducer element.
    """
    x = (np.arange(N_ELEMENTS) - (N_ELEMENTS - 1) / 2) * PITCH
    return x