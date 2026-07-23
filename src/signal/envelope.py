import numpy as np
from scipy.signal import hilbert


def envelope_detect(rf_image):
    """
    Compute RF envelope using Hilbert transform.
    """
    analytic = hilbert(rf_image, axis=0)
    envelope = np.abs(analytic)
    return envelope

def log_compress(envelope, dynamic_range=60):
    envelope = envelope / envelope.max()

    image = 20 * np.log10(envelope + 1e-12)

    image = np.clip(image, -dynamic_range, 0)

    image = (image + dynamic_range) / dynamic_range

    return image