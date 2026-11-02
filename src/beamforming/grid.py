import numpy as np


def create_grid(x_min=-0.015, x_max=0.015, z_min=0.005, z_max=0.060, nx=256, nz=512):
    """Cartesian image grid. Returns X, Z of shape (nz, nx)."""
    x = np.linspace(x_min, x_max, nx)
    z = np.linspace(z_min, z_max, nz)
    return np.meshgrid(x, z)
