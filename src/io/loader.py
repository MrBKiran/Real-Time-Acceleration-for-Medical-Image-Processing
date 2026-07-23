import h5py
import numpy as np


def load_rf(filepath):
    """Load RF data from a MATLAB v7.3 file."""
    with h5py.File(filepath, "r") as f:
        return np.array(f["USDATA"])


def load_header(filepath):
    """Load the USHEADER dataset from a MATLAB v7.3 file."""
    with h5py.File(filepath, "r") as f:
        header = np.array(f["USHEADER"])

    return header