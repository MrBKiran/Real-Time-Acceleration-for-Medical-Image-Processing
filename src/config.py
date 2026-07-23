import numpy as np

# Physical constants
C = 1540.0                 # Speed of sound (m/s)
FS = 31.25e6               # Sampling frequency (Hz)
FC = 7.813e6               # Center frequency (Hz)
F_NUMBER = 1.5

# Probe geometry
PITCH = 195.3e-6           # Element spacing (m)
N_ELEMENTS = 128

# Acquisition
N_ANGLES = 75
N_SAMPLES = 2944

# Steering angles (degrees)
ANGLES_DEG = np.linspace(-16.0, 16.0, N_ANGLES)

# Steering angles (radians)
ANGLES_RAD = np.deg2rad(ANGLES_DEG)