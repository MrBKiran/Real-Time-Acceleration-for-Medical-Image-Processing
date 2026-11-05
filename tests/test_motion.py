import numpy as np

from src.motion.flow import bulk_motion, dense_flow
from src.simulation.ultrasound import fast_bmode_phantom


def test_optical_flow_recovers_known_shift():
    img, _, _ = fast_bmode_phantom(256, 160, n_lesions=1, rng=np.random.default_rng(3), randomize=False)
    a = img[:, 10:138]
    b = img[:, 8:136]                       # content moves +2 px laterally
    dx, dz = bulk_motion(dense_flow(a, b), 0.2, 0.15)
    assert abs(dx - 0.4) < 0.06 and abs(dz) < 0.05


def test_no_motion_on_identical_frames():
    img, _, _ = fast_bmode_phantom(128, 96, rng=np.random.default_rng(0))
    assert np.allclose(bulk_motion(dense_flow(img, img), 0.2, 0.2), (0.0, 0.0), atol=1e-3)
