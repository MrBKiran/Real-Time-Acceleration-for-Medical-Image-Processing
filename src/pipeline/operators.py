"""Concrete operators: RF source -> beamform -> enhance -> infer -> sink."""
from __future__ import annotations

import numpy as np
import torch

from src.pipeline.runtime import Operator, SourceOperator
from src.signal.torch_ops import envelope, log_compress


class RFSource(SourceOperator):
    """Streams pre-recorded / simulated RF frames, mimicking an ADC front end.

    ``fps`` throttles to a real acquisition rate (None = as fast as possible).
    """
    name = "rf_source"

    def __init__(self, frames, n_frames=None, fps=None):
        self.frames = frames          # list of (n_angles, nch, ns) arrays
        self.n_frames, self.fps = n_frames, fps
        self._next = 0.0

    def generate(self, i):
        import time
        if self.n_frames is not None and i >= self.n_frames:
            return None
        if self.fps:
            now = time.perf_counter()
            self._next = max(self._next, now) + 1.0 / self.fps if i else now + 1.0 / self.fps
            if self._next > now:
                time.sleep(self._next - now)
        return {"idx": i, "rf": self.frames[i % len(self.frames)]}


class BeamformOp(Operator):
    name = "beamform"

    def __init__(self, das, dynamic_range=60.0):
        self.das, self.dr = das, dynamic_range

    def setup(self):          # warm-up builds the look-up tables
        pass  # LUTs are built lazily on the first frame (needs the record length)

    def compute(self, f):
        env = self.das.envelope(f["rf"])
        f["bmode_raw"] = log_compress(env, self.dr)
        f["env"] = env
        return f


class EnhanceOp(Operator):
    name = "enhance"

    def __init__(self, enhancer):
        self.enh = enhancer

    def compute(self, f):
        f["bmode"] = self.enh(f["bmode_raw"])
        return f


class InferOp(Operator):
    name = "inference"

    def __init__(self, detector, thr=0.5):
        self.det, self.thr = detector, thr

    def setup(self):
        self.det(torch.zeros(256, 128))

    def compute(self, f):
        if getattr(self.det, "classes", None):
            f["prob"], f["class_prob"] = self.det.segment_and_classify(f["bmode"])
        else:
            f["prob"] = self.det(f["bmode"])
        return f


class MotionOp(Operator):
    """Speckle-tracking motion estimate between consecutive frames (optical flow).

    Adds ``motion_mm`` = (lateral, axial) displacement since the previous frame.
    Runs on the raw (unfiltered) B-mode: speckle is the pattern being tracked.
    """
    name = "motion"

    def __init__(self, dx_mm, dz_mm, key="bmode_raw"):
        self.dx, self.dz, self.key = dx_mm, dz_mm, key
        self._prev = None

    def setup(self):
        self._prev = None

    def compute(self, f):
        from src.motion.flow import bulk_motion, dense_flow
        cur = f[self.key].detach().cpu().numpy()
        f["motion_mm"] = (0.0, 0.0) if self._prev is None else bulk_motion(dense_flow(self._prev, cur), self.dx, self.dz)
        self._prev = cur
        return f


class CollectSink(Operator):
    """Keeps (a decimated subset of) result frames for display / video."""
    name = "sink"

    def __init__(self, keep_every=1, to_cpu=True):
        self.keep_every, self.to_cpu = keep_every, to_cpu
        self.frames = []

    def compute(self, f):
        if f["idx"] % self.keep_every == 0:
            out = {"idx": f["idx"], "motion_mm": f.get("motion_mm"), "class_prob": f.get("class_prob")}
            for k in ("bmode_raw", "bmode", "prob"):
                if k in f:
                    out[k] = f[k].detach().cpu().numpy() if self.to_cpu else f[k]
            self.frames.append(out)
        return f
