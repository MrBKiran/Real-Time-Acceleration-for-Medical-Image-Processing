"""NVIDIA Holoscan deployment of the same pipeline (Holoscan SDK, Python API).

STATUS: written against the public Holoscan Python API but NOT executed in the
development container (it has no NVIDIA GPU / Holoscan SDK).  Run it inside the
official container, e.g.

    docker run --rm -it --gpus all nvcr.io/nvidia/clara-holoscan/holoscan:v2.9.0-dgpu \
        python -m src.pipeline.holoscan_app --rf frames.npz --ckpt models/tinyunet.pt

The operators only delegate to the framework-independent callables used by the
standalone runtime (TorchDAS / Enhancer / LesionDetector), so numerical results
are identical.  The Holoscan/CUDA hand-off point is the torch tensor on the GPU.
"""
from __future__ import annotations

import argparse

import numpy as np

try:
    from holoscan.conditions import CountCondition
    from holoscan.core import Application, Operator, OperatorSpec
    HAVE_HOLOSCAN = True
except ImportError:                     # pragma: no cover
    HAVE_HOLOSCAN = False
    Operator = object


if HAVE_HOLOSCAN:
    class RFSourceOp(Operator):
        def __init__(self, fragment, frames, *a, **k):
            super().__init__(fragment, *a, **k)
            self.frames, self.i = frames, 0

        def setup(self, spec: OperatorSpec):
            spec.output("rf")

        def compute(self, op_input, op_output, context):
            op_output.emit({"idx": self.i, "rf": self.frames[self.i % len(self.frames)]}, "rf")
            self.i += 1

    class _Wrap(Operator):
        """Holoscan operator that delegates to a runtime Operator."""
        def __init__(self, fragment, inner, *a, **k):
            super().__init__(fragment, *a, **k)
            self.inner = inner

        def start(self):
            self.inner.setup()

        def setup(self, spec: OperatorSpec):
            spec.input("in")
            spec.output("out")

        def compute(self, op_input, op_output, context):
            op_output.emit(self.inner.compute(op_input.receive("in")), "out")

    class SinkOp(Operator):
        def __init__(self, fragment, inner, *a, **k):
            super().__init__(fragment, *a, **k)
            self.inner = inner

        def setup(self, spec: OperatorSpec):
            spec.input("in")

        def compute(self, op_input, op_output, context):
            self.inner.compute(op_input.receive("in"))

    class UltrasoundApp(Application):
        def __init__(self, frames, ops, n_frames=200):
            super().__init__()
            self._frames, self._ops, self._n = frames, ops, n_frames

        def compose(self):
            src = RFSourceOp(self, self._frames, CountCondition(self, self._n), name="rf_source")
            bf, enh, inf, sink = self._ops
            bf_, enh_, inf_ = (_Wrap(self, o, name=o.name) for o in (bf, enh, inf))
            sink_ = SinkOp(self, sink, name="sink")
            self.add_flow(src, bf_, {("rf", "in")})
            self.add_flow(bf_, enh_, {("out", "in")})
            self.add_flow(enh_, inf_, {("out", "in")})
            self.add_flow(inf_, sink_, {("out", "in")})


def main():                              # pragma: no cover
    if not HAVE_HOLOSCAN:
        raise SystemExit("Holoscan SDK not installed - use src.pipeline.app for the standalone runtime.")
    from scripts.common import build_components
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf", required=True)
    ap.add_argument("--ckpt", default="models/tinyunet.pt")
    ap.add_argument("--frames", type=int, default=200)
    a = ap.parse_args()
    frames, ops = build_components(a.rf, a.ckpt)
    UltrasoundApp(frames, ops, a.frames).run()


if __name__ == "__main__":
    main()
