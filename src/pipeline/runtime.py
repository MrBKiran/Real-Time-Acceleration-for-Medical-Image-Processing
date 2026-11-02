"""Minimal Holoscan-style streaming runtime.

Mirrors the Holoscan programming model (Operator -> flows -> Application) so the
operators can be dropped into a real Holoscan app unchanged (see
``holoscan_app.py``).  Each operator runs in its own thread connected by small
bounded queues (back-pressure, like Holoscan's double-buffer connectors), so
acquisition, GPU processing, inference and display overlap in time.
"""
from __future__ import annotations

import queue
import threading
import time

from src.metrics import LatencyStats

_STOP = object()


class Operator:
    name = "op"

    def setup(self):          # allocate / warm up
        pass

    def compute(self, frame):  # frame: dict -> dict (or None to drop)
        raise NotImplementedError

    def teardown(self):
        pass


class SourceOperator(Operator):
    """Generates frames. Return None from ``generate`` to end the stream."""

    def generate(self, i):
        raise NotImplementedError

    def compute(self, frame):
        return frame


class Application:
    def __init__(self, sync=None, queue_depth=2):
        self.ops: list[Operator] = []
        self.sync = sync or (lambda: None)
        self.depth = queue_depth
        self.stats = LatencyStats()

    def add_flow(self, *ops):
        """Linear flow: add_flow(src, a, b, sink)."""
        self.ops.extend(o for o in ops if o not in self.ops)
        return self

    # ---------- pipelined (threaded) execution ----------
    def run(self, max_frames=100):
        src, rest = self.ops[0], self.ops[1:]
        for o in self.ops:
            o.setup()
        qs = [queue.Queue(self.depth) for _ in self.ops]
        t_start = time.perf_counter()
        n_out = [0]

        def src_loop():
            for i in range(max_frames):
                t = time.perf_counter()
                f = src.generate(i)
                if f is None:
                    break
                f["_t_in"] = time.perf_counter()
                self.stats.add(src.name, f["_t_in"] - t)
                qs[0].put(f)
            qs[0].put(_STOP)

        def op_loop(k):
            op, qin = rest[k], qs[k]
            qout = qs[k + 1] if k + 1 < len(rest) else None
            while True:
                f = qin.get()
                if f is _STOP:
                    if qout is not None:
                        qout.put(_STOP)
                    return
                t = time.perf_counter()
                out = op.compute(f)
                self.sync()
                self.stats.add(op.name, time.perf_counter() - t)
                if qout is None:
                    self.stats.add("end_to_end", time.perf_counter() - f["_t_in"])
                    n_out[0] += 1
                elif out is not None:
                    qout.put(out)

        # qs[k] feeds rest[k]; qs[len(rest)] unused
        threads = [threading.Thread(target=src_loop, daemon=True)] + [
            threading.Thread(target=op_loop, args=(k,), daemon=True) for k in range(len(rest))
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t_start
        for o in self.ops:
            o.teardown()
        return dict(frames=n_out[0], wall_s=wall, fps=n_out[0] / wall, latency=self.stats.summary())

    # ---------- serial execution: true per-frame latency ----------
    def run_serial(self, max_frames=50):
        src, rest = self.ops[0], self.ops[1:]
        for o in self.ops:
            o.setup()
        t_start = time.perf_counter()
        n = 0
        for i in range(max_frames):
            f = src.generate(i)
            if f is None:
                break
            t0 = f["_t_in"] = time.perf_counter()
            for op in rest:
                t = time.perf_counter()
                f = op.compute(f)
                self.sync()
                self.stats.add(op.name, time.perf_counter() - t)
            self.stats.add("end_to_end", time.perf_counter() - t0)
            n += 1
        wall = time.perf_counter() - t_start
        for o in self.ops:
            o.teardown()
        return dict(frames=n, wall_s=wall, fps=n / wall, latency=self.stats.summary())
