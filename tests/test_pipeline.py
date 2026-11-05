import numpy as np
import torch

from scripts.common import grid
from src.ai.model import TinyUNet, n_params
from src.beamforming.geometry import element_positions
from src.beamforming.torch_das import TorchDAS
from src.enhancement.filters import Enhancer
from src.pipeline.operators import BeamformOp, CollectSink, EnhanceOp, InferOp, RFSource
from src.pipeline.runtime import Application
from src.simulation.ultrasound import default_t0, make_scatterers, simulate_plane_wave_rf


class _Det:
    def __call__(self, img):
        return torch.zeros_like(torch.as_tensor(img))


def _frames(n=3):
    ex = element_positions()
    ang = np.array([-4.0, 4.0])
    x, z, a = make_scatterers((-0.006, 0.006), (0.01, 0.025), 20, seed=1)
    t0 = default_t0(ex, ang)
    rf, _ = simulate_plane_wave_rf(x, z, a, ex, ang, 31.25e6, 7.813e6, 1540.0, 1400, t0)
    X, Z = grid(32, 64)
    return [rf] * n, TorchDAS(ex, X, Z, ang, 31.25e6, 1540.0, t0, device="cpu")


def _app():
    frames, das = _frames()
    sink = CollectSink()
    return Application().add_flow(RFSource(frames, 6), BeamformOp(das), EnhanceOp(Enhancer(["median"])),
                                  InferOp(_Det()), sink), sink


def test_pipeline_threaded_processes_all_frames_in_order():
    app, sink = _app()
    r = app.run(6)
    assert r["frames"] == 6 and [f["idx"] for f in sink.frames] == list(range(6))
    assert {"beamform", "enhance", "inference", "end_to_end"} <= set(r["latency"])


def test_pipeline_serial_latency_is_sum_of_stages():
    app, _ = _app()
    r = app.run_serial(5)
    st = r["latency"]
    parts = sum(st[k]["mean_ms"] for k in ("beamform", "enhance", "inference"))
    assert st["end_to_end"]["mean_ms"] >= 0.9 * parts


def test_unet_shapes_and_size():
    m = TinyUNet().eval()
    for hw in ((256, 128), (100, 77)):
        assert m(torch.rand(1, 1, *hw)).shape == (1, 1, *hw)
    assert n_params(m) < 300_000


def test_transformer_segmenter_shapes_and_guard():
    import pytest
    from src.ai.model import build_model
    m = build_model("transformer").eval()
    for hw in ((256, 128), (256, 256), (100, 77)):
        assert m(torch.rand(1, 1, *hw)).shape == (1, 1, *hw)
    with pytest.raises(ValueError):
        build_model("transformer", n_classes=3)
