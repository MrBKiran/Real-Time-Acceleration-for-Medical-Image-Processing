"""Export a trained segmenter to ONNX (deploy with TensorRT / ONNX Runtime / Holoscan inference op).

    python -m scripts.export_onnx --ckpt models/tinyunet.pt --out models/tinyunet.onnx

Exports the segmentation output (``logits``); a checkpoint trained with
--classify additionally exports ``class_logits``.  Needs: pip install onnx onnxscript
"""
import argparse

import torch

from src.ai.model import build_model


class _Both(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.forward_both(x)


ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="models/tinyunet.pt")
ap.add_argument("--out", default="models/tinyunet.onnx")
a = ap.parse_args()
st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
n_classes = st.get("n_classes", 0)
m = build_model(st.get("arch", "unet"), n_classes).eval()
m.load_state_dict(st["state"])
net, outs = (_Both(m), ["logits", "class_logits"]) if n_classes else (m, ["logits"])
dyn = {"bmode": {0: "n"}} | {o: {0: "n"} for o in outs}
torch.onnx.export(net, torch.zeros(1, 1, 256, 128), a.out, input_names=["bmode"], output_names=outs,
                  dynamic_axes=dyn, opset_version=17)
print("exported", a.out, "outputs", outs)
