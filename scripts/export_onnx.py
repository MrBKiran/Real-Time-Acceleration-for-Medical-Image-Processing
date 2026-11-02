"""Export the trained segmenter to ONNX (deploy with TensorRT / ONNX Runtime / Holoscan inference op).

    python -m scripts.export_onnx --ckpt models/tinyunet.pt --out models/tinyunet.onnx
"""
import argparse

import torch

from src.ai.model import TinyUNet

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="models/tinyunet.pt")
ap.add_argument("--out", default="models/tinyunet.onnx")
a = ap.parse_args()
m = TinyUNet().eval()
m.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["state"])
torch.onnx.export(m, torch.zeros(1, 1, 256, 128), a.out, input_names=["bmode"], output_names=["logits"],
                  dynamic_axes={"bmode": {0: "n"}, "logits": {0: "n"}}, opset_version=17)
print("exported", a.out)
