import numpy as np
import torch

from src.ai.model import TinyUNet
from src.ai.normalize import normalize_bmode


class LesionDetector:
    """Wraps the trained U-Net; returns probability map, binary mask and detections."""

    def __init__(self, ckpt="models/tinyunet.pt", device="auto", half=False, min_area_px=40):
        from src.beamforming.torch_das import get_device
        self.device = get_device(device)
        self.model = TinyUNet().to(self.device).eval()
        state = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["state"] if "state" in state else state)
        self.half = half and self.device.type == "cuda"
        if self.half:
            self.model.half()
        self.min_area_px = min_area_px

    @torch.no_grad()
    def __call__(self, img):
        x = normalize_bmode(torch.as_tensor(img, dtype=torch.float32, device=self.device))[None, None]
        if self.half:
            x = x.half()
        return torch.sigmoid(self.model(x))[0, 0].float()

    @staticmethod
    def detections(prob, thr=0.5, min_area_px=40):
        """Connected components -> list of dict(bbox, area, score)."""
        import cv2
        m = (prob > thr).cpu().numpy().astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        p = prob.cpu().numpy()
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area >= min_area_px:
                out.append(dict(bbox=(int(x), int(y), int(w), int(h)), area=int(area),
                                score=float(p[lab == i].mean())))
        return out
