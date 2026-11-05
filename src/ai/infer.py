import numpy as np
import torch

from src.ai.model import build_model
from src.ai.normalize import normalize_bmode


class LesionDetector:
    """Wraps the trained U-Net; returns probability map, binary mask and detections."""

    def __init__(self, ckpt="models/tinyunet.pt", device="auto", half=False, min_area_px=40):
        from src.beamforming.torch_das import get_device
        self.device = get_device(device)
        state = torch.load(ckpt, map_location=self.device, weights_only=False)
        n_classes = state.get("n_classes", 0) if isinstance(state, dict) else 0
        self.classes = list(state.get("classes", [])) if n_classes else []
        self.arch = state.get("arch", "unet") if isinstance(state, dict) else "unet"
        self.model = build_model(self.arch, n_classes).to(self.device).eval()
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

    @torch.no_grad()
    def segment_and_classify(self, img):
        """(probability map, {class: probability}) - needs a checkpoint trained with --classify."""
        if not self.classes:
            return self(img), None
        x = normalize_bmode(torch.as_tensor(img, dtype=torch.float32, device=self.device))[None, None]
        if self.half:
            x = x.half()
        seg, cl = self.model.forward_both(x)
        p = torch.softmax(cl.float(), 1)[0].tolist()
        return torch.sigmoid(seg)[0, 0].float(), dict(zip(self.classes, p))

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
