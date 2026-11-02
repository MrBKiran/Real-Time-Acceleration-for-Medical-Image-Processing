"""Lightweight U-Net for lesion segmentation / detection on B-mode frames."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class TinyUNet(nn.Module):
    """~120k parameters at base=8 - small enough for real-time edge inference."""

    def __init__(self, base=8, in_ch=1, out_ch=1):
        super().__init__()
        b = base
        self.e1, self.e2, self.e3 = _block(in_ch, b), _block(b, 2 * b), _block(2 * b, 4 * b)
        self.bott = _block(4 * b, 8 * b)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, 2), _block(8 * b, 4 * b)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, 2), _block(4 * b, 2 * b)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * b, b, 2, 2), _block(2 * b, b)
        self.head = nn.Conv2d(b, out_ch, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        ph, pw = (-H) % 8, (-W) % 8                       # pad to multiple of 8
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.bott(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.head(d1)[..., :H, :W]


def n_params(m):
    return sum(p.numel() for p in m.parameters())
