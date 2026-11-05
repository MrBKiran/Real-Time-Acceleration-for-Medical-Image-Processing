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
    """~120k parameters at base=8 - small enough for real-time edge inference.

    ``n_classes > 0`` adds an image-level classification head (e.g. BUSI
    benign / malignant / normal) on the globally pooled bottleneck, so one
    forward pass gives both the lesion mask and the class.  ``forward`` always
    returns only the segmentation logits (pipeline / ONNX compatible);
    ``forward_both`` returns (seg_logits, class_logits).
    """

    def __init__(self, base=8, in_ch=1, out_ch=1, n_classes=0):
        super().__init__()
        self.n_classes = n_classes
        b = base
        self.e1, self.e2, self.e3 = _block(in_ch, b), _block(b, 2 * b), _block(2 * b, 4 * b)
        self.bott = _block(4 * b, 8 * b)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, 2), _block(8 * b, 4 * b)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, 2), _block(4 * b, 2 * b)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * b, b, 2, 2), _block(2 * b, b)
        self.head = nn.Conv2d(b, out_ch, 1)
        if n_classes:
            # input: [global mean, mask-weighted mean, global max] of bottleneck features + lesion area
            self.cls_head = nn.Sequential(nn.Linear(3 * 8 * b + 1, 32), nn.ReLU(inplace=True),
                                          nn.Dropout(0.2), nn.Linear(32, n_classes))

    def forward(self, x):
        return self._run(x)[0]

    def forward_both(self, x):
        seg, b = self._run(x)
        if not self.n_classes:
            raise RuntimeError("model has no classification head (n_classes=0)")
        # lesion-focused pooling: the class depends on what is inside the lesion, its size and shape
        H, W = x.shape[-2:]
        m = torch.sigmoid(F.pad(seg, (0, b.shape[-1] * 8 - W, 0, b.shape[-2] * 8 - H))).detach()
        m = F.avg_pool2d(m, 8)                                            # to bottleneck resolution
        area = m.mean((-2, -1))                                           # (N, 1) lesion area fraction
        masked = (b * m).sum((-2, -1)) / (m.sum((-2, -1)) + 1e-3)
        feats = torch.cat([b.mean((-2, -1)), masked, b.amax((-2, -1)), area], 1)
        return seg, self.cls_head(feats)

    def _run(self, x):
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
        return self.head(d1)[..., :H, :W], b


def n_params(m):
    return sum(p.numel() for p in m.parameters())


class _PosConv(nn.Module):
    """Conditional positional encoding (depthwise 3x3 conv on the token grid) - works at any input size."""

    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, t, hw):
        n, L, c = t.shape
        g = t.transpose(1, 2).reshape(n, c, *hw)
        return t + self.dw(g).flatten(2).transpose(1, 2)


class TinySegFormer(nn.Module):
    """Small transformer segmenter for the CNN-vs-transformer comparison.

    Conv stem to 1/4 resolution (skip), patch embedding to 1/8 (256x128 input ->
    32x16 = 512 tokens), ``depth`` pre-norm transformer blocks with conditional
    positional encoding, light conv decoder with one skip.  Segmentation only.
    """

    def __init__(self, dim=64, depth=4, heads=4, n_classes=0):
        super().__init__()
        if n_classes:
            raise ValueError("TinySegFormer supports segmentation only (use TinyUNet for --classify)")
        self.n_classes = 0
        cbr = lambda i, o, s: nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.stem = nn.Sequential(cbr(1, 16, 2), cbr(16, 32, 2))                 # 1/4
        self.embed = nn.Conv2d(32, dim, 2, 2)                                    # 1/8 tokens
        self.pos = _PosConv(dim)
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout=0.0, batch_first=True,
                                                               norm_first=True, activation="gelu") for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.up1 = cbr(dim, 32, 1)
        self.fuse = cbr(64, 32, 1)
        self.head = nn.Sequential(cbr(32, 16, 1), nn.Conv2d(16, 1, 1))

    def forward(self, x):
        H, W = x.shape[-2:]
        ph, pw = (-H) % 8, (-W) % 8
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        s4 = self.stem(x)
        g = self.embed(s4)
        n, c, h, w = g.shape
        t = g.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.blocks):
            t = blk(t)
            if i == 0:
                t = self.pos(t, (h, w))
        g = self.norm(t).transpose(1, 2).reshape(n, c, h, w)
        y = F.interpolate(self.up1(g), size=s4.shape[-2:], mode="bilinear", align_corners=False)
        y = self.fuse(torch.cat([y, s4], 1))
        y = F.interpolate(self.head(y), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return y[..., :H, :W]


ARCHS = {"unet": TinyUNet, "transformer": TinySegFormer}


def build_model(arch="unet", n_classes=0):
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; choose from {list(ARCHS)}")
    return ARCHS[arch](n_classes=n_classes)
