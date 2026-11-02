"""Train the segmentation model.

    python -m src.ai.train --data synthetic --epochs 8
    python -m src.ai.train --data busi --busi-root /path/to/Dataset_BUSI_with_GT
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from src.ai.data import BUSI, SyntheticLesions
from src.ai.model import TinyUNet, n_params


def soft_dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    d, io = [], []
    for x, y in loader:
        pred = (torch.sigmoid(model(x.to(device))) > 0.5).cpu()
        for p, g in zip(pred, y):
            p, g = p.bool(), g.bool()
            if g.sum() == 0 and p.sum() == 0:
                d.append(1.0); io.append(1.0); continue
            inter = (p & g).sum().item()
            d.append(2 * inter / (p.sum().item() + g.sum().item() + 1e-9))
            io.append(inter / ((p | g).sum().item() + 1e-9))
    return float(np.mean(d)), float(np.mean(io))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "busi"], default="synthetic")
    ap.add_argument("--busi-root", default="Dataset_BUSI_with_GT")
    ap.add_argument("--n", type=int, default=1500, help="synthetic samples")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="models/tinyunet.pt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.data == "synthetic":
        train_ds = SyntheticLesions(a.n, seed=1)
        val_ds = SyntheticLesions(max(100, a.n // 8), seed=2, augment=False)
    else:
        full = BUSI(a.busi_root)
        nv = max(1, int(0.15 * len(full)))
        train_ds, val_ds = random_split(full, [len(full) - nv, nv], torch.Generator().manual_seed(a.seed))
    tl = DataLoader(train_ds, a.batch, shuffle=True, num_workers=2, persistent_workers=True)
    vl = DataLoader(val_ds, 32, num_workers=2)

    model = TinyUNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(tl))
    print(f"TinyUNet params: {n_params(model):,}  device={device}  train={len(train_ds)} val={len(val_ds)}")
    hist = []
    for ep in range(a.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            lg = model(x)
            loss = F.binary_cross_entropy_with_logits(lg, y) + soft_dice_loss(lg, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        dice, iou = evaluate(model, vl, device)
        hist.append(dict(epoch=ep + 1, loss=tot / len(tl), val_dice=dice, val_iou=iou))
        print(f"epoch {ep+1}/{a.epochs} loss {tot/len(tl):.4f} val dice {dice:.3f} iou {iou:.3f} ({time.time()-t0:.0f}s)")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "data": a.data, "history": hist}, out)
    (out.with_suffix(".json")).write_text(json.dumps(hist, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
