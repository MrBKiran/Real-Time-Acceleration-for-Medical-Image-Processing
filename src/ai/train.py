"""Train the lesion segmenter.

    # everything that has lesion masks: real BUSI + synthetic phantoms mixed per epoch
    python -m src.ai.train --data all --busi-root path/to/Dataset_BUSI_with_GT

    # synthetic only / BUSI only
    python -m src.ai.train --data synthetic
    python -m src.ai.train --data busi --busi-root path/to/Dataset_BUSI_with_GT

Only BUSI (and the synthetic phantoms) carry lesion masks.  The Zenodo raw-RF
datasets have no segmentation labels, so they are used to test the pipeline, not
to train this model.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from src.ai.data import BUSI, SyntheticLesions
from src.ai.model import TinyUNet, n_params


def soft_dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    """Mean Dice / IoU per image (empty-vs-empty counts as perfect)."""
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


def split_busi(root, val_frac, seed):
    """Deterministic image-level split.  NOTE: BUSI has no patient IDs, so images
    of the same patient can land on both sides -> validation Dice is optimistic."""
    full = BUSI(root)
    idx = np.random.default_rng(seed).permutation(len(full))
    nv = max(1, int(val_frac * len(full)))
    return (BUSI(root, augment=True, indices=idx[nv:]),
            BUSI(root, augment=False, indices=idx[:nv]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "busi", "all"], default="synthetic")
    ap.add_argument("--busi-root", default="Dataset_BUSI_with_GT")
    ap.add_argument("--n", type=int, default=1500, help="synthetic samples per epoch pool")
    ap.add_argument("--busi-fraction", type=float, default=0.5,
                    help="with --data all: share of each epoch drawn from BUSI (oversampled)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--init", default=None, help="checkpoint to fine-tune from, e.g. models/tinyunet.pt")
    ap.add_argument("--workers", type=int, default=0 if os.name == "nt" else 2,
                    help="DataLoader processes (default 0 on Windows: spawn + paging-file limits)")
    ap.add_argument("--amp", action="store_true", help="fp16 mixed precision on CUDA (off by default: the model is tiny)")
    ap.add_argument("--clip", type=float, default=1.0, help="gradient-norm clip (0 = off)")
    ap.add_argument("--out", default="models/tinyunet.pt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and a.amp

    vals = {}                                   # name -> loader
    if a.data == "synthetic":
        train_ds = SyntheticLesions(a.n, seed=1)
        vals["synthetic"] = SyntheticLesions(max(100, a.n // 8), seed=2, augment=False, enhance_prob=1.0)
        sampler = None
    else:
        busi_tr, busi_va = split_busi(a.busi_root, a.val_frac, a.seed)
        vals["busi"] = busi_va
        if a.data == "busi":
            train_ds, sampler = busi_tr, None
        else:
            syn = SyntheticLesions(a.n, seed=1)
            train_ds = ConcatDataset([busi_tr, syn])
            w_b = a.busi_fraction / len(busi_tr)
            w_s = (1 - a.busi_fraction) / len(syn)
            weights = [w_b] * len(busi_tr) + [w_s] * len(syn)
            sampler = WeightedRandomSampler(weights, num_samples=int(max(len(busi_tr) / a.busi_fraction, a.n)), replacement=True)
            vals["synthetic"] = SyntheticLesions(150, seed=2, augment=False, enhance_prob=1.0)
    nw = a.workers
    kw = dict(num_workers=nw, persistent_workers=nw > 0, pin_memory=device.type == "cuda")
    tl = DataLoader(train_ds, a.batch, shuffle=sampler is None, sampler=sampler, drop_last=True, **kw)
    vls = {k: DataLoader(v, 16, num_workers=0) for k, v in vals.items()}   # small; avoids extra worker processes

    model = TinyUNet().to(device)
    if a.init:
        st = torch.load(a.init, map_location=device, weights_only=False)
        model.load_state_dict(st["state"] if "state" in st else st)
        print("initialised from", a.init)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(tl))
    scaler = torch.amp.GradScaler(enabled=amp)
    sizes = {k: len(v) for k, v in vals.items()}
    print(f"TinyUNet params: {n_params(model):,}  device={device}  amp={amp}  train batches/epoch={len(tl)}  val={sizes}")

    # data sanity check: catches NaN/inf or wrong-range inputs before they poison the weights
    xb, yb = next(iter(tl))
    print(f"batch check: x {tuple(xb.shape)} [{xb.min():.3f}, {xb.max():.3f}] mean {xb.mean():.3f} finite={bool(torch.isfinite(xb).all())} | "
          f"y {tuple(yb.shape)} lesion-pixel fraction {yb.mean():.3f} finite={bool(torch.isfinite(yb).all())}")
    if not (torch.isfinite(xb).all() and torch.isfinite(yb).all()):
        raise SystemExit("non-finite values in the training data - check the dataset files")
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise SystemExit("initial weights contain NaN/inf - delete the --init checkpoint and retry")

    key = "busi" if "busi" in vls else "synthetic"      # model selection metric
    best, hist = -1.0, []
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(a.epochs):
        model.train()
        t0, tot, nbad, nok = time.time(), 0.0, 0, 0
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
                nbad += 1                                    # skip corrupt samples instead of training on them
                continue
            with torch.autocast(device.type, enabled=amp):
                lg = model(x)
            lg = lg.float()
            loss = F.binary_cross_entropy_with_logits(lg, y) + soft_dice_loss(lg, y)
            if not torch.isfinite(loss):
                nbad += 1
                opt.zero_grad(set_to_none=True)
                if nbad > 20:
                    raise SystemExit("loss became NaN/inf repeatedly. Try without --amp, a lower --lr (e.g. 1e-3), or check the data.")
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if a.clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            nok += 1
        if nbad:
            print(f"  warning: skipped {nbad} bad batches this epoch")
        if nok == 0:
            raise SystemExit("no valid batches in this epoch")
        tot_loss = tot / nok
        rec = dict(epoch=ep + 1, loss=tot_loss)
        for k, vl in vls.items():
            rec[f"{k}_dice"], rec[f"{k}_iou"] = evaluate(model, vl, device)
        hist.append(rec)
        msg = " ".join(f"{k} dice {rec[k + '_dice']:.3f}" for k in vls)
        flag = ""
        if rec[f"{key}_dice"] > best:
            best = rec[f"{key}_dice"]
            torch.save({"state": model.state_dict(), "data": a.data, "history": hist, "best_epoch": ep + 1}, out)
            flag = "  *saved (best)"
        print(f"epoch {ep+1}/{a.epochs} loss {rec['loss']:.4f} {msg} ({time.time()-t0:.0f}s){flag}")
    out.with_suffix(".json").write_text(json.dumps(hist, indent=1))
    print(f"done. best {key} dice {best:.3f} -> {out}")


if __name__ == "__main__":
    main()
