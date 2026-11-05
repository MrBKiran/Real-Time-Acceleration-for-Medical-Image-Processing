"""Train the lesion segmenter (and optionally a benign/malignant/normal classifier).

    # everything that has lesion masks: real BUSI + synthetic phantoms mixed per epoch
    python -m src.ai.train --data all --busi-root path/to/Dataset_BUSI_with_GT

    # + image-level classification head trained on the BUSI class labels
    python -m src.ai.train --data all --busi-root path/to/Dataset_BUSI_with_GT --classify

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


def _unpack(batch):
    x, y = batch[0], batch[1]
    return x, y, (batch[2] if len(batch) > 2 else None)


@torch.no_grad()
def evaluate(model, loader, device, n_classes=0):
    """Mean Dice / IoU per image (empty-vs-empty counts as perfect); with a class
    head also accuracy and a confusion matrix (rows = true, cols = predicted)."""
    model.eval()
    d, io = [], []
    conf = np.zeros((n_classes, n_classes), int) if n_classes else None
    for batch in loader:
        x, y, lab = _unpack(batch)
        x = x.to(device)
        if n_classes:
            seg, cl = model.forward_both(x)
            for t, p in zip(lab.tolist(), cl.argmax(1).cpu().tolist()):
                if t >= 0:
                    conf[t, p] += 1
        else:
            seg = model(x)
        pred = (torch.sigmoid(seg) > 0.5).cpu()
        for p, g in zip(pred, y):
            p, g = p.bool(), g.bool()
            if g.sum() == 0 and p.sum() == 0:
                d.append(1.0); io.append(1.0); continue
            inter = (p & g).sum().item()
            d.append(2 * inter / (p.sum().item() + g.sum().item() + 1e-9))
            io.append(inter / ((p | g).sum().item() + 1e-9))
    out = dict(dice=float(np.mean(d)), iou=float(np.mean(io)))
    if conf is not None and conf.sum():
        out["acc"] = float(np.trace(conf) / conf.sum())
        out["balanced_acc"] = float(np.mean([conf[i, i] / conf[i].sum() for i in range(n_classes) if conf[i].sum()]))
        out["confusion"] = conf.tolist()
    return out


def split_busi(root, val_frac, seed, classify=False):
    """Deterministic, class-stratified image-level split.  NOTE: BUSI has no patient
    IDs, so images of the same patient can land on both sides -> validation scores
    are optimistic."""
    full = BUSI(root)
    labels = np.array(full.labels())
    rng = np.random.default_rng(seed)
    va = []
    for c in np.unique(labels):
        ids = rng.permutation(np.flatnonzero(labels == c))
        va.extend(ids[:max(1, int(round(val_frac * len(ids))))].tolist())
    va = np.array(sorted(va))
    tr = np.setdiff1d(np.arange(len(full)), va)
    # when classifying, don't crop: a crop can cut the lesion out of a 'benign' image
    crop = 0.0 if classify else 0.7
    return (BUSI(root, augment=True, indices=tr, return_label=classify, crop_prob=crop),
            BUSI(root, augment=False, indices=va, return_label=classify))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "busi", "all"], default="synthetic")
    ap.add_argument("--busi-root", default="Dataset_BUSI_with_GT")
    ap.add_argument("--classify", action="store_true",
                    help="also train a benign/malignant/normal head (needs BUSI)")
    ap.add_argument("--cls-weight", type=float, default=0.5, help="weight of the classification loss")
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
    if a.classify and a.data == "synthetic":
        ap.error("--classify needs BUSI class labels: use --data busi or --data all")

    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and a.amp
    n_classes = len(BUSI.CLASSES) if a.classify else 0
    lab = dict(return_label=a.classify)

    vals = {}                                   # name -> dataset
    class_w = None
    if a.data == "synthetic":
        train_ds = SyntheticLesions(a.n, seed=1)
        vals["synthetic"] = SyntheticLesions(max(100, a.n // 8), seed=2, augment=False, enhance_prob=1.0)
        sampler = None
    else:
        busi_tr, busi_va = split_busi(a.busi_root, a.val_frac, a.seed, a.classify)
        vals["busi"] = busi_va
        if a.classify:
            counts = np.bincount(busi_tr.labels(), minlength=n_classes).astype(float)
            class_w = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), dtype=torch.float32, device=device)
            print("BUSI train class counts", dict(zip(BUSI.CLASSES, counts.astype(int).tolist())),
                  "-> loss weights", [round(w, 2) for w in class_w.tolist()])
        if a.data == "busi":
            train_ds, sampler = busi_tr, None
        else:
            syn = SyntheticLesions(a.n, seed=1, **lab)
            train_ds = ConcatDataset([busi_tr, syn])
            w_b = a.busi_fraction / len(busi_tr)
            w_s = (1 - a.busi_fraction) / len(syn)
            weights = [w_b] * len(busi_tr) + [w_s] * len(syn)
            sampler = WeightedRandomSampler(weights, num_samples=int(max(len(busi_tr) / a.busi_fraction, a.n)), replacement=True)
            vals["synthetic"] = SyntheticLesions(150, seed=2, augment=False, enhance_prob=1.0, **lab)
    nw = a.workers
    kw = dict(num_workers=nw, persistent_workers=nw > 0, pin_memory=device.type == "cuda")
    tl = DataLoader(train_ds, a.batch, shuffle=sampler is None, sampler=sampler, drop_last=True, **kw)
    vls = {k: DataLoader(v, 16, num_workers=0) for k, v in vals.items()}   # small; avoids extra worker processes

    model = TinyUNet(n_classes=n_classes).to(device)
    if a.init:
        st = torch.load(a.init, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(st["state"] if "state" in st else st, strict=False)
        bad = [k for k in missing if not k.startswith("cls_head")]
        if bad or unexpected:
            raise SystemExit(f"--init checkpoint does not match the model (missing {bad}, unexpected {unexpected})")
        print("initialised from", a.init, "(new classification head)" if missing else "")
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(tl))
    scaler = torch.amp.GradScaler(enabled=amp)
    sizes = {k: len(v) for k, v in vals.items()}
    print(f"TinyUNet params: {n_params(model):,}  device={device}  amp={amp}  classes={n_classes or 'off'}  "
          f"train batches/epoch={len(tl)}  val={sizes}")

    # data sanity check: catches NaN/inf or wrong-range inputs before they poison the weights
    xb, yb, _ = _unpack(next(iter(tl)))
    print(f"batch check: x {tuple(xb.shape)} [{xb.min():.3f}, {xb.max():.3f}] mean {xb.mean():.3f} finite={bool(torch.isfinite(xb).all())} | "
          f"y {tuple(yb.shape)} lesion-pixel fraction {yb.mean():.3f} finite={bool(torch.isfinite(yb).all())}")
    if not (torch.isfinite(xb).all() and torch.isfinite(yb).all()):
        raise SystemExit("non-finite values in the training data - check the dataset files")
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise SystemExit("initial weights contain NaN/inf - delete the --init checkpoint and retry")

    key = "busi" if "busi" in vls else "synthetic"      # model selection set

    def score(r):                                       # Dice, or mean of Dice and balanced accuracy when classifying
        return (r[f"{key}_dice"] + r[f"{key}_balanced_acc"]) / 2 if f"{key}_balanced_acc" in r else r[f"{key}_dice"]

    best, hist = -1.0, []
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(a.epochs):
        model.train()
        t0, tot, nbad, nok = time.time(), 0.0, 0, 0
        for batch in tl:
            x, y, lb = _unpack(batch)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
                nbad += 1                                    # skip corrupt samples instead of training on them
                continue
            with torch.autocast(device.type, enabled=amp):
                if n_classes:
                    lg, cl = model.forward_both(x)
                else:
                    lg, cl = model(x), None
            lg = lg.float()
            loss = F.binary_cross_entropy_with_logits(lg, y) + soft_dice_loss(lg, y)
            if cl is not None:
                lb = lb.to(device)
                if (lb >= 0).any():                          # synthetic samples carry label -1 (no class)
                    loss = loss + a.cls_weight * F.cross_entropy(cl.float(), lb, weight=class_w, ignore_index=-1)
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
        rec = dict(epoch=ep + 1, loss=tot / nok)
        for k, vl in vls.items():
            for m, v in evaluate(model, vl, device, n_classes if k == "busi" else 0).items():
                rec[f"{k}_{m}"] = v
        hist.append(rec)
        msg = " ".join(f"{k} dice {rec[k + '_dice']:.3f}" for k in vls)
        if "busi_acc" in rec:
            msg += f" | busi acc {rec['busi_acc']:.3f} (balanced {rec['busi_balanced_acc']:.3f})"
        flag = ""
        if score(rec) > best:
            best = score(rec)
            torch.save({"state": model.state_dict(), "data": a.data, "history": hist, "best_epoch": ep + 1,
                        "n_classes": n_classes, "classes": list(BUSI.CLASSES) if n_classes else []}, out)
            flag = "  *saved (best)"
        print(f"epoch {ep+1}/{a.epochs} loss {rec['loss']:.4f} {msg} ({time.time()-t0:.0f}s){flag}")
    out.with_suffix(".json").write_text(json.dumps(hist, indent=1))
    best_rec = max(hist, key=score)
    if "busi_confusion" in best_rec:
        print("confusion matrix at best epoch (rows true, cols predicted:", ", ".join(BUSI.CLASSES) + ")")
        for name, row in zip(BUSI.CLASSES, best_rec["busi_confusion"]):
            print(f"  {name:10s} {row}")
    sel = f"(dice + balanced acc)/2 {best:.3f}" if "busi_balanced_acc" in best_rec else f"dice {best:.3f}"
    print(f"done. best epoch {best_rec['epoch']}: {key} dice {best_rec[key + '_dice']:.3f}"
          + (f", balanced acc {best_rec['busi_balanced_acc']:.3f}" if "busi_balanced_acc" in best_rec else "")
          + f" [selected on {sel}] -> {out}")


if __name__ == "__main__":
    main()
