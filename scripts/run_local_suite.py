"""One command for the GPU / real-data steps that cannot run in the cloud sandbox.

    python -m scripts.run_local_suite --busi-root Dataset_BUSI_with_GT

Runs, in order (each step's full output is saved; a failing step is reported and the
suite continues with the next one):
  1. environment check (torch, CUDA, GPU name)
  2. simulated RF sequence (only if data/sim_sequence.npz is missing)
  3. streaming pipeline on the GPU (per-stage latency)          -> docs/results/
  4. stage benchmarks                                           -> docs/results/benchmark.json
  5. BUSI + synthetic training with the classification head     -> models/tinyunet_busi.pt
     (the synthetic-only models/tinyunet.pt is kept)
  6. cross-simulator evaluation of both checkpoints
  7. CNN vs transformer comparison on this device (if models/tinysegformer.pt exists)
  8. optional: your real Zenodo files -> converter (--t0 auto) -> pipeline

Results: docs/results/local_run/summary.json and summary.md (paste-ready).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "local_run"


def run(name, args, log):
    print(f"\n=== {name} ===\n  {' '.join(args)}", flush=True)
    t = time.time()
    p = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    (OUT / f"{name}.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr, encoding="utf-8")
    ok = p.returncode == 0
    tail = "\n".join((p.stdout.strip().splitlines() or [""])[-12:])
    print(tail if ok else f"FAILED (exit {p.returncode}) - see {OUT / (name + '.log')}\n" + p.stderr[-2000:], flush=True)
    log[name] = dict(ok=ok, seconds=round(time.time() - t, 1), returncode=p.returncode)
    return ok


def jload(p):
    p = ROOT / p
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--busi-root", default="Dataset_BUSI_with_GT")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--no-classify", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--zenodo-glob", default=None, help='e.g. "7986407/CIRS073_RUMC/**/USDATA_*.mat"')
    ap.add_argument("--zenodo-header", default=None, help='e.g. "7986407/CIRS040GSE/CIRS040GSE/USHEADER_*.mat"')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log, summary = {}, {}

    import torch
    env = dict(python=sys.version.split()[0], torch=torch.__version__, cuda_available=torch.cuda.is_available(),
               cuda_build=torch.version.cuda,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    summary["environment"] = env
    print("environment:", env)
    if not env["cuda_available"]:
        print("WARNING: CUDA not available - timings will be CPU timings. See README for installing the CUDA build of torch.")

    if not (ROOT / "data" / "sim_sequence.npz").exists():
        run("simulate", ["-m", "scripts.simulate_dataset", "--out", "data/sim_sequence.npz", "--frames", "12"], log)
    if run("pipeline", ["-m", "scripts.run_pipeline", "--data", "data/sim_sequence.npz", "--frames", "36"], log):
        pr = jload("docs/results/pipeline_results.json")
        summary["pipeline"] = dict(device=pr["device"], n_angles=pr["n_angles"],
                                   per_stage_ms={k: round(v["mean_ms"], 2) for k, v in pr["serial"]["latency"].items()},
                                   end_to_end_p95_ms=round(pr["serial"]["latency"]["end_to_end"]["p95_ms"], 1),
                                   pipelined_fps=round(pr["pipelined"]["fps"], 2), motion=pr.get("motion"),
                                   quality=pr.get("quality_frame0"))
    if run("benchmark", ["-m", "scripts.benchmark"], log):
        summary["benchmark"] = jload("docs/results/benchmark.json")

    busi = (ROOT / a.busi_root) if not Path(a.busi_root).is_absolute() else Path(a.busi_root)
    ck = "models/tinyunet_busi.pt"
    if a.skip_train:
        print("\n(skipping training)")
    elif not (busi / "benign").is_dir():
        print(f"\nBUSI not found at {busi} (needs benign/malignant/normal sub-folders) - skipping training")
        log["train_busi"] = dict(ok=False, skipped="BUSI not found")
    else:
        args = ["-m", "src.ai.train", "--data", "all", "--busi-root", str(busi), "--epochs", str(a.epochs),
                "--init", "models/tinyunet.pt", "--out", ck]
        if not a.no_classify:
            args.append("--classify")
        if run("train_busi", args, log):
            st = torch.load(ROOT / ck, map_location="cpu", weights_only=False)
            best = st["history"][st["best_epoch"] - 1]
            summary["busi_training"] = dict(best_epoch=st["best_epoch"], epochs=len(st["history"]),
                                            **{k: v for k, v in best.items() if k.startswith("busi_") or k.startswith("synthetic_")})

    for name, c in (("eval_synthetic_model", "models/tinyunet.pt"), ("eval_busi_model", ck)):
        if (ROOT / c).exists() and run(name, ["-m", "scripts.evaluate_ai", "--ckpt", c,
                                              "--out", f"docs/results/local_run/{name}.json"], log):
            summary[name] = jload(f"docs/results/local_run/{name}.json")
    if (ROOT / "models" / "tinysegformer.pt").exists() and run(
            "compare_models", ["-m", "scripts.compare_models", "--out", "docs/results/local_run/model_comparison.json"], log):
        summary["model_comparison"] = jload("docs/results/local_run/model_comparison.json")

    if a.zenodo_glob and a.zenodo_header:
        if run("convert_zenodo", ["-m", "scripts.convert_zenodo_mat", "--glob", a.zenodo_glob, "--header", a.zenodo_header,
                                  "--n-angles", "5", "--t0", "auto", "--out", "data/zenodo.npz"], log):
            summary["zenodo_conversion"] = jload("data/zenodo_conversion.json")
            if run("pipeline_zenodo", ["-m", "scripts.run_pipeline", "--data", "data/zenodo.npz", "--frames", "30",
                                       "--out", "docs/results/zenodo"], log):
                pz = jload("docs/results/zenodo/pipeline_results.json")
                summary["pipeline_zenodo"] = dict(per_stage_ms={k: round(v["mean_ms"], 2) for k, v in pz["serial"]["latency"].items()},
                                                  pipelined_fps=round(pz["pipelined"]["fps"], 2))

    summary["steps"] = log
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (OUT / "summary.md").write_text(to_markdown(summary), encoding="utf-8")
    print(f"\nDone. {sum(v.get('ok', False) for v in log.values())}/{len(log)} steps OK. "
          f"Results: {OUT / 'summary.md'} and summary.json")


def to_markdown(s):
    L = ["# Local run summary", ""]
    e = s["environment"]
    L.append(f"- torch {e['torch']} (CUDA build {e['cuda_build']}), CUDA available: {e['cuda_available']}, GPU: {e['gpu']}")
    if "pipeline" in s:
        p = s["pipeline"]
        L += ["", f"## Pipeline ({p['device']}, {p['n_angles']} angles, 256x128)", "", "| stage | mean ms |", "|---|---|"]
        L += [f"| {k} | {v} |" for k, v in p["per_stage_ms"].items()]
        L += ["", f"End-to-end p95 {p['end_to_end_p95_ms']} ms; pipelined throughput {p['pipelined_fps']} fps."]
        if p.get("motion"):
            L.append(f"Motion: {p['motion']['est_lateral_mean']:.4f} mm/frame (true {p['motion']['true_lateral_mm_per_frame']}).")
    if "busi_training" in s:
        b = s["busi_training"]
        L += ["", f"## BUSI training (best epoch {b['best_epoch']}/{b['epochs']})", ""]
        for k in ("busi_dice", "busi_iou", "busi_acc", "busi_balanced_acc", "synthetic_dice"):
            if k in b:
                L.append(f"- {k}: {b[k]:.3f}")
        if "busi_confusion" in b:
            L += ["", "Confusion (rows true, cols predicted: benign, malignant, normal):", ""]
            L += [f"    {r}" for r in b["busi_confusion"]]
    for k in ("eval_synthetic_model", "eval_busi_model"):
        if k in s:
            L.append(f"- {k}: cross-simulator Dice (enhanced) {s[k]['enhanced']['dice']:.3f}, "
                     f"detections/frame {s[k]['mean_detections_per_frame']:.2f}")
    if "model_comparison" in s:
        L += ["", "## CNN vs transformer", "", "| model | params | in-dist Dice | cross-sim Dice | ms/frame |", "|---|---|---|---|---|"]
        for k, r in s["model_comparison"].items():
            L.append(f"| {k} | {r['params']:,} | {r['in_distribution_dice']:.3f} | {r['cross_simulator_dice_enhanced']:.3f} | {r['latency_ms_batch1']:.2f} |")
    L += ["", "## Steps", ""] + [f"- {k}: {'OK' if v.get('ok') else 'FAILED/SKIPPED'} ({v.get('seconds', '-')} s)" for k, v in s["steps"].items()]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
