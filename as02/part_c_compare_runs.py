#!/usr/bin/env python3
"""
AS02 Part C: compare finetuning runs from data/evaluations/.

For each run bundle written by the patched modes.py it collects the economic
and performance metrics, the learning curve (train/eval loss per step), and
the run info (trainable params, steps, config). Writes a comparison table and
an overlay figure of the learning curves.

Run: python3 as02/part_c_compare_runs.py [--runs exp1_baseline exp2_...]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data/evaluations"
OUT = ROOT / "data/as02/part_c"; OUT.mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument("--runs", nargs="*", default=None, help="run_name filters (default: all bundles)")
args = parser.parse_args()

bundles = []
for info in sorted(EVAL.glob("*_run_info.json")):
    ri = json.loads(info.read_text())
    if args.runs and ri.get("run_name") not in args.runs:
        continue
    stem = info.name.replace("_run_info.json", "")
    metrics = json.loads(Path(ri["metrics_file"]).read_text())
    hist = json.loads((EVAL / f"{stem}_log_history.json").read_text())
    cfg = yaml.safe_load((EVAL / f"{stem}_config.yaml").read_text())
    bundles.append((stem, ri, metrics, hist, cfg))

rows = []
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for stem, ri, m, hist, cfg in bundles:
    econ, perf = m["LLM Training Job Economic Metrics"], m["LLM Training Performance"]
    tr = cfg["training"]; op = cfg["optimizer"]
    train_pts = [(h["step"], h["loss"]) for h in hist if "loss" in h]
    eval_pts = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
    first_loss = train_pts[0][1] if train_pts else None
    best_eval = min((v for _, v in eval_pts), default=None)
    rows.append({
        "run": ri.get("run_name") or stem,
        "method": tr["method"], "epochs": tr["epochs"], "batch_size": tr["batch_size"],
        "grad_accum": tr["grad_accum_steps"], "effective_batch": tr["batch_size"] * tr["grad_accum_steps"],
        "lr": op["learning_rate"], "warmup_ratio": tr["warmup_ratio"], "max_length": tr["max_length"],
        "lora_r": tr["lora_r"], "lora_alpha": tr["lora_alpha"],
        "target_modules": ",".join(tr.get("lora_target_modules", [])),
        "trainable_params": ri["trainable_params"], "trainable_pct": ri["trainable_pct"],
        "train_examples": ri["train_examples"], "eval_examples": ri["eval_examples"],
        "optimizer_steps": ri["global_steps"],
        "elapsed_s": round(econ["elapsed_time"], 1), "elapsed_min": round(econ["elapsed_time"] / 60, 1),
        "sec_per_step": round(econ["elapsed_time"] / max(ri["global_steps"], 1), 2),
        "peak_mem_GB": round(econ["max_memory_bytes"] / 1024**3, 2), "mem_pct": round(econ["memory_used_percent"], 1),
        "avg_cpu_pct": round(econ["avg_cpu_util_percent"] or 0, 1),
        "tokens_processed": econ["tokens_processed"], "tokens_source": econ["tokens_source"],
        "token_budget_used_pct": round(econ["token_budget_used_percent"], 2),
        "flops": econ["flops_estimate"], "flops_level": econ["flops_level"],
        "energy_kWh": econ["energy_kwh"], "carbon_kg": econ["carbon_kg_estimate"],
        "first_train_loss": first_loss, "final_train_loss": perf["loss"], "final_perplexity": perf["perplexity"],
        "final_eval_loss": perf["eval_loss"], "final_eval_perplexity": perf["eval_perplexity"],
        "best_eval_loss": best_eval, "n_eval_points": len(eval_pts),
    })
    label = ri.get("run_name") or stem
    if train_pts:
        axes[0].plot(*zip(*train_pts), label=label, alpha=0.85)
    if eval_pts:
        axes[1].plot(*zip(*eval_pts), marker="o", label=label)
axes[0].set_title("Train loss (label tokens only)"); axes[0].set_xlabel("optimizer step"); axes[0].set_ylabel("loss")
axes[1].set_title("Eval loss"); axes[1].set_xlabel("optimizer step")
for ax in axes:
    ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "learning_curves.png", dpi=150); plt.close(fig)

df = pd.DataFrame(rows)
df.to_csv(OUT / "run_comparison.csv", index=False)
(OUT / "run_comparison.md").write_text(df.T.to_markdown())
with pd.option_context("display.max_rows", 100, "display.width", 200):
    print(df.T)
print("\n->", OUT)
