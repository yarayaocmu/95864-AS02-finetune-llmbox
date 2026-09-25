#!/usr/bin/env python3
"""
AS02 Part A, steps 4-5: how the tokenizer "sees" the train set, and what
batching does to it.

Uses LLMBox's own TrainingDataLoader (verified_prefix masking) so the token
counts are exactly what the finetune job will feed the model.

Outputs (data/as02/part_a/):
    tokens_<model>.csv              per-example token counts (total / trainable)
    tokenizer_probe_<model>.json    how domain strings (drug names, doses,
                                    abbreviations) get split into pieces
    batch_analysis_<model>.csv      batches per epoch, padding waste, tokens per
                                    optimizer step for several batch sizes
    figures/F6_token_length_<model>.png

Run: python3 as02/part_a_tokenize_batches.py --model models/llms/google/gemma-3-270m-it
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_services import TrainingDataLoader, IGNORE_INDEX  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="local model / tokenizer directory")
parser.add_argument("--train", default=str(ROOT / "data/as02/splits/train.jsonl"))
parser.add_argument("--max-length", type=int, default=512)
parser.add_argument("--out", default=str(ROOT / "data/as02/part_a"))
args = parser.parse_args()
OUT = Path(args.out); FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
tag = Path(args.model).name.replace("-", "_")

tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
loader = TrainingDataLoader(assistant_mask_strategy="verified_prefix")

# ---- 1. tokenize the whole train split exactly as the finetune job does --
convs = loader._load_conversations_from_jsonl(args.train)
ds = loader._build_dataset(convs, tok, args.max_length)
with open(args.train) as f:
    labels = [json.loads(l)["completion"] for l in f if l.strip()]
rows = []
for ex, lab in zip(ds.examples, labels):
    n_train = sum(1 for t in ex["labels"] if t != IGNORE_INDEX)
    rows.append({"label": lab, "tokens_total": len(ex["input_ids"]),
                 "tokens_trainable": n_train,
                 "truncated": int(len(ex["input_ids"]) >= args.max_length)})
df = pd.DataFrame(rows)
df.to_csv(OUT / f"tokens_{tag}.csv", index=False)
summary = {
    "model": args.model, "vocab_size": tok.vocab_size, "max_length": args.max_length,
    "examples": int(len(df)),
    "tokens_total": df["tokens_total"].describe().round(1).to_dict(),
    "tokens_trainable": df["tokens_trainable"].describe().round(2).to_dict(),
    "share_of_tokens_that_carry_loss_pct": round(100 * df["tokens_trainable"].sum()
                                                  / df["tokens_total"].sum(), 2),
    "examples_truncated": int(df["truncated"].sum()),
    "total_tokens_one_epoch": int(df["tokens_total"].sum()),
}
# how each label is tokenised (a 1-token label is easiest to learn)
ex0 = ds.examples[0]
summary["label_token_pieces"] = {
    lab: tok.convert_ids_to_tokens(tok(lab, add_special_tokens=False)["input_ids"])
    for lab in ["POSITIVE", "NEUTRAL", "NEGATIVE"]}
summary["trainable_tokens_decoded_example"] = tok.decode(
    [t for t in ex0["labels"] if t != IGNORE_INDEX])

# ---- 2. tokenizer probe on organisationally important strings ----------
probes = ["Levofloxacin", "Etonogestrel", "Nexplanon", "Lisdexamfetamine", "Ibuprofen",
          "37.5 mg", "10-20mg", "IBS-D", "ADHD", "Type 2 diabetes", "HbA1c", "BP 140/90",
          "b.i.d.", "PRN", "&#039;", "didn&#039;t", "Bipolar Disorde", "5'6 and 188 LBS",
          "NEGATIVE", "negative", "Negative", " NEGATIVE"]
probe = {p: tok.convert_ids_to_tokens(tok(p, add_special_tokens=False)["input_ids"]) for p in probes}
summary["tokenizer_probe"] = probe

# ---- 3. batching analysis ------------------------------------------------
# Right-padding to the longest sequence in each batch (CausalLMCollator), so
# padding waste depends on batch composition. We simulate the shuffled order.
import random
lengths = df["tokens_total"].tolist()
random.Random(42).shuffle(lengths)
brows = []
for bs in [1, 2, 4, 8, 16, 32]:
    batches = [lengths[i:i + bs] for i in range(0, len(lengths), bs)]
    padded = sum(len(b) * max(b) for b in batches)
    real = sum(lengths)
    for accum in [1, 4, 8]:
        brows.append({"batch_size": bs, "grad_accum_steps": accum,
                      "effective_batch": bs * accum,
                      "micro_batches_per_epoch": len(batches),
                      "optimizer_steps_per_epoch": -(-len(batches) // accum),
                      "avg_padded_tokens_per_micro_batch": round(padded / len(batches)),
                      "padding_waste_pct": round(100 * (padded - real) / padded, 1),
                      "real_tokens_per_optimizer_step": round(real / -(-len(batches) // accum))})
bdf = pd.DataFrame(brows)
bdf.to_csv(OUT / f"batch_analysis_{tag}.csv", index=False)
summary["batch_analysis_bs_x_accum1"] = bdf[bdf["grad_accum_steps"] == 1].to_dict("records")

# ---- 4. figure ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.hist(df["tokens_total"], bins=40, color="#4C72B0")
ax.axvline(args.max_length, color="red", ls="--", label=f"max_length={args.max_length}")
ax.set_xlabel("tokens per training example (prompt + label)"); ax.set_ylabel("examples")
ax.set_title(f"F6. Token length of train examples ({Path(args.model).name})"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / f"F6_token_length_{tag}.png", dpi=150); plt.close(fig)

(OUT / f"tokenizer_probe_{tag}.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
