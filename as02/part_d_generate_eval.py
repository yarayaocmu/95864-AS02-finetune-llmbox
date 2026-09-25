#!/usr/bin/env python3
"""
AS02 Part D: run a (base or finetuned) model over an evaluation set and score it.

Why a script and not only the interactive chat? The assignment asks for 100+
evaluation prompts. This script sends the prompts, records every response in
the SAME chat-log JSON shape that LLMBox's `mode=chat` writes (so
`mode=evaluate` can consume it), and computes the organisational metrics:
    * label-format compliance (did the model answer with exactly one label?)
    * accuracy, macro-F1, per-class recall, confusion matrix
    * NEGATIVE recall (our safety-critical class: a missed NEGATIVE report
      hides a poorly tolerated drug from the clinician)
    * accuracy on the chronic-condition subgroup and on unseen drugs

Inputs are LLMBox JSONL records (prompt/completion/messages[/meta]); the
`completion` is used as the gold label and never shown to the model.

Run (from llmbox root):
    python3 as02/part_d_generate_eval.py --model finetuned_exp1/merged \
        --data data/as02/splits/test.jsonl --n 100 --tag exp1_test
    python3 as02/part_d_generate_eval.py --model models/llms/google/gemma-3-270m-it \
        --data data/as02/splits/test.jsonl --n 100 --tag base_test
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["POSITIVE", "NEUTRAL", "NEGATIVE"]

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="model dir (base, or finetuned/merged)")
parser.add_argument("--adapter", default=None, help="optional LoRA adapter dir to load on top of --model")
parser.add_argument("--data", required=True, help="LLMBox jsonl with prompt/completion")
parser.add_argument("--n", type=int, default=100, help="number of prompts (random sample, seed 42)")
parser.add_argument("--tag", required=True, help="label for output files")
parser.add_argument("--max-new-tokens", type=int, default=8)
parser.add_argument("--greedy", action="store_true", default=True,
                    help="deterministic decoding (default). Use --sample to sample.")
parser.add_argument("--sample", dest="greedy", action="store_false")
parser.add_argument("--system-prompt", default="",
                    help="optional system prompt (the instruction is already inside each prompt)")
parser.add_argument("--out", default=str(ROOT / "data/as02/part_d"))
parser.add_argument("--username", default="yaoyuan")
args = parser.parse_args()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)

# ---- load model ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, local_files_only=True)
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
model.to(device).eval()

# ---- load data ----------------------------------------------------------
records = [json.loads(l) for l in open(args.data) if l.strip()]
random.Random(42).shuffle(records)
records = records[:args.n]

def parse_label(text: str) -> str | None:
    """First label word in the response; None if the model did not comply."""
    m = re.search(r"\b(POSITIVE|NEUTRAL|NEGATIVE)\b", text.upper())
    return m.group(1) if m else None

# ---- generate -----------------------------------------------------------
started = datetime.now(timezone.utc).isoformat()
session = {"session": {"session_id": uuid.uuid4().hex[:8], "started_at": started,
                       "model": {"path": args.model, "adapter": args.adapter, "dtype": "bfloat16", "device": device},
                       "generation": {"max_new_tokens": args.max_new_tokens, "do_sample": not args.greedy},
                       "system_prompt": args.system_prompt, "data": args.data, "n": len(records)},
           "turns": []}
rows = []
t0 = time.time()
for i, rec in enumerate(records, 1):
    msgs = ([{"role": "system", "content": args.system_prompt}] if args.system_prompt else [])
    msgs.append({"role": "user", "content": rec["prompt"]})
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=not args.greedy,
                             pad_token_id=tok.pad_token_id)
    answer = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    pred = parse_label(answer)
    gold = rec["completion"].strip().upper()
    meta = rec.get("meta", {})
    session["turns"].append({"turn": i,
                             "user": {"username": args.username, "content": rec["prompt"],
                                      "timestamp": datetime.now(timezone.utc).isoformat()},
                             "assistant": {"content": answer, "timestamp": datetime.now(timezone.utc).isoformat()},
                             "gold": gold, "pred": pred, "meta": meta})
    rows.append({"i": i, "gold": gold, "pred": pred, "raw": answer, "exact_format": answer.upper() == gold or answer.upper() in LABELS,
                 "is_chronic": meta.get("is_chronic"), "drug": meta.get("drug"), "rating": meta.get("rating")})
    if i % 20 == 0:
        print(f"[{i}/{len(records)}] {time.time() - t0:.0f}s  last: gold={gold} pred={pred!r} raw={answer[:40]!r}")
elapsed = time.time() - t0

# ---- score --------------------------------------------------------------
n = len(rows)
compliant = sum(1 for r in rows if r["raw"].strip().upper() in LABELS)
parsed = sum(1 for r in rows if r["pred"] is not None)
correct = sum(1 for r in rows if r["pred"] == r["gold"])
conf = {g: {p: 0 for p in LABELS + ["NONE"]} for g in LABELS}
for r in rows:
    conf[r["gold"]][r["pred"] or "NONE"] += 1
per_class = {}
f1s = []
for c in LABELS:
    tp = conf[c][c]; fn = sum(conf[c].values()) - tp
    fp = sum(conf[g][c] for g in LABELS if g != c)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec_ = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0
    per_class[c] = {"support": tp + fn, "precision": round(prec, 3), "recall": round(rec_, 3), "f1": round(f1, 3)}
    f1s.append(f1)
def acc(sub):
    return round(sum(1 for r in sub if r["pred"] == r["gold"]) / len(sub), 3) if sub else None
scores = {
    "tag": args.tag, "model": args.model, "adapter": args.adapter, "data": args.data, "n": n,
    "seconds": round(elapsed, 1), "sec_per_prompt": round(elapsed / n, 2),
    "format_compliance_exact_label_only": round(compliant / n, 3),
    "parseable_label_anywhere": round(parsed / n, 3),
    "accuracy": round(correct / n, 3),
    "macro_f1": round(sum(f1s) / len(f1s), 3),
    "negative_recall": per_class["NEGATIVE"]["recall"],
    "majority_class_baseline_accuracy": round(max(sum(1 for r in rows if r["gold"] == c) for c in LABELS) / n, 3),
    "per_class": per_class, "confusion(gold->pred)": conf,
    "accuracy_chronic_subgroup": acc([r for r in rows if r.get("is_chronic") == 1]),
    "accuracy_non_chronic": acc([r for r in rows if r.get("is_chronic") == 0]),
    "off_by_two_errors (POSITIVE<->NEGATIVE)": conf["POSITIVE"]["NEGATIVE"] + conf["NEGATIVE"]["POSITIVE"],
    "example_non_compliant_outputs": [r["raw"][:120] for r in rows if r["raw"].strip().upper() not in LABELS][:5],
}
(OUT / f"chatlog_{args.tag}.json").write_text(json.dumps(session, indent=2, ensure_ascii=False))
(OUT / f"scores_{args.tag}.json").write_text(json.dumps(scores, indent=2))
# flat jsonl (query/response/gold) for mode=evaluate and the LLM judge
with open(OUT / f"evalset_{args.tag}.jsonl", "w") as f:
    for t in session["turns"]:
        f.write(json.dumps({"query": t["user"]["content"], "response": t["assistant"]["content"],
                            "gold": t["gold"], "pred": t["pred"], "meta": t["meta"]}, ensure_ascii=False) + "\n")
print(json.dumps(scores, indent=2))
print(f"\nwrote {OUT / f'chatlog_{args.tag}.json'}, scores_{args.tag}.json, evalset_{args.tag}.jsonl")
