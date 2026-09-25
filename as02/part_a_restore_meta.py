#!/usr/bin/env python3
"""
AS02 Part A helper. LLMBox `mode=prepare_data` re-standardises records in
"existing prompt/completion" mode and keeps only prompt/completion/text/messages,
so the `meta` block (uniqueID, rating, TLP label, chronic flag, drug ...) that
part_a_transform.py attached is dropped from train.jsonl / test.jsonl.

This script re-attaches `meta` by matching on the prompt (prompts are unique;
checked in part_a_inspect.py). The model never sees `meta`; the training loader
only reads `messages`. Run after `mode=prepare_data`:
    python3 as02/part_a_restore_meta.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = [json.loads(l) for l in open(ROOT / "data/as02/transformed/medrx_experience.jsonl") if l.strip()]
by_prompt = {r["prompt"]: r["meta"] for r in src}
for name in ("train", "test"):
    p = ROOT / "data/as02/splits" / f"{name}.jsonl"
    rows = [json.loads(l) for l in open(p) if l.strip()]
    missing = 0
    for r in rows:
        if "meta" not in r:
            m = by_prompt.get(r["prompt"])
            if m is None:
                missing += 1
            else:
                r["meta"] = m
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)} rows, meta restored, {missing} without match")
