#!/usr/bin/env python3
"""
AS02 Part D (LLM Judge, non-interactive runner).

Runs the Pydantic-validated local judge (src/pydantic_models/experience_judge.py)
over an evaluation set produced by part_d_generate_eval.py, and compares the
judge's Likert ratings with (a) the gold labels and (b) an optional human
labeled file from `mode=evaluate` (labeled_evaluation_dataset.json) using
percent agreement and Cohen's kappa from LLMBox's AgreementMeasures.

The interactive `mode=evaluate` path is still the graded one: there, paste
    from src.pydantic_models.experience_judge import judge_fn
when asked for a labeling function. This script exists so the whole set can
be judged reproducibly and the agreement numbers land in a file.

Run: python3 as02/part_d_llm_judge.py --evalset data/as02/part_d/evalset_exp2_test.jsonl \
        --criteria "..." [--human data/as02/part_d/labeled_evaluation_dataset.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluator import AgreementMeasures  # noqa: E402
from src.pydantic_models.experience_judge import judge_structured, likert_from_labels  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--evalset", required=True)
parser.add_argument("--criteria", default=("The response is exactly one of POSITIVE / NEUTRAL / NEGATIVE "
                                           "and matches the experience a clinician would read from the report."))
parser.add_argument("--human", default=None, help="labeled_evaluation_dataset.json from mode=evaluate")
parser.add_argument("--n", type=int, default=None)
args = parser.parse_args()

rows = [json.loads(l) for l in open(args.evalset) if l.strip()]
if args.n:
    rows = rows[:args.n]
out_rows, judge_lk, gold_lk = [], [], []
for i, ex in enumerate(rows, 1):
    v = judge_structured(ex, args.criteria)
    g_lk, _, _ = likert_from_labels(ex["gold"], ex["response"])  # rubric applied with the GOLD label
    out_rows.append({**ex, "judge_label": v.likert, "judge_verdict": v.model_dump(), "gold_likert": g_lk})
    judge_lk.append(v.likert); gold_lk.append(g_lk)
    if i % 10 == 0:
        print(f"[{i}/{len(rows)}] judge_class={v.judge_label} gold={ex['gold']} likert={v.likert}")

res = {
    "evalset": args.evalset, "n": len(rows), "criteria": args.criteria,
    "judge_likert_distribution": dict(sorted(Counter(judge_lk).items())),
    "gold_rubric_likert_distribution": dict(sorted(Counter(gold_lk).items())),
    "judge_class_agreement_with_gold": round(sum(1 for r in out_rows if r["judge_verdict"]["judge_label"] == r["gold"]) / len(rows), 3),
    "judge_vs_gold_rubric_percent_agreement": round(AgreementMeasures.percent_agreement(judge_lk, gold_lk), 3),
    "judge_vs_gold_rubric_cohen_kappa": round(AgreementMeasures.cohen_kappa(judge_lk, gold_lk), 3),
    "mean_judge_likert": round(sum(judge_lk) / len(judge_lk), 3),
    "share_rated_5": round(sum(1 for x in judge_lk if x == 5) / len(judge_lk), 3),
}
if args.human:
    human = json.load(open(args.human))
    h_by_q = {h["query"]: h["rating"] for h in human}
    pairs = [(h_by_q[r["query"]], r["judge_label"]) for r in out_rows if r["query"] in h_by_q]
    if pairs:
        h, j = zip(*pairs)
        res["human_vs_judge_n"] = len(pairs)
        res["human_vs_judge_percent_agreement"] = round(AgreementMeasures.percent_agreement(list(h), list(j)), 3)
        res["human_vs_judge_cohen_kappa"] = round(AgreementMeasures.cohen_kappa(list(h), list(j)), 3)
out_dir = ROOT / "data/evaluations/llmjudge"; out_dir.mkdir(parents=True, exist_ok=True)
tag = Path(args.evalset).stem.replace("evalset_", "")
with open(out_dir / f"llmjudge_{tag}.jsonl", "w") as f:
    for r in out_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
(out_dir / f"llmjudge_{tag}_results.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
