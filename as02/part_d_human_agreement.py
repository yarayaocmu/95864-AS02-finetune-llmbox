#!/usr/bin/env python3
"""Human ratings (labeled_evaluation_dataset.json) vs the LLM judge and vs the reference rubric.
Uses LLMBox's AgreementMeasures. Run: python3 as02/part_d_human_agreement.py"""
import json, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.evaluator import AgreementMeasures as AM
from src.pydantic_models.experience_judge import likert_from_labels
D = ROOT / "data/as02/part_d"
human = json.load(open(D / "labeled_evaluation_dataset.json"))
judge = {r["query"]: r for r in (json.loads(l) for l in open(ROOT / "data/evaluations/llmjudge/llmjudge_exp2_test.jsonl"))}
h, j, g, m = [], [], [], []
for ex in human:
    jr = judge[ex["query"]]
    h.append(ex["rating"]); j.append(jr["judge_label"])
    g.append(likert_from_labels(jr["gold"], ex["response"])[0])
    m.append(jr["pred"] == jr["gold"])
n = len(h)
res = {"n": n, "human_likert_distribution": dict(sorted(Counter(h).items())), "human_mean_likert": round(sum(h) / n, 2),
       "human_share_rated_5": round(sum(1 for x in h if x == 5) / n, 2), "human_share_rated_le2": round(sum(1 for x in h if x <= 2) / n, 2),
       "human_vs_judge_percent_agreement": round(AM.percent_agreement(h, j), 3), "human_vs_judge_cohen_kappa": round(AM.cohen_kappa(h, j), 3),
       "human_vs_reference_rubric_percent_agreement": round(AM.percent_agreement(h, g), 3), "human_vs_reference_rubric_cohen_kappa": round(AM.cohen_kappa(h, g), 3),
       "model_accuracy_vs_reference_on_same_100": round(sum(m) / n, 2),
       "cases_human5_judge3_or_less": sum(1 for a, b in zip(h, j) if a == 5 and b <= 3), "cases_human3_or_less_judge5": sum(1 for a, b in zip(h, j) if a <= 3 and b == 5)}
(D / "human_vs_judge_agreement.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))
