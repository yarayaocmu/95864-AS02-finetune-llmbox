#!/usr/bin/env python3
"""Convert the filled human_rating_sheet.xlsx into LLMBox's labeled_evaluation_dataset.json
(same shape HumanEvaluation.run() writes) plus evaluation_criteria.md.
Run: python3 as02/part_d_sheet_to_labels.py --criteria "..." """
import argparse, json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "data/as02/part_d"
ap = argparse.ArgumentParser(); ap.add_argument("--criteria", required=True)
ap.add_argument("--sheet", default=str(D / "human_rating_sheet_rated.xlsx")); a = ap.parse_args()
sheet = pd.read_excel(a.sheet)
rows = [json.loads(l) for l in open(D / "evalset_exp2_test.jsonl")]
assert len(sheet) == len(rows)
out = []
for r, rating in zip(rows, sheet["your_rating_1_to_5"]):
    assert rating in (1, 2, 3, 4, 5), f"row missing/invalid rating: {rating}"
    out.append({"query": r["query"], "response": r["response"], "rating": int(rating)})
(D / "labeled_evaluation_dataset.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
(D / "evaluation_criteria.md").write_text(f"# Evaluation Criteria\n\n{a.criteria}\n\nLikert: 5 correct single label; 4 correct with extra words; 3 adjacent class; 2 opposite polarity; 1 no valid label.\n\nAgreement metric chosen: Cohen's kappa\n")
print("wrote", D / "labeled_evaluation_dataset.json")
