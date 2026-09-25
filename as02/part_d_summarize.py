#!/usr/bin/env python3
"""
AS02 Part D: collect all scores_*.json into one table + figure, and break the
edge-case run down by category.

Run: python3 as02/part_d_summarize.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/as02/part_d"
LABELS = ["POSITIVE", "NEUTRAL", "NEGATIVE"]

rows = []
for f in sorted(D.glob("scores_*.json")):
    s = json.loads(f.read_text())
    rows.append({"run": s["tag"], "model": Path(s["model"]).parts[-2] if "merged" in s["model"] else "gemma-3-270m-it (base)",
                 "data": Path(s["data"]).name, "n": s["n"],
                 "format_ok": s["format_compliance_exact_label_only"], "accuracy": s["accuracy"],
                 "macro_f1": s["macro_f1"], "majority_baseline": s["majority_class_baseline_accuracy"],
                 "POS_recall": s["per_class"]["POSITIVE"]["recall"], "NEU_recall": s["per_class"]["NEUTRAL"]["recall"],
                 "NEG_recall": s["per_class"]["NEGATIVE"]["recall"], "NEG_precision": s["per_class"]["NEGATIVE"]["precision"],
                 "polarity_flips": s["off_by_two_errors (POSITIVE<->NEGATIVE)"],
                 "acc_chronic": s["accuracy_chronic_subgroup"], "acc_non_chronic": s["accuracy_non_chronic"],
                 "sec_per_prompt": s["sec_per_prompt"]})
df = pd.DataFrame(rows)
order = ["base_test", "exp1_test", "exp2_test", "exp2_unseen", "exp2_edge"]
df["o"] = df["run"].map({k: i for i, k in enumerate(order)}); df = df.sort_values("o").drop(columns="o")
df.to_csv(D / "part_d_summary.csv", index=False)
(D / "part_d_summary.md").write_text(df.to_markdown(index=False))
print(df.to_string(index=False))

# figure: accuracy / macro-F1 / NEG recall per run
fig, ax = plt.subplots(figsize=(9, 4))
x = range(len(df)); w = 0.25
ax.bar([i - w for i in x], df["accuracy"], w, label="accuracy")
ax.bar(list(x), df["macro_f1"], w, label="macro-F1")
ax.bar([i + w for i in x], df["NEG_recall"], w, label="NEGATIVE recall")
ax.plot(list(x), df["majority_baseline"], "k--", label="majority-class baseline")
ax.set_xticks(list(x)); ax.set_xticklabels(df["run"]); ax.set_ylim(0, 1.05); ax.legend(loc="lower right")
ax.set_title("Part D: base vs finetuned Gemma-3-270m-it (100 prompts each; edge = 24)")
fig.tight_layout(); fig.savefig(D / "part_d_scores.png", dpi=150); plt.close(fig)

# edge cases by category
edge = [json.loads(l) for l in open(D / "evalset_exp2_edge.jsonl")]
er = pd.DataFrame([{"category": r["meta"]["category"], "expected": r["meta"]["expected_behaviour"],
                    "gold": r["gold"] if r["meta"]["gold_defined"] else "(n/a)", "pred": r["pred"],
                    "correct": (r["pred"] == r["gold"]) if r["meta"]["gold_defined"] else None,
                    "raw": r["response"][:60]} for r in edge])
er.to_csv(D / "edge_cases_by_category.csv", index=False)
(D / "edge_cases_by_category.md").write_text(er.to_markdown(index=False))
print("\nEdge cases:\n", er[["category", "gold", "pred", "correct"]].to_string(index=False))

# confusion matrices side by side
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, tag in zip(axes, ["base_test", "exp1_test", "exp2_test"]):
    s = json.loads((D / f"scores_{tag}.json").read_text()); c = s["confusion(gold->pred)"]
    m = [[c[g][p] for p in LABELS] for g in LABELS]
    ax.imshow(m, cmap="Blues"); ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(LABELS, fontsize=8); ax.set_yticklabels(LABELS, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("gold"); ax.set_title(f"{tag} (acc {s['accuracy']:.2f})")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, m[i][j], ha="center", va="center", color="white" if m[i][j] > 15 else "black")
fig.tight_layout(); fig.savefig(D / "part_d_confusion.png", dpi=150); plt.close(fig)
print("\n->", D)
