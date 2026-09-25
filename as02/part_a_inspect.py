#!/usr/bin/env python3
"""
AS02 Part A, step 3: inspect the TRAIN split that LLMBox wrote.

Produces descriptive statistics, distribution plots, a duplicate check and a
sensitive-information scan (things we do not want the network to memorise).
Note: the rating / TLP / chronic-condition statistics reuse the same feature
definitions as our AS01 datasheet (same dataset, same cleaning).

Run: python3 as02/part_a_inspect.py [--train data/as02/splits/train.jsonl]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--train", default=str(ROOT / "data/as02/splits/train.jsonl"))
parser.add_argument("--test", default=str(ROOT / "data/as02/splits/test.jsonl"))
parser.add_argument("--out", default=str(ROOT / "data/as02/part_a"))
args = parser.parse_args()
OUT = Path(args.out); FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                m = r.get("meta", {})
                rows.append({"prompt": r["prompt"], "completion": r["completion"],
                             "prompt_chars": len(r["prompt"]), "prompt_words": len(r["prompt"].split()),
                             "text_chars": len(r["text"]), **m})
    return pd.DataFrame(rows)

train, test = load(args.train), load(args.test)
report = {}

# ---- 1. label / feature distributions ----------------------------------
report["n_train"], report["n_test"] = int(len(train)), int(len(test))
report["train_label_counts"] = train["completion"].value_counts().to_dict()
report["test_label_counts"] = test["completion"].value_counts().to_dict()
report["train_rating_by_label"] = (train.groupby("completion")["rating"]
                                   .agg(["min", "max", "mean"]).round(2).to_dict("index"))
report["train_tlp_counts"] = train["tlp"].value_counts().to_dict()
report["train_chronic_share"] = round(float(train["is_chronic"].mean()), 3)
report["train_n_unique_drugs"] = int(train["drug"].nunique())
report["train_n_unique_conditions"] = int(train["condition"].nunique())
report["train_top_conditions"] = train["condition"].value_counts().head(15).to_dict()
report["train_top_drugs"] = train["drug"].value_counts().head(15).to_dict()
report["train_length_stats"] = (train[["review_word_len", "prompt_words", "prompt_chars"]]
                                .describe().round(1).to_dict())
report["train_length_by_label"] = (train.groupby("completion")["review_word_len"]
                                   .describe().round(1).to_dict("index"))

# ---- 2. duplication / artifacts ----------------------------------------
dup_prompt_train = int(train["prompt"].duplicated().sum())
overlap = len(set(train["prompt"]) & set(test["prompt"]))
report["duplicates"] = {
    "exact_duplicate_prompts_in_train": dup_prompt_train,
    "train_test_prompt_overlap": overlap,
    "same_review_text_under_multiple_drugs_in_train": 0,  # removed upstream
}
art = {
    "condition_truncated_name (e.g. 'Bipolar Disorde')": int(train["condition"].str.match(
        r".*(Disorde|Cance|Diseas|Syndrom|Infectio)$").sum()),
    "report_contains_html_entity": int(train["prompt"].str.contains(r"&#\d+;|&amp;|&quot;").sum()),
    "report_contains_url": int(train["prompt"].str.contains(r"https?://|www\.").sum()),
    "report_non_ascii_chars": int(train["prompt"].str.contains(r"[^\x00-\x7F]").sum()),
    "report_all_caps_words(>=3)": int(train["prompt"].str.count(r"\b[A-Z]{4,}\b").ge(3).sum()),
    "report_shorter_than_10_words": int((train["review_word_len"] < 10).sum()),
}
report["artifacts"] = art

# ---- 3. sensitive-information scan --------------------------------------
patterns = {
    "explicit_age (e.g. 'I am 34', '34 years old', '34 yo')":
        r"\b(i am|i'm|im|am a|age)\s?\d{1,2}\b|\b\d{1,2}\s?(years? old|yrs? old|yo|y/o)\b",
    "named_doctor ('Dr. Smith')": r"\bDr\.?\s+[A-Z][a-z]+",
    "third_party_family (my son/daughter/wife/husband/mother/father)":
        r"\bmy (son|daughter|wife|husband|mother|mom|father|dad|kid|child|children|boyfriend|girlfriend|partner)\b",
    "pregnancy / breastfeeding": r"\b(pregnan|breastfeed|nursing my)\w*",
    "email_address": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "phone_number": r"\b\d{3}[-. ]\d{3}[-. ]\d{4}\b",
    "full_date (mm/dd/yyyy)": r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    "weight_or_height ('188 lbs', \"5'6\")": r"\b\d{2,3}\s?(lbs?|pounds|kg)\b|\b\d'\d{1,2}\b",
    "location_mention (city/state/country words)":
        r"\b(in|from|at)\s+(Texas|California|Florida|New York|Ohio|Canada|UK|Australia|India|London|Chicago)\b",
    "crisis_words (suicid/overdose)": r"\b(suicid\w*|overdos\w*|kill myself)\b",
    "substance_dependence (addict/withdrawal/rehab)": r"\b(addict\w*|withdrawal|rehab)\b",
    "explicit_sexual_health_terms": r"\b(sex|sexual|libido|erection|orgasm)\w*",
}
sens = {}
examples = {}
for name, pat in patterns.items():
    hit = train["prompt"].str.contains(pat, case=False, regex=True)
    sens[name] = {"rows": int(hit.sum()), "pct": round(100 * hit.mean(), 2)}
    if hit.any():
        ex = train.loc[hit, "prompt"].iloc[0]
        m = re.search(pat, ex, flags=re.I)
        s = max(0, m.start() - 60); e = min(len(ex), m.end() + 60)
        examples[name] = "..." + ex[s:e].replace("\n", " ") + "..."
report["sensitive_scan"] = sens
report["sensitive_scan_examples"] = examples

# ---- 4. tables ----------------------------------------------------------
lab = pd.DataFrame({"train": train["completion"].value_counts(),
                    "test": test["completion"].value_counts()}).fillna(0).astype(int)
lab["train_pct"] = (100 * lab["train"] / lab["train"].sum()).round(1)
lab.to_csv(OUT / "label_distribution.csv")
pd.DataFrame(sens).T.to_csv(OUT / "sensitive_scan.csv")
pd.Series(art).to_csv(OUT / "artifacts.csv", header=["rows"])
train["condition"].value_counts().head(25).to_csv(OUT / "top_conditions.csv", header=["rows"])
train.groupby("completion")["review_word_len"].describe().round(1).to_csv(OUT / "length_by_label.csv")

# ---- 5. figures ---------------------------------------------------------
# F1 label distribution: balanced train vs natural share (from transform log)
nat = json.loads((ROOT / "data/as02/part_a_transform_log.json").read_text())["natural_class_share_in_pool"]
order = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].bar(order, [lab.loc[o, "train_pct"] for o in order], color="#4C72B0")
ax[0].set_title("F1a. Train set label share (%)"); ax[0].set_ylim(0, 70)
ax[1].bar(order, [100 * nat[o] for o in order], color="#DD8452")
ax[1].set_title("F1b. Natural label share in source pool (%)"); ax[1].set_ylim(0, 70)
fig.tight_layout(); fig.savefig(FIG / "F1_label_distribution.png", dpi=150); plt.close(fig)

# F2 review length by label
fig, ax = plt.subplots(figsize=(7, 3.8))
for o in order:
    ax.hist(train.loc[train["completion"] == o, "review_word_len"], bins=40, alpha=0.5, label=o)
ax.set_xlabel("patient report length (words)"); ax.set_ylabel("reviews"); ax.legend()
ax.set_title("F2. Train report length by label"); fig.tight_layout()
fig.savefig(FIG / "F2_length_by_label.png", dpi=150); plt.close(fig)

# F3 top conditions
top = train["condition"].value_counts().head(20)[::-1]
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.barh(top.index, top.values, color="#55A868"); ax.set_title("F3. Top 20 conditions in train")
ax.set_xlabel("reviews"); fig.tight_layout(); fig.savefig(FIG / "F3_top_conditions.png", dpi=150); plt.close(fig)

# F4 rating vs label (sanity: label derived from rating)
ct = pd.crosstab(train["rating"], train["completion"])[order]
fig, ax = plt.subplots(figsize=(7, 3.8))
ct.plot(kind="bar", stacked=True, ax=ax, color=["#C44E52", "#8172B2", "#55A868"])
ax.set_title("F4. Star rating vs. label in train (label is derived from rating)")
ax.set_xlabel("rating (1-10)"); ax.set_ylabel("reviews"); fig.tight_layout()
fig.savefig(FIG / "F4_rating_vs_label.png", dpi=150); plt.close(fig)

# F5 sensitive scan
s = pd.DataFrame(sens).T.sort_values("pct")
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh([k.split(" (")[0] for k in s.index], s["pct"], color="#C44E52")
ax.set_xlabel("% of train prompts matching pattern"); ax.set_title("F5. Sensitive-content scan of train prompts")
fig.tight_layout(); fig.savefig(FIG / "F5_sensitive_scan.png", dpi=150); plt.close(fig)

(OUT / "part_a_inspect_report.json").write_text(json.dumps(report, indent=2, default=str))
print(json.dumps(report, indent=2, default=str))
print("\nFigures ->", FIG)
