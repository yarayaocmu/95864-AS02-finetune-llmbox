#!/usr/bin/env python3
"""
AS02 Part A, step 1-2: transform the AS01 Drugs.com review dataset into the
prompt/completion format accepted by LLMBox, then split it into train/test
JSONL with LLMBox's own DataTransformer.

Organisational task (Scenario 01, medication decision support):
    input  = drug name + condition + a patient's free-text report
    output = the patient-reported experience class: POSITIVE / NEUTRAL / NEGATIVE

Design choices (explained in the memo):
    * Patient-data protection: only TLP:GREEN and TLP:CLEAR reviews are used
      for training. TLP:RED (crisis / highly sensitive) and TLP:AMBER
      (third parties, explicit age, named doctors, pregnancy) are excluded so
      that the network never memorises them.
    * Brand/generic duplicate reviews are removed BEFORE sampling, otherwise
      the same text can land in both train and test.
    * Class balance: the raw data is 66% POSITIVE. We sample an equal number
      of reviews per class so the model cannot succeed by always answering
      POSITIVE. The natural imbalance is documented separately.
    * Unseen-drug hold-out: 10% of drugs are removed from the sampling pool
      and kept as a separate evaluation file (Part D) to test whether the
      model generalises to drugs it never saw during finetuning.
    * The label instruction is baked into the prompt (LLMBox `instruction`)
      so the finetuned model does not depend on a specific system prompt.

Run (from the llmbox root, inside the llmbox .venv):
    python3 as02/part_a_transform.py --per-class 1000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LLMBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLMBOX_ROOT))
from src.data_services import DataTransformer  # noqa: E402

AS01_PARQUET = (LLMBOX_ROOT.parents[1] / "AS01" / "as01_v02" / "data" / "clean"
                / "reviews_clean.parquet")
OUT_DIR = LLMBOX_ROOT / "data" / "as02"
TRANSFORMED = OUT_DIR / "transformed" / "medrx_experience.jsonl"
SPLIT_DIR = OUT_DIR / "splits"
UNSEEN = OUT_DIR / "eval" / "unseen_drugs_eval.jsonl"

from as02.part_a_transform_instruction import INSTRUCTION  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--per-class", type=int, default=1000,
                    help="reviews sampled per class for the train/test pool")
parser.add_argument("--unseen-per-class", type=int, default=50,
                    help="reviews per class in the unseen-drug evaluation file")
parser.add_argument("--train-fraction", type=float, default=0.8)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max-words", type=int, default=300,
                    help="drop reviews longer than this (token budget control)")
args = parser.parse_args()
rng = np.random.default_rng(args.seed)

log: dict = {"source": str(AS01_PARQUET), "seed": args.seed}
df = pd.read_parquet(AS01_PARQUET)
log["rows_clean_as01"] = int(len(df))

# ---- 1. filters ---------------------------------------------------------
df = df[df["tlp_label"].isin(["TLP:GREEN", "TLP:CLEAR"])]
log["rows_after_tlp_filter(GREEN,CLEAR)"] = int(len(df))
df = df.drop_duplicates(subset=["review"], keep="first")
log["rows_after_brand_generic_dedup"] = int(len(df))
df = df.dropna(subset=["condition_clean"])
log["rows_after_drop_missing_condition"] = int(len(df))
df = df[(df["review_word_len"] >= 5) & (df["review_word_len"] <= args.max_words)]
log[f"rows_after_length_filter(5..{args.max_words} words)"] = int(len(df))

# ---- 2. unseen-drug hold-out -------------------------------------------
drugs = np.sort(df["drugName"].unique())
unseen_drugs = set(rng.choice(drugs, size=int(0.10 * len(drugs)), replace=False))
pool = df[~df["drugName"].isin(unseen_drugs)]
unseen = df[df["drugName"].isin(unseen_drugs)]
log["n_drugs_total"] = int(len(drugs))
log["n_drugs_held_out"] = int(len(unseen_drugs))
log["rows_unseen_drug_pool"] = int(len(unseen))

# ---- 3. balanced sample per class ---------------------------------------
def balanced(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    parts = [g.sample(min(len(g), n), random_state=seed)
             for _, g in frame.groupby("sentiment_label")]
    return pd.concat(parts).sample(frac=1.0, random_state=seed)

sample = balanced(pool, args.per_class, args.seed)
unseen_sample = balanced(unseen, args.unseen_per_class, args.seed)
log["natural_class_share_in_pool"] = (pool["sentiment_label"].value_counts(normalize=True)
                                      .round(3).to_dict())
log["sampled_rows"] = int(len(sample))
log["sampled_class_counts"] = sample["sentiment_label"].value_counts().to_dict()

# ---- 4. rename features to the names the model will see -----------------
def to_rows(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for r in frame.itertuples():
        rows.append({
            "drug": r.drugName,
            "condition": r.condition_clean,
            "patient_report": r.review,
            "experience": r.sentiment_label,
            # meta is NOT shown to the model; kept for analysis / evaluation
            "uniqueID": int(r.uniqueID), "rating": int(r.rating),
            "tlp": r.tlp_label, "is_chronic": int(r.is_chronic_condition),
            "review_word_len": int(r.review_word_len),
        })
    return rows

META = ["uniqueID", "rating", "tlp", "is_chronic", "review_word_len",
        "drug", "condition", "experience"]

# ---- 5. LLMBox transformation -------------------------------------------
dt = DataTransformer()

def standardize(rows: list[dict]) -> list[dict]:
    examples = list(dt.standardize_llm_dataset(
        rows,
        text_columns=["drug", "condition", "patient_report"],
        target_columns="experience",
        instruction=INSTRUCTION,
    ))
    # standardize_llm_dataset only returns prompt/completion/text/messages;
    # re-attach the meta fields (same order) for later analysis.
    for ex, row in zip(examples, rows):
        ex["meta"] = {k: row[k] for k in META}
    return examples

transformed = standardize(to_rows(sample))
unseen_examples = standardize(to_rows(unseen_sample))

n = dt.write_jsonl(transformed, TRANSFORMED)
log["transformed_written"] = n
n_u = dt.write_jsonl(unseen_examples, UNSEEN)
log["unseen_drug_eval_written"] = n_u

# ---- 6. train/test split with LLMBox ------------------------------------
# Same function, seed and fraction that `mode=prepare_data` uses, so the
# files it writes later are byte-identical to these.
result = dt.split_and_save_jsonl(transformed, output_directory=SPLIT_DIR,
                                 train_fraction=args.train_fraction,
                                 seed=args.seed, overwrite=True)
log["split"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in result.items()}

for name in ("train", "test"):
    with open(SPLIT_DIR / f"{name}.jsonl") as f:
        labels = [json.loads(l)["completion"] for l in f if l.strip()]
    log[f"{name}_class_counts"] = pd.Series(labels).value_counts().to_dict()

(OUT_DIR / "part_a_transform_log.json").write_text(json.dumps(log, indent=2))
(OUT_DIR / "part_a_example_record.json").write_text(json.dumps(transformed[0], indent=2))
print(json.dumps(log, indent=2))
print("\nExample record:\n", json.dumps(transformed[0], indent=2)[:1500])
