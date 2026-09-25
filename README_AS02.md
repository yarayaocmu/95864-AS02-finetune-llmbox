# AS02 — Finetuning Gemma-3-270m-it for medication-experience triage (Scenario 01)

Team dataset: Drugs.com patient reviews (AS01). Task: given `drug`, `condition`
and a `patient_report`, answer `POSITIVE` / `NEUTRAL` / `NEGATIVE`.

This folder is Prof. Kingsley's LLMBox (GPL-3.0, https://github.com/sarakingsley/llmbox)
plus our AS02 additions. **All modifications to LLMBox are marked `AS02 edit`
in the source** (GPL attribution requirement):

| File | Change |
|---|---|
| `src/schema.py` | `TrainingConfig.lora_target_modules` (which layers get LoRA) and `TrainingConfig.run_name` |
| `src/modes.py` | LoRA `target_modules` read from config; `include_num_input_tokens_seen=True` so the economic metrics count real tokens instead of `steps x batch x max_length`; after every run a reproducibility bundle is written to `data/evaluations/` (`*_log_history.json`, `*_config.yaml`, `*_run_info.json`, `*_loss_curve.png`) |
| `src/pydantic_models/experience_judge.py` | NEW: Pydantic model `ExperienceJudgement` + `judge_fn` for the LLM-Judge evaluation (local Phi-4-mini as judge) |
| `as02/*.py` | NEW: our Part A-D scripts (below) |

## Environment (Apple M4 Pro, 48 GB, MPS; no GPU)

```bash
cd AS/AS02/llmbox
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m venv .venv
.venv/bin/pip install torch transformers peft accelerate datasets hydra-core omegaconf \
    huggingface_hub psutil matplotlib pandas pyarrow scikit-learn seaborn pydantic fire tabulate pyyaml
# model (gated on HF: accept the Gemma licence, then `hf auth login`)
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-3-270m-it', local_dir='models/llms/google/gemma-3-270m-it')"
```
Versions used: torch 2.14.0, transformers 5.17.0, peft 0.21.0, Python 3.13.7.

## Part A — data

```bash
.venv/bin/python as02/part_a_transform.py --per-class 1000 --seed 42     # AS01 parquet -> LLMBox jsonl + train/test
.venv/bin/python as02/part_a_inspect.py                                   # stats, plots, duplicate + sensitive-info scan
# LLMBox starter script (re-splits with the same seed -> identical files, then tokenises):
HF_HUB_OFFLINE=1 .venv/bin/python -m startllm mode=prepare_data data=jsonl \
    data.path=data/as02/transformed/medrx_experience.jsonl data.output_dir=data/as02/splits \
    data.train_fraction=0.8 model.local_path=models/llms/google/gemma-3-270m-it training.max_length=512
.venv/bin/python as02/part_a_restore_meta.py                               # prepare_data drops the meta block; put it back (model never sees it)
.venv/bin/python as02/part_a_tokenize_batches.py --model models/llms/google/gemma-3-270m-it   # tokens + batch analysis
```
Outputs: `data/as02/transformed/`, `data/as02/splits/{train,test}.jsonl`,
`data/as02/eval/unseen_drugs_eval.jsonl`, `data/as02/part_a/` (csv + json + `figures/`).

## Part B/C — finetuning experiments

See `as02/run_experiments.sh` for the exact commands of Experiment 1 (baseline)
and Experiment 2. Every run writes its metrics JSON to `data/evaluations/`.
Compare runs: `.venv/bin/python as02/part_c_compare_runs.py` -> `data/as02/part_c/`.

## Part D — evaluation

```bash
# 100-prompt evaluation of base vs finetuned model (chat-log JSON + scores)
.venv/bin/python as02/part_d_generate_eval.py --model models/llms/google/gemma-3-270m-it --data data/as02/splits/test.jsonl --n 100 --tag base_test
.venv/bin/python as02/part_d_generate_eval.py --model finetuned_exp2/merged --data data/as02/splits/test.jsonl --n 100 --tag exp2_test
.venv/bin/python as02/part_d_generate_eval.py --model finetuned_exp2/merged --data data/as02/eval/unseen_drugs_eval.jsonl --n 100 --tag exp2_unseen
# interactive chat with the finetuned model (LLMBox)
HF_HUB_OFFLINE=1 .venv/bin/python -m startllm mode=chat model.local_path=finetuned_exp2/merged system_prompt="" generation.do_sample=false generation.max_new_tokens=8
# human evaluation (LLMBox, interactive; individual task)
.venv/bin/python -m startllm mode=evaluate data=jsonl data.path=data/as02/part_d/evalset_exp2_test.jsonl model.local_path=finetuned_exp2/merged
#   -> when asked for an LLM-judge function paste:  from src.pydantic_models.experience_judge import judge_fn
# reproducible LLM-judge run + agreement with human labels
.venv/bin/python as02/part_d_llm_judge.py --evalset data/as02/part_d/evalset_exp2_test.jsonl --human data/as02/part_d/labeled_evaluation_dataset.json
```

## Running on Colab (A100)

Open `as02/AS02_colab.ipynb` in Colab (File -> Open notebook -> GitHub, or upload). It clones this
repo, installs `requirements_as02.txt`, downloads Gemma (needs `HF_TOKEN` secret) and Phi-4-mini
(judge), then runs Part A prepare_data, the experiments (`PY=python3 DTYPE=bfloat16`), the
comparison, the Part D evaluations and the LLM judge, and pushes `data/evaluations` + `data/as02`
back to GitHub. The laptop (MPS) path and the Colab (CUDA) path use identical code and data; only
`model.dtype` differs (float32 on MPS, bfloat16 on A100).
Local MPS note: the first two laptop attempts of Experiment 1 ran out of MPS memory after ~25 steps
(logs in `data/as02/smoke/`); `src/modes.py` now empties the MPS cache every step.
