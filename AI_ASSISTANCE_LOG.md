# AS02 — AI code-assistance log

AI code assistance (Claude Code, Anthropic) was used for the software in this project: the AS02 scripts
under `as02/`, the marked `AS02 edit` patches to LLMBox, the Pydantic judge module, the Colab notebook and
the run commands. This file lists every AI-generated file and the prompt that produced it. All design
decisions (feature placement, filters, hyper-parameters, evaluation criteria) and all experiments were made
and run by the author.

Tool: Claude Code (Anthropic), one session on 2026-09-25 inside VS Code.

| # | Prompt (summary) | AI output used | File(s) |
|---|---|---|---|
| 1 | "Complete AS02 step by step" with the assignment text; explore the AS01 repo and LLMBox | Cloned LLMBox, read the modules, created `.venv`, installed dependencies | `.venv/`, `README_AS02.md` |
| 2 | Transform the AS01 dataset with LLMBox `DataTransformer`: balanced sample, TLP filter, unseen-drug hold-out | `as02/part_a_transform.py` | Part A |
| 3 | Descriptive statistics, plots, duplicate and sensitive-information scan of the train split | `as02/part_a_inspect.py` | Part A |
| 4 | Tokenizer and batch analysis using LLMBox `TrainingDataLoader` | `as02/part_a_tokenize_batches.py` | Part A |
| 5 | Make LoRA `target_modules` configurable, count tokens exactly, save learning curves per run | patches in `src/schema.py`, `src/modes.py` (marked `AS02 edit`) | Part B/C |
| 6 | Compare runs and overlay learning curves | `as02/part_c_compare_runs.py` | Part C |
| 7 | Batch evaluation of base vs finetuned model in chat-log format with accuracy, macro-F1, NEGATIVE recall | `as02/part_d_generate_eval.py` | Part D |
| 8 | Pydantic LLM-judge model and a local judge function compatible with `mode=evaluate`; agreement runner | `src/pydantic_models/experience_judge.py`, `as02/part_d_llm_judge.py` | Part D |
| 9 | Experiment shell script | `as02/run_experiments.sh` | Part C |
| 10 | Edge-case prompts for Part D (sarcasm, negation, off-task dosing questions, prompt injection, PII, non-English) | `as02/part_d_edge_cases.py`, `as02/part_a_transform_instruction.py` | Part D |
| 11 | Restore the `meta` block after `mode=prepare_data` | `as02/part_a_restore_meta.py` | Part A |
| 12 | "Would Colab A100 be faster? Prepare the code, upload to GitHub, I will train in Colab" | CUDA support in the evaluation and judge scripts, MPS empty-cache callback and `prediction_loss_only` in `src/modes.py` (laptop OOM fix), `requirements_as02.txt`, Colab notebook, GitHub repository | `as02/AS02_colab.ipynb`, `README_AS02.md` |
| 13 | "I updated the code and results" (Colab results pulled) | Part D summary table and figure script; fixed the trainable-parameter count (taken after `merge_and_unload`, values restored from logs); fixed the judge default-path crash seen on Colab; local LLM-judge run on the 100 exp2 answers | `as02/part_d_summarize.py`, `src/modes.py`, `src/pydantic_models/experience_judge.py`, `data/as02/part_d/`, `data/evaluations/llmjudge/` |
| 14 | Submission requirements | Submission folder layout, `training_configurations.jsonl`, regenerated `adapter_config.json` for the two checkpoints (verified against the Colab outputs), Pydantic JSON schema exports, `MEMO_MATERIALS_INDEX.md` | `submission/`, `data/pydantic_models/` |
| 15 | Human-evaluation tooling | Rating spreadsheet generator, converter to LLMBox `labeled_evaluation_dataset.json`, human-vs-judge agreement script (the 100 ratings were entered by the author) | `as02/part_d_sheet_to_labels.py`, `as02/part_d_human_agreement.py` |
