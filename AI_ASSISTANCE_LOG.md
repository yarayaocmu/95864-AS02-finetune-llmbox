# AS02 — AI code-assistance log (course policy: code assistance allowed, must be documented)

Tool: Claude Code (Anthropic, model Claude Fable 5.1), used on 2026-09-25 inside VS Code.
Scope of AI use: **software code only** (scripts, LLMBox patches, run commands, debugging).
The memo / report text, the analysis and all interpretation are written by the student.
The full prompt/response transcript is exported from the Claude Code session and attached as
Appendix B; this file is the index of what was generated.

| # | Student prompt (summary) | AI output used | File(s) |
|---|---|---|---|
| 1 | "一步一步完成AS02" with the assignment text pasted; explore AS01 repo and LLMBox | Cloned LLMBox, read modules, created `.venv`, installed deps | `.venv/`, `README_AS02.md` |
| 2 | (same session) transform AS01 dataset with LLMBox `DataTransformer`, balanced sample, TLP filter, unseen-drug hold-out | `as02/part_a_transform.py` | Part A |
| 3 | (same session) descriptive statistics, plots, duplicate + sensitive-info scan of train split | `as02/part_a_inspect.py` | Part A |
| 4 | (same session) tokenizer / batch analysis using LLMBox `TrainingDataLoader` | `as02/part_a_tokenize_batches.py` | Part A |
| 5 | (same session) make LoRA `target_modules` configurable, exact token counting, save learning curves per run | patches in `src/schema.py`, `src/modes.py` (marked `AS02 edit`) | Part B/C |
| 6 | (same session) compare runs / overlay learning curves | `as02/part_c_compare_runs.py` | Part C |
| 7 | (same session) batch evaluation of base vs finetuned model, chat-log format, accuracy / macro-F1 / NEGATIVE recall | `as02/part_d_generate_eval.py` | Part D |
| 8 | (same session) Pydantic LLM-judge model + local-judge function compatible with `mode=evaluate`; agreement runner | `src/pydantic_models/experience_judge.py`, `as02/part_d_llm_judge.py` | Part D |
| 9 | (same session) experiment shell script | `as02/run_experiments.sh` | Part C |

Design decisions (which features go in the prompt, TLP filter, class balance,
hyper-parameters of Experiment 1/2, evaluation criterion) were made by the student
and implemented with AI help; the AI did not research or write any memo content.
| 10 | (same session) edge-case prompts for Part D (sarcasm, negation, off-task dosing questions, prompt injection, PII, non-English) | `as02/part_d_edge_cases.py`, `as02/part_a_transform_instruction.py` | Part D |
| 11 | (same session) restore `meta` after `mode=prepare_data` | `as02/part_a_restore_meta.py` | Part A |
| 12 | "是不是用colab A100 train会更快一点 / 先把代码准备好，上传github，我在colab里train" | CUDA support in eval/judge scripts, MPS empty-cache callback + `prediction_loss_only` in `src/modes.py` (laptop OOM fix), `requirements_as02.txt`, Colab notebook, GitHub repo | `as02/AS02_colab.ipynb`, `README_AS02.md` |
| 13 | "我更新了代码和结果" (Colab results pulled) | Part D summary table/figures script; fixed trainable-param count (was taken after `merge_and_unload`, values restored from logs); fixed judge default path crash seen on Colab; local LLM-judge run on the 100 exp2 answers | `as02/part_d_summarize.py`, `src/modes.py`, `src/pydantic_models/experience_judge.py`, `data/as02/part_d/`, `data/evaluations/llmjudge/` |
