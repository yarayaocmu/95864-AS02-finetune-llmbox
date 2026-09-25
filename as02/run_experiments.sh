#!/usr/bin/env bash
# AS02 Part C: finetuning experiments (run from the llmbox root, inside .venv).
# Every run writes: data/evaluations/finetune_gemma3_270m_it_<UTC>_metrics.json
# plus the AS02 bundle (<stem>_<run_name>_log_history.json / _config.yaml / _run_info.json / _loss_curve.png).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}          # Colab: PY=python3
DTYPE=${DTYPE:-float32}            # Colab (A100): DTYPE=bfloat16
MODEL=models/llms/google/gemma-3-270m-it
TRAIN=data/as02/splits/train.jsonl
COMMON="mode=finetune training.enabled=true training.method=lora data=jsonl data.path=$TRAIN data.eval_split=0.1 \
  model=gemma3_270m model.source=local model.local_path=$MODEL model.dtype=$DTYPE \
  training.max_length=512 training.logging_steps=5 training.eval_steps=25 training.save_steps=1000 training.merge_adapter=true"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

run() {  # $1 = run name, rest = overrides
  local name=$1; shift
  echo "=============== $name ==============="
  $PY -m startllm $COMMON training.run_name=$name training.output_dir=./finetuned_$name "$@" 2>&1 | tee data/as02/part_c/log_$name.txt | grep -E "trainable|Train:|'loss'|eval_loss|train_runtime|\[info\]|Error|error"
}
mkdir -p data/as02/part_c

case "${1:-all}" in
  timing)  # 12 optimizer steps to measure seconds/step before committing to full runs
    run timing training.epochs=0.08 training.batch_size=2 training.grad_accum_steps=8 training.eval_steps=6 ;;
  exp1)    # EXPERIMENT 1 = LLMBox TrainingConfig defaults (baseline)
    run exp1_baseline training.epochs=3 training.batch_size=2 training.grad_accum_steps=8 training.warmup_ratio=0.03 \
        optimizer=adamw optimizer.learning_rate=2e-5 training.lora_r=8 training.lora_alpha=16 training.lora_dropout=0.05 ;;
  exp2)    # EXPERIMENT 2 = updated config (decided after reading exp1 metrics; see memo)
    run exp2_updated training.epochs=2 training.batch_size=8 training.grad_accum_steps=2 training.warmup_ratio=0.03 \
        optimizer=adamw optimizer.learning_rate=2e-4 training.lora_r=8 training.lora_alpha=16 training.lora_dropout=0.05 ;;
  exp3)    # OPTIONAL: same as exp2 but LoRA on attention projections only (what target_modules changes)
    run exp3_attention_only training.epochs=2 training.batch_size=8 training.grad_accum_steps=2 training.warmup_ratio=0.03 \
        optimizer=adamw optimizer.learning_rate=2e-4 'training.lora_target_modules=[q_proj,k_proj,v_proj,o_proj]' ;;
  all) "$0" exp1; "$0" exp2 ;;
  *) echo "usage: $0 {timing|exp1|exp2|exp3|all}"; exit 1 ;;
esac
