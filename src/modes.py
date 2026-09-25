'''
    LLMBox -- A Software Application for Building Customized and Affordable AI Solutions.
    Copyright (C) 2026  Sara Kingsley

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

"""
modes.py

Implementations of the six run modes dispatched by configurator.py:
chat, generate, tool_calling, structured_output, train, finetune.

torch/transformers/peft are imported lazily inside each function, so that
composing or printing a config (`python configurator.py --cfg job`) never
requires them to be installed -- only actually running a mode does.
"""

import json
import logging
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from omegaconf import OmegaConf

# import LLMBOX application software:
from src.sys_logger import Logger
from src.schema import (
    ModelConfig,
    GenerationConfig,
    ToolDef,
    ToolCallingConfig,
    StructuredOutputConfig,
    OptimizerConfig,
    TrainingConfig,
    DataConfig,
    ModeConfig,
    Config,
    register_configs
)

from src.generation import GenerationManager
from src.data_services import (
    DataTransformer,
    CausalLMCollator,
    TokenizedChatDataset,
    TrainingDataLoader
)

from src import metrics as training_metrics

# The following import is the minimal external evaluation interface.
# It is assumed to provide MetricsTracker with .start()/.stop(trainer) as in original src.metrics.
#try:
   # from src.evaluator import MetricsTracker
#except ImportError:
   # MetricsTracker = None

class Modes:

    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.IGNORE_INDEX = -100
        self.generator = GenerationManager()
        self.datamanager = DataTransformer()
        self.dataloader = TrainingDataLoader()
        self.logger = Logger()

    def _make_dataloader(self, cfg):
        strategy = getattr(cfg.model, "assistant_mask_strategy", "verified_prefix")
        return TrainingDataLoader(assistant_mask_strategy=strategy)

    # --------------------------------------------------------------------------
    # chat -- interactive multi-turn session, logged the same way gemma_chat.py
    # logs sessions (one JSON file per session under <output_dir>/chat_log).
    # --------------------------------------------------------------------------
    def run_chat(self, cfg) -> None:
        model, tokenizer, device = self.generator._load_model_and_tokenizer(cfg)
        log_dir = Path(cfg.output_dir) / "chat_log"
        log_dir.mkdir(parents=True, exist_ok=True)
        session_started_at =  self.logger._utc_now_iso()
        session_id = uuid.uuid4().hex[:8]
        log_path = log_dir / f"{session_started_at.replace(':', '-')}_{session_id}.json"
        session_record = {
            "session": {
                "session_id": session_id,
                "started_at": session_started_at,
                "model": OmegaConf.to_container(cfg.model, resolve=True),
                "generation": OmegaConf.to_container(cfg.generation, resolve=True),
                "system_prompt": cfg.system_prompt,
            },
            "turns": [],
        }
        messages = []
        if cfg.system_prompt:
            messages.append({"role": "system", "content": cfg.system_prompt})
        username = cfg.username or "user"
        print(f"[info] Logging this session to: {log_path}", file=sys.stderr)
        print("[info] Type your message and press Enter. Type /exit or Ctrl-D to quit.\n", file=sys.stderr)
        turn_number = 0
        while True:
            try:
                user_content = input(f"{username}> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[info] Exiting.", file=sys.stderr)
                break
            if not user_content:
                continue
            if user_content.lower() in {"/exit", "/quit"}:
                break
            messages.append({"role": "user", "content": user_content})
            answer = self.generator._generate_once(model, tokenizer, device, messages, cfg)
            messages.append({"role": "assistant", "content": answer})
            turn_number += 1
            session_record["turns"].append({
                "turn": turn_number,
                "user": {"username": username, "content": user_content, "timestamp": self.logger._utc_now_iso()},
                "assistant": {"content": answer, "timestamp": self.logger._utc_now_iso()},
            })
            log_path.write_text(json.dumps(session_record, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"assistant> {answer}\n")
        print(f"[info] Session saved to: {log_path}", file=sys.stderr)

    # --------------------------------------------------------------------------
    # generate -- single prompt in, single response out, no history kept.
    # --------------------------------------------------------------------------
    def run_generate(self, cfg) -> None:
        model, tokenizer, device = self.generator._load_model_and_tokenizer(cfg)
        prompt = self.generator._resolve_prompt(cfg)
        messages = []
        if cfg.system_prompt:
            messages.append({"role": "system", "content": cfg.system_prompt})
        messages.append({"role": "user", "content": prompt})
        print(self.generator._generate_once(model, tokenizer, device, messages, cfg))

    # --------------------------------------------------------------------------
    # tool_calling -- single-turn generation with tool/function definitions
    # passed through the chat template.
    # --------------------------------------------------------------------------
    def run_tool_calling(self, cfg) -> None:
        if not cfg.tool_calling.enabled:
            raise ValueError("mode=tool_calling requires tool_calling.enabled=true")
        if not cfg.model.supports_tool_calling:
            self.log.warning(
                "Model '%s' is not marked as supporting tool calling "
                "(model.supports_tool_calling=false); its chat template may "
                "ignore the 'tools' argument entirely.", cfg.model.name,
            )
        model, tokenizer, device = self.generator._load_model_and_tokenizer(cfg)
        prompt = self.generator._resolve_prompt(cfg)
        tools = OmegaConf.to_container(cfg.tool_calling.tools, resolve=True)

        if cfg.model.tool_calling_format == "functools_prompt":
            system_content = self.generator.build_functools_system_prompt(cfg.system_prompt, tools)
            messages = [{"role": "system", "content": system_content}]
            messages.append({"role": "user", "content": prompt})

            answer, tool_call_records = self.generator.run_tool_turn(model, tokenizer, device, messages, cfg)
            if tool_call_records:
                print("[info] Tool call(s) made:", file=sys.stderr)
                for record in tool_call_records:
                    print(f"  {record['name']}({record['arguments']}) -> {record['result']}", file=sys.stderr)
            print(answer)
            return

        messages = []
        if cfg.system_prompt:
            messages.append({"role": "system", "content": cfg.system_prompt})
        messages.append({"role": "user", "content": prompt})

        answer = self.generator._generate_once(
            model, tokenizer, device, messages, cfg,
            tools=tools, tool_choice=cfg.tool_calling.tool_choice,
        )
        print(answer)

    # --------------------------------------------------------------------------
    # structured_output -- single-turn generation constrained to a JSON schema.
    # --------------------------------------------------------------------------
    def run_structured_output(self, cfg) -> None:
        if not cfg.structured_output.enabled:
            raise ValueError("mode=structured_output requires structured_output.enabled=true")
        if not cfg.model.supports_structured_output:
            self.log.warning(
                "Model '%s' is not marked as supporting structured output "
                "(model.supports_structured_output=false); results may not "
                "reliably follow the schema.", cfg.model.name,
            )
        model, tokenizer, device = self.generator._load_model_and_tokenizer(cfg)
        prompt = self.generator._resolve_prompt(cfg)
        schema_instruction = ""
        if cfg.structured_output.schema_path:
            schema_text = Path(cfg.structured_output.schema_path).read_text(encoding="utf-8")
            schema_instruction = (
                "\n\nRespond with ONLY a single JSON object that strictly matches "
                f"this JSON Schema, with no other text:\n{schema_text}"
            )
        system_content = (cfg.system_prompt or "") + schema_instruction
        messages = []
        if system_content.strip():
            messages.append({"role": "system", "content": system_content.strip()})
        messages.append({"role": "user", "content": prompt})
        answer = self.generator._generate_once(model, tokenizer, device, messages, cfg)
        if cfg.structured_output.strict:
            try:
                print(json.dumps(json.loads(answer), indent=2))
                return
            except json.JSONDecodeError as e:
                self.log.warning("Model output was not valid JSON (%s); printing raw output instead.", e)
        print(answer)

    # --------------------------------------------------------------------------
    # prepare_data -- process an original dataset for LLMBox compatibility,
    # optionally format for the target model, add specialtokens/prefixes,
    # split, tokenize, and verify readiness for training, finetuning or eval.
    # --------------------------------------------------------------------------
    @staticmethod
    def _load_json_or_jsonl(path: Path) -> list:
        """Load a .json or .jsonl file, whichever it actually is."""
        text = path.read_text(encoding="utf-8-sig")
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as whole_file_err:
            # Not a single JSON document -> try JSON Lines (one object per line)
            examples = []
            for lineno, line in enumerate(text.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"{path} is neither valid JSON ({whole_file_err}) nor valid "
                        f"JSON Lines (line {lineno}: {e.msg})."
                    ) from e
            return examples

        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            if "train" in raw:                      # HuggingFace-style splits
                return list(raw["train"])
            if raw and all(isinstance(v, list) for v in raw.values()):
                return [ex for v in raw.values() for ex in v]
            return [raw]                            # a single record
        raise ValueError("Unrecognized JSON dataset structure.")

    def run_prepare_data(self, cfg) -> None:
        """
        Prepare a dataset for LLMBox jobs: format/standardize, optionally add special tokens/prefixes/suffixes,
        split into train/test, tokenize and report readiness for downstream tasks. The config (cfg) should supply
        the necessary arguments to indicate: (1) where the input dataset is; (2) where to write outputs; (3) columns; (4) etc.
        """
        import sys
        from pathlib import Path
        # 1. Load original input dataset:
        # Accept either JSONL, JSON or CSV as original format for illustration

        input_path = getattr(cfg.data, "raw_path", None) or getattr(cfg.data, "original_path", None) or getattr(cfg.data, "path", None)
        if not input_path:
            raise ValueError("No data.raw_path or data.original_path or data.path set in configuration.")
        input_path = Path(input_path).expanduser()
        output_dir = getattr(cfg.data, "output_dir", None) or getattr(cfg, "output_dir", None) or input_path.parent
        '''
         # Detect file extension
        ext = input_path.suffix.lower()
        if ext == ".jsonl":
            with input_path.open("r", encoding="utf-8") as f:
                examples = [json.loads(line) for line in f if line.strip()]
        elif ext == ".json":
            with input_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            # If dict of splits, default to train or concat all
            if isinstance(raw, dict):
                if "train" in raw:  # HuggingFace convention
                    examples = list(raw["train"])
                else:
                    # Concatenate all split lists
                    examples = []
                    for val in raw.values():
                        if isinstance(val, list):
                            examples.extend(val)
            elif isinstance(raw, list):
                examples = raw
            else:
                raise ValueError("Unrecognized JSON dataset structure.")
        elif ext == ".csv":
            import csv
            with input_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                examples = list(reader)
        '''
        # Detect file format. Content wins over extension: a ".json" file
                # may really be JSON Lines.
        ext = input_path.suffix.lower()
        if ext in (".json", ".jsonl"):
            examples = self._load_json_or_jsonl(input_path)
        elif ext == ".csv":
            import csv
            with input_path.open("r", encoding="utf-8") as f:
                examples = list(csv.DictReader(f))
        else:
            raise ValueError(f"prepare_data: Unknown input file extension: {ext}")
        #else:
            #raise ValueError(f"prepare_data: Unknown input file extension: {ext}")
        print(f"[info] Loaded {len(examples)} original records from {input_path}")

        # 2. Standardize dataset format using DataTransformer:
        #   Prefer explicit config for text_columns, target_columns, label_maps, instruction
        #text_columns = getattr(cfg.data, "text_columns", None)       # sk edited sept. 19 2026 around 6;26 pm ET
        #target_columns = getattr(cfg.data, "target_columns", None)   # sk edited sept. 19 2026 around 6;26 pm ET
        #label_maps = getattr(cfg.data, "label_maps", None)           # sk edited sept. 19 2026 around 6;26 pm ET
        #instruction = getattr(cfg.data, "instruction", "")           # sk edited sept. 19 2026 around 6;26 pm ET
        def _plain(value):
            """OmegaConf container -> plain Python; empty ([], {}, "") -> None."""
            if value is None:
                return None
            if OmegaConf.is_config(value):
                value = OmegaConf.to_container(value, resolve=True)
            return value or None

        text_columns = _plain(getattr(cfg.data, "text_columns", None))
        target_columns = _plain(getattr(cfg.data, "target_columns", None))
        label_maps = _plain(getattr(cfg.data, "label_maps", None))
        instruction = getattr(cfg.data, "instruction", "") or ""

        # Users may override by passing these explicitly
        standardized_iter = self.datamanager.standardize_llm_dataset(
            examples,
            text_columns=text_columns,
            target_columns=target_columns,
            instruction=instruction,
            label_maps=label_maps,
        )
        # (2.5) Optionally, postprocess for model-specific special tokens, prefixes, suffixes
        # We'll check if cfg.model has any such attributes
        add_prefix = getattr(cfg.model, "add_prefix", None)
        add_suffix = getattr(cfg.model, "add_suffix", None)
        add_special = getattr(cfg.model, "add_specialtokens", None)
        standardized_examples = []
        for ex in standardized_iter:
            # prefix/suffix/completion formatting:
            if add_prefix:
                ex["prompt"] = add_prefix + ex["prompt"]
            if add_suffix:
                ex["completion"] = ex["completion"] + add_suffix
            # Optionally, add special tokens or custom callbacks if required
            if add_special:
                special_tokens = add_special if isinstance(add_special, list) else [add_special]
                ex["special_tokens"] = special_tokens
            standardized_examples.append(ex)
        print(f"[info] Standardized to {len(standardized_examples)} prompt/completion records.")
        # 3. Split dataset into train and test splits & write JSONL output
        train_fraction = getattr(cfg.data, "train_fraction", None) or getattr(cfg.data, "split", None)
        if not train_fraction:
            train_fraction = 0.8  # Default
        seed = getattr(cfg, "seed", 42)
        result = self.datamanager.split_and_save_jsonl(
            standardized_examples,
            output_directory=output_dir,
            train_fraction=train_fraction,
            seed=seed,
            overwrite=True
        )
        print(f"[info] Train/test split complete: train={result['train_count']} test={result['test_count']} saved at {output_dir}")
        # 4. Tokenize train/test to verify readiness for train/finetune/eval
        # Load a tokenizer (respecting model config)
        from transformers import AutoTokenizer
        model_path = getattr(cfg.model, "local_path", None) or getattr(cfg.model, "model_id", None)
        local_files_only = cfg.model.source == "local"
        if not model_path:
            raise ValueError("prepare_data: model.local_path or model.model_id not set.")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=getattr(cfg.model, "trust_remote_code", False),
            local_files_only=local_files_only,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        #dataloader = TrainingDataLoader()                  # sk edit: sept. 19 2026 around 7:17 PM EST
        dataloader = self._make_dataloader(cfg)

        # Tokenize a small sample from train & test
        train_path = result['train_path']
        test_path = result['test_path']
        for split_name, path in [("train", train_path), ("test", test_path)]:
            # Only check if file is nonempty
            with open(path, 'r', encoding='utf-8') as f:
                covered = [json.loads(line) for line in f if line.strip()]
            if not covered:
                print(f"[warn] {split_name} split is empty!", file=sys.stderr)
                continue
            # Wrap each record as a conversation for loader
            conversations = []
            for rec in covered:
                try:
                    conversation = dataloader._record_to_conversation(rec)
                    conversations.append(conversation)
                except Exception as e:
                    print(f"[warn] Could not convert record to conversation: {e}", file=sys.stderr)
            # Tokenize a subset (or all if small)
            max_length = getattr(cfg.training, "max_length", 1024)
            n_check = min(10, len(conversations))
            subset = conversations[:n_check]
            try:
                dataset = dataloader._build_dataset(subset, tokenizer, max_length)
                print(f"[info] {split_name} split: Successfully tokenized {len(dataset)} examples; ready for training/eval.")
            except Exception as e:
                print(f"[error] {split_name}: tokenization failed: {e}", file=sys.stderr)
                raise
        print("[info] Data preparation complete. Dataset is ready for LLMBox jobs.")

    # --------------------------------------------------------------------------
    # train / finetune -- shared SFT trainer. 'train' only allows full weight
    # training (continued pretraining); 'finetune' also allows LoRA.
    # --------------------------------------------------------------------------
    def _build_optimizer(self, cfg, model):
        import torch
        opt = cfg.optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        if opt.name == "adamw":
            return torch.optim.AdamW(params, lr=opt.learning_rate, weight_decay=opt.weight_decay,
                                    betas=tuple(opt.betas), eps=opt.eps)
        if opt.name == "sgd":
            return torch.optim.SGD(params, lr=opt.learning_rate, weight_decay=opt.weight_decay,
                                    momentum=opt.momentum)
        if opt.name == "adafactor":
            from transformers.optimization import Adafactor
            return Adafactor(params, lr=opt.learning_rate, weight_decay=opt.weight_decay,
                            scale_parameter=False, relative_step=False)
        raise ValueError(f"Unknown optimizer.name '{opt.name}'.")


    def _run_training(self, cfg, allow_lora: bool) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
        import inspect
        import shutil
        model_metrics_dir = Path("data/evaluations")
        model_metrics_dir.mkdir(parents=True, exist_ok=True)
        if cfg.training.method == "lora" and not allow_lora:
            raise ValueError("mode=train does not support training.method=lora; use mode=finetune for LoRA.")
        device, dtype = self.generator._resolve_device_and_dtype(cfg)
        model_path, local_files_only = self.generator._resolve_model_path(cfg)
        self.log.info("Loading '%s' (source=%s, %s) for training...", cfg.model.name, cfg.model.source, model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=cfg.model.trust_remote_code, local_files_only=local_files_only
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if cfg.model.trust_remote_code:
            self.generator._ensure_remote_code_compat()
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, trust_remote_code=cfg.model.trust_remote_code, local_files_only=local_files_only,
        )
        # VERY FLEXIBLE PEFT HANDLING TO MATCH TrainingConfig AND SCHEMA:
        if hasattr(cfg.training, 'method') and cfg.training.method in {"lora", "adapters", "bitfit", "freeze", "prefix", "qlora"}:
            if cfg.training.method == "lora":
                try:
                    from peft import LoraConfig, get_peft_model
                except ImportError as e:
                    raise RuntimeError("training.method=lora requires: pip install -U peft") from e
                lora_config = LoraConfig(
                    r=getattr(cfg.training, 'lora_r', 8),
                    lora_alpha=getattr(cfg.training, 'lora_alpha', 16),
                    lora_dropout=getattr(cfg.training, 'lora_dropout', 0.05),
                    bias="none", task_type="CAUSAL_LM",
                    target_modules=list(getattr(cfg.training, "lora_target_modules", None) or
                                        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),  # AS02 edit: configurable
                )
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()
            elif cfg.training.method == "adapters":
                try:
                    from peft import AdapterConfig, get_peft_model
                except ImportError as e:
                    raise RuntimeError("training.method=adapters requires: pip install -U peft") from e
                adapters_dim = getattr(cfg.training, 'adapters_dim', 64)
                adapter_config = AdapterConfig(
                    adapters_dim=adapters_dim, task_type="CAUSAL_LM"
                )
                model = get_peft_model(model, adapter_config)
            elif cfg.training.method == "prefix":
                try:
                    from peft import PrefixTuningConfig, get_peft_model
                except ImportError as e:
                    raise RuntimeError("training.method=prefix requires: pip install -U peft") from e
                prefix_length = getattr(cfg.training, 'prefix_length', 30)
                prefix_config = PrefixTuningConfig(
                    task_type="CAUSAL_LM", num_virtual_tokens=prefix_length
                )
                model = get_peft_model(model, prefix_config)
            elif cfg.training.method == "bitfit":
                try:
                    from peft import BitFitConfig, get_peft_model
                except ImportError as e:
                    raise RuntimeError("training.method=bitfit requires: pip install -U peft") from e
                bitfit_bias_params = getattr(cfg.training, 'bitfit_bias_params', [])
                bitfit_config = BitFitConfig(task_type="CAUSAL_LM", bias=bitfit_bias_params or "all")
                model = get_peft_model(model, bitfit_config)
            elif cfg.training.method == "freeze":
                freeze_modules = getattr(cfg.training, 'freeze_modules', [])
                for name, param in model.named_parameters():
                    if any(module in name for module in freeze_modules):
                        param.requires_grad = False
            elif cfg.training.method == "qlora":
                try:
                    from peft import LoraConfig, get_peft_model
                except ImportError as e:
                    raise RuntimeError("training.method=qlora requires: pip install -U peft") from e
                # qlora-specific settings: use lora, but user is assumed to set quantization elsewhere
                lora_config = LoraConfig(
                    r=getattr(cfg.training, 'lora_r', 8),
                    lora_alpha=getattr(cfg.training, 'lora_alpha', 16),
                    lora_dropout=getattr(cfg.training, 'lora_dropout', 0.05),
                    bias="none", task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                )
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()
            else:
                pass  # Default: do nothing (full fine-tune)

        '''  SK EDITED: September 21, 2026 around 7:54 AM EST
        if cfg.training.method == "lora" and not allow_lora:
            raise ValueError("mode=train does not support training.method=lora; use mode=finetune for LoRA.")
        device, dtype = self.generator._resolve_device_and_dtype(cfg)
        model_path, local_files_only = self.generator._resolve_model_path(cfg)
        self.log.info("Loading '%s' (source=%s, %s) for training...", cfg.model.name, cfg.model.source, model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=cfg.model.trust_remote_code, local_files_only=local_files_only
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if cfg.model.trust_remote_code:
            self.generator._ensure_remote_code_compat()
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, trust_remote_code=cfg.model.trust_remote_code, local_files_only=local_files_only,
        )
        if cfg.training.method == "lora":
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as e:
                raise RuntimeError("training.method=lora requires: pip install -U peft") from e
            lora_config = LoraConfig(
                r=cfg.training.lora_r, lora_alpha=cfg.training.lora_alpha,
                lora_dropout=cfg.training.lora_dropout, bias="none", task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        '''

        model.to(device)
        self.dataloader = self._make_dataloader(cfg)                    # sk edited: sept. 19 2026 around 7:18 PM EST
        conversations = self.dataloader._load_conversations(cfg)
        if not conversations:
            raise ValueError(f"No conversations found under data.path='{cfg.data.path}' (data.type='{cfg.data.type}').")
        self.log.info("Loaded %d conversation(s).", len(conversations))
        random.seed(cfg.seed)
        random.shuffle(conversations)
        num_eval = (
            max(1, int(len(conversations) * cfg.data.eval_split))
            if cfg.data.eval_split > 0 and len(conversations) > 1 else 0
        )
        eval_conversations = conversations[:num_eval]
        train_conversations = conversations[num_eval:]
        self.log.info("Train: %d  Eval: %d", len(train_conversations), len(eval_conversations))
        train_dataset = self.dataloader._build_dataset(train_conversations, tokenizer, cfg.training.max_length)
        eval_dataset = self.dataloader._build_dataset(eval_conversations, tokenizer, cfg.training.max_length) if eval_conversations else None
        collator = self.dataloader._make_collator(tokenizer.pad_token_id)
        optimizer = self._build_optimizer(cfg, model)

        import inspect                                                         # sk edited: sept. 19 2026 around 7:40 PM EST
        _ta_params = inspect.signature(TrainingArguments.__init__).parameters  # sk edited: sept. 19 2026 around 7:40 PM EST
        if "warmup_ratio" in _ta_params:            # older transformers       # sk edited: sept. 19 2026 around 7:40 PM EST
            warmup_kwargs = {"warmup_ratio": cfg.training.warmup_ratio}        # sk edited: sept. 19 2026 around 7:40 PM EST
        else:                                       # newer: float < 1 = ratio of total steps   # sk edited: sept. 19 2026 around 7:40 PM EST
            warmup_kwargs = {"warmup_steps": cfg.training.warmup_ratio}        # sk edited: sept. 19 2026 around 7:40 PM EST

        training_args = TrainingArguments(
            output_dir=cfg.training.output_dir,
            num_train_epochs=cfg.training.epochs,
            per_device_train_batch_size=cfg.training.batch_size,
            per_device_eval_batch_size=cfg.training.batch_size,
            gradient_accumulation_steps=cfg.training.grad_accum_steps,
            #warmup_ratio=cfg.training.warmup_ratio,                        # sk edited: sept. 19 2026 around 7:40 PM EST
            **warmup_kwargs,                                                # sk edited: sept. 19 2026 around 7:40 PM EST
            logging_steps=cfg.training.logging_steps,
            save_steps=cfg.training.save_steps,
            save_total_limit=2,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=cfg.training.eval_steps if eval_dataset else None,
            bf16=(dtype.__str__() == "torch.bfloat16" and device != "mps"),
            fp16=(dtype.__str__() == "torch.float16" and device != "mps"),
            report_to=[],
            remove_unused_columns=False,
            include_num_input_tokens_seen=True,   # AS02 edit: exact token count for MetricsTracker
            prediction_loss_only=True,            # AS02 edit: do not accumulate eval logits (262k vocab x seq x batch
                                                  #   blew MPS memory); eval_loss is all we report
        )
        trainer = Trainer(
            model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset,
            data_collator=collator, optimizers=(optimizer, None),
        )
        if device == "mps":                                   # AS02 edit: MPS allocator kept growing
            import torch                                      #   (OOM after ~25 steps with the 262k vocab);
            from transformers import TrainerCallback          #   release cached blocks after every step.

            class _MpsEmptyCache(TrainerCallback):
                def on_step_end(self, args, state, control, **kwargs):
                    torch.mps.empty_cache()
                def on_evaluate(self, args, state, control, **kwargs):
                    torch.mps.empty_cache()
            trainer.add_callback(_MpsEmptyCache())
        self.log.info("Starting %s (method=%s, optimizer=%s)...", cfg.mode.name, cfg.training.method, cfg.optimizer.name)

        _n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)   # AS02 edit: count before
        _n_total = sum(p.numel() for p in model.parameters())                          #   merge_and_unload()
        metrics_tracker = training_metrics.MetricsTracker(cfg)  # sk edited: sept. 20 2026 around 10:10 AM EST
        if cfg.training.report_metrics:                         # sk edited: sept. 20 2026 around 10:10 AM EST
            metrics_tracker.start()                             # sk edited: sept. 20 2026 around 10:10 AM EST

        trainer.train()
        trainer.save_model(cfg.training.output_dir)
        tokenizer.save_pretrained(cfg.training.output_dir)
        self.log.info("Saved to '%s'.", cfg.training.output_dir)
        if cfg.training.method == "lora" and cfg.training.merge_adapter:
            merged_model = model.merge_and_unload()
            merged_dir = Path(cfg.training.output_dir) / "merged"
            merged_model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            self.log.info("Merged model saved to '%s'.", merged_dir)

        # Write metrics as JSON (not MD) to /data/evaluation/.
        if cfg.training.report_metrics:                                # sk edited: sept. 20 2026 around 10:10 AM EST
            metrics_obj = metrics_tracker.stop(trainer)                # sk edited: sept. 20 2026 around 10:10 AM EST
            # Only save JSON to /data/evaluation; do not write MD (to preserve legacy MD path).
            metrics_file = model_metrics_dir / f"{cfg.mode.name}_{cfg.model.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_metrics.json" # sk edited: sept. 20 2026 around 10:10 AM EST
            # Save the grouped metrics dictionary to metrics.json                 # sk edited: sept. 20 2026 around 10:10 AM EST
            metrics_file.write_text(json.dumps(metrics_obj.to_grouped_dict(), indent=2, ensure_ascii=False), encoding="utf-8") # sk edited: sept. 20 2026 around 10:10 AM EST
            print(f"[info] Training metrics saved for analysis at {metrics_file}")  # sk edited: sept. 20 2026 around 10:10 AM EST
            # ---- AS02 edit (Y. Yao, Sept 2026): reproducibility bundle per run ----
            stem = metrics_file.with_suffix('').name.replace('_metrics', '')
            run_name = getattr(cfg.training, 'run_name', None)
            if run_name:
                stem = f"{stem}_{run_name}"
            (model_metrics_dir / f"{stem}_log_history.json").write_text(
                json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
            (model_metrics_dir / f"{stem}_config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
            curve = training_metrics.plot_loss_curve(trainer, str(model_metrics_dir / f"{stem}_loss_curve.png"))
            trainable, total = _n_trainable, _n_total
            extra = {"run_name": run_name, "trainable_params": trainable, "total_params": total,
                     "trainable_pct": round(100 * trainable / total, 4), "global_steps": trainer.state.global_step,
                     "train_examples": len(train_dataset), "eval_examples": len(eval_dataset) if eval_dataset else 0,
                     "device": device, "dtype": str(dtype), "metrics_file": str(metrics_file)}
            (model_metrics_dir / f"{stem}_run_info.json").write_text(json.dumps(extra, indent=2), encoding="utf-8")
            print(f"[info] AS02 run bundle saved: {stem}_log_history.json / _config.yaml / _run_info.json" + (f" / loss curve {curve}" if curve else ""))
         # The training_report.md is handled (unchanged) in the current output_dir by other pipelines if needed.
         # No change to MD report in output_dir.

    def run_train(self, cfg) -> None:
        if not cfg.training.enabled:
            raise ValueError("mode=train requires training.enabled=true")
        self._run_training(cfg, allow_lora=False)

    def run_finetune(self, cfg) -> None:
        if not cfg.training.enabled:
            raise ValueError("mode=finetune requires training.enabled=true")
        self._run_training(cfg, allow_lora=True)

    # --------------------------------------------------------------------------
    # evaluation -- distinct evaluation mode using src/evaluator.py.
    # Must NEVER run training_metrics logic from train or finetune.
    # Walk user through evaluation tasks one by one interactively in the terminal.
    # --------------------------------------------------------------------------
    def run_evaluation(self, cfg) -> None:
        """Run LLM evaluation (distinct mode). Calls src.evaluator functions.
        This mode is SEPARATE from train/finetune and never runs training_metrics.
        Walks the user through EVALUATION tasks ONE BY ONE interactively.
        """
        import src.evaluator as llm_evaluation
        import os
        import sys
        from pathlib import Path

        print("[EVALUATION MODE] Entering interactive terminal evaluation suite.")
        # Step 1: Prepare evaluation inputs (select test split or other evaluation file)
        data_path = ""
        sample_size = 10
        print("\nStep 1: Prepare Evaluation Inputs")
        print("-------------------------------")
        # Auto-pick test.jsonl if present in cfg or default, or prompt user
        eval_candidates = []
        possible_eval_paths = []
        data_cfg = getattr(cfg, 'data', None)
        if data_cfg is not None:
            base_dir = None
            if hasattr(data_cfg, 'output_dir') and data_cfg.output_dir:
                base_dir = Path(data_cfg.output_dir).expanduser()
            elif hasattr(cfg, 'output_dir') and cfg.output_dir:
                base_dir = Path(cfg.output_dir).expanduser()
            if base_dir:
                possible_eval_paths.extend([
                    base_dir / "test.jsonl",
                    base_dir / "evaluation_dataset.json",
                ])
            # Also suggest data_cfg.path if it exists and is not a train file
            if hasattr(data_cfg, 'path') and data_cfg.path:
                possible_eval_paths.append(Path(data_cfg.path).expanduser())

        # Add cwd options for fallback
        for fn in ["test.jsonl", "evaluation_dataset.json", "data/test.jsonl", "eval.jsonl"]:
            possible_eval_paths.append(Path(fn))
        for p in possible_eval_paths:
            if p.exists() and p.is_file():
                eval_candidates.append(p)
        data_path = ""
        if eval_candidates:
            print("Available evaluation dataset(s):")
            for i, c in enumerate(eval_candidates, 1):
                print(f"  [{i}] {c}")
            sel = input(f"Select evaluation data file [1-{len(eval_candidates)}] or enter path: ").strip()
            if sel.isdigit() and 1 <= int(sel) <= len(eval_candidates):
                data_path = str(eval_candidates[int(sel) - 1])
            else:
                data_path = sel or str(eval_candidates[0])
        else:
            data_path = input("Enter path to evaluation input file (.jsonl/.json): ").strip()
        while not (os.path.isfile(data_path) and data_path.lower().endswith(('.json', '.jsonl'))):
            data_path = input("  File not found or not a .json/.jsonl. Enter path: ").strip()

        while True:
            inp = input("Sample size for evaluation (default=10): ").strip()
            if not inp:
                sample_size = 10
                break
            if inp.isdigit() and int(inp) > 0:
                sample_size = int(inp)
                break
            print("  Please enter a positive integer.")
        kind = "auto"
        pi = llm_evaluation.PrepareEvaluationInput(data_path, sample_size=sample_size, kind=kind)
        formatted_eval_data = pi.format_for_evaluation()
        eval_out_path = Path(data_path).parent / "evaluation_dataset.json"
        pi.save_evaluation_dataset(str(eval_out_path))
        print(f"[OK] Wrote formatted eval sample to {eval_out_path}")

        # Step 2: Human Evaluation
        print("\nStep 2: Human Rating of LLM Responses")
        print("------------------------------------")
        print("You will now label the evaluation sample via interactive ratings.")
        h = llm_evaluation.HumanEvaluation(str(eval_out_path))
        labeled_path = h.run()
        print(f"[INFO] Human-labeled evaluation sample saved as {labeled_path}")

        # Step 3: Agreement measure (if possible)
        print("\nStep 3: (Optional) Inter-rater Agreement")
        print("----------------------------------------")
        print("If you have TWO labeled human judgment files (e.g., from two raters), you can measure agreement.")
        want_agreement = input("Compute agreement between two raters? [y/N]: ").strip().lower().startswith('y')
        if want_agreement:
            l1 = input("Path to first labeled dataset (JSON): ").strip()
            l2 = input("Path to second labeled dataset (JSON): ").strip()
            try:
                with open(l1, encoding="utf-8") as f1, open(l2, encoding="utf-8") as f2:
                    d1 = json.load(f1)
                    d2 = json.load(f2)
                labels1 = [ex.get("rating") for ex in d1]
                labels2 = [ex.get("rating") for ex in d2]
                pa = llm_evaluation.AgreementMeasures.percent_agreement(labels1, labels2)
                print(f"  Percent agreement: {pa*100:.2f}%")
                try:
                    kappa = llm_evaluation.AgreementMeasures.cohen_kappa(labels1, labels2)
                    print(f"  Cohen's kappa: {kappa:.4f}")
                except Exception as _:
                    print("  Cohen's kappa unavailable.")
            except Exception as e:
                print(f"[WARN] Agreement measure failed: {e}")
        else:
            print("(Skipping agreement calculation)")

        # Step 4: LLM Judge Evaluation (if user wants/has a judge function)
        print("\nStep 4: (Optional) LLM Judge Evaluation")
        print("---------------------------------------")
        print("If you wish to run an LLM JUDGE function/class over your evaluation set, you may do so here (advanced).")
        llmjudge_response = input("Run LLM Judge evaluation? [y/N]: ").strip().lower().startswith('y')
        if llmjudge_response:
            print("Provide a function (Python code) that takes (example, criteria) -> label. You may manually edit or implement it in src/evaluator.py>LLMJudgeEvaluation usage.")
            # For simplicity, let user define a very simple inline judge in REPL, otherwise skip
            user_code = input("Paste code for a labeling function? Leave blank to use a trivial judge that returns 5:").strip()
            judge_schema_source = None
            if not user_code:
                def trivial_judge(example, criteria):
                    return 5
                judge_fn = trivial_judge
            else:
                import types
                local_ctx = {}
                try:
                    exec(user_code, {}, local_ctx)
                    judge_fn = None
                    for k, v in local_ctx.items():
                        if callable(v):
                            judge_fn = v
                            break
                    if not judge_fn:
                        print("[WARN] No function found in user code. Defaulting to 'always 5'.")
                        judge_fn = lambda ex, cr: 5
                    else:
                        judge_schema_source = "user_repl_function"
                except Exception as e:
                    print(f"[WARN] Could not parse judge function: {e}. Using default.")
                    judge_fn = lambda ex, cr: 5
            llmjudge = llm_evaluation.LLMJudgeEvaluation(str(eval_out_path), judge_fn, judge_schema_source=judge_schema_source)
            judge_results_path = llmjudge.run()
            print(f"[INFO] LLM Judge evaluation complete: {judge_results_path}")
        else:
            print("(Skipping LLM Judge evaluation)")

        print("\nEVALUATION SUITE COMPLETE!")
        print("All outputs available under:")
        print("  ", Path(eval_out_path).parent.absolute())

        # Optionally, print location and instructions
        print("\nNext steps:")
        print("  - Edit evaluation_criteria.md to describe your rubric.")
        print("  - Email outputs as instructed")

        # No return needed; end of evaluation
        return
