import argparse
import datetime as dt
import inspect
import json
import random
import re
import shutil
import types
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def slugify_model_name(model_name):
    slug = model_name.lower().replace("/", "__")
    slug = re.sub(r"[^a-z0-9_.-]+", "-", slug)
    return slug.strip("-")


def make_run_id(model_name):
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{slugify_model_name(model_name)}_dpo"


def format_output(output):
    return json.dumps(output, ensure_ascii=False, indent=2)


def make_prompt(record, template):
    return template.format(
        instruction=record["instruction"],
        passage=record["input"]["passage"],
        question=record["input"]["question"],
        question_type=record["input"]["question_type"],
        output_json="",
    ).rstrip()


def split_stratified(records, stratify_key, train_ratio, seed):
    rng = random.Random(seed)
    groups = {}
    for index, record in enumerate(records):
        meta = record.get("meta", {})
        key = meta.get(stratify_key) or record.get(stratify_key) or "unknown"
        groups.setdefault(key, []).append(index)

    train_indexes = []
    valid_indexes = []
    for indexes in groups.values():
        rng.shuffle(indexes)
        if len(indexes) == 1:
            train_count = 1
        else:
            train_count = max(1, round(len(indexes) * train_ratio))
            train_count = min(train_count, len(indexes) - 1)
        train_indexes.extend(indexes[:train_count])
        valid_indexes.extend(indexes[train_count:])

    rng.shuffle(train_indexes)
    rng.shuffle(valid_indexes)
    return [records[i] for i in train_indexes], [records[i] for i in valid_indexes]


def to_preference_rows(records, template):
    rows = []
    for record in records:
        meta = record.get("meta", {})
        rows.append(
            {
                "prompt": make_prompt(record, template),
                "chosen": format_output(record["chosen"]),
                "rejected": format_output(record["rejected"]),
                "source_index": meta.get("source_index"),
                "requested_failure_type": meta.get("requested_failure_type"),
            }
        )
    return rows


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def find_embedding_module(model):
    vocab_size = getattr(model.config, "vocab_size", None)
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            score = 0
            if vocab_size is not None and module.num_embeddings == vocab_size:
                score += 10
            if any(token in name.lower() for token in ["embed", "wte", "tok"]):
                score += 3
            candidates.append((score, name, module))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: (item[0], item[2].num_embeddings), reverse=True)
    _, name, module = candidates[0]
    return name, module


def set_module_by_name(model, module_name, new_module):
    parent = model
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def patch_input_embeddings_if_needed(model, model_name):
    if "exaone" not in model_name.lower():
        return
    embedding_name, embedding_module = find_embedding_module(model)
    if embedding_module is None:
        print("[WARN] EXAONE input embedding patch skipped: embedding module not found")
        return

    def get_input_embeddings(self):
        return embedding_module

    def set_input_embeddings(self, value):
        set_module_by_name(model, embedding_name, value)

    model.get_input_embeddings = types.MethodType(get_input_embeddings, model)
    model.set_input_embeddings = types.MethodType(set_input_embeddings, model)
    print(f"[INFO] Patched EXAONE input embeddings: {embedding_name}")


def resolve_sft_adapter(model_name, config, explicit_adapter):
    if explicit_adapter:
        return explicit_adapter
    return (config.get("sft_adapters") or {}).get(model_name)


def load_model(model_name, config, sft_adapter=None):
    training_config = config["training"]
    quantization_config = None
    if training_config["load_in_4bit"]:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    patch_input_embeddings_if_needed(model, model_name)
    if training_config["load_in_4bit"]:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_config["gradient_checkpointing"],
        )
    if sft_adapter:
        model = PeftModel.from_pretrained(model, sft_adapter, is_trainable=True)
        print(f"[INFO] Loaded trainable SFT adapter: {sft_adapter}")
    return model


def build_dpo_config(config, run_dir, run_id, max_steps=None):
    training_config = dict(config["training"])
    if max_steps is not None:
        training_config["max_steps"] = max_steps
        training_config["save_strategy"] = "steps"
        training_config["eval_strategy"] = "steps"
        training_config["save_steps"] = max_steps
        training_config["eval_steps"] = max_steps

    kwargs = {
        "output_dir": str(run_dir / "checkpoints"),
        "run_name": run_id,
        "num_train_epochs": training_config["num_train_epochs"],
        "learning_rate": training_config["learning_rate"],
        "per_device_train_batch_size": training_config["per_device_train_batch_size"],
        "per_device_eval_batch_size": training_config["per_device_eval_batch_size"],
        "gradient_accumulation_steps": training_config["gradient_accumulation_steps"],
        "warmup_ratio": training_config["warmup_ratio"],
        "weight_decay": training_config["weight_decay"],
        "logging_steps": training_config["logging_steps"],
        "save_strategy": training_config["save_strategy"],
        "bf16": torch.cuda.is_available(),
        "fp16": False,
        "report_to": "none",
        "remove_unused_columns": False,
        "gradient_checkpointing": training_config["gradient_checkpointing"],
        "beta": training_config["beta"],
        "max_length": training_config["max_length"],
        "max_prompt_length": training_config["max_prompt_length"],
    }
    if "max_steps" in training_config:
        kwargs["max_steps"] = training_config["max_steps"]
    if "save_steps" in training_config:
        kwargs["save_steps"] = training_config["save_steps"]
    if "eval_steps" in training_config:
        kwargs["eval_steps"] = training_config["eval_steps"]

    signature = inspect.signature(DPOConfig.__init__).parameters
    if "eval_strategy" in signature:
        kwargs["eval_strategy"] = training_config["eval_strategy"]
    elif "evaluation_strategy" in signature:
        kwargs["evaluation_strategy"] = training_config["eval_strategy"]

    filtered = {key: value for key, value in kwargs.items() if key in signature}
    return DPOConfig(**filtered)


def train(args):
    config = load_json(resolve_path(args.config))
    model_name = args.model_name or config["models"][0]
    sft_adapter = resolve_sft_adapter(model_name, config, args.sft_adapter)
    run_id = args.run_id or make_run_id(model_name)
    output_root = resolve_path(config["outputs"]["root_dir"])
    run_dir = output_root / run_id
    adapter_dir = run_dir / config["outputs"]["adapter_dir_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    records = load_json(resolve_path(config["dataset"]["path"]))
    records = [record for record in records if record.get("meta", {}).get("quality_tier") != "exclude"]
    template = resolve_path(config["prompt"]["template_path"]).read_text(encoding="utf-8")
    train_records, valid_records = split_stratified(
        records,
        config["dataset"]["stratify_by"],
        config["dataset"]["train_ratio"],
        config["dataset"]["seed"],
    )
    train_rows = to_preference_rows(train_records, template)
    valid_rows = to_preference_rows(valid_records, template)

    write_json(run_dir / "train_split.json", train_records)
    write_json(run_dir / "valid_split.json", valid_records)
    write_json(run_dir / "train_preferences.json", train_rows)
    write_json(run_dir / "valid_preferences.json", valid_rows)
    write_json(run_dir / config["outputs"]["train_config_file"], config)

    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, config, sft_adapter)
    peft_config = None
    if sft_adapter is None:
        peft_config = LoraConfig(
            r=config["training"]["lora_r"],
            lora_alpha=config["training"]["lora_alpha"],
            lora_dropout=config["training"]["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
    dpo_config = build_dpo_config(config, run_dir, run_id, args.max_steps)

    train_dataset = Dataset.from_list(train_rows)
    valid_dataset = Dataset.from_list(valid_rows)
    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "args": dpo_config,
        "train_dataset": train_dataset,
        "eval_dataset": valid_dataset,
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config
    trainer_signature = inspect.signature(DPOTrainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = DPOTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics = trainer.evaluate()
    eval_summary = {
        "train_size": len(train_rows),
        "valid_size": len(valid_rows),
        "dataset_path": config["dataset"]["path"],
        "model_name": model_name,
        "sft_adapter": sft_adapter,
        "metrics": metrics,
    }
    eval_results = {
        "run_id": run_id,
        "model_name": model_name,
        "sft_adapter": sft_adapter,
        "dataset_path": config["dataset"]["path"],
        "eval_summary": eval_summary,
    }
    write_json(run_dir / config["outputs"]["eval_results_file"], eval_results)
    write_json(run_dir / config["outputs"]["eval_summary_file"], eval_summary)

    if args.copy_config:
        shutil.copy2(resolve_path(args.config), run_dir / Path(args.config).name)

    print(f"run_dir={run_dir}")
    print(json.dumps(eval_summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_dpo_config_v1.json")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--sft_adapter", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--copy_config", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
