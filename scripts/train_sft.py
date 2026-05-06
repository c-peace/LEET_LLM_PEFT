import argparse
import copy
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


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
    return f"{timestamp}_{slugify_model_name(model_name)}"


def split_stratified(records, stratify_key, train_ratio, seed):
    rng = random.Random(seed)
    groups = {}
    for index, record in enumerate(records):
        key = record["input"][stratify_key]
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
    train_records = [records[index] for index in train_indexes]
    valid_records = [records[index] for index in valid_indexes]
    return train_records, valid_records


def format_output(output):
    return json.dumps(output, ensure_ascii=False, indent=2)


def make_prompt(record, template, include_output):
    output_json = format_output(record["output"]) if include_output else ""
    return template.format(
        instruction=record["instruction"],
        passage=record["input"]["passage"],
        question=record["input"]["question"],
        question_type=record["input"]["question_type"],
        output_json=output_json,
    )


def make_inference_prompt(record, template):
    prompt = make_prompt(record, template, include_output=False)
    return prompt.rstrip()


def tokenize_records(records, tokenizer, template, max_seq_length):
    tokenized = []
    eos = tokenizer.eos_token or ""

    for record in records:
        prompt = make_inference_prompt(record, template)
        answer = format_output(record["output"])
        full_text = f"{prompt}\n{answer}{eos}"

        full = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=True,
        )
        prompt_ids = tokenizer(
            f"{prompt}\n",
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=True,
        )["input_ids"]

        labels = copy.deepcopy(full["input_ids"])
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        full["labels"] = labels
        tokenized.append(full)

    return Dataset.from_list(tokenized)


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

    base_model_prefix = getattr(model, "base_model_prefix", None)
    base_model = getattr(model, base_model_prefix, None) if base_model_prefix else None
    if base_model is None:
        for attr in ["model", "transformer", "backbone"]:
            candidate = getattr(model, attr, None)
            if candidate is not None:
                base_model = candidate
                break

    if base_model is not None:
        base_model.get_input_embeddings = types.MethodType(get_input_embeddings, base_model)
        base_model.set_input_embeddings = types.MethodType(set_input_embeddings, base_model)

    print(f"[INFO] Patched EXAONE input embeddings: {embedding_name}")


def load_model(model_name, config):
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
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=training_config["gradient_checkpointing"],
    )

    lora_config = LoraConfig(
        r=training_config["lora_r"],
        lora_alpha=training_config["lora_alpha"],
        lora_dropout=training_config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    return get_peft_model(model, lora_config)


def extract_json_object(raw_text):
    raw_text = raw_text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.S)
    if fenced:
        raw_text = fenced.group(1)

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("JSON object not found")

    return json.loads(raw_text[start : end + 1])


def is_sentence_like(choice):
    choice = choice.strip()
    if not choice:
        return False
    if choice[-1] in ".!?":
        return True
    return choice.endswith(
        (
            "다",
            "이다",
            "한다",
            "된다",
            "있다",
            "없다",
            "않다",
            "아니다",
            "수 있다",
            "수 없다",
        )
    )


def evaluate_rules(raw_output):
    result = {
        "json_parse_success": False,
        "valid_choice_count": False,
        "valid_answer_index": False,
        "no_empty_choices": False,
        "no_duplicate_choices": False,
        "sentence_like_choices": False,
        "all_passed": False,
        "errors": [],
    }

    try:
        parsed = extract_json_object(raw_output)
        result["json_parse_success"] = True
    except Exception as exc:
        result["errors"].append(f"parse_error: {exc}")
        return result, None

    choices = parsed.get("choices")
    answer_index = parsed.get("answer_index")

    if isinstance(choices, list) and len(choices) == 5:
        result["valid_choice_count"] = True
    else:
        result["errors"].append("choices must be a list of exactly 5 items")

    if isinstance(answer_index, int) and 1 <= answer_index <= 5:
        result["valid_answer_index"] = True
    else:
        result["errors"].append("answer_index must be an integer from 1 to 5")

    if isinstance(choices, list):
        normalized_choices = [str(choice).strip() for choice in choices]
        if all(normalized_choices):
            result["no_empty_choices"] = True
        else:
            result["errors"].append("choices contain empty item")

        if len(set(normalized_choices)) == len(normalized_choices):
            result["no_duplicate_choices"] = True
        else:
            result["errors"].append("choices contain duplicate item")

        if len(normalized_choices) == 5 and all(is_sentence_like(choice) for choice in normalized_choices):
            result["sentence_like_choices"] = True
        else:
            result["errors"].append("choices are not all sentence-like")

    checked_keys = [
        "json_parse_success",
        "valid_choice_count",
        "valid_answer_index",
        "no_empty_choices",
        "no_duplicate_choices",
        "sentence_like_choices",
    ]
    result["all_passed"] = all(result[key] for key in checked_keys)
    return result, parsed


def summarize_rule_results(items):
    keys = [
        "json_parse_success",
        "valid_choice_count",
        "valid_answer_index",
        "no_empty_choices",
        "no_duplicate_choices",
        "sentence_like_choices",
        "all_passed",
    ]
    summary = {"total": len(items)}
    for key in keys:
        summary[key] = sum(1 for item in items if item["rule_eval"][key])
    return summary


def generate_eval_results(model, tokenizer, valid_records, template, config, run_id, model_name):
    evaluation_config = config["evaluation"]
    sample_size = min(evaluation_config["generation_sample_size"], len(valid_records))
    eval_records = valid_records[:sample_size]
    items = []

    model.eval()
    for index, record in enumerate(eval_records):
        prompt = make_inference_prompt(record, template)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config["training"]["max_seq_length"])
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=evaluation_config["max_new_tokens"],
                do_sample=evaluation_config["do_sample"],
                temperature=evaluation_config["temperature"] if evaluation_config["do_sample"] else None,
                top_p=evaluation_config["top_p"] if evaluation_config["do_sample"] else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_length = inputs["input_ids"].shape[-1]
        generated_ids = generated[0][prompt_length:]
        raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        rule_eval, parsed_output = evaluate_rules(raw_output)

        items.append(
            {
                "id": index,
                "input": record["input"],
                "gold_output": record["output"],
                "model_raw_output": raw_output,
                "model_output": parsed_output,
                "rule_eval": rule_eval,
                "llm_judge": None,
            }
        )

    return {
        "run_id": run_id,
        "model_name": model_name,
        "dataset_path": config["dataset"]["path"],
        "eval_summary": summarize_rule_results(items),
        "items": items,
    }


def train(args):
    config = load_json(resolve_path(args.config))
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
        config["training"]["save_strategy"] = "steps"
        config["training"]["eval_strategy"] = "steps"
        config["training"]["save_steps"] = args.max_steps
        config["training"]["eval_steps"] = args.max_steps

    if args.eval_sample_size is not None:
        config["evaluation"]["generation_sample_size"] = args.eval_sample_size

    model_name = args.model_name or config["models"][0]
    run_id = args.run_id or make_run_id(model_name)
    output_root = resolve_path(config["outputs"]["root_dir"])
    run_dir = output_root / run_id
    adapter_dir = run_dir / config["outputs"]["adapter_dir_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_records = load_json(resolve_path(config["dataset"]["path"]))
    template = resolve_path(config["prompt"]["template_path"]).read_text(encoding="utf-8")
    train_records, valid_records = split_stratified(
        dataset_records,
        config["dataset"]["stratify_by"],
        config["dataset"]["train_ratio"],
        config["dataset"]["seed"],
    )

    write_json(run_dir / "train_split.json", train_records)
    write_json(run_dir / "valid_split.json", valid_records)
    write_json(run_dir / config["outputs"]["train_config_file"], config)

    tokenizer = load_tokenizer(model_name)
    train_dataset = tokenize_records(
        train_records,
        tokenizer,
        template,
        config["training"]["max_seq_length"],
    )
    valid_dataset = tokenize_records(
        valid_records,
        tokenizer,
        template,
        config["training"]["max_seq_length"],
    )

    model = load_model(model_name, config)
    model.print_trainable_parameters()

    training_config = config["training"]
    training_args_kwargs = {
        "output_dir": str(run_dir / "checkpoints"),
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
    }
    if "max_steps" in training_config:
        training_args_kwargs["max_steps"] = training_config["max_steps"]
    if "save_steps" in training_config:
        training_args_kwargs["save_steps"] = training_config["save_steps"]
    if "eval_steps" in training_config:
        training_args_kwargs["eval_steps"] = training_config["eval_steps"]
    training_args_signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in training_args_signature:
        training_args_kwargs["eval_strategy"] = training_config["eval_strategy"]
    else:
        training_args_kwargs["evaluation_strategy"] = training_config["eval_strategy"]
    training_args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    eval_results = generate_eval_results(
        model,
        tokenizer,
        valid_records,
        template,
        config,
        run_id,
        model_name,
    )
    write_json(run_dir / config["outputs"]["eval_results_file"], eval_results)
    write_json(run_dir / config["outputs"]["eval_summary_file"], eval_results["eval_summary"])

    if args.copy_config:
        shutil.copy2(resolve_path(args.config), run_dir / Path(args.config).name)

    print(f"run_dir={run_dir}")
    print(json.dumps(eval_results["eval_summary"], ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config_v1.json")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--eval_sample_size", type=int, default=None)
    parser.add_argument("--copy_config", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
