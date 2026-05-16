import argparse
import json
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


FAILURE_PLAN = [
    ("not_grounded", 35),
    ("multiple_correct", 30),
    ("wrong_answer_index", 20),
    ("question_type_mismatch", 10),
    ("format_or_duplicate", 5),
]


DPO_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "rejected": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "choices": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 5,
                    "maxItems": 5,
                },
                "answer_index": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["choices", "answer_index"],
        },
        "failure_types": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
    },
    "required": ["rejected", "failure_types"],
}


FAILURE_GUIDES = {
    "not_grounded": "Make one or more choices plausible but unsupported by the passage.",
    "multiple_correct": "Make at least two choices reasonably correct, breaking the single-answer condition.",
    "wrong_answer_index": "Keep the choices mostly usable, but set answer_index to a non-answer choice.",
    "question_type_mismatch": "Make choices superficially related but mismatched to the asked question type.",
    "format_or_duplicate": "Keep JSON valid, but include duplicate or over-repetitive choices.",
}


PASSAGE_REQUIRED_FAILURE_TYPES = {"not_grounded", "multiple_correct"}


GENERATOR_PROMPT_TEMPLATE = """Create one plausible but flawed rejected output for Korean LEET DPO data.
Return only JSON matching the schema. Do not copy the chosen output.

Failure target: {failure_type}
Guide: {failure_guide}

Constraints:
- rejected.choices: exactly 5 Korean strings.
- rejected.answer_index: integer 1..5.
- The rejected output should be worse than chosen, not random garbage.
- Use only schema fields.

[Instruction]
{instruction}

[Input]
question_type: {question_type}
question: {question}
{passage_block}

[Chosen/gold output]
answer_index: {chosen_answer_index}
choices:
{chosen_choices}
"""


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


def format_choices(choices):
    return "\n".join(f"{index}. {choice}" for index, choice in enumerate(choices, start=1))


def extract_json_object(text):
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("JSON object not found")
    return json.loads(text[start : end + 1])


def weighted_failure_types(total, seed):
    weighted = []
    for failure_type, weight in FAILURE_PLAN:
        weighted.extend([failure_type] * weight)
    rng = random.Random(seed)
    result = []
    for _ in range(total):
        result.append(rng.choice(weighted))
    return result


def should_include_passage(failure_type, keep_passage):
    return keep_passage or failure_type in PASSAGE_REQUIRED_FAILURE_TYPES


def build_prompt(record, failure_type, keep_passage=False):
    output = record["output"]
    item_input = record["input"]
    passage_block = ""
    if should_include_passage(failure_type, keep_passage):
        passage_block = f"\npassage:\n{item_input['passage']}\n"
    return GENERATOR_PROMPT_TEMPLATE.format(
        failure_type=failure_type,
        failure_guide=FAILURE_GUIDES[failure_type],
        instruction=record["instruction"],
        question_type=item_input["question_type"],
        question=item_input["question"],
        passage_block=passage_block,
        chosen_answer_index=output["answer_index"],
        chosen_choices=format_choices(output["choices"]),
    )


def normalize_generation(raw_result):
    rejected = raw_result["rejected"]
    choices = [str(choice).strip() for choice in rejected["choices"]]
    answer_index = int(rejected["answer_index"])
    if len(choices) != 5:
        raise ValueError("rejected.choices must contain exactly 5 items")
    if not 1 <= answer_index <= 5:
        raise ValueError("rejected.answer_index must be from 1 to 5")

    failure_types = raw_result.get("failure_types", [])
    if not isinstance(failure_types, list):
        failure_types = [str(failure_types)]

    return {
        "rejected": {
            "choices": choices,
            "answer_index": answer_index,
        },
        "failure_types": [str(item).strip() for item in failure_types if str(item).strip()],
    }


class CodexExecGenerator:
    def __init__(self, codex_bin, model, profile, sandbox, timeout, extra_args):
        self.codex_bin = codex_bin
        self.model = model
        self.profile = profile
        self.sandbox = sandbox
        self.timeout = timeout
        self.extra_args = extra_args or []
        self.temp_dir = tempfile.TemporaryDirectory(prefix="dpo-codex-")
        self.schema_path = Path(self.temp_dir.name) / "dpo_schema.json"
        write_json(self.schema_path, DPO_OUTPUT_SCHEMA)

    def close(self):
        self.temp_dir.cleanup()

    def generate(self, prompt):
        output_path = Path(self.temp_dir.name) / f"dpo_output_{time.time_ns()}.json"
        command = [
            self.codex_bin,
            "exec",
            "--cd",
            str(ROOT),
            "--sandbox",
            self.sandbox,
            "--skip-git-repo-check",
            "--ephemeral",
            "--color",
            "never",
            "--output-schema",
            str(self.schema_path),
            "--output-last-message",
            str(output_path),
        ]
        if self.model:
            command.extend(["--model", self.model])
        if self.profile:
            command.extend(["--profile", self.profile])
        command.extend(self.extra_args)
        command.append("-")

        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=self.timeout,
            check=False,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"codex exec failed with exit code {completed.returncode}: {message}")
        if not output_path.exists():
            raise RuntimeError("codex exec did not write an output-last-message file")
        return normalize_generation(extract_json_object(output_path.read_text(encoding="utf-8")))


def make_pair(record, generated, source_path, index, requested_failure_type, model, passage_in_prompt):
    pair = {
        "instruction": record["instruction"],
        "input": record["input"],
        "chosen": record["output"],
        "rejected": generated["rejected"],
        "meta": {
            "source": source_path,
            "source_index": index,
            "rejected_generator": "codex exec",
            "generator_model": model,
            "requested_failure_type": requested_failure_type,
            "failure_types": generated["failure_types"],
            "passage_in_prompt": passage_in_prompt,
            "quality_tier": "unverified",
        },
    }
    if generated.get("reason"):
        pair["meta"]["reason"] = generated["reason"]
    return pair


def load_existing_candidates(path):
    if not path.exists():
        return []
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return payload


def existing_indexes(candidates):
    indexes = set()
    for item in candidates:
        meta = item.get("meta") or {}
        if "source_index" in meta:
            indexes.add(meta["source_index"])
    return indexes


def build_train_subset(candidates, target_size, seed):
    eligible = [item for item in candidates if item.get("meta", {}).get("quality_tier") != "exclude"]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    return eligible[: min(target_size, len(eligible))]


def parse_args():
    parser = argparse.ArgumentParser(description="Build DPO candidates with codex exec generated rejected outputs.")
    parser.add_argument("--source", default="sft_train_aug_v1.json")
    parser.add_argument("--output", default="dpo_candidates_v1.json")
    parser.add_argument("--train-output", default="dpo_train_1k_v1.json")
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--limit", type=int, help="Generate only N new candidates.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Regenerate even if source_index already exists.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument(
        "--model",
        default=os.environ.get("CODEX_DPO_MODEL") or os.environ.get("DPO_MODEL") or "gpt-5.4-mini",
    )
    parser.add_argument(
        "--keep-passage",
        action="store_true",
        help="Include passage for every failure type. By default passage is only included when needed.",
    )
    parser.add_argument("--profile")
    parser.add_argument("--sandbox", default="read-only", choices=["read-only", "workspace-write", "danger-full-access"])
    parser.add_argument("--codex-arg", action="append", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    source_path = resolve_path(args.source)
    output_path = resolve_path(args.output)
    train_output_path = resolve_path(args.train_output)

    records = load_json(source_path)
    candidates = [] if args.force else load_existing_candidates(output_path)
    done_indexes = set() if args.force else existing_indexes(candidates)
    failure_types = weighted_failure_types(len(records), args.seed)

    generated_count = 0
    skipped_existing = 0
    generator = CodexExecGenerator(
        args.codex_bin,
        args.model,
        args.profile,
        args.sandbox,
        args.timeout,
        args.codex_arg,
    )
    try:
        for index, record in enumerate(records):
            if index < args.offset:
                continue
            if args.limit is not None and generated_count >= args.limit:
                break
            if index in done_indexes:
                skipped_existing += 1
                continue

            requested_failure_type = failure_types[index]
            passage_in_prompt = should_include_passage(requested_failure_type, args.keep_passage)
            print(
                f"[dpo] source_index={index} failure_type={requested_failure_type} "
                f"passage_in_prompt={passage_in_prompt}"
            )
            prompt = build_prompt(record, requested_failure_type, args.keep_passage)
            last_error = None
            for attempt in range(args.retries + 1):
                try:
                    generated = generator.generate(prompt)
                    candidates.append(
                        make_pair(
                            record,
                            generated,
                            str(source_path.relative_to(ROOT)) if source_path.is_relative_to(ROOT) else str(source_path),
                            index,
                            requested_failure_type,
                            args.model,
                            passage_in_prompt,
                        )
                    )
                    generated_count += 1
                    done_indexes.add(index)
                    write_json(output_path, candidates)
                    break
                except (
                    KeyError,
                    ValueError,
                    json.JSONDecodeError,
                    subprocess.SubprocessError,
                    TimeoutError,
                    RuntimeError,
                ) as exc:
                    last_error = exc
                    if attempt < args.retries:
                        time.sleep(args.retry_sleep)
            else:
                print(f"[error] source_index={index} {last_error}")

            if args.sleep:
                time.sleep(args.sleep)
    finally:
        generator.close()

    train_subset = build_train_subset(candidates, args.train_size, args.seed)
    write_json(train_output_path, train_subset)
    print(
        json.dumps(
            {
                "source": str(source_path),
                "candidates": len(candidates),
                "generated_now": generated_count,
                "skipped_existing": skipped_existing,
                "model": args.model,
                "output": str(output_path),
                "train_output": str(train_output_path),
                "train_size": len(train_subset),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
