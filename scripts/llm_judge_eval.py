import argparse
import json
import os
import subprocess
import time
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


JUDGE_SYSTEM_PROMPT = """You are a strict Korean LEET multiple-choice question evaluator.
Evaluate whether the generated choices form a valid, usable question for the given passage and question.
Use only the passage as evidence. Do not reward plausible but unsupported claims.
Return only a JSON object."""


JUDGE_USER_TEMPLATE = """다음은 LEET 지문 기반 객관식 선지 생성 결과입니다.

[평가 목표]
생성된 5개 선지와 answer_index가 실제 문제로 사용할 수 있는지 평가하세요.

[입력]
question_type: {question_type}
question: {question}

passage:
{passage}

[gold reference]
gold_answer_index: {gold_answer_index}
gold_choices:
{gold_choices}

[model output]
answer_index: {answer_index}
choices:
{choices}

[평가 기준]
1. selected_answer_valid: answer_index가 가리키는 선지가 질문 기준으로 정답/부정답 역할을 올바르게 하는가?
2. distractors_valid: 나머지 선지들이 명확히 오답이며, 정답으로도 해석될 여지가 없는가?
3. grounded: 모든 선지가 지문 근거 안에서 판단 가능한가?
4. unambiguous: 정답이 하나로 명확한가?
5. choice_quality: 선지들이 너무 어색하거나, 사소한 말장난이거나, 서로 과도하게 반복되지 않는가?
6. usable: 실제 평가 문항으로 바로 사용할 수 있는가?

[출력 형식]
아래 JSON 스키마를 지키세요. 설명은 reason에 짧게 쓰세요.
{{
  "selected_answer_valid": true,
  "distractors_valid": true,
  "grounded": true,
  "unambiguous": true,
  "choice_quality": true,
  "usable": true,
  "score": 5,
  "reason": "짧은 한국어 근거",
  "problem_choice_indexes": []
}}

score는 1~5 정수입니다.
problem_choice_indexes는 문제가 있는 선지 번호를 1-based 배열로 쓰세요."""


JUDGE_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "selected_answer_valid": {"type": "boolean"},
        "distractors_valid": {"type": "boolean"},
        "grounded": {"type": "boolean"},
        "unambiguous": {"type": "boolean"},
        "choice_quality": {"type": "boolean"},
        "usable": {"type": "boolean"},
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "reason": {"type": "string"},
        "problem_choice_indexes": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1, "maximum": 5},
        },
    },
    "required": [
        "selected_answer_valid",
        "distractors_valid",
        "grounded",
        "unambiguous",
        "choice_quality",
        "usable",
        "score",
        "reason",
        "problem_choice_indexes",
    ],
}


SUMMARY_KEYS = [
    "selected_answer_valid",
    "distractors_valid",
    "grounded",
    "unambiguous",
    "choice_quality",
    "usable",
]


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
        raise ValueError("JSON object not found in judge response")
    return json.loads(text[start : end + 1])


def normalize_judge_result(raw_result):
    result = {}
    for key in SUMMARY_KEYS:
        result[key] = bool(raw_result.get(key, False))

    score = raw_result.get("score", 1)
    try:
        score = int(score)
    except (TypeError, ValueError):
        score = 1
    result["score"] = max(1, min(5, score))

    reason = raw_result.get("reason", "")
    result["reason"] = str(reason).strip()

    problem_indexes = raw_result.get("problem_choice_indexes", [])
    if not isinstance(problem_indexes, list):
        problem_indexes = []
    normalized_indexes = []
    for item in problem_indexes:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= index <= 5:
            normalized_indexes.append(index)
    result["problem_choice_indexes"] = sorted(set(normalized_indexes))
    return result


def make_automatic_failure(reason):
    return {
        "selected_answer_valid": False,
        "distractors_valid": False,
        "grounded": False,
        "unambiguous": False,
        "choice_quality": False,
        "usable": False,
        "score": 1,
        "reason": reason,
        "problem_choice_indexes": [],
        "judge_status": "skipped",
    }


def item_has_judgeable_output(item):
    output = item.get("model_output")
    if not isinstance(output, dict):
        return False
    choices = output.get("choices")
    answer_index = output.get("answer_index")
    return isinstance(choices, list) and len(choices) == 5 and isinstance(answer_index, int)


def build_judge_prompt(item):
    model_output = item["model_output"]
    gold_output = item.get("gold_output") or {}
    item_input = item.get("input") or {}
    user_prompt = JUDGE_USER_TEMPLATE.format(
        question_type=item_input.get("question_type", ""),
        question=item_input.get("question", ""),
        passage=item_input.get("passage", ""),
        gold_answer_index=gold_output.get("answer_index", ""),
        gold_choices=format_choices(gold_output.get("choices", [])),
        answer_index=model_output.get("answer_index", ""),
        choices=format_choices(model_output.get("choices", [])),
    )
    return f"{JUDGE_SYSTEM_PROMPT}\n\n{user_prompt}"


class CodexExecClient:
    def __init__(self, codex_bin, model, profile, sandbox, timeout, extra_args):
        self.codex_bin = codex_bin
        self.model = model
        self.profile = profile
        self.sandbox = sandbox
        self.timeout = timeout
        self.extra_args = extra_args or []
        self.temp_dir = tempfile.TemporaryDirectory(prefix="llm-judge-codex-")
        self.schema_path = Path(self.temp_dir.name) / "judge_schema.json"
        write_json(self.schema_path, JUDGE_OUTPUT_SCHEMA)

    def close(self):
        self.temp_dir.cleanup()

    def judge(self, prompt):
        output_path = Path(self.temp_dir.name) / f"judge_output_{time.time_ns()}.json"
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
        return output_path.read_text(encoding="utf-8")


def judge_item(client, item, retries, retry_sleep):
    if not item_has_judgeable_output(item):
        return make_automatic_failure("model_output이 없거나 choices/answer_index 형식이 유효하지 않음")

    prompt = build_judge_prompt(item)
    last_error = None
    for attempt in range(retries + 1):
        try:
            content = client.judge(prompt)
            result = normalize_judge_result(extract_json_object(content))
            result["judge_status"] = "ok"
            return result
        except (
            KeyError,
            ValueError,
            json.JSONDecodeError,
            subprocess.SubprocessError,
            TimeoutError,
            RuntimeError,
        ) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep)

    failure = make_automatic_failure(f"judge 호출 또는 응답 파싱 실패: {last_error}")
    failure["judge_status"] = "error"
    return failure


def summarize_judged_items(items):
    judged_items = [item for item in items if isinstance(item.get("llm_judge"), dict)]
    summary = {
        "total": len(items),
        "judged": len(judged_items),
        "judge_ok": sum(1 for item in judged_items if item["llm_judge"].get("judge_status") == "ok"),
        "judge_skipped": sum(1 for item in judged_items if item["llm_judge"].get("judge_status") == "skipped"),
        "judge_error": sum(1 for item in judged_items if item["llm_judge"].get("judge_status") == "error"),
    }
    for key in SUMMARY_KEYS:
        summary[key] = sum(1 for item in judged_items if item["llm_judge"].get(key) is True)
    if judged_items:
        summary["average_score"] = round(
            sum(float(item["llm_judge"].get("score", 0)) for item in judged_items) / len(judged_items),
            3,
        )
    else:
        summary["average_score"] = 0.0
    return summary


def find_eval_result_paths(args):
    if args.inputs:
        return [resolve_path(path) for path in args.inputs]
    experiments_dir = resolve_path(args.experiments_dir)
    return sorted(experiments_dir.glob(args.glob))


def output_path_for(input_path, args):
    if args.in_place:
        return input_path
    return input_path.with_name(args.output_name)


def should_skip_item(item, args):
    if args.force:
        return False
    return isinstance(item.get("llm_judge"), dict) and item["llm_judge"].get("judge_status") == "ok"


def evaluate_file(client, path, args):
    payload = load_json(path)
    items = payload.get("items", [])
    if args.limit is not None:
        item_indexes = range(min(args.limit, len(items)))
    else:
        item_indexes = range(len(items))

    processed = 0
    for index in item_indexes:
        item = items[index]
        if should_skip_item(item, args):
            continue
        item["llm_judge"] = judge_item(client, item, args.retries, args.retry_sleep)
        processed += 1
        if args.sleep:
            time.sleep(args.sleep)

    payload["llm_judge_summary"] = summarize_judged_items(items)
    output_path = output_path_for(path, args)
    write_json(output_path, payload)

    summary_path = output_path.with_name(args.summary_name)
    write_json(summary_path, payload["llm_judge_summary"])

    return {
        "input": str(path),
        "output": str(output_path),
        "summary": str(summary_path),
        "processed": processed,
        "llm_judge_summary": payload["llm_judge_summary"],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated choices with an LLM judge.")
    parser.add_argument("--inputs", nargs="*", help="Specific eval_results.json files to judge.")
    parser.add_argument("--experiments-dir", default="experiments", help="Experiments directory.")
    parser.add_argument("--glob", default="*/eval_results.json", help="Glob under experiments-dir.")
    parser.add_argument("--output-name", default="eval_results.judged.json")
    parser.add_argument("--summary-name", default="llm_judge_summary.json")
    parser.add_argument("--in-place", action="store_true", help="Overwrite input eval_results.json files.")
    parser.add_argument("--force", action="store_true", help="Re-judge items even if llm_judge already exists.")
    parser.add_argument("--limit", type=int, help="Judge only the first N items per file.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between judge calls.")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument(
        "--model",
        default=os.environ.get("CODEX_JUDGE_MODEL") or os.environ.get("JUDGE_MODEL") or "gpt-5.4",
        help="Model passed to `codex exec --model`.",
    )
    parser.add_argument("--profile", help="Optional profile passed to `codex exec --profile`.")
    parser.add_argument("--sandbox", default="read-only", choices=["read-only", "workspace-write", "danger-full-access"])
    parser.add_argument(
        "--codex-arg",
        action="append",
        default=[],
        help="Extra argument passed through to `codex exec`. Repeat for multiple args.",
    )
    parser.add_argument("--report-path", default="experiments/llm_judge_report.json")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = find_eval_result_paths(args)
    if not paths:
        raise SystemExit("평가할 eval_results.json 파일을 찾지 못했습니다.")

    client = CodexExecClient(args.codex_bin, args.model, args.profile, args.sandbox, args.timeout, args.codex_arg)
    try:
        reports = []
        for path in paths:
            print(f"[judge] {path}")
            reports.append(evaluate_file(client, path, args))
    finally:
        client.close()

    report_path = resolve_path(args.report_path)
    write_json(
        report_path,
        {
            "judge_backend": "codex exec",
            "codex_bin": args.codex_bin,
            "model": args.model,
            "profile": args.profile,
            "sandbox": args.sandbox,
            "files": reports,
        },
    )
    print(json.dumps({"report_path": str(report_path), "files": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
