import argparse
import json
import re
from pathlib import Path

from huggingface_hub import HfApi, upload_file


MODEL_REPO_NAMES = {
    "Qwen/Qwen3.5-4B": "qwen35-4b",
    "google/gemma-4-E4B-it": "gemma4-e4b-it",
    "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct": "exaone35-24b-instruct",
}


def hf_safe_slug(value):
    slug = value.lower()
    slug = slug.replace("/", "-").replace(":", "-").replace(".", "")
    slug = re.sub(r"[^a-z0-9_-]+", "-", slug)
    slug = re.sub(r"[-_]+", "-", slug)
    return slug.strip("-")


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def latest_run_dir(runs_root):
    candidates = [path for path in runs_root.glob("*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no run directories found under: {runs_root}")
    return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]


def load_model_name(run_dir):
    eval_results_path = run_dir / "eval_results.json"
    if not eval_results_path.exists():
        raise FileNotFoundError(f"eval_results.json not found: {eval_results_path}")
    eval_results = json.loads(eval_results_path.read_text(encoding="utf-8"))
    return eval_results["model_name"]


def build_repo_name(project_name, model_name, experiment_version):
    model_repo_name = MODEL_REPO_NAMES.get(model_name, hf_safe_slug(model_name))
    return "-".join(
        [
            hf_safe_slug(project_name),
            hf_safe_slug(model_repo_name),
            hf_safe_slug(experiment_version),
        ]
    )


def write_experiment_note(
    run_dir,
    project_name,
    model_name,
    experiment_version,
    experiment_title,
    experiment_hypothesis,
):
    model_repo_name = MODEL_REPO_NAMES.get(model_name, hf_safe_slug(model_name))
    experiment_note = f"""# {project_name} {model_repo_name} {experiment_version}

## Experiment
- Project name: {project_name}
- Model name: {model_name}
- Model repo name: {model_repo_name}
- Version: {experiment_version}
- Title: {experiment_title}
- Run ID: {run_dir.name}

## Hypothesis
{experiment_hypothesis}

## Files
- Adapter files are stored at the repository root.
- Run artifacts are stored under `run_artifacts/{run_dir.name}/`.
- If present, test generation outputs are stored as `test_generation_results.json` and `test_generation_summary.json` in the run artifacts.
"""
    note_path = run_dir / "experiment_note.md"
    note_path.write_text(experiment_note, encoding="utf-8")
    return note_path


def upload_run(args):
    runs_root = Path(args.runs_root)
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir(runs_root)
    adapter_dir = run_dir / "adapter"

    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter folder not found: {adapter_dir}")

    model_name = load_model_name(run_dir)
    repo_name = build_repo_name(args.project_name, model_name, args.experiment_version)
    repo_id = f"{args.hf_username}/{repo_name}"

    write_experiment_note(
        run_dir=run_dir,
        project_name=args.project_name,
        model_name=model_name,
        experiment_version=args.experiment_version,
        experiment_title=args.experiment_title,
        experiment_hypothesis=args.experiment_hypothesis,
    )

    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=repo_id,
        repo_type="model",
    )

    uploaded_artifacts = []
    for artifact_path in sorted(run_dir.iterdir()):
        if not artifact_path.is_file():
            continue
        upload_file(
            path_or_fileobj=str(artifact_path),
            path_in_repo=f"run_artifacts/{run_dir.name}/{artifact_path.name}",
            repo_id=repo_id,
            repo_type="model",
        )
        uploaded_artifacts.append(artifact_path.name)

    print(f"uploaded adapter: https://huggingface.co/{repo_id}")
    print(f"uploaded run artifacts from: {run_dir}")
    print(f"uploaded artifact files: {uploaded_artifacts}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_username", required=True)
    parser.add_argument("--project_name", required=True)
    parser.add_argument("--experiment_version", required=True)
    parser.add_argument("--experiment_title", required=True)
    parser.add_argument("--experiment_hypothesis", required=True)
    parser.add_argument("--runs_root", default="/content/sft_outputs/runs")
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--private", type=str_to_bool, default=True)
    return parser.parse_args()


def main():
    upload_run(parse_args())


if __name__ == "__main__":
    main()
