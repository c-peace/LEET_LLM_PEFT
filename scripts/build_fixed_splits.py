import argparse
import json
import random
from pathlib import Path


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


def stratified_test_indexes(records, stratify_key, test_size, seed):
    rng = random.Random(seed)
    groups = {}
    for index, record in enumerate(records):
        key = record["input"][stratify_key]
        groups.setdefault(key, []).append(index)

    for indexes in groups.values():
        rng.shuffle(indexes)

    total = len(records)
    exact_counts = {
        key: len(indexes) * test_size / total
        for key, indexes in groups.items()
    }
    test_counts = {}
    for key, count in exact_counts.items():
        if len(groups[key]) > 1:
            test_counts[key] = max(1, int(count))
        else:
            test_counts[key] = 0

    remaining = test_size - sum(test_counts.values())
    if remaining >= 0:
        remainders = sorted(
            groups,
            key=lambda key: (exact_counts[key] - int(exact_counts[key]), len(groups[key])),
            reverse=True,
        )
        for key in remainders:
            if remaining <= 0:
                break
            if test_counts[key] < len(groups[key]):
                test_counts[key] += 1
                remaining -= 1
    else:
        overage = abs(remaining)
        reducible = sorted(
            groups,
            key=lambda key: (test_counts[key], len(groups[key])),
            reverse=True,
        )
        for key in reducible:
            if overage <= 0:
                break
            if test_counts[key] > 1:
                test_counts[key] -= 1
                overage -= 1

    selected = []
    for key, indexes in groups.items():
        selected.extend(indexes[: test_counts[key]])

    rng.shuffle(selected)
    return set(selected)


def summarize(records):
    summary = {}
    for record in records:
        question_type = record["input"]["question_type"]
        summary[question_type] = summary.get(question_type, 0) + 1
    return dict(sorted(summary.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sft_dataset.json")
    parser.add_argument("--train-output", default="sft_train_v1.json")
    parser.add_argument("--test-output", default="test_set_v1.json")
    parser.add_argument("--manifest-output", default="dataset_split_v1_manifest.json")
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--stratify-by", default="question_type")
    args = parser.parse_args()

    records = load_json(resolve_path(args.input))
    if args.test_size <= 0 or args.test_size >= len(records):
        raise ValueError("test-size must be greater than 0 and smaller than dataset size")

    test_indexes = stratified_test_indexes(
        records,
        args.stratify_by,
        args.test_size,
        args.seed,
    )
    train_records = [
        record for index, record in enumerate(records)
        if index not in test_indexes
    ]
    test_records = [
        record for index, record in enumerate(records)
        if index in test_indexes
    ]

    manifest = {
        "version": "v1",
        "source": args.input,
        "seed": args.seed,
        "stratify_by": args.stratify_by,
        "total_records": len(records),
        "train_records": len(train_records),
        "test_records": len(test_records),
        "test_indexes": sorted(test_indexes),
        "train_type_counts": summarize(train_records),
        "test_type_counts": summarize(test_records),
    }

    write_json(resolve_path(args.train_output), train_records)
    write_json(resolve_path(args.test_output), test_records)
    write_json(resolve_path(args.manifest_output), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
