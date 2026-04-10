import argparse
import importlib.util
import json
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

warnings.warn(
    "[DEPRECATED] run_batch_collision_audit.py ist als deprecated markiert und wird in einem "
    "zukuenftigen Release entfernt. Batch-Sicherheit wird durch UniquePositiveBatchSampler "
    "in train_bge_m3.py gewaehrleistet.",
    DeprecationWarning,
    stacklevel=2,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT_PATH = PROJECT_ROOT / "Training" / "train_bge_m3.py"
MATERIAL_ID_FIELDS = [
    "material_id",
    "materialId",
    "materialID",
    "kbob_id",
    "kbobId",
    "material_code",
    "materialCode",
]


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_module(script_path: Path, module_name: str) -> Any:
    if not script_path.is_file():
        raise FileNotFoundError(f"Skript nicht gefunden: {script_path}")

    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    root_dir = str(PROJECT_ROOT)
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Modul konnte nicht geladen werden: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_material_ids(row: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for field in MATERIAL_ID_FIELDS:
        if field not in row:
            continue
        raw = row.get(field)
        if isinstance(raw, (str, int, float)):
            token = normalize_text(str(raw))
            if token:
                values.add(token)
        elif isinstance(raw, list):
            for item in raw:
                token = normalize_text(str(item))
                if token:
                    values.add(token)
    return values


def load_raw_row_lookup(pairs_file: Path) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with pairs_file.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Ungueltiges JSON in Zeile {line_no}: {exc}") from exc

            query = str(row.get("query", "")).strip()
            positive = str(row.get("positive", "")).strip()
            if not query or not positive:
                continue

            key = f"{normalize_text(query)}|||{normalize_text(positive)}"
            row_copy = dict(row)
            row_copy["_material_ids"] = sorted(parse_material_ids(row_copy))
            lookup[key].append(row_copy)
    return dict(lookup)


def tokenize_for_semantic(value: str) -> set[str]:
    normalized = normalize_text(value)
    if not normalized:
        return set()
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    if tokens:
        return tokens
    return {normalized}


def token_jaccard(left: str, right: str) -> float:
    left_tokens = tokenize_for_semantic(left)
    right_tokens = tokenize_for_semantic(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = left_tokens.intersection(right_tokens)
    union = left_tokens.union(right_tokens)
    return len(intersection) / len(union)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch collision audit with class-level breakdown.")
    parser.add_argument("--pairs-file", required=True, help="Finales Trainingsartefakt fuer Audit.")
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--batch-audit-epochs", type=int, default=3)
    parser.add_argument("--hard-negative-mode", choices=["off", "fallback", "strict"], default="strict")
    parser.add_argument("--hard-negative-selection", choices=["first", "random", "random_preselected"], default="random")
    parser.add_argument("--num-hard-negatives", type=int, default=1)
    parser.add_argument("--report-file", required=True, help="Output JSON report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_hard_negatives <= 0:
        raise ValueError("--num-hard-negatives muss > 0 sein.")

    pairs_file = resolve_path(args.pairs_file)
    report_file = resolve_path(args.report_file)

    train_module = load_module(TRAIN_SCRIPT_PATH, "train_bge_m3_batch_collision_audit")

    raw_lookup = load_raw_row_lookup(pairs_file)
    all_records = train_module.read_records(pairs_file)
    train_records, _ = train_module.split_records(all_records, dev_ratio=args.dev_ratio, seed=args.seed)

    train_examples, _ = train_module.build_train_examples(
        train_records=train_records,
        hard_negative_mode=args.hard_negative_mode,
        hard_negative_selection=args.hard_negative_selection,
        seed=args.seed,
        num_hard_negatives=args.num_hard_negatives,
    )

    query_to_positives: dict[str, set[str]] = defaultdict(set)
    for record in train_records:
        query_to_positives[normalize_text(record.query)].add(normalize_text(record.positive))

    try:
        sampler = train_module.UniquePositiveBatchSampler(
            train_examples,
            batch_size=args.batch_size,
            seed=args.seed,
            query_positive_union=query_to_positives,
        )
        sampler_query_union_aware = True
    except TypeError:
        sampler = train_module.UniquePositiveBatchSampler(
            train_examples,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        sampler_query_union_aware = False

    violations_by_class: Counter[str] = Counter()
    query_violation_counts: Counter[str] = Counter()

    total_violations = 0
    duplicate_samples_total = 0
    duplicate_positive_total = 0
    batches_checked = 0
    records_checked = 0

    example_batches: list[dict[str, Any]] = []
    example_conflicting_records: list[dict[str, Any]] = []

    for epoch_index in range(max(1, args.batch_audit_epochs)):
        for batch_idx, batch_indices in enumerate(sampler):
            batches_checked += 1
            records_checked += len(batch_indices)

            batch_examples = [train_examples[index] for index in batch_indices]
            batch_rows: list[dict[str, Any]] = []
            batch_keys: list[str] = []
            batch_positive_keys: list[str] = []

            for local_idx, example in enumerate(batch_examples):
                texts = list(getattr(example, "texts", []))
                query = texts[0] if len(texts) > 0 else ""
                positive = texts[1] if len(texts) > 1 else ""
                hard_negative = texts[2] if len(texts) > 2 else ""
                query_key = normalize_text(query)
                positive_key = normalize_text(positive)
                pair_key = f"{query_key}|||{positive_key}"
                material_ids = []
                if pair_key in raw_lookup and raw_lookup[pair_key]:
                    material_ids = list(raw_lookup[pair_key][0].get("_material_ids", []))

                row = {
                    "local_index": local_idx,
                    "global_index": batch_indices[local_idx],
                    "query": query,
                    "positive": positive,
                    "hard_negative": hard_negative,
                    "query_key": query_key,
                    "positive_key": positive_key,
                    "pair_key": pair_key,
                    "material_ids": material_ids,
                }
                batch_rows.append(row)
                batch_keys.append(pair_key)
                batch_positive_keys.append(positive_key)

            duplicate_sample_keys = [key for key, count in Counter(batch_keys).items() if count > 1]
            duplicate_positive_keys = [key for key, count in Counter(batch_positive_keys).items() if count > 1]

            duplicate_samples_total += len(batch_keys) - len(set(batch_keys))
            duplicate_positive_total += len(batch_positive_keys) - len(set(batch_positive_keys))

            batch_violations: list[dict[str, Any]] = []

            for left in batch_rows:
                left_query_key = left["query_key"]
                known_positives = query_to_positives.get(left_query_key, set())
                for right in batch_rows:
                    if left["local_index"] == right["local_index"]:
                        continue

                    if right["positive_key"] not in known_positives:
                        continue

                    classes: set[str] = {"query_level_positive_union_conflict"}

                    if left["query_key"] == right["query_key"]:
                        classes.add("same_query")

                    if left["positive_key"] == right["positive_key"]:
                        classes.add("same_positive_canonical_form")

                    if set(left["material_ids"]).intersection(set(right["material_ids"])):
                        classes.add("same_material_or_entity")

                    semantic_score = token_jaccard(left["positive"], right["positive"])
                    if semantic_score >= 0.8:
                        classes.add("semantic_equivalent_positive")

                    if duplicate_sample_keys or duplicate_positive_keys:
                        classes.add("duplicate_in_batch")

                    if "duplicate_in_batch" not in classes:
                        classes.add("sampler_induced_collision")

                    total_violations += 1
                    for cls in classes:
                        violations_by_class[cls] += 1

                    query_violation_counts[left["query"]] += 1

                    violation_payload = {
                        "epoch": epoch_index,
                        "batch_index": batch_idx,
                        "left": {
                            "query": left["query"],
                            "positive": left["positive"],
                            "material_ids": left["material_ids"],
                            "global_index": left["global_index"],
                        },
                        "right": {
                            "query": right["query"],
                            "positive": right["positive"],
                            "material_ids": right["material_ids"],
                            "global_index": right["global_index"],
                        },
                        "classes": sorted(classes),
                        "semantic_score": semantic_score,
                    }
                    batch_violations.append(violation_payload)

                    if len(example_conflicting_records) < 120:
                        example_conflicting_records.append(violation_payload)

            if batch_violations and len(example_batches) < 25:
                example_batches.append(
                    {
                        "epoch": epoch_index,
                        "batch_index": batch_idx,
                        "batch_size": len(batch_rows),
                        "duplicate_sample_keys": duplicate_sample_keys,
                        "duplicate_positive_keys": duplicate_positive_keys,
                        "violations_in_batch": len(batch_violations),
                        "rows": [
                            {
                                "global_index": row["global_index"],
                                "query": row["query"],
                                "positive": row["positive"],
                                "material_ids": row["material_ids"],
                            }
                            for row in batch_rows
                        ],
                    }
                )

    example_queries = [
        {"query": query, "violations": count}
        for query, count in query_violation_counts.most_common(25)
    ]

    report = {
        "meta": {
            "pairs_file": str(pairs_file),
            "seed": args.seed,
            "sampler_mode": "UniquePositiveBatchSampler",
            "sampler_query_positive_union_aware": sampler_query_union_aware,
            "batch_size": args.batch_size,
            "batch_audit_epochs": max(1, args.batch_audit_epochs),
            "hard_negative_mode": args.hard_negative_mode,
            "hard_negative_selection": args.hard_negative_selection,
        },
        "summary": {
            "total_violations": total_violations,
            "violations_by_class": dict(violations_by_class),
            "batches_checked": batches_checked,
            "records_checked": records_checked,
            "duplicate_samples_total": duplicate_samples_total,
            "duplicate_positive_in_batch_total": duplicate_positive_total,
            "train_records_total": len(train_records),
            "train_examples_total": len(train_examples),
            "train_unique_queries": len({normalize_text(record.query) for record in train_records}),
        },
        "example_batches": example_batches,
        "example_queries": example_queries,
        "example_conflicting_records": example_conflicting_records,
    }

    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Batch collision report: {report_file}")
    print(f"Total violations: {total_violations}")


if __name__ == "__main__":
    main()
