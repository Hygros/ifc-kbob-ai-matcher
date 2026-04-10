"""
Für kleine Datensätze (bis ca. 5000 unique-Paare):

python Training/run_training_pipeline.py `
    --query-file Training/query_generation/generated_queries/generated_queries_without_exposure.txt `
    --expected-file Training/query_generation/generated_queries/mapping_generated_queries_without_exposure.txt `
  --base-model BAAI/bge-m3 `
  --output-dir Training/artifacts/models/bge-m3-finetuned-generated_queries_without_exposure `
  --deduplicate --max-per-positive 30 `
  --epochs 4

epochs 2:
     #Anzahl Trainingsdurchläufe über den Trainingssplit.
batch-size 8:
    Anzahl Trainingspaare pro Schritt; größer = schneller, aber mehr VRAM.
lr 2e-5:
    Learning Rate; bestimmt, wie stark Gewichte pro Update geändert werden.
max-length 512:
    Maximale Tokenlänge pro Text (Query/Positive); längere Texte werden abgeschnitten.
device auto:
    cuda wenn verfügbar, sonst cpu
deduplicate:
    Entfernt identische (query, positive)-Paare, damit doppelte Beispiele das Training nicht verzerren.
max-per-postive n:
    Cap pro unique Positive (z.B. 30-50), überzählige Paare zufällig entfernen.

Weitere (default):    
warmup-ratio 0.1:
    10% der Trainingsschritte werden als Warmup genutzt (LR steigt zunächst an, stabileres Training).
dev-ratio 0.1:
    10% der Paare gehen in den Dev-Split, 90 % in Train (für Monitoring/Eval während Training).
seed 42:
    Fixiert Zufall (Shuffle/Split) für besser reproduzierbare Runs.
fp16 (Default: aus):
    Mixed Precision ist nur aktiv, wenn Flag explizit gesetzt. Sonst läuft Training in voller Präzision.


usage: run_training_pipeline.py [-h] --query-file QUERY_FILE --expected-file EXPECTED_FILE [--base-model BASE_MODEL]
                                [--pairs-out PAIRS_OUT] [--output-dir OUTPUT_DIR] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--lr LR]
                                [--warmup-ratio WARMUP_RATIO] [--max-length MAX_LENGTH] [--dev-ratio DEV_RATIO] [--seed SEED]
                                [--device DEVICE] [--fp16] [--deduplicate]
run_training_pipeline.py: error: the following arguments are required: --query-file, --expected-file
"""




import argparse
import hashlib
import importlib.util
import json
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "Training"
VALIDATE_SCRIPT = TRAINING_DIR / "validate_training_data.py"
PREPARE_SCRIPT = TRAINING_DIR / "prepare_training_data.py"
TRAIN_SCRIPT = TRAINING_DIR / "train_bge_m3.py"
QA_PREFLIGHT_SCRIPT = TRAINING_DIR / "run_data_qa_preflight.py"


def run_command(command: list[str]) -> None:
    print(f"\n> {' '.join(command)}")
    result = subprocess.run(command, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Befehl fehlgeschlagen mit Exit-Code {result.returncode}: {' '.join(command)}")


def run_command_timed(command: list[str]) -> float:
    started = time.perf_counter()
    run_command(command)
    return time.perf_counter() - started


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def resolve_data_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def sha1_file(path: Path) -> str:
    if not path.is_file():
        return ""

    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_non_empty_lines(path: Path) -> int:
    if not path.is_file():
        return 0

    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def canonical_record_key(query: str, positive: str) -> str:
    key = f"{normalize_text(query)}|||{normalize_text(positive)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def collect_record_keys_from_jsonl(path: Path) -> list[str]:
    if not path.is_file():
        return []

    keys: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Ungueltiges JSON in {path} Zeile {line_no}: {exc}") from exc

            query = str(row.get("query", "")).strip()
            positive = str(row.get("positive", "")).strip()
            if not query or not positive:
                continue
            keys.append(canonical_record_key(query, positive))

    return keys


def record_keys_sha1(keys: list[str]) -> str:
    payload = "\n".join(keys).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def artifact_entry(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": "", "present": False, "sha1": "", "line_count": 0}

    resolved = resolve_data_path(path)
    return {
        "path": str(resolved),
        "present": resolved.is_file(),
        "sha1": sha1_file(resolved),
        "line_count": count_non_empty_lines(resolved),
    }


def input_entry(path: Path) -> dict[str, Any]:
    resolved = resolve_data_path(path)
    return {
        "path": str(resolved),
        "sha1": sha1_file(resolved),
        "line_count": count_non_empty_lines(resolved),
    }


def optional_input_entry(path_value: str) -> dict[str, Any]:
    raw = path_value.strip()
    if not raw:
        return {"path": "", "present": False, "sha1": "", "line_count": 0}

    resolved = resolve_data_path(Path(raw))
    return {
        "path": str(resolved),
        "present": resolved.is_file(),
        "sha1": sha1_file(resolved),
        "line_count": count_non_empty_lines(resolved),
    }


def optional_artifact_entry(path_value: Path | None) -> dict[str, Any]:
    if path_value is None:
        return {"path": "", "present": False, "sha1": "", "line_count": 0}
    return artifact_entry(path_value)


def resolve_optional_path(path_value: str) -> Path | None:
    raw = path_value.strip()
    if not raw:
        return None
    return resolve_data_path(Path(raw))


def compute_rule_traceability(
    *,
    args: argparse.Namespace,
    prefix_strategy: dict[str, Any],
    rule_policy_file: Path | None,
) -> dict[str, Any]:
    components: list[dict[str, str]] = []

    scalar_components = {
        "hard_negative_mode": str(args.hard_negative_mode),
        "hard_negative_selection": str(args.hard_negative_selection),
        "num_hard_negatives": str(args.num_hard_negatives),
        "model_selection_metric": str(args.model_selection_metric),
        "qa_fn_strict_stop_rate": str(args.qa_fn_strict_stop_rate),
        "qa_fn_cross_query_stop_rate": str(args.qa_fn_cross_query_stop_rate),
        "qa_fn_cross_query_scope": str(args.qa_fn_cross_query_scope),
        "qa_fn_cross_query_near_jaccard_threshold": str(args.qa_fn_cross_query_near_jaccard_threshold),
        "qa_fn_any_scope_stop_count": str(args.qa_fn_any_scope_stop_count),
        "qa_batch_query_family_stop_rate": str(args.qa_batch_query_family_stop_rate),
        "qa_multi_positive_retention_stop_rate": str(args.qa_multi_positive_retention_stop_rate),
        "prefix_mode": str(prefix_strategy.get("prefix_mode", "")),
        "legacy_prefix_experiment_active": str(prefix_strategy.get("legacy_prefix_experiment_active", False)),
        "legacy_query_prefix": str(prefix_strategy.get("legacy_query_prefix", "")),
    }

    for name, value in sorted(scalar_components.items()):
        components.append({"name": name, "kind": "scalar", "value": value})

    file_components = {
        "rule_policy_file": rule_policy_file,
    }
    for name, path in sorted(file_components.items()):
        if path is None:
            components.append({"name": name, "kind": "file", "path": "", "sha1": "", "present": "False"})
            continue
        components.append(
            {
                "name": name,
                "kind": "file",
                "path": str(path),
                "sha1": sha1_file(path),
                "present": str(path.is_file()),
            }
        )

    digest_payload_lines: list[str] = []
    for component in components:
        digest_payload_lines.append("|".join(f"{key}={component[key]}" for key in sorted(component.keys())))
    rule_hash = hashlib.sha1("\n".join(digest_payload_lines).encode("utf-8")).hexdigest()

    return {
        "rule_hash": rule_hash,
        "components": components,
    }


def safe_read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def split_record_keys_from_pairs(
    pairs_file: Path,
    dev_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    resolved = resolve_data_path(pairs_file)
    if not resolved.is_file():
        return [], []

    if not 0 <= dev_ratio < 1:
        raise ValueError("--dev-ratio muss im Bereich [0, 1) liegen.")

    records: list[tuple[str, str]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Ungueltiges JSON in {resolved} Zeile {line_no}: {exc}") from exc

            query = str(row.get("query", "")).strip()
            positive = str(row.get("positive", "")).strip()
            if not query or not positive:
                continue
            records.append((query, positive))

    if not records:
        return [], []

    if dev_ratio == 0:
        keys = [canonical_record_key(query, positive) for query, positive in records]
        return keys, []

    query_to_records: dict[str, list[tuple[str, str]]] = {}
    for query, positive in records:
        query_to_records.setdefault(query, []).append((query, positive))

    unique_queries = list(query_to_records.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_queries)

    dev_query_count = int(len(unique_queries) * dev_ratio)
    if dev_query_count == 0 and len(unique_queries) > 5:
        dev_query_count = 1

    dev_queries = set(unique_queries[:dev_query_count])

    train_keys: list[str] = []
    dev_keys: list[str] = []
    for query, positive in records:
        key = canonical_record_key(query, positive)
        if query in dev_queries:
            dev_keys.append(key)
        else:
            train_keys.append(key)

    return train_keys, dev_keys


def load_pairs_rows(path: Path) -> list[dict[str, Any]]:
    resolved = resolve_data_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Pairs-Datei nicht gefunden: {resolved}")

    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Ungueltiges JSON in {resolved} Zeile {line_no}: {exc}") from exc

            query = str(row.get("query", "")).strip()
            positive = str(row.get("positive", "")).strip()
            if not query or not positive:
                raise ValueError(
                    f"Ungueltiger Datensatz in {resolved} Zeile {line_no}: query/positive fehlt."
                )
            rows.append(dict(row))

    if not rows:
        raise ValueError(f"Pairs-Datei enthaelt keine Datensaetze: {resolved}")
    return rows


def split_row_indices_by_query(
    rows: list[dict[str, Any]],
    dev_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0 <= dev_ratio < 1:
        raise ValueError("--dev-ratio muss im Bereich [0, 1) liegen.")

    if dev_ratio == 0:
        return list(range(len(rows))), []

    query_to_indices: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        query = str(row.get("query", "")).strip()
        query_to_indices.setdefault(query, []).append(index)

    unique_queries = list(query_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_queries)

    dev_query_count = int(len(unique_queries) * dev_ratio)
    if dev_query_count == 0 and len(unique_queries) > 5:
        dev_query_count = 1

    dev_queries = set(unique_queries[:dev_query_count])

    train_indices: list[int] = []
    dev_indices: list[int] = []
    for index, row in enumerate(rows):
        query = str(row.get("query", "")).strip()
        if query in dev_queries:
            dev_indices.append(index)
        else:
            train_indices.append(index)

    if not train_indices:
        raise ValueError("Nach dem Split sind keine Trainingsdaten uebrig.")

    return train_indices, dev_indices


def stable_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        key = normalize_text(value)
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def row_query_positive_values(row: dict[str, Any]) -> list[str]:
    values = coerce_str_list(row.get("pos"))
    values.extend(coerce_str_list(row.get("query_positives")))

    positive = str(row.get("positive", "")).strip()
    if positive:
        values.append(positive)

    return stable_unique(values)


def clean_row_hard_negatives(
    row: dict[str, Any],
    query_positive_keys: set[str] | None = None,
) -> list[str]:
    raw = coerce_str_list(row.get("hard_negatives", []))
    raw.extend(coerce_str_list(row.get("neg", [])))

    positive_key = normalize_text(str(row.get("positive", "")))
    positive_union = query_positive_keys or set()
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in raw:
        candidate = str(value).strip()
        candidate_key = normalize_text(candidate)
        if (
            not candidate_key
            or candidate_key == positive_key
            or candidate_key in positive_union
            or candidate_key in seen
        ):
            continue
        seen.add(candidate_key)
        cleaned.append(candidate)
    return cleaned


def deterministic_choice_index(seed: int, record_identity: str, population_size: int, channel: str) -> int:
    if population_size <= 0:
        raise ValueError("population_size muss > 0 sein.")
    digest = hashlib.sha1(f"{seed}|{record_identity}|{channel}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % population_size


def write_rows_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    resolved = resolve_data_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_random_preselected_pairs(
    *,
    pairs_file: Path,
    output_file: Path,
    seed: int,
    dev_ratio: float,
    num_hard_negatives: int,
) -> dict[str, Any]:
    if num_hard_negatives <= 0:
        raise ValueError("num_hard_negatives muss > 0 sein.")

    rows = load_pairs_rows(pairs_file)
    train_indices, dev_indices = split_row_indices_by_query(rows=rows, dev_ratio=dev_ratio, seed=seed)

    query_positive_union: dict[str, set[str]] = defaultdict(set)
    for index in train_indices:
        row = rows[index]
        query = str(row.get("query", "")).strip()
        query_key = normalize_text(query)
        if not query_key:
            continue

        for query_positive in row_query_positive_values(row):
            query_positive_key = normalize_text(query_positive)
            if query_positive_key:
                query_positive_union[query_key].add(query_positive_key)

    train_positives = stable_unique([
        str(rows[index].get("positive", "")).strip()
        for index in train_indices
        if str(rows[index].get("positive", "")).strip()
    ])

    global_hard_pool = stable_unique([
        candidate
        for index in train_indices
        for candidate in clean_row_hard_negatives(
            rows[index],
            query_positive_keys=query_positive_union.get(normalize_text(str(rows[index].get("query", "")).strip()), set()),
        )
    ])

    preselected_count = 0
    preselected_from_hn = 0
    preselected_from_fallback_positive = 0
    preselected_from_fallback_pool = 0
    preselected_missing = 0
    preselected_count_per_slot = [0 for _ in range(num_hard_negatives)]
    preselected_counts_per_record: list[int] = []

    for index in train_indices:
        row = dict(rows[index])
        query = str(row.get("query", "")).strip()
        positive = str(row.get("positive", "")).strip()
        query_key = normalize_text(query)
        query_positive_keys = query_positive_union.get(query_key, set())
        record_identity = f"{canonical_record_key(query, positive)}|line:{index}"

        hard_negatives = clean_row_hard_negatives(row, query_positive_keys=query_positive_keys)
        selected_values: list[str] = []
        selected_keys: set[str] = set()

        for slot_index in range(num_hard_negatives):
            selected: str | None = None

            hn_candidates = [
                candidate
                for candidate in hard_negatives
                if normalize_text(candidate) not in selected_keys
            ]
            if hn_candidates:
                choice_index = deterministic_choice_index(
                    seed=seed,
                    record_identity=record_identity,
                    population_size=len(hn_candidates),
                    channel=f"record_hard_negatives_{slot_index}",
                )
                selected = hn_candidates[choice_index]
                preselected_from_hn += 1
            else:
                fallback_candidates = [
                    value
                    for value in train_positives
                    if normalize_text(value) not in query_positive_keys
                    and normalize_text(value) not in selected_keys
                ]
                if fallback_candidates:
                    choice_index = deterministic_choice_index(
                        seed=seed,
                        record_identity=record_identity,
                        population_size=len(fallback_candidates),
                        channel=f"fallback_positives_{slot_index}",
                    )
                    selected = fallback_candidates[choice_index]
                    preselected_from_fallback_positive += 1
                else:
                    pool_candidates = [
                        value
                        for value in global_hard_pool
                        if normalize_text(value) not in query_positive_keys
                        and normalize_text(value) not in selected_keys
                    ]
                    if pool_candidates:
                        choice_index = deterministic_choice_index(
                            seed=seed,
                            record_identity=record_identity,
                            population_size=len(pool_candidates),
                            channel=f"fallback_global_pool_{slot_index}",
                        )
                        selected = pool_candidates[choice_index]
                        preselected_from_fallback_pool += 1

            if selected:
                selected_key = normalize_text(selected)
                if selected_key:
                    selected_keys.add(selected_key)
                    selected_values.append(selected)
                    preselected_count_per_slot[slot_index] += 1
                    preselected_count += 1
            else:
                preselected_missing += 1

        if selected_values:
            row["preselected_hard_negatives"] = list(selected_values)
            row["preselected_hard_negative"] = selected_values[0]
        else:
            row.pop("preselected_hard_negatives", None)
            row.pop("preselected_hard_negative", None)

        preselected_counts_per_record.append(len(selected_values))

        rows[index] = row

    for index in dev_indices:
        row = dict(rows[index])
        row.pop("preselected_hard_negatives", None)
        row.pop("preselected_hard_negative", None)
        rows[index] = row

    write_rows_jsonl(output_file, rows)

    output_resolved = resolve_data_path(output_file)
    return {
        "pairs_file": str(resolve_data_path(pairs_file)),
        "output_file": str(output_resolved),
        "seed": int(seed),
        "dev_ratio": float(dev_ratio),
        "num_hard_negatives_requested": int(num_hard_negatives),
        "rows_total": len(rows),
        "rows_train": len(train_indices),
        "rows_dev": len(dev_indices),
        "preselected_count": preselected_count,
        "preselected_count_per_slot": preselected_count_per_slot,
        "preselected_median_per_record": median(preselected_counts_per_record),
        "preselected_max_per_record": float(max(preselected_counts_per_record)) if preselected_counts_per_record else 0.0,
        "preselected_missing": preselected_missing,
        "preselected_from_hard_negatives": preselected_from_hn,
        "preselected_from_fallback_positive": preselected_from_fallback_positive,
        "preselected_from_fallback_global_pool": preselected_from_fallback_pool,
        "train_positives_unique": len(train_positives),
        "global_hard_pool_unique": len(global_hard_pool),
        "output_sha1": sha1_file(output_resolved),
        "output_line_count": count_non_empty_lines(output_resolved),
    }


def median(values: list[int]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return float(sorted_values[mid])
    return float((sorted_values[mid - 1] + sorted_values[mid]) / 2)


def build_hn_query_stats_from_pairs(pairs_file: Path) -> dict[str, Any]:
    resolved_file = resolve_data_path(pairs_file)
    if not resolved_file.is_file():
        raise FileNotFoundError(f"Pairs-Datei nicht gefunden: {resolved_file}")

    query_to_hn: dict[str, set[str]] = defaultdict(set)
    rows_total = 0

    with resolved_file.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue

            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Ungueltiges JSON in Zeile {line_no}: {exc}") from exc

            query_key = normalize_text(str(row.get("query", "")))
            if not query_key:
                continue
            rows_total += 1
            query_to_hn.setdefault(query_key, set())

            for candidate in coerce_str_list(row.get("hard_negatives", [])) + coerce_str_list(row.get("neg", [])):
                candidate_key = normalize_text(candidate)
                if candidate_key:
                    query_to_hn[query_key].add(candidate_key)

    total_queries = len(query_to_hn)
    queries_with_hn = sum(1 for negatives in query_to_hn.values() if negatives)
    queries_without_hn = total_queries - queries_with_hn
    queries_without_hn_rate = (queries_without_hn / total_queries) if total_queries else 0.0

    hn_counts = [len(negatives) for negatives in query_to_hn.values() if negatives]

    return {
        "pairs_file": str(resolved_file),
        "rows_total": rows_total,
        "total_queries": total_queries,
        "queries_with_hn": queries_with_hn,
        "queries_without_hn": queries_without_hn,
        "queries_without_hn_rate": queries_without_hn_rate,
        "median_hn_per_query": median(hn_counts),
        "min_hn_per_query": float(min(hn_counts)) if hn_counts else 0.0,
        "max_hn_per_query": float(max(hn_counts)) if hn_counts else 0.0,
    }


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


def compute_strict_hn_viability(
    pairs_file: Path,
    dev_ratio: float,
    seed: int,
    hard_negative_selection: str,
    num_hard_negatives: int,
) -> dict[str, Any]:
    train_module = load_module(TRAIN_SCRIPT, "train_bge_m3_pipeline_viability")
    resolved_file = resolve_data_path(pairs_file)

    all_records = train_module.read_records(resolved_file)
    train_records, _ = train_module.split_records(all_records, dev_ratio=dev_ratio, seed=seed)

    train_queries_total = len({normalize_text(record.query) for record in train_records})
    train_queries_with_hn_in_records = len(
        {
            normalize_text(record.query)
            for record in train_records
            if getattr(record, "hard_negatives", ())
        }
    )

    try:
        examples, stats = train_module.build_train_examples(
            train_records=train_records,
            hard_negative_mode="strict",
            hard_negative_selection=hard_negative_selection,
            seed=seed,
            num_hard_negatives=num_hard_negatives,
        )
        queries_with_hn_in_trainer_input = len(
            {
                normalize_text(example.texts[0])
                for example in examples
                if len(getattr(example, "texts", [])) >= 3
            }
        )
        return {
            "strict_hn_usable": True,
            "strict_error": "",
            "strict_examples_total": len(examples),
            "strict_stats": stats,
            "train_queries_total": train_queries_total,
            "train_queries_with_hn_in_records": train_queries_with_hn_in_records,
            "train_queries_with_hn_in_trainer_input": queries_with_hn_in_trainer_input,
        }
    except ValueError as exc:
        return {
            "strict_hn_usable": False,
            "strict_error": str(exc),
            "strict_examples_total": 0,
            "strict_stats": {},
            "train_queries_total": train_queries_total,
            "train_queries_with_hn_in_records": train_queries_with_hn_in_records,
            "train_queries_with_hn_in_trainer_input": 0,
        }


def safe_slug(value: str, max_len: int = 24) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not slug:
        return "na"
    return slug[:max_len]


def build_run_id(args: argparse.Namespace) -> str:
    query_label = safe_slug(Path(args.query_file).stem or "query")
    expected_label = safe_slug(Path(args.expected_file).stem or "expected")
    model_label = safe_slug(args.base_model.split("/")[-1], max_len=18)
    dedup = "d1" if args.deduplicate else "d0"

    signature = (
        f"q={args.query_file}|e={args.expected_file}|m={args.base_model}|ep={args.epochs}|"
        f"bs={args.batch_size}|lr={args.lr}|mx={args.max_length}|dv={args.dev_ratio}|"
        f"sd={args.seed}|wu={args.warmup_ratio}|{dedup}|dev={args.device}|fp16={args.fp16}|"
        f"msm={args.model_selection_metric}|"
        f"mpp={args.max_per_positive}|hnf={args.hard_negatives_file}|"
        f"rpf={args.rule_policy_file}|"
        f"hnm={args.hard_negative_mode}|hns={args.hard_negative_selection}|nhn={args.num_hard_negatives}|"
        f"qafn={args.qa_fn_strict_stop_rate},{args.qa_fn_cross_query_stop_rate},{args.qa_fn_any_scope_stop_count}|"
        f"qafnscope={args.qa_fn_cross_query_scope}|"
        f"qafnnear={args.qa_fn_cross_query_near_jaccard_threshold}|"
        f"qafam={args.qa_batch_query_family_stop_rate}|qamp={args.qa_multi_positive_retention_stop_rate}|"
        f"qaab={args.qa_run_instruction_ab}|qalp={args.qa_legacy_query_prefix}"
    )
    hash8 = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:8]
    return (
        f"{query_label}-{expected_label}-{model_label}-"
        f"e{args.epochs}-b{args.batch_size}-lr{str(args.lr).replace('.', 'p')}-"
        f"d{str(args.dev_ratio).replace('.', 'p')}-s{args.seed}-{dedup}-{hash8}"
    )


def resolve_prefix_strategy(
    *,
    run_instruction_ab: bool,
    legacy_query_prefix: str,
) -> dict[str, Any]:
    legacy_prefix = legacy_query_prefix.strip()
    legacy_experiment_active = bool(run_instruction_ab)

    if legacy_experiment_active and not legacy_prefix:
        raise ValueError(
            "Legacy Prefix Experiment angefordert, aber --qa-legacy-query-prefix ist leer. "
            "Bitte Prefix explizit setzen oder --no-qa-run-instruction-ab verwenden."
        )

    if legacy_experiment_active:
        return {
            "prefix_mode": "legacy_prefix",
            "source_of_prefix_setting": "explicit_cli_opt_in",
            "dense_only_bge_m3_default_applied": False,
            "legacy_prefix_experiment_active": True,
            "legacy_query_prefix": legacy_prefix,
        }

    return {
        "prefix_mode": "no_prefix",
        "source_of_prefix_setting": "dense_only_bge_m3_default",
        "dense_only_bge_m3_default_applied": True,
        "legacy_prefix_experiment_active": False,
        "legacy_query_prefix": "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Orchestriert die Bi-Encoder-Trainingspipeline: "
            "validate -> prepare -> qa -> train."
        )
    )
    parser.add_argument("--query-file", required=True, help="Pfad zur Query-TXT.")
    parser.add_argument("--expected-file", required=True, help="Pfad zur Expected-TXT.")
    parser.add_argument("--base-model", default="BAAI/bge-m3", help="Startmodell für Fine-Tuning.")
    parser.add_argument(
        "--pairs-out",
        default="Training/artifacts/training_pairs.jsonl",
        help="Ausgabe-JSONL für Trainingspaare.",
    )
    parser.add_argument(
        "--output-dir",
        default="Training/artifacts/models/bge-m3-finetuned",
        help="Output-Verzeichnis des feinjustierten Modells.",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Epochen für Fine-Tuning.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch-Size.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning Rate.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup-Anteil.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximale Token-Länge.")
    parser.add_argument("--dev-ratio", type=float, default=0.1, help="Dev-Split-Anteil.")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed.")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
    parser.add_argument("--fp16", action="store_true", help="Mixed precision Training aktivieren.")
    parser.add_argument("--deduplicate", action="store_true", help="Identische query/positive-Paare entfernen.")
    parser.add_argument(
        "--max-per-positive",
        type=int,
        default=0,
        help="Maximale Anzahl Paare pro unique Positive (0 = unbegrenzt).",
    )
    parser.add_argument(
        "--hard-negatives-file",
        default="",
        help="Optionales JSONL mit query + hard_negatives (z. B. aus mine_hard_negatives.py).",
    )
    parser.add_argument(
        "--rule-policy-file",
        default="",
        help="Optionales JSON/CFG mit Query-/Rule-Policy fuer Hash-Traceability.",
    )

    parser.add_argument(
        "--hard-negative-mode",
        choices=["off", "fallback", "strict"],
        default="fallback",
        help="Weitergabe an train_bge_m3.py.",
    )
    parser.add_argument(
        "--hard-negative-selection",
        choices=["first", "random", "random_preselected"],
        default="random_preselected",
        help="Weitergabe an train_bge_m3.py.",
    )
    parser.add_argument(
        "--num-hard-negatives",
        type=int,
        default=1,
        help="Anzahl Hard-Negatives pro Record (K) fuer Multi-HN-Training.",
    )
    parser.add_argument(
        "--model-selection-metric",
        choices=["default", "hit5_mrr10"],
        default="default",
        help="Metrik fuer best-model Auswahl im Train-Skript.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Deterministische Run-ID (optional). Wenn leer, wird automatisch eine erzeugt.",
    )
    parser.add_argument(
        "--save-each-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Zusätzlich zum besten Modell einen Checkpoint je Epoche speichern.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        default="",
        help="Optionaler Ordner für Epochen-Checkpoints (Default: <output-dir>/epochs).",
    )
    parser.add_argument(
        "--qa-preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fuehrt einen maschinellen QA-Preflight vor dem Training aus.",
    )
    parser.add_argument(
        "--qa-report-dir",
        default="Training/outputs/qa",
        help="Output-Ordner fuer QA-Reports.",
    )
    parser.add_argument(
        "--qa-eval-query-file",
        default="",
        help="Optionales Eval-Query-TXT fuer Leakage-Checks Train/Dev/Eval.",
    )
    parser.add_argument(
        "--qa-eval-expected-file",
        default="",
        help="Optionales Eval-Expected-TXT fuer Leakage-Checks Train/Dev/Eval.",
    )
    parser.add_argument(
        "--qa-details-file",
        default="",
        help="Optionales details_*.csv fuer HN-Qualitaet und Rank/Score-Analysen.",
    )
    parser.add_argument(
        "--qa-run-instruction-ab",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fuehrt kleinen A/B-Test ohne Prefix vs Legacy-Prefix aus.",
    )
    parser.add_argument(
        "--qa-legacy-query-prefix",
        default="",
        help="Legacy Query Prefix fuer den QA A/B-Test (nur mit --qa-run-instruction-ab).",
    )
    parser.add_argument(
        "--qa-ab-max-cases",
        type=int,
        default=200,
        help="Maximale Anzahl Cases fuer den QA A/B-Test.",
    )
    parser.add_argument("--qa-fn-strict-stop-rate", type=float, default=0.01)
    parser.add_argument("--qa-fn-cross-query-stop-rate", type=float, default=0.01)
    parser.add_argument(
        "--qa-fn-cross-query-scope",
        choices=["off", "family", "global"],
        default="family",
    )
    parser.add_argument("--qa-fn-cross-query-near-jaccard-threshold", type=float, default=0.60)
    parser.add_argument("--qa-fn-any-scope-stop-count", type=int, default=0)
    parser.add_argument("--qa-batch-query-family-stop-rate", type=float, default=0.0)
    parser.add_argument("--qa-multi-positive-retention-stop-rate", type=float, default=1.0)
    parser.add_argument(
        "--qa-fail-on-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pipeline bricht bei STOP-Kriterien aus dem QA-Report ab.",
    )
    parser.add_argument(
        "--split-manifest-dir",
        default="Training/outputs/manifests",
        help="Output-Ordner fuer maschinenlesbare Split-Manifeste.",
    )
    parser.add_argument(
        "--stop-before-train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fuehrt Pipeline bis inkl. QA/Manifest aus und beendet vor dem Training.",
    )
    parser.add_argument(
        "--full-runtime-profile-out",
        default="",
        help="Optionales JSON-Ausgabefile fuer Voll-Laufzeitprofil.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_hard_negatives <= 0:
        raise ValueError("--num-hard-negatives muss > 0 sein.")

    prefix_strategy = resolve_prefix_strategy(
        run_instruction_ab=bool(args.qa_run_instruction_ab),
        legacy_query_prefix=args.qa_legacy_query_prefix,
    )

    default_rule_policy_candidate = PROJECT_ROOT / "Training" / "query_generation" / "query_generation_policy.json"
    rule_policy_file = resolve_optional_path(args.rule_policy_file)
    if rule_policy_file is None and default_rule_policy_candidate.is_file():
        rule_policy_file = default_rule_policy_candidate.resolve()

    rule_traceability = compute_rule_traceability(
        args=args,
        prefix_strategy=prefix_strategy,
        rule_policy_file=rule_policy_file,
    )
    rule_hash = str(rule_traceability["rule_hash"])
    print(f"Rule hash: {rule_hash}")

    if args.hard_negative_mode == "strict" and not args.hard_negatives_file.strip():
        raise ValueError(
            "--hard-negative-mode strict erfordert --hard-negatives-file. "
            "Ohne HN-Input ist strict technisch nicht nutzbar."
        )

    scripts_to_check = [VALIDATE_SCRIPT, PREPARE_SCRIPT, TRAIN_SCRIPT]
    if args.qa_preflight:
        scripts_to_check.append(QA_PREFLIGHT_SCRIPT)

    for script in scripts_to_check:
        if not script.is_file():
            raise FileNotFoundError(f"Skript nicht gefunden: {script}")

    query_file = resolve_data_path(Path(args.query_file))
    expected_file = resolve_data_path(Path(args.expected_file))
    pairs_out = resolve_data_path(Path(args.pairs_out))
    output_dir = resolve_data_path(Path(args.output_dir))

    resolved_run_id = args.run_id.strip() or build_run_id(args)
    print(f"Run-ID: {resolved_run_id}")

    split_manifest_dir = resolve_data_path(Path(args.split_manifest_dir))
    split_manifest_dir.mkdir(parents=True, exist_ok=True)
    split_manifest_file = split_manifest_dir / f"split_manifest_{resolved_run_id}.json"

    qa_dir = resolve_data_path(Path(args.qa_report_dir))
    qa_report_file = qa_dir / f"qa_report_{resolved_run_id}.json"
    qa_gate_file = qa_dir / f"qa_gate_{resolved_run_id}.csv"

    full_runtime_profile_path: Path | None = None
    if args.full_runtime_profile_out.strip():
        full_runtime_profile_path = resolve_data_path(Path(args.full_runtime_profile_out))

    stage_runtime_seconds: dict[str, float] = {
        "prepare": 0.0,
        "qa_preflight": 0.0,
        "preselect_stage": 0.0,
        "train_script_total": 0.0,
    }
    trainer_full_runtime_profile_file: Path | None = None

    random_preselected_pairs_file: Path | None = None
    random_preselected_report_file: Path | None = None

    manifest_eval_query_file = args.qa_eval_query_file.strip()
    manifest_eval_expected_file = args.qa_eval_expected_file.strip()

    run_command(
        [
            sys.executable,
            str(VALIDATE_SCRIPT),
            "--query-file",
            str(query_file),
            "--expected-file",
            str(expected_file),
        ]
    )

    prepare_command = [
        sys.executable,
        str(PREPARE_SCRIPT),
        "--query-file",
        str(query_file),
        "--expected-file",
        str(expected_file),
        "--out",
        str(pairs_out),
        "--seed",
        str(args.seed),
    ]
    if args.deduplicate:
        prepare_command.append("--deduplicate")
    if args.max_per_positive > 0:
        prepare_command.extend(["--max-per-positive", str(args.max_per_positive)])
    if args.hard_negatives_file.strip():
        prepare_command.extend(["--hard-negatives-file", args.hard_negatives_file])

    stage_runtime_seconds["prepare"] = run_command_timed(prepare_command)

    run_command([sys.executable, str(VALIDATE_SCRIPT), "--pairs-file", str(pairs_out)])

    train_pairs_file = pairs_out

    if args.hard_negative_selection == "random_preselected":
        preselect_started = time.perf_counter()
        phase12_dir = resolve_data_path(Path("Training/outputs/phase12"))
        phase12_dir.mkdir(parents=True, exist_ok=True)

        random_preselected_pairs_file = phase12_dir / f"random_preselected_pairs_{resolved_run_id}.jsonl"
        random_preselected_report_file = phase12_dir / f"random_preselected_pairs_{resolved_run_id}.json"

        preselected_summary = build_random_preselected_pairs(
            pairs_file=train_pairs_file,
            output_file=random_preselected_pairs_file,
            seed=args.seed,
            dev_ratio=args.dev_ratio,
            num_hard_negatives=args.num_hard_negatives,
        )

        random_preselected_report_file.write_text(
            json.dumps(preselected_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        run_command([sys.executable, str(VALIDATE_SCRIPT), "--pairs-file", str(random_preselected_pairs_file)])
        train_pairs_file = random_preselected_pairs_file
        print(f"Random-preselected pairs: {random_preselected_pairs_file}")
        print(f"Random-preselected report: {random_preselected_report_file}")
        stage_runtime_seconds["preselect_stage"] = time.perf_counter() - preselect_started

    final_hn_stats = build_hn_query_stats_from_pairs(train_pairs_file)
    strict_hn_viability = compute_strict_hn_viability(
        pairs_file=train_pairs_file,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
        hard_negative_selection=args.hard_negative_selection,
        num_hard_negatives=args.num_hard_negatives,
    )

    hn_report_dir = Path(args.qa_report_dir)
    hn_report_file = hn_report_dir / f"hn_viability_{resolved_run_id}.json"
    hn_report_file = resolve_data_path(hn_report_file)
    hn_report_file.parent.mkdir(parents=True, exist_ok=True)

    hn_report_payload = {
        "run_id": resolved_run_id,
        "rule_hash": rule_hash,
        "hard_negative_mode": args.hard_negative_mode,
        "hard_negative_selection": args.hard_negative_selection,
        "num_hard_negatives": int(args.num_hard_negatives),
        "model_selection_metric": args.model_selection_metric,
        "hard_negatives_file": args.hard_negatives_file,
        "final_train_pairs_file": str(resolve_data_path(train_pairs_file)),
        "final_hn_stats": final_hn_stats,
        "strict_hn_viability": strict_hn_viability,
        "prefix_mode": prefix_strategy["prefix_mode"],
        "source_of_prefix_setting": prefix_strategy["source_of_prefix_setting"],
        "dense_only_bge_m3_default_applied": prefix_strategy["dense_only_bge_m3_default_applied"],
        "legacy_prefix_experiment_active": prefix_strategy["legacy_prefix_experiment_active"],
        "legacy_query_prefix": prefix_strategy["legacy_query_prefix"],
    }
    hn_report_file.write_text(json.dumps(hn_report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"HN-Viability report: {hn_report_file}")
    print(
        "Prefix strategy: "
        f"prefix_mode={prefix_strategy['prefix_mode']}, "
        f"source_of_prefix_setting={prefix_strategy['source_of_prefix_setting']}, "
        "dense_only_bge_m3_default_applied="
        f"{prefix_strategy['dense_only_bge_m3_default_applied']}, "
        "legacy_prefix_experiment_active="
        f"{prefix_strategy['legacy_prefix_experiment_active']}"
    )

    if args.hard_negative_mode == "strict" and not strict_hn_viability["strict_hn_usable"]:
        raise RuntimeError(
            "Strict hard-negative mode ist vor dem Training nicht nutzbar. "
            f"train_queries_total={strict_hn_viability['train_queries_total']}, "
            f"train_queries_with_hn_in_records={strict_hn_viability['train_queries_with_hn_in_records']}, "
            f"train_queries_with_hn_in_trainer_input={strict_hn_viability['train_queries_with_hn_in_trainer_input']}, "
            f"queries_without_hn_rate={final_hn_stats['queries_without_hn_rate']:.6f}. "
            f"Grund: {strict_hn_viability['strict_error']}"
        )

    if args.qa_preflight:
        qa_started = time.perf_counter()
        qa_command = [
            sys.executable,
            str(QA_PREFLIGHT_SCRIPT),
            "--pairs-file",
            str(train_pairs_file),
            "--prepare-pairs-file",
            str(pairs_out),
            "--query-file",
            str(query_file),
            "--expected-file",
            str(expected_file),
            "--base-model",
            args.base_model,
            "--report-dir",
            str(qa_dir),
            "--run-id",
            resolved_run_id,
            "--rule-hash",
            rule_hash,
            "--dev-ratio",
            str(args.dev_ratio),
            "--seed",
            str(args.seed),
            "--batch-size",
            str(args.batch_size),
            "--max-length",
            str(args.max_length),
            "--hard-negative-mode",
            args.hard_negative_mode,
            "--hard-negative-selection",
            args.hard_negative_selection,
            "--num-hard-negatives",
            str(args.num_hard_negatives),
            "--ab-max-cases",
            str(args.qa_ab_max_cases),
            "--fn-strict-stop-rate",
            str(args.qa_fn_strict_stop_rate),
            "--fn-cross-query-stop-rate",
            str(args.qa_fn_cross_query_stop_rate),
            "--fn-cross-query-scope",
            args.qa_fn_cross_query_scope,
            "--fn-cross-query-near-jaccard-threshold",
            str(args.qa_fn_cross_query_near_jaccard_threshold),
            "--fn-any-scope-stop-count",
            str(args.qa_fn_any_scope_stop_count),
            "--batch-query-family-stop-rate",
            str(args.qa_batch_query_family_stop_rate),
            "--multi-positive-retention-stop-rate",
            str(args.qa_multi_positive_retention_stop_rate),
        ]

        if prefix_strategy["legacy_prefix_experiment_active"]:
            qa_command.extend(["--legacy-query-prefix", prefix_strategy["legacy_query_prefix"]])

        if args.hard_negatives_file.strip():
            qa_command.extend(["--hard-negatives-file", args.hard_negatives_file])
        if args.qa_details_file.strip():
            qa_command.extend(["--details-file", args.qa_details_file])
        if args.qa_eval_query_file.strip() and args.qa_eval_expected_file.strip():
            qa_eval_query_path = resolve_data_path(Path(args.qa_eval_query_file.strip()))
            qa_eval_expected_path = resolve_data_path(Path(args.qa_eval_expected_file.strip()))
            qa_command.extend(["--eval-query-file", str(qa_eval_query_path)])
            qa_command.extend(["--eval-expected-file", str(qa_eval_expected_path)])

        if args.qa_run_instruction_ab:
            qa_command.append("--run-instruction-ab")
        else:
            qa_command.append("--no-run-instruction-ab")

        if args.qa_fail_on_stop:
            qa_command.append("--fail-on-stop")
        else:
            qa_command.append("--no-fail-on-stop")

        run_command(qa_command)
        stage_runtime_seconds["qa_preflight"] = time.perf_counter() - qa_started

    final_pairs_resolved = resolve_data_path(train_pairs_file)
    final_record_keys = collect_record_keys_from_jsonl(final_pairs_resolved)

    train_record_keys, dev_record_keys = split_record_keys_from_pairs(
        final_pairs_resolved,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )
    train_artifact_ref = f"derived_split::{final_pairs_resolved}#train"
    dev_artifact_ref = f"derived_split::{final_pairs_resolved}#dev"

    preselect_stage_outputs: list[str] = []
    if random_preselected_pairs_file is not None:
        preselect_stage_outputs = ["random_preselected_pairs", "random_preselected_report"]

    qa_stage_outputs: list[str] = []
    if args.qa_preflight:
        qa_stage_outputs = ["qa_report", "qa_gate_csv"]

    if random_preselected_pairs_file is not None:
        final_source_artifact = "random_preselected_pairs"
    else:
        final_source_artifact = "prepare_pairs"
    manifest_payload = {
        "schema_version": "phase2-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_id": resolved_run_id,
        "seed": args.seed,
        "prefix_mode": prefix_strategy["prefix_mode"],
        "source_of_prefix_setting": prefix_strategy["source_of_prefix_setting"],
        "dense_only_bge_m3_default_applied": prefix_strategy["dense_only_bge_m3_default_applied"],
        "legacy_prefix_experiment_active": prefix_strategy["legacy_prefix_experiment_active"],
        "inputs": {
            "query_file": input_entry(query_file),
            "expected_file": input_entry(expected_file),
            "hard_negatives_file": optional_input_entry(args.hard_negatives_file),
            "rule_policy_file": optional_artifact_entry(rule_policy_file),
            "eval_query_file": optional_input_entry(manifest_eval_query_file),
            "eval_expected_file": optional_input_entry(manifest_eval_expected_file),
        },
        "execution_flags": {
            "qa_preflight": bool(args.qa_preflight),
            "stop_before_train": bool(args.stop_before_train),
            "hard_negative_mode": args.hard_negative_mode,
            "hard_negative_selection": args.hard_negative_selection,
            "num_hard_negatives": int(args.num_hard_negatives),
            "model_selection_metric": args.model_selection_metric,
            "dev_ratio": args.dev_ratio,
        },
        "rule_traceability": rule_traceability,
        "filter_rules": {
            "prepare": {
                "deduplicate": bool(args.deduplicate),
                "max_per_positive": int(args.max_per_positive),
            },
            "qa": {
                "enabled": bool(args.qa_preflight),
                "run_instruction_ab": bool(args.qa_run_instruction_ab),
            },
        },
        "artifacts": {
            "prepare_pairs": artifact_entry(pairs_out),
            "random_preselected_pairs": artifact_entry(random_preselected_pairs_file),
            "random_preselected_report": artifact_entry(random_preselected_report_file),
            "qa_report": artifact_entry(qa_report_file),
            "qa_gate_csv": artifact_entry(qa_gate_file),
        },
        "lineage": {
            "prepare_stage": {
                "input_artifacts": ["query_file", "expected_file", "hard_negatives_file"],
                "output_artifacts": ["prepare_pairs"],
            },
            "preselect_stage": {
                "input_artifacts": ["prepare_pairs"],
                "output_artifacts": preselect_stage_outputs,
            },
            "final_stage": {
                "input_artifacts": [final_source_artifact],
                "output_artifacts": ["final_merged", "train", "dev"],
            },
            "qa_stage": {
                "input_artifacts": [final_source_artifact, "query_file", "expected_file"],
                "output_artifacts": qa_stage_outputs,
            },
        },
        "splits": {
            "final_merged": {
                "artifact": str(final_pairs_resolved),
                "row_count": len(final_record_keys),
                "record_keys": final_record_keys,
                "record_keys_sha1": record_keys_sha1(final_record_keys),
            },
            "train": {
                "artifact": train_artifact_ref,
                "row_count": len(train_record_keys),
                "record_keys": train_record_keys,
                "record_keys_sha1": record_keys_sha1(train_record_keys),
            },
            "dev": {
                "artifact": dev_artifact_ref,
                "row_count": len(dev_record_keys),
                "record_keys": dev_record_keys,
                "record_keys_sha1": record_keys_sha1(dev_record_keys),
            },
        },
    }

    split_manifest_file.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Split manifest: {split_manifest_file}")

    def write_full_runtime_profile() -> None:
        if full_runtime_profile_path is None:
            return

        trainer_runtime_payload = (
            safe_read_json(trainer_full_runtime_profile_file)
            if trainer_full_runtime_profile_file is not None
            else {}
        )
        trainer_blocks_raw = trainer_runtime_payload.get("runtime_blocks_seconds", {})
        trainer_blocks = trainer_blocks_raw if isinstance(trainer_blocks_raw, dict) else {}

        runtime_blocks_seconds = {
            "prepare": float(stage_runtime_seconds.get("prepare", 0.0)),
            "qa_preflight": float(stage_runtime_seconds.get("qa_preflight", 0.0)),
            "preselect_stage": float(stage_runtime_seconds.get("preselect_stage", 0.0)),
            "train_setup": float(trainer_blocks.get("train_setup", 0.0)),
            "dataloader_batch_collation": float(trainer_blocks.get("dataloader_batch_collation", 0.0)),
            "tokenization_feature_prep": float(trainer_blocks.get("tokenization_feature_prep", 0.0)),
            "forward_backward": float(trainer_blocks.get("forward_backward", 0.0)),
            "epoch_evaluation": float(trainer_blocks.get("epoch_evaluation", 0.0)),
            "checkpoint_saving": float(trainer_blocks.get("checkpoint_saving", 0.0)),
            "final_save_post_processing": float(trainer_blocks.get("final_save_post_processing", 0.0)),
        }

        payload = {
            "profile_name": "full_run_runtime_profile",
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "run_id": resolved_run_id,
            "hard_negative_mode": args.hard_negative_mode,
            "hard_negative_selection": args.hard_negative_selection,
            "num_hard_negatives": int(args.num_hard_negatives),
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "runtime_blocks_seconds": runtime_blocks_seconds,
            "pipeline_blocks_seconds": {
                "prepare": float(stage_runtime_seconds.get("prepare", 0.0)),
                "qa_preflight": float(stage_runtime_seconds.get("qa_preflight", 0.0)),
                "preselect_stage": float(stage_runtime_seconds.get("preselect_stage", 0.0)),
                "train_script_total": float(stage_runtime_seconds.get("train_script_total", 0.0)),
            },
            "trainer_blocks_seconds": trainer_blocks,
            "artifacts": {
                "split_manifest": str(split_manifest_file),
                "qa_report": str(qa_report_file),
                "qa_gate_csv": str(qa_gate_file),
                "trainer_runtime_profile": (
                    str(trainer_full_runtime_profile_file) if trainer_full_runtime_profile_file is not None else ""
                ),
            },
        }

        full_runtime_profile_path.parent.mkdir(parents=True, exist_ok=True)
        full_runtime_profile_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Full runtime profile: {full_runtime_profile_path}")

    if args.stop_before_train:
        print("Stop-before-train aktiv: Pipeline endet nach QA und Manifest-Schreiben.")
        write_full_runtime_profile()
        return

    train_command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--train-file",
        str(train_pairs_file),
        "--base-model",
        args.base_model,
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--max-length",
        str(args.max_length),
        "--dev-ratio",
        str(args.dev_ratio),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--run-id",
        resolved_run_id,
        "--rule-hash",
        rule_hash,
        "--hard-negative-mode",
        args.hard_negative_mode,
        "--hard-negative-selection",
        args.hard_negative_selection,
        "--num-hard-negatives",
        str(args.num_hard_negatives),
        "--model-selection-metric",
        args.model_selection_metric,
    ]
    if args.save_each_epoch:
        train_command.append("--save-each-epoch")
    else:
        train_command.append("--no-save-each-epoch")
    if args.checkpoints_dir:
        train_command.extend(["--checkpoints-dir", str(args.checkpoints_dir)])
    if args.fp16:
        train_command.append("--fp16")

    if full_runtime_profile_path is not None:
        trainer_full_runtime_profile_file = full_runtime_profile_path.parent / (
            f"{full_runtime_profile_path.stem}__trainer_{resolved_run_id}.json"
        )
        train_command.extend(["--full-runtime-profile-out", str(trainer_full_runtime_profile_file)])

    stage_runtime_seconds["train_script_total"] = run_command_timed(train_command)
    write_full_runtime_profile()
    print("\nTrainingspipeline abgeschlossen.")


if __name__ == "__main__":
    main()
