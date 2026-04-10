"""
Machine-readable QA preflight for dense-only BGE-M3 training.

This script validates leakage, hard-negative quality, batch safety,
truncation, and instruction-prefix A/B behavior before training.
It is intended to run after prepare_training_data.py and before train_bge_m3.py.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

from text_normalization import query_family_key
from text_normalization import query_semantic_tokens


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT_PATH = PROJECT_ROOT / "Training" / "train_bge_m3.py"
EVAL_SCRIPT_PATH = PROJECT_ROOT / "Evaluation" / "evaluate_material_models.py"
DEFAULT_DB_CANDIDATES = [
    PROJECT_ROOT / "Oekobilanzdaten.sqlite3",
    PROJECT_ROOT / "Ökobilanzdaten.sqlite3",
    PROJECT_ROOT.parent / "Oekobilanzdaten.sqlite3",
    PROJECT_ROOT.parent / "Ökobilanzdaten.sqlite3",
]
DISCRIMINATIVE_TAIL_PATTERN = re.compile(
    r"Ifc[A-Za-z0-9_]+|C\d+/\d+|INSITU|PRECAST|NPK\s+[A-Z]|Beton|Stahl", re.IGNORECASE
)


@dataclass
class Criterion:
    criterion_id: str
    gate_type: str
    status: str
    value: float | int | str
    threshold: float | int | str
    comparator: str
    message: str


@dataclass
class OverlapResult:
    left_size: int
    right_size: int
    intersection_size: int
    jaccard: float
    overlap_rate_left: float
    overlap_rate_right: float


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(values)
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def parse_pipe_tokens(value: str) -> list[str]:
    return [token.strip() for token in str(value).split("|") if token.strip()]


def parse_expected_tokens_line(line: str) -> list[str]:
    raw = str(line).strip()
    if not raw:
        return []

    tokens: list[str] = []
    parts = re.split(r"[|;]", raw)
    for part in parts:
        token = part.strip()
        if not token:
            continue
        if "::" in token:
            token = token.rsplit("::", 1)[0].strip()
        if token:
            tokens.append(token)
    return tokens


def parse_bool(value: str) -> bool:
    return normalize_text(value) in {"1", "true", "yes", "y"}


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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


def load_non_empty_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")
    values: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                values.append(line)
    return values


def load_query_expected_pairs(query_file: Path, expected_file: Path) -> list[tuple[str, list[str]]]:
    queries = load_non_empty_lines(query_file)
    expected_lines = load_non_empty_lines(expected_file)
    if len(queries) != len(expected_lines):
        raise ValueError(
            "Anzahl Query-Zeilen passt nicht zur Anzahl Expected-Zeilen: "
            f"{len(queries)} != {len(expected_lines)}"
        )

    pairs: list[tuple[str, list[str]]] = []
    for index, query in enumerate(queries):
        tokens = parse_expected_tokens_line(expected_lines[index])
        pairs.append((query, tokens))
    return pairs


def build_expected_positive_map(pairs: list[tuple[str, list[str]]]) -> dict[str, set[str]]:
    by_query: dict[str, set[str]] = defaultdict(set)
    for query, tokens in pairs:
        query_key = normalize_text(query)
        for token in tokens:
            token_key = normalize_text(token)
            if token_key:
                by_query[query_key].add(token_key)
    return dict(by_query)


def load_hard_negatives_jsonl(path: Path) -> dict[str, dict[str, set[str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Hard-negatives Datei nicht gefunden: {path}")

    by_query: dict[str, dict[str, set[str]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Ungultiges JSON in Zeile {line_number}: {exc}") from exc

            query = str(row.get("query", "")).strip()
            if not query:
                continue
            query_key = normalize_text(query)

            positives_raw = row.get("positives", [])
            hard_negatives_raw = row.get("hard_negatives", [])
            if isinstance(positives_raw, str):
                positives_raw = [positives_raw]
            if isinstance(hard_negatives_raw, str):
                hard_negatives_raw = [hard_negatives_raw]
            if not isinstance(positives_raw, list):
                positives_raw = []
            if not isinstance(hard_negatives_raw, list):
                hard_negatives_raw = []

            payload = by_query.setdefault(
                query_key,
                {
                    "positives": set(),
                    "hard_negatives": set(),
                    "source_files": set(),
                },
            )

            for value in positives_raw:
                token = normalize_text(str(value))
                if token:
                    payload["positives"].add(token)

            for value in hard_negatives_raw:
                token = normalize_text(str(value))
                if token:
                    payload["hard_negatives"].add(token)

            source_details_file = str(row.get("source_details_file", "")).strip()
            if source_details_file:
                payload["source_files"].add(source_details_file)

    return by_query


def infer_details_file_from_hn(hn_file: Path) -> Path | None:
    if not hn_file.is_file():
        return None

    source_files: set[str] = set()
    with hn_file.open("r", encoding="utf-8") as handle:
        for _, line in zip(range(200), handle):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            source_name = str(row.get("source_details_file", "")).strip()
            if source_name:
                source_files.add(source_name)

    if len(source_files) != 1:
        return None

    source_name = next(iter(source_files))
    candidates = [
        PROJECT_ROOT / "Evaluation" / "outputs" / "results" / source_name,
        PROJECT_ROOT / "Training" / "outputs" / source_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_details_csv(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Details CSV nicht gefunden: {path}")

    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "query",
            "relevant_resolved",
            "predicted_top1_score",
            "expected_rank",
            "top1_correct",
            "top10_materials",
            "top10_scores",
        }
        missing = required_columns.difference(set(reader.fieldnames or []))
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Details CSV fehlt erforderliche Spalten: {missing_text}")

        for row in reader:
            query = str(row.get("query", "")).strip()
            if not query:
                continue
            query_key = normalize_text(query)

            relevant_resolved = parse_pipe_tokens(row.get("relevant_resolved", ""))
            top10_materials = parse_pipe_tokens(row.get("top10_materials", ""))
            raw_scores = parse_pipe_tokens(row.get("top10_scores", ""))
            top10_scores: list[float] = []
            for token in raw_scores:
                top10_scores.append(parse_float(token, default=0.0))

            row_payload = {
                "query": query,
                "relevant_resolved": relevant_resolved,
                "predicted_top1_score": parse_float(row.get("predicted_top1_score", "0"), default=0.0),
                "expected_rank": parse_int(row.get("expected_rank", "0"), default=0),
                "top1_correct": parse_bool(row.get("top1_correct", "False")),
                "top10_materials": top10_materials,
                "top10_scores": top10_scores,
            }
            by_query[query_key].append(row_payload)

    return dict(by_query)


def build_query_positive_union(
    expected_map: dict[str, set[str]],
    all_records: list[Any],
    hard_negatives_by_query: dict[str, dict[str, set[str]]],
    details_by_query: dict[str, list[dict[str, Any]]],
) -> dict[str, set[str]]:
    union_map: dict[str, set[str]] = defaultdict(set)

    for query_key, values in expected_map.items():
        union_map[query_key].update(values)

    for record in all_records:
        query_key = normalize_text(record.query)
        positive_key = normalize_text(record.positive)
        if positive_key:
            union_map[query_key].add(positive_key)

    for query_key, payload in hard_negatives_by_query.items():
        union_map[query_key].update(payload.get("positives", set()))

    for query_key, rows in details_by_query.items():
        for row in rows:
            for token in row.get("relevant_resolved", []):
                token_key = normalize_text(token)
                if token_key:
                    union_map[query_key].add(token_key)

    return dict(union_map)


def build_split_sets(records: list[Any]) -> dict[str, set[str]]:
    queries: set[str] = set()
    positives: set[str] = set()
    query_positive_pairs: set[str] = set()
    material_ids: set[str] = set()

    for record in records:
        query_key = normalize_text(record.query)
        positive_key = normalize_text(record.positive)
        if not query_key or not positive_key:
            continue
        queries.add(query_key)
        positives.add(positive_key)
        query_positive_pairs.add(f"{query_key}|||{positive_key}")
        material_ids.add(positive_key)

    return {
        "queries": queries,
        "positives": positives,
        "query_positive_pairs": query_positive_pairs,
        "material_ids": material_ids,
    }


def build_eval_sets_from_pairs(pairs: list[tuple[str, list[str]]]) -> dict[str, set[str]]:
    queries: set[str] = set()
    positives: set[str] = set()
    query_positive_pairs: set[str] = set()
    material_ids: set[str] = set()

    for query, tokens in pairs:
        query_key = normalize_text(query)
        if not query_key:
            continue
        queries.add(query_key)
        for token in tokens:
            token_key = normalize_text(token)
            if not token_key:
                continue
            positives.add(token_key)
            query_positive_pairs.add(f"{query_key}|||{token_key}")
            material_ids.add(token_key)

    return {
        "queries": queries,
        "positives": positives,
        "query_positive_pairs": query_positive_pairs,
        "material_ids": material_ids,
    }


def overlap_metrics(left: set[str], right: set[str]) -> OverlapResult:
    intersection = left.intersection(right)
    union = left.union(right)
    return OverlapResult(
        left_size=len(left),
        right_size=len(right),
        intersection_size=len(intersection),
        jaccard=safe_div(len(intersection), len(union)),
        overlap_rate_left=safe_div(len(intersection), len(left)),
        overlap_rate_right=safe_div(len(intersection), len(right)),
    )


def token_set_for_text_near_dup(text: str) -> set[str]:
    normalized = normalize_text(text)
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    if tokens:
        return tokens
    if normalized:
        return {normalized}
    return set()


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = left.intersection(right)
    union = left.union(right)
    return safe_div(len(intersection), len(union))


def text_near_duplicate_left_rate(
    left_values: Iterable[str],
    right_values: Iterable[str],
    threshold: float,
    example_limit: int = 10,
) -> dict[str, Any]:
    left_unique = sorted({normalize_text(value) for value in left_values if normalize_text(value)})
    right_unique = sorted({normalize_text(value) for value in right_values if normalize_text(value)})

    if not left_unique or not right_unique:
        return {
            "left_total": len(left_unique),
            "left_with_match": 0,
            "left_rate": 0.0,
            "examples": [],
        }

    right_tokens = [token_set_for_text_near_dup(value) for value in right_unique]
    inverted_index: dict[str, set[int]] = defaultdict(set)
    for index, token_set in enumerate(right_tokens):
        for token in token_set:
            inverted_index[token].add(index)

    left_with_match = 0
    examples: list[dict[str, Any]] = []

    for left_value in left_unique:
        left_tokens = token_set_for_text_near_dup(left_value)
        candidates: set[int] = set()
        for token in left_tokens:
            candidates.update(inverted_index.get(token, set()))

        best_score = 0.0
        best_match = ""
        for candidate_index in candidates:
            score = jaccard_similarity(left_tokens, right_tokens[candidate_index])
            if score > best_score:
                best_score = score
                best_match = right_unique[candidate_index]

        if best_score >= threshold:
            left_with_match += 1
            if len(examples) < example_limit:
                examples.append(
                    {
                        "left": left_value,
                        "right": best_match,
                        "jaccard": round(best_score, 6),
                    }
                )

    return {
        "left_total": len(left_unique),
        "left_with_match": left_with_match,
        "left_rate": safe_div(left_with_match, len(left_unique)),
        "examples": examples,
    }


def resolve_device(device: str) -> str:
    selected = device.strip().lower()
    if selected in {"cpu", "cuda"}:
        return selected
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_sentence_transformer(model_name: str, device: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def embedding_near_duplicate_left_rate(
    left_values: Iterable[str],
    right_values: Iterable[str],
    model: Any,
    threshold: float,
    chunk_size: int = 256,
    example_limit: int = 10,
) -> dict[str, Any]:
    import numpy as np

    left_unique = sorted({normalize_text(value) for value in left_values if normalize_text(value)})
    right_unique = sorted({normalize_text(value) for value in right_values if normalize_text(value)})

    if not left_unique or not right_unique:
        return {
            "left_total": len(left_unique),
            "left_with_match": 0,
            "left_rate": 0.0,
            "examples": [],
            "threshold": threshold,
        }

    left_emb = model.encode(left_unique, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    right_emb = model.encode(right_unique, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    left_with_match = 0
    max_scores: list[tuple[float, str, str]] = []

    for start in range(0, left_emb.shape[0], chunk_size):
        stop = min(start + chunk_size, left_emb.shape[0])
        sims = np.matmul(left_emb[start:stop], right_emb.T)
        best_indices = sims.argmax(axis=1)
        best_scores = sims.max(axis=1)

        for offset, score in enumerate(best_scores):
            left_text = left_unique[start + offset]
            right_text = right_unique[int(best_indices[offset])]
            if float(score) >= threshold:
                left_with_match += 1
            max_scores.append((float(score), left_text, right_text))

    max_scores.sort(key=lambda item: item[0], reverse=True)
    examples: list[dict[str, Any]] = []
    for score, left_text, right_text in max_scores[:example_limit]:
        examples.append(
            {
                "left": left_text,
                "right": right_text,
                "cosine": round(score, 6),
            }
        )

    return {
        "left_total": len(left_unique),
        "left_with_match": left_with_match,
        "left_rate": safe_div(left_with_match, len(left_unique)),
        "examples": examples,
        "threshold": threshold,
    }


def distribution_stats(values: Iterable[int | float]) -> dict[str, float]:
    numeric_values = [int(value) for value in values]
    if not numeric_values:
        return {"count": 0, "min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(numeric_values),
        "min": float(min(numeric_values)),
        "p50": percentile(numeric_values, 0.50),
        "p95": percentile(numeric_values, 0.95),
        "max": float(max(numeric_values)),
        "mean": safe_div(sum(numeric_values), len(numeric_values)),
    }


def hard_negative_quality(records: list[Any]) -> dict[str, Any]:
    by_query_all: dict[str, list[str]] = defaultdict(list)
    by_query_unique: dict[str, set[str]] = defaultdict(set)

    duplicate_within_record_total = 0
    negative_entries_total = 0

    for record in records:
        query_key = normalize_text(record.query)
        if not query_key:
            continue

        negatives = [normalize_text(value) for value in record.hard_negatives if normalize_text(value)]
        unique_negatives = set(negatives)

        duplicate_within_record_total += len(negatives) - len(unique_negatives)
        negative_entries_total += len(negatives)

        by_query_all[query_key].extend(negatives)
        by_query_unique[query_key].update(unique_negatives)

    unique_queries = sorted(by_query_unique.keys())
    negatives_per_query: list[int] = []
    queries_without_hn = 0
    cross_record_repetition_total = 0
    cross_record_negative_total = 0
    all_negative_total = 0
    all_negative_unique: set[str] = set()

    for query_key in unique_queries:
        values_all = by_query_all.get(query_key, [])
        values_unique = by_query_unique.get(query_key, set())

        negatives_per_query.append(len(values_unique))
        if not values_unique:
            queries_without_hn += 1
            continue

        cross_record_repetition_total += len(values_all) - len(set(values_all))
        cross_record_negative_total += len(values_all)
        all_negative_total += len(values_unique)
        all_negative_unique.update(values_unique)

    return {
        "unique_queries": len(unique_queries),
        "queries_without_hn": queries_without_hn,
        "queries_without_hn_rate": safe_div(queries_without_hn, len(unique_queries)),
        "negatives_per_query": distribution_stats(negatives_per_query),
        "duplicate_negatives_total": duplicate_within_record_total,
        "duplicate_negatives_rate": safe_div(duplicate_within_record_total, negative_entries_total),
        "cross_record_negative_repetition_total": cross_record_repetition_total,
        "cross_record_negative_repetition_rate": safe_div(cross_record_repetition_total, cross_record_negative_total),
        "unique_negative_diversity": safe_div(len(all_negative_unique), all_negative_total),
        "all_negative_total": all_negative_total,
        "all_negative_entries_total": negative_entries_total,
    }


def query_to_hn_map(records: list[Any]) -> dict[str, set[str]]:
    by_query: dict[str, set[str]] = defaultdict(set)
    for record in records:
        query_key = normalize_text(record.query)
        if not query_key:
            continue
        for negative in getattr(record, "hard_negatives", ()):
            negative_key = normalize_text(negative)
            if negative_key:
                by_query[query_key].add(negative_key)
        by_query.setdefault(query_key, set())
    return dict(by_query)


def hn_count_summary(query_to_hn: dict[str, set[str]]) -> dict[str, float]:
    values = [len(negatives) for negatives in query_to_hn.values() if negatives]
    if not values:
        return {"median_hn_per_query": 0.0, "min_hn_per_query": 0.0, "max_hn_per_query": 0.0}

    return {
        "median_hn_per_query": percentile(values, 0.5),
        "min_hn_per_query": float(min(values)),
        "max_hn_per_query": float(max(values)),
    }


def build_hn_viability_report(
    hard_negatives_by_query: dict[str, dict[str, set[str]]],
    prepare_records: list[Any] | None,
    final_records: list[Any],
    trainer_query_to_hn: dict[str, set[str]],
    strict_viability_error: str,
) -> dict[str, Any]:
    input_queries_with_hn = {
        query_key
        for query_key, payload in hard_negatives_by_query.items()
        if payload.get("hard_negatives")
    }

    final_query_to_hn = query_to_hn_map(final_records)
    final_queries = set(final_query_to_hn.keys())
    final_queries_with_hn = {query_key for query_key, negatives in final_query_to_hn.items() if negatives}

    if prepare_records is not None:
        prepare_query_to_hn = query_to_hn_map(prepare_records)
        prepare_queries_with_hn = {
            query_key for query_key, negatives in prepare_query_to_hn.items() if negatives
        }
    else:
        prepare_queries_with_hn = set(final_queries_with_hn)

    trainer_queries_with_hn = {
        query_key for query_key, negatives in trainer_query_to_hn.items() if negatives
    }

    total_queries = len(final_queries)
    queries_with_hn_in_final = len(final_queries_with_hn)
    queries_without_hn_rate = safe_div(total_queries - queries_with_hn_in_final, total_queries)

    count_summary = hn_count_summary(final_query_to_hn)

    return {
        "total_queries": total_queries,
        "queries_with_hn_input": len(input_queries_with_hn),
        "queries_with_hn_after_prepare": len(prepare_queries_with_hn),
        "queries_with_hn_in_final_train_artifact": queries_with_hn_in_final,
        "queries_with_hn_in_trainer_input": len(trainer_queries_with_hn),
        "queries_without_hn_rate": queries_without_hn_rate,
        "strict_hn_usable": not strict_viability_error and len(trainer_queries_with_hn) > 0,
        "strict_hn_reason_if_not_usable": strict_viability_error,
        "median_hn_per_query": count_summary["median_hn_per_query"],
        "min_hn_per_query": count_summary["min_hn_per_query"],
        "max_hn_per_query": count_summary["max_hn_per_query"],
        "queries_with_hn_lost_after_prepare": len(input_queries_with_hn - prepare_queries_with_hn),
        "queries_with_hn_lost_before_final_artifact": len(prepare_queries_with_hn - final_queries_with_hn),
        "queries_with_hn_lost_before_trainer_input": len(final_queries_with_hn - trainer_queries_with_hn),
    }


def build_query_positive_map(records: list[Any]) -> dict[str, set[str]]:
    by_query: dict[str, set[str]] = defaultdict(set)
    for record in records:
        query_key = normalize_text(record.query)
        positive_key = normalize_text(record.positive)
        if query_key and positive_key:
            by_query[query_key].add(positive_key)
    return dict(by_query)


def compute_multi_positive_retention(
    prepare_records: list[Any] | None,
    final_records: list[Any],
) -> dict[str, Any]:
    if prepare_records is None:
        return {
            "status": "skipped",
            "reason": "prepare_pairs_file not provided",
            "prepare_multi_positive_queries": 0,
            "final_multi_positive_queries": 0,
            "retained_multi_positive_queries": 0,
            "retention_rate": 1.0,
        }

    prepare_map = build_query_positive_map(prepare_records)
    final_map = build_query_positive_map(final_records)

    prepare_multi = {query for query, positives in prepare_map.items() if len(positives) > 1}
    final_multi = {query for query, positives in final_map.items() if len(positives) > 1}
    retained_multi = prepare_multi & final_multi

    return {
        "status": "ok",
        "prepare_multi_positive_queries": len(prepare_multi),
        "final_multi_positive_queries": len(final_multi),
        "retained_multi_positive_queries": len(retained_multi),
        "retention_rate": safe_div(len(retained_multi), len(prepare_multi)) if prepare_multi else 1.0,
    }


def cross_query_false_negative_match(
    *,
    query_key: str,
    positive_queries_for_negative: set[str],
    cross_query_scope: str,
    query_family_by_key: dict[str, str],
    query_tokens_by_key: dict[str, set[str]],
    cross_query_near_jaccard_threshold: float,
) -> bool:
    if not positive_queries_for_negative or query_key in positive_queries_for_negative:
        return False

    if cross_query_scope == "off":
        return False

    if cross_query_scope == "global":
        return True

    if cross_query_scope == "family":
        query_family = query_family_by_key.get(query_key, "")
        query_tokens = query_tokens_by_key.get(query_key, set())

        for other_query_key in positive_queries_for_negative:
            if other_query_key == query_key:
                continue

            same_family = bool(query_family) and query_family == query_family_by_key.get(other_query_key, "")
            if same_family:
                return True

            other_tokens = query_tokens_by_key.get(other_query_key, set())
            if jaccard_similarity(query_tokens, other_tokens) >= cross_query_near_jaccard_threshold:
                return True

        return False

    return True


def compute_false_negative_metrics(
    records: list[Any],
    query_positive_union: dict[str, set[str]],
    details_by_query: dict[str, list[dict[str, Any]]],
    cross_query_scope: str,
    cross_query_near_jaccard_threshold: float,
) -> dict[str, Any]:
    unique_query_negative_pairs: set[str] = set()
    strict_false_negative_pairs: set[str] = set()
    cross_query_false_negative_pairs: set[str] = set()
    strict_examples: list[dict[str, str]] = []
    cross_query_examples: list[dict[str, str]] = []

    positive_queries_by_material: dict[str, set[str]] = defaultdict(set)
    for query_key, positives in query_positive_union.items():
        for positive_key in positives:
            if positive_key:
                positive_queries_by_material[positive_key].add(query_key)

    query_text_by_key: dict[str, str] = {}
    for record in records:
        query_key = normalize_text(record.query)
        if query_key and query_key not in query_text_by_key:
            query_text_by_key[query_key] = str(record.query)
    for query_key in query_positive_union:
        query_text_by_key.setdefault(query_key, query_key)

    query_family_by_key = {
        query_key: query_family_key(query_text) for query_key, query_text in query_text_by_key.items()
    }
    query_tokens_by_key = {
        query_key: set(query_semantic_tokens(query_text)) for query_key, query_text in query_text_by_key.items()
    }

    rank_score_rows: list[dict[str, Any]] = []
    difficulty_counter: Counter[str] = Counter()

    for record in records:
        query_key = normalize_text(record.query)
        positives_for_query = query_positive_union.get(query_key, set())
        if not record.hard_negatives:
            continue

        details_rows = details_by_query.get(query_key, [])
        details_row = details_rows[0] if details_rows else None

        top10_materials_norm: list[str] = []
        top10_scores: list[float] = []
        if details_row is not None:
            top10_materials_norm = [normalize_text(value) for value in details_row.get("top10_materials", [])]
            top10_scores = [float(score) for score in details_row.get("top10_scores", [])]

        for negative in record.hard_negatives:
            negative_norm = normalize_text(negative)
            if not negative_norm:
                continue

            pair_key = f"{query_key}|||{negative_norm}"
            if pair_key in unique_query_negative_pairs:
                continue
            unique_query_negative_pairs.add(pair_key)

            is_strict_false_negative = negative_norm in positives_for_query
            if is_strict_false_negative:
                strict_false_negative_pairs.add(pair_key)
                if len(strict_examples) < 30:
                    strict_examples.append(
                        {
                            "query": record.query,
                            "hard_negative": negative,
                            "reason": "hard_negative in query positive union",
                        }
                    )

            positive_queries_for_negative = positive_queries_by_material.get(negative_norm, set())
            is_cross_query_false_negative = cross_query_false_negative_match(
                query_key=query_key,
                positive_queries_for_negative=positive_queries_for_negative,
                cross_query_scope=cross_query_scope,
                query_family_by_key=query_family_by_key,
                query_tokens_by_key=query_tokens_by_key,
                cross_query_near_jaccard_threshold=cross_query_near_jaccard_threshold,
            )
            if is_cross_query_false_negative:
                cross_query_false_negative_pairs.add(pair_key)
                if len(cross_query_examples) < 30:
                    reason = "hard_negative is positive for another query"
                    if cross_query_scope == "family":
                        reason = "hard_negative is positive for a related query (family/near-query)"
                    cross_query_examples.append(
                        {
                            "query": record.query,
                            "hard_negative": negative,
                            "reason": reason,
                        }
                    )

            hn_rank = None
            hn_score = None
            pos_score = None
            score_gap = None

            if top10_materials_norm:
                if negative_norm in top10_materials_norm:
                    hn_index = top10_materials_norm.index(negative_norm)
                    hn_rank = hn_index + 1
                    if hn_index < len(top10_scores):
                        hn_score = top10_scores[hn_index]

                positive_scores: list[float] = []
                for index, material_norm in enumerate(top10_materials_norm):
                    if material_norm in positives_for_query and index < len(top10_scores):
                        positive_scores.append(top10_scores[index])
                if positive_scores:
                    pos_score = max(positive_scores)

                if pos_score is not None and hn_score is not None:
                    score_gap = float(pos_score - hn_score)

            difficulty_band = classify_hn_difficulty(hn_rank, score_gap)
            difficulty_counter[difficulty_band] += 1

            rank_score_rows.append(
                {
                    "query": record.query,
                    "hard_negative": negative,
                    "hn_rank": hn_rank,
                    "hn_score": hn_score,
                    "pos_score": pos_score,
                    "score_gap": score_gap,
                    "difficulty_band": difficulty_band,
                }
            )

    denominator = len(unique_query_negative_pairs)
    strict_count = len(strict_false_negative_pairs)
    cross_query_count = len(cross_query_false_negative_pairs)
    any_scope_count = len(strict_false_negative_pairs | cross_query_false_negative_pairs)

    score_gap_values = [
        float(row["score_gap"]) for row in rank_score_rows if isinstance(row.get("score_gap"), float)
    ]

    return {
        "fn_strict_count": strict_count,
        "fn_strict_rate": safe_div(strict_count, denominator),
        "fn_cross_query_count": cross_query_count,
        "fn_cross_query_rate": safe_div(cross_query_count, denominator),
        "fn_any_scope_count": any_scope_count,
        "fn_any_scope_rate": safe_div(any_scope_count, denominator),
        "fn_denominator_pairs": denominator,
        "fn_cross_query_scope": cross_query_scope,
        "fn_cross_query_near_jaccard_threshold": cross_query_near_jaccard_threshold,
        "fn_examples": strict_examples,
        "fn_cross_query_examples": cross_query_examples,
        "score_gap_stats": distribution_stats([int(value * 1000000) for value in score_gap_values]) if score_gap_values else distribution_stats([]),
        "rank_score_rows": rank_score_rows,
        "difficulty_bands": dict(difficulty_counter),
    }


def classify_hn_difficulty(rank: int | None, score_gap: float | None) -> str:
    if rank is None:
        return "mixed"

    if score_gap is not None:
        if rank <= 3 and score_gap <= 0.02:
            return "very_hard"
        if rank <= 5 and score_gap <= 0.05:
            return "hard"
        if rank <= 10 and score_gap <= 0.10:
            return "medium"
        return "mixed"

    if rank <= 3:
        return "very_hard"
    if rank <= 5:
        return "hard"
    if rank <= 10:
        return "medium"
    return "mixed"


def resolve_db_path() -> Path | None:
    env_candidates = [
        os.environ.get("KBOB_DB_PATH", "").strip(),
        os.environ.get("ECOBILANZ_DB_PATH", "").strip(),
    ]
    for candidate in env_candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.is_file():
            return path

    for candidate in DEFAULT_DB_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def pick_column(columns: list[str], preferred: list[str]) -> str | None:
    normalized_map = {column.casefold(): column for column in columns}
    for option in preferred:
        key = option.casefold()
        if key in normalized_map:
            return normalized_map[key]
    return None


def load_material_annotations() -> dict[str, dict[str, str]]:
    db_path = resolve_db_path()
    if db_path is None:
        return {}

    with sqlite3.connect(str(db_path)) as connection:
        cursor = connection.cursor()
        cursor.execute("PRAGMA table_info(Oekobilanzdaten)")
        columns = [str(row[1]) for row in cursor.fetchall() if len(row) > 1]

        material_column = pick_column(columns, ["Material"])
        group_column = pick_column(columns, ["Materialgruppe", "MaterialGroup", "Group", "Gruppe"])
        category_column = pick_column(columns, ["KBOB_Kategorie", "KBOB Kategorie", "Kategorie", "Category"])

        if material_column is None:
            return {}

        selected_columns = [material_column]
        if group_column is not None:
            selected_columns.append(group_column)
        if category_column is not None and category_column not in selected_columns:
            selected_columns.append(category_column)

        quoted_columns = ", ".join(f"[{name}]" for name in selected_columns)
        query = f"SELECT {quoted_columns} FROM Oekobilanzdaten WHERE [{material_column}] IS NOT NULL"
        cursor.execute(query)

        annotations: dict[str, dict[str, str]] = {}
        for row in cursor.fetchall():
            if not row:
                continue
            material = normalize_text(str(row[0]))
            if not material:
                continue

            group_value = "UNKNOWN"
            category_value = "UNKNOWN"
            if group_column is not None and len(row) >= 2:
                group_value = str(row[1]).strip() or "UNKNOWN"
            if category_column is not None:
                index = 2 if group_column is not None else 1
                if len(row) > index:
                    category_value = str(row[index]).strip() or "UNKNOWN"

            annotations[material] = {
                "material_group": group_value,
                "kbob_category": category_value,
            }

    return annotations


def extract_query_type(query: str) -> str:
    token = str(query).strip().split(" ", 1)[0]
    if re.match(r"^Ifc[A-Za-z0-9_]+$", token):
        return token
    return "UNKNOWN"


def detect_language(query: str) -> str:
    text = str(query).strip()
    if not text:
        return "unknown"

    try:
        langdetect_module = __import__("langdetect")
        detect_func = getattr(langdetect_module, "detect", None)
        if callable(detect_func):
            return str(detect_func(text))
    except Exception:
        pass

    normalized = text.casefold()
    if any(char in normalized for char in ["ä", "ö", "ü", "ß"]):
        return "de"
    german_markers = ["beton", "stahl", "wand", "decke", "pfahl", "vorgefertigt"]
    english_markers = ["wall", "slab", "beam", "column", "precast", "concrete"]
    if any(marker in normalized for marker in german_markers):
        return "de"
    if any(marker in normalized for marker in english_markers):
        return "en"
    return "unknown"


def char_length_bin(length: int) -> str:
    if length <= 32:
        return "0-32"
    if length <= 64:
        return "33-64"
    if length <= 128:
        return "65-128"
    return "129+"


def token_length_bin(length: int) -> str:
    if length <= 8:
        return "0-8"
    if length <= 16:
        return "9-16"
    if length <= 32:
        return "17-32"
    if length <= 64:
        return "33-64"
    return "65+"


def build_balance_report(records: list[Any], tokenizer: Any | None) -> dict[str, Any]:
    material_annotations = load_material_annotations()

    query_type_counter: Counter[str] = Counter()
    language_counter: Counter[str] = Counter()
    query_char_bin_counter: Counter[str] = Counter()
    query_token_bin_counter: Counter[str] = Counter()
    material_group_counter: Counter[str] = Counter()
    kbob_category_counter: Counter[str] = Counter()

    for record in records:
        query = str(record.query)
        positive_norm = normalize_text(record.positive)

        query_type_counter[extract_query_type(query)] += 1
        language_counter[detect_language(query)] += 1

        query_char_len = len(query)
        query_char_bin_counter[char_length_bin(query_char_len)] += 1

        if tokenizer is not None:
            token_ids = tokenizer(query, add_special_tokens=True, truncation=False).get("input_ids", [])
            query_token_len = len(token_ids)
        else:
            query_token_len = len(query.split())
        query_token_bin_counter[token_length_bin(query_token_len)] += 1

        annotation = material_annotations.get(positive_norm, {})
        material_group_counter[annotation.get("material_group", "UNKNOWN")] += 1
        kbob_category_counter[annotation.get("kbob_category", "UNKNOWN")] += 1

    return {
        "query_type": dict(query_type_counter),
        "language": dict(language_counter),
        "query_char_length_bins": dict(query_char_bin_counter),
        "query_token_length_bins": dict(query_token_bin_counter),
        "material_group": dict(material_group_counter),
        "kbob_category": dict(kbob_category_counter),
    }


def batch_safety_audit(
    train_records: list[Any],
    train_examples: list[Any],
    sampler_cls: Any,
    batch_size: int,
    seed: int,
    epochs: int,
) -> dict[str, Any]:
    query_to_positives: dict[str, set[str]] = defaultdict(set)
    for record in train_records:
        query_to_positives[normalize_text(record.query)].add(normalize_text(record.positive))

    try:
        sampler = sampler_cls(
            train_examples,
            batch_size=batch_size,
            seed=seed,
            query_positive_union=query_to_positives,
        )
        sampler_query_union_aware = True
    except TypeError:
        sampler = sampler_cls(train_examples, batch_size=batch_size, seed=seed)
        sampler_query_union_aware = False

    duplicate_samples = 0
    implicit_positive_negative_violations = 0
    directed_comparisons = 0
    duplicate_positive_in_batch = 0
    query_family_collision_pairs = 0
    total_query_pairs = 0
    example_violations: list[dict[str, str]] = []
    query_family_examples: list[dict[str, str]] = []

    for _ in range(max(1, epochs)):
        for batch_indices in sampler:
            batch_examples = [train_examples[index] for index in batch_indices]
            batch_keys = ["|||".join(normalize_text(text) for text in example.texts) for example in batch_examples]
            duplicate_samples += len(batch_keys) - len(set(batch_keys))

            positive_keys = [normalize_text(example.texts[1]) for example in batch_examples]
            duplicate_positive_in_batch += len(positive_keys) - len(set(positive_keys))

            query_families = [query_family_key(example.texts[0]) for example in batch_examples]
            for i in range(len(batch_examples)):
                for j in range(i + 1, len(batch_examples)):
                    total_query_pairs += 1
                    left_family = query_families[i]
                    right_family = query_families[j]
                    if left_family and right_family and left_family == right_family:
                        query_family_collision_pairs += 1
                        if len(query_family_examples) < 30:
                            query_family_examples.append(
                                {
                                    "query_left": batch_examples[i].texts[0],
                                    "query_right": batch_examples[j].texts[0],
                                    "family_key": left_family,
                                }
                            )

            for i, left in enumerate(batch_examples):
                left_query = normalize_text(left.texts[0])
                known_positives = query_to_positives.get(left_query, set())
                for j, right in enumerate(batch_examples):
                    if i == j:
                        continue
                    directed_comparisons += 1
                    right_positive = normalize_text(right.texts[1])
                    if right_positive in known_positives:
                        implicit_positive_negative_violations += 1
                        if len(example_violations) < 30:
                            example_violations.append(
                                {
                                    "query": left.texts[0],
                                    "candidate_negative": right.texts[1],
                                    "reason": "known positive appears as implicit in-batch negative",
                                }
                            )

    return {
        "epochs_simulated": max(1, epochs),
        "sampler_query_positive_union_aware": sampler_query_union_aware,
        "duplicate_samples": duplicate_samples,
        "duplicate_positive_in_batch": duplicate_positive_in_batch,
        "query_family_collision_pairs": query_family_collision_pairs,
        "total_query_pairs": total_query_pairs,
        "query_family_collision_rate": safe_div(query_family_collision_pairs, total_query_pairs),
        "implicit_positive_negative_violations": implicit_positive_negative_violations,
        "directed_comparisons": directed_comparisons,
        "implicit_positive_negative_rate": safe_div(implicit_positive_negative_violations, directed_comparisons),
        "violation_examples": example_violations,
        "query_family_examples": query_family_examples,
    }


def analyze_truncation_field(
    values: list[str],
    tokenizer: Any,
    max_length: int,
) -> dict[str, Any]:
    lengths: list[int] = []
    truncated_count = 0
    high_risk_count = 0

    for text in values:
        encoded = tokenizer(text, add_special_tokens=True, truncation=False)
        input_ids = encoded.get("input_ids", [])
        length = len(input_ids)
        lengths.append(length)

        if length <= max_length:
            continue

        truncated_count += 1

        high_risk = False
        if getattr(tokenizer, "is_fast", False):
            detailed = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
            offsets = detailed.get("offset_mapping", [])
            if offsets:
                cutoff_index = min(max_length - 1, len(offsets) - 1)
                if cutoff_index >= 0:
                    cutoff_char = offsets[cutoff_index][1]
                    tail = text[cutoff_char:]
                    if DISCRIMINATIVE_TAIL_PATTERN.search(tail):
                        high_risk = True
        else:
            if DISCRIMINATIVE_TAIL_PATTERN.search(text):
                high_risk = True

        if high_risk:
            high_risk_count += 1

    return {
        "count": len(values),
        "length_stats": distribution_stats(lengths),
        "truncated_count": truncated_count,
        "truncated_rate": safe_div(truncated_count, len(values)),
        "high_risk_count": high_risk_count,
        "high_risk_rate": safe_div(high_risk_count, len(values)),
    }


def run_instruction_ab_test(
    model_name: str,
    eval_query_file: Path,
    eval_expected_file: Path,
    legacy_prefix: str,
    max_cases: int,
    device: str,
) -> dict[str, Any]:
    if not legacy_prefix.strip():
        return {"status": "skipped", "reason": "legacy prefix empty"}

    eval_module = load_module(EVAL_SCRIPT_PATH, "eval_models_preflight_ab")

    query_lines = load_non_empty_lines(eval_query_file)
    expected_lines = load_non_empty_lines(eval_expected_file)
    if len(query_lines) != len(expected_lines):
        raise ValueError(
            "A/B Evaluation: query/expected line mismatch: "
            f"{len(query_lines)} != {len(expected_lines)}"
        )

    if max_cases > 0:
        query_lines = query_lines[:max_cases]
        expected_lines = expected_lines[:max_cases]

    with tempfile.TemporaryDirectory(prefix="qa_ab_") as temp_dir:
        temp_path = Path(temp_dir)
        query_a = temp_path / "query_a.txt"
        query_b = temp_path / "query_b.txt"
        expected = temp_path / "expected.txt"

        query_a.write_text("\n".join(query_lines) + "\n", encoding="utf-8")
        query_b.write_text("\n".join(f"{legacy_prefix}{line}" for line in query_lines) + "\n", encoding="utf-8")
        expected.write_text("\n".join(expected_lines) + "\n", encoding="utf-8")

        setattr(eval_module, "SBERT_DEVICE", "" if device == "auto" else device)
        setattr(eval_module, "SBERT_CROSS_ENCODER_MODEL", "")

        cases_a = eval_module.build_evaluation_cases(query_file=query_a, expected_file=expected)
        cases_b = eval_module.build_evaluation_cases(query_file=query_b, expected_file=expected)

        database_path = eval_module.resolve_database_path(PROJECT_ROOT)
        with sqlite3.connect(str(database_path)) as connection:
            materials = eval_module.fetch_materials_from_db(connection)

        exact_index: dict[str, list[int]] = {}
        normalized_index: dict[str, list[int]] = {}
        for index, material in enumerate(materials):
            exact_index.setdefault(material, []).append(index)
            normalized_index.setdefault(eval_module.normalize(material), []).append(index)

        result_a = eval_module.evaluate_model(
            model_name=model_name,
            materials=materials,
            cases=cases_a,
            exact_index=exact_index,
            normalized_index=normalized_index,
            project_root=PROJECT_ROOT,
            cross_encoder_model="",
            rerank_top_n=30,
        )
        result_b = eval_module.evaluate_model(
            model_name=model_name,
            materials=materials,
            cases=cases_b,
            exact_index=exact_index,
            normalized_index=normalized_index,
            project_root=PROJECT_ROOT,
            cross_encoder_model="",
            rerank_top_n=30,
        )

    summary_a = result_a.summaries[0] if result_a.summaries else {}
    summary_b = result_b.summaries[0] if result_b.summaries else {}

    hit1_a = parse_float(summary_a.get("hit@1", "0"), default=0.0)
    hit1_b = parse_float(summary_b.get("hit@1", "0"), default=0.0)
    mrr_a = parse_float(summary_a.get("mrr", "0"), default=0.0)
    mrr_b = parse_float(summary_b.get("mrr", "0"), default=0.0)
    recall_a = parse_float(summary_a.get("recall@10", "0"), default=0.0)
    recall_b = parse_float(summary_b.get("recall@10", "0"), default=0.0)

    return {
        "status": "ok",
        "cases": len(query_lines),
        "arm_a": {
            "label": "no_prefix",
            "hit@1": hit1_a,
            "mrr@10": mrr_a,
            "recall@10": recall_a,
        },
        "arm_b": {
            "label": "legacy_prefix",
            "hit@1": hit1_b,
            "mrr@10": mrr_b,
            "recall@10": recall_b,
        },
        "delta_b_minus_a": {
            "hit@1": hit1_b - hit1_a,
            "mrr@10": mrr_b - mrr_a,
            "recall@10": recall_b - recall_a,
        },
    }


def resolve_prefix_strategy(
    *,
    run_instruction_ab: bool,
    legacy_query_prefix: str,
) -> dict[str, Any]:
    legacy_prefix = legacy_query_prefix.strip()
    legacy_experiment_active = bool(run_instruction_ab)

    if legacy_experiment_active and not legacy_prefix:
        raise ValueError(
            "Legacy Prefix Experiment angefordert, aber --legacy-query-prefix ist leer. "
            "Bitte Prefix explizit setzen oder --no-run-instruction-ab verwenden."
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
    parser = argparse.ArgumentParser(description="QA preflight before dense-only training.")
    parser.add_argument("--pairs-file", required=True, help="Training pairs JSONL (prepared).")
    parser.add_argument(
        "--prepare-pairs-file",
        default="",
        help="Optional pairs JSONL direkt nach prepare step, fuer HN-Lineage-Metriken.",
    )
    parser.add_argument("--query-file", required=True, help="Raw query TXT used for prepare step.")
    parser.add_argument("--expected-file", required=True, help="Raw expected TXT used for prepare step.")
    parser.add_argument("--base-model", default="BAAI/bge-m3", help="Model used for tokenization/embedding checks.")
    parser.add_argument("--hard-negatives-file", default="", help="Optional hard-negatives JSONL.")
    parser.add_argument("--details-file", default="", help="Optional evaluation details CSV.")
    parser.add_argument("--eval-query-file", default="", help="Optional evaluation query TXT for split-level leakage.")
    parser.add_argument("--eval-expected-file", default="", help="Optional evaluation expected TXT for split-level leakage.")
    parser.add_argument("--report-dir", default="Training/outputs/qa", help="Output directory for QA reports.")
    parser.add_argument("--run-id", default="", help="Optional run id for deterministic file naming.")
    parser.add_argument(
        "--rule-hash",
        default="",
        help="Optionaler deterministischer Hash ueber Regel-/Policy-Konfigurationen.",
    )

    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--hard-negative-mode", choices=["off", "fallback", "strict"], default="fallback")
    parser.add_argument("--hard-negative-selection", choices=["first", "random", "random_preselected"], default="first")
    parser.add_argument("--num-hard-negatives", type=int, default=1)
    parser.add_argument("--batch-audit-epochs", type=int, default=3)

    parser.add_argument("--text-near-dup-threshold", type=float, default=0.9)
    parser.add_argument("--embedding-near-dup-threshold", type=float, default=0.92)
    parser.add_argument("--skip-embedding-near-duplicates", action="store_true")
    parser.add_argument("--embedding-device", default="auto", help="auto|cpu|cuda")

    parser.add_argument("--run-instruction-ab", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--legacy-query-prefix",
        default="",
        help="Legacy query prefix for A/B comparison (only with --run-instruction-ab).",
    )
    parser.add_argument("--ab-max-cases", type=int, default=200)

    parser.add_argument("--fn-strict-stop-rate", type=float, default=0.01)
    parser.add_argument("--fn-cross-query-stop-rate", type=float, default=0.01)
    parser.add_argument(
        "--fn-cross-query-scope",
        choices=["off", "family", "global"],
        default="family",
    )
    parser.add_argument("--fn-cross-query-near-jaccard-threshold", type=float, default=0.60)
    parser.add_argument("--fn-any-scope-stop-count", type=int, default=0)
    parser.add_argument("--batch-query-family-stop-rate", type=float, default=0.0)
    parser.add_argument("--multi-positive-retention-stop-rate", type=float, default=1.0)
    parser.add_argument("--near-dup-text-warn-rate", type=float, default=0.02)
    parser.add_argument("--near-dup-emb-warn-rate", type=float, default=0.02)
    parser.add_argument("--queries-without-hn-warn-rate", type=float, default=0.25)
    parser.add_argument("--truncation-warn-rate", type=float, default=0.10)
    parser.add_argument("--high-risk-tail-warn-rate", type=float, default=0.02)

    parser.add_argument("--fail-on-stop", action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args()


def criterion_to_row(criterion: Criterion) -> dict[str, Any]:
    return {
        "criterion_id": criterion.criterion_id,
        "gate_type": criterion.gate_type,
        "status": criterion.status,
        "value": criterion.value,
        "threshold": criterion.threshold,
        "comparator": criterion.comparator,
        "message": criterion.message,
    }


def add_stop_criterion(
    criteria: list[Criterion],
    criterion_id: str,
    value: float,
    threshold: float,
    comparator: str,
    message: str,
) -> None:
    failed = False
    if comparator == "<=":
        failed = value > threshold
    elif comparator == "==":
        failed = value != threshold
    elif comparator == "<":
        failed = value >= threshold
    elif comparator == ">":
        failed = value <= threshold

    criteria.append(
        Criterion(
            criterion_id=criterion_id,
            gate_type="STOP",
            status="FAIL" if failed else "PASS",
            value=value,
            threshold=threshold,
            comparator=comparator,
            message=message,
        )
    )


def add_warn_criterion(
    criteria: list[Criterion],
    criterion_id: str,
    value: float,
    threshold: float,
    comparator: str,
    message: str,
) -> None:
    warned = False
    if comparator == "<=":
        warned = value > threshold
    elif comparator == ">=":
        warned = value < threshold
    elif comparator == ">":
        warned = value <= threshold
    elif comparator == "<":
        warned = value >= threshold

    criteria.append(
        Criterion(
            criterion_id=criterion_id,
            gate_type="WARN",
            status="WARN" if warned else "PASS",
            value=value,
            threshold=threshold,
            comparator=comparator,
            message=message,
        )
    )


def write_gate_csv(path: Path, criteria: list[Criterion]) -> None:
    rows = [criterion_to_row(item) for item in criteria]
    fieldnames = [
        "criterion_id",
        "gate_type",
        "status",
        "value",
        "threshold",
        "comparator",
        "message",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if args.num_hard_negatives <= 0:
        raise ValueError("--num-hard-negatives muss > 0 sein.")

    prefix_strategy = resolve_prefix_strategy(
        run_instruction_ab=bool(args.run_instruction_ab),
        legacy_query_prefix=args.legacy_query_prefix,
    )

    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = PROJECT_ROOT / report_dir
    report_dir = report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id.strip() or f"qa_{Path(args.pairs_file).stem}_{args.seed}"
    run_id = re.sub(r"[^A-Za-z0-9._-]+", "_", run_id).strip("._-") or "qa"

    pairs_file = Path(args.pairs_file)
    if not pairs_file.is_absolute():
        pairs_file = PROJECT_ROOT / pairs_file
    pairs_file = pairs_file.resolve()

    prepare_pairs_file: Path | None = None
    if args.prepare_pairs_file.strip():
        prepare_pairs_file = Path(args.prepare_pairs_file)
        if not prepare_pairs_file.is_absolute():
            prepare_pairs_file = PROJECT_ROOT / prepare_pairs_file
        prepare_pairs_file = prepare_pairs_file.resolve()

    query_file = Path(args.query_file)
    if not query_file.is_absolute():
        query_file = PROJECT_ROOT / query_file
    query_file = query_file.resolve()

    expected_file = Path(args.expected_file)
    if not expected_file.is_absolute():
        expected_file = PROJECT_ROOT / expected_file
    expected_file = expected_file.resolve()

    hard_negatives_file = Path(args.hard_negatives_file).resolve() if args.hard_negatives_file.strip() else None

    details_file: Path | None = None
    if args.details_file.strip():
        details_file = Path(args.details_file)
        if not details_file.is_absolute():
            details_file = PROJECT_ROOT / details_file
        details_file = details_file.resolve()
    elif hard_negatives_file is not None:
        details_file = infer_details_file_from_hn(hard_negatives_file)

    eval_query_file: Path | None = None
    eval_expected_file: Path | None = None
    if args.eval_query_file.strip() and args.eval_expected_file.strip():
        eval_query_file = Path(args.eval_query_file)
        if not eval_query_file.is_absolute():
            eval_query_file = PROJECT_ROOT / eval_query_file
        eval_query_file = eval_query_file.resolve()

        eval_expected_file = Path(args.eval_expected_file)
        if not eval_expected_file.is_absolute():
            eval_expected_file = PROJECT_ROOT / eval_expected_file
        eval_expected_file = eval_expected_file.resolve()

    train_module = load_module(TRAIN_SCRIPT_PATH, "train_bge_m3_preflight")

    all_records = train_module.read_records(pairs_file)
    prepare_records: list[Any] | None = None
    if prepare_pairs_file is not None:
        prepare_records = train_module.read_records(prepare_pairs_file)

    train_records, dev_records = train_module.split_records(all_records, dev_ratio=args.dev_ratio, seed=args.seed)
    train_example_generation_error = ""
    try:
        train_examples, hard_negative_stats = train_module.build_train_examples(
            train_records=train_records,
            hard_negative_mode=args.hard_negative_mode,
            hard_negative_selection=args.hard_negative_selection,
            seed=args.seed,
            num_hard_negatives=args.num_hard_negatives,
        )
    except ValueError as exc:
        train_example_generation_error = str(exc)
        # Keep preflight executable even when strict mode has no usable hard negatives.
        train_examples, hard_negative_stats = train_module.build_train_examples(
            train_records=train_records,
            hard_negative_mode="off",
            hard_negative_selection=args.hard_negative_selection,
            seed=args.seed,
            num_hard_negatives=args.num_hard_negatives,
        )
        hard_negative_stats = dict(hard_negative_stats)
        hard_negative_stats["requested_hard_negative_mode"] = args.hard_negative_mode
        hard_negative_stats["fallback_hard_negative_mode_used"] = "off"
        hard_negative_stats["train_example_generation_error"] = train_example_generation_error

    strict_viability_error = ""
    strict_viability_examples: list[Any] = []
    try:
        strict_viability_examples, _ = train_module.build_train_examples(
            train_records=train_records,
            hard_negative_mode="strict",
            hard_negative_selection=args.hard_negative_selection,
            seed=args.seed,
            num_hard_negatives=args.num_hard_negatives,
        )
    except ValueError as exc:
        strict_viability_error = str(exc)

    trainer_query_to_hn: dict[str, set[str]] = defaultdict(set)
    for example in strict_viability_examples:
        texts = list(getattr(example, "texts", []))
        if len(texts) < 3:
            continue
        query_key = normalize_text(texts[0])
        hard_negative_key = normalize_text(texts[2])
        if query_key and hard_negative_key:
            trainer_query_to_hn[query_key].add(hard_negative_key)

    expected_pairs = load_query_expected_pairs(query_file=query_file, expected_file=expected_file)
    expected_map = build_expected_positive_map(expected_pairs)

    hard_negatives_by_query: dict[str, dict[str, set[str]]] = {}
    if hard_negatives_file is not None and hard_negatives_file.is_file():
        hard_negatives_by_query = load_hard_negatives_jsonl(hard_negatives_file)

    details_by_query: dict[str, list[dict[str, Any]]] = {}
    if details_file is not None and details_file.is_file():
        details_by_query = load_details_csv(details_file)

    query_positive_union = build_query_positive_union(
        expected_map=expected_map,
        all_records=all_records,
        hard_negatives_by_query=hard_negatives_by_query,
        details_by_query=details_by_query,
    )

    split_sets = {
        "train": build_split_sets(train_records),
        "dev": build_split_sets(dev_records),
    }

    eval_sets: dict[str, set[str]] | None = None
    if eval_query_file is not None and eval_expected_file is not None:
        eval_pairs = load_query_expected_pairs(eval_query_file, eval_expected_file)
        eval_sets = build_eval_sets_from_pairs(eval_pairs)

    overlap_matrix: dict[str, dict[str, dict[str, Any]]] = {}
    split_pairs = [("train", "dev")]
    if eval_sets is not None:
        split_pairs.extend([("train", "eval"), ("dev", "eval")])

    for left_name, right_name in split_pairs:
        left_sets = split_sets[left_name] if left_name != "eval" else eval_sets
        right_sets = split_sets[right_name] if right_name != "eval" else eval_sets
        assert left_sets is not None
        assert right_sets is not None

        pair_key = f"{left_name}_vs_{right_name}"
        overlap_matrix[pair_key] = {}
        for field in ("queries", "positives", "query_positive_pairs", "material_ids"):
            metrics = overlap_metrics(left_sets[field], right_sets[field])
            overlap_matrix[pair_key][field] = asdict(metrics)

    near_duplicate_matrix: dict[str, dict[str, Any]] = {}
    sentence_model = None
    embedding_device = resolve_device(args.embedding_device)

    for left_name, right_name in split_pairs:
        left_sets = split_sets[left_name] if left_name != "eval" else eval_sets
        right_sets = split_sets[right_name] if right_name != "eval" else eval_sets
        assert left_sets is not None
        assert right_sets is not None

        pair_key = f"{left_name}_vs_{right_name}"
        near_duplicate_matrix[pair_key] = {}

        for field in ("queries", "positives"):
            text_left = text_near_duplicate_left_rate(
                left_sets[field],
                right_sets[field],
                threshold=args.text_near_dup_threshold,
            )
            text_right = text_near_duplicate_left_rate(
                right_sets[field],
                left_sets[field],
                threshold=args.text_near_dup_threshold,
            )

            field_payload: dict[str, Any] = {
                "text_based": {
                    "left_to_right": text_left,
                    "right_to_left": text_right,
                    "threshold": args.text_near_dup_threshold,
                    "method": "token_jaccard",
                }
            }

            if not args.skip_embedding_near_duplicates:
                if sentence_model is None:
                    sentence_model = load_sentence_transformer(args.base_model, embedding_device)

                emb_left = embedding_near_duplicate_left_rate(
                    left_sets[field],
                    right_sets[field],
                    model=sentence_model,
                    threshold=args.embedding_near_dup_threshold,
                )
                emb_right = embedding_near_duplicate_left_rate(
                    right_sets[field],
                    left_sets[field],
                    model=sentence_model,
                    threshold=args.embedding_near_dup_threshold,
                )
                field_payload["embedding_based"] = {
                    "left_to_right": emb_left,
                    "right_to_left": emb_right,
                    "threshold": args.embedding_near_dup_threshold,
                    "method": "cosine_sentence_transformer",
                    "model": args.base_model,
                }
            else:
                field_payload["embedding_based"] = {
                    "status": "skipped",
                    "reason": "--skip-embedding-near-duplicates enabled",
                }

            near_duplicate_matrix[pair_key][field] = field_payload

    hn_quality = hard_negative_quality(all_records)
    hn_viability = build_hn_viability_report(
        hard_negatives_by_query=hard_negatives_by_query,
        prepare_records=prepare_records,
        final_records=all_records,
        trainer_query_to_hn=dict(trainer_query_to_hn),
        strict_viability_error=strict_viability_error,
    )
    multi_positive_retention = compute_multi_positive_retention(
        prepare_records=prepare_records,
        final_records=all_records,
    )

    queries_without_hn_rate_requested_mode = float(hn_quality["queries_without_hn_rate"])
    if args.hard_negative_mode == "strict" and not train_example_generation_error:
        # In strict mode, examples are only built from records with hard negatives.
        queries_without_hn_rate_requested_mode = 0.0
    hn_quality["queries_without_hn_rate_raw"] = float(hn_quality["queries_without_hn_rate"])
    hn_quality["queries_without_hn_rate_requested_mode"] = queries_without_hn_rate_requested_mode

    fn_metrics = compute_false_negative_metrics(
        records=all_records,
        query_positive_union=query_positive_union,
        details_by_query=details_by_query,
        cross_query_scope=args.fn_cross_query_scope,
        cross_query_near_jaccard_threshold=args.fn_cross_query_near_jaccard_threshold,
    )

    batch_audit = batch_safety_audit(
        train_records=train_records,
        train_examples=train_examples,
        sampler_cls=train_module.UniquePositiveBatchSampler,
        batch_size=args.batch_size,
        seed=args.seed,
        epochs=args.batch_audit_epochs,
    )

    tokenizer = None
    tokenizer_error = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    except Exception as exc:
        tokenizer_error = str(exc)

    truncation_report: dict[str, Any] = {}
    if tokenizer is not None:
        queries = [str(record.query) for record in all_records]
        positives = [str(record.positive) for record in all_records]
        negatives = [
            str(negative)
            for record in all_records
            for negative in record.hard_negatives
            if str(negative).strip()
        ]
        truncation_report = {
            "query": analyze_truncation_field(queries, tokenizer=tokenizer, max_length=args.max_length),
            "positive": analyze_truncation_field(positives, tokenizer=tokenizer, max_length=args.max_length),
            "negative": analyze_truncation_field(negatives, tokenizer=tokenizer, max_length=args.max_length),
            "tokenizer": args.base_model,
        }
    else:
        truncation_report = {
            "status": "skipped",
            "reason": f"Tokenizer konnte nicht geladen werden: {tokenizer_error}",
        }

    balance_report = build_balance_report(all_records, tokenizer=tokenizer)

    instruction_ab_report: dict[str, Any]
    if prefix_strategy["legacy_prefix_experiment_active"] and eval_query_file is not None and eval_expected_file is not None:
        try:
            instruction_ab_report = run_instruction_ab_test(
                model_name=args.base_model,
                eval_query_file=eval_query_file,
                eval_expected_file=eval_expected_file,
                legacy_prefix=prefix_strategy["legacy_query_prefix"],
                max_cases=args.ab_max_cases,
                device=args.embedding_device,
            )
        except Exception as exc:
            instruction_ab_report = {
                "status": "failed",
                "reason": str(exc),
            }
    else:
        skip_reason = "legacy prefix experiment disabled (--no-run-instruction-ab)"
        if prefix_strategy["legacy_prefix_experiment_active"]:
            skip_reason = "eval files missing"
        instruction_ab_report = {
            "status": "skipped",
            "reason": skip_reason,
        }

    compatibility_report = {
        "training_path": "dense-only",
        "loss": "MultipleNegativesRankingLoss",
        "sampler": "UniquePositiveBatchSampler",
        "data_format": {
            "required_fields": ["query", "positive"],
            "optional_fields": [
                "hard_negatives",
                "preselected_hard_negative",
                "preselected_hard_negatives",
                "query_index",
                "weight",
            ],
            "records_total": len(all_records),
        },
        "checked_compatibility": {
            "loss_sampler_pair": "checked",
            "batch_duplicate_samples": batch_audit["duplicate_samples"],
            "batch_known_positive_as_negative": batch_audit["implicit_positive_negative_violations"],
            "hard_negative_mode": args.hard_negative_mode,
            "hard_negative_selection": args.hard_negative_selection,
            "num_hard_negatives": args.num_hard_negatives,
        },
    }

    criteria: list[Criterion] = []

    if "train_vs_eval" in overlap_matrix:
        add_stop_criterion(
            criteria,
            "stop_exact_overlap_train_eval_query_positive",
            overlap_matrix["train_vs_eval"]["query_positive_pairs"]["intersection_size"],
            0,
            "==",
            "Exact query-positive overlap Train vs Eval must be zero.",
        )
    else:
        criteria.append(
            Criterion(
                criterion_id="stop_exact_overlap_train_eval_query_positive",
                gate_type="STOP",
                status="PASS",
                value="n/a",
                threshold=0,
                comparator="==",
                message="Eval split missing, criterion skipped.",
            )
        )

    if "dev_vs_eval" in overlap_matrix:
        add_stop_criterion(
            criteria,
            "stop_exact_overlap_dev_eval_query_positive",
            overlap_matrix["dev_vs_eval"]["query_positive_pairs"]["intersection_size"],
            0,
            "==",
            "Exact query-positive overlap Dev vs Eval must be zero.",
        )
    else:
        criteria.append(
            Criterion(
                criterion_id="stop_exact_overlap_dev_eval_query_positive",
                gate_type="STOP",
                status="PASS",
                value="n/a",
                threshold=0,
                comparator="==",
                message="Eval split missing, criterion skipped.",
            )
        )

    add_stop_criterion(
        criteria,
        "stop_batch_duplicate_samples",
        float(batch_audit["duplicate_samples"]),
        0.0,
        "==",
        "Batch duplicate samples must be zero.",
    )
    add_stop_criterion(
        criteria,
        "stop_batch_known_positive_as_negative",
        float(batch_audit["implicit_positive_negative_violations"]),
        0.0,
        "==",
        "Known positives as implicit negatives must be zero.",
    )
    add_stop_criterion(
        criteria,
        "stop_false_negative_rate_strict",
        float(fn_metrics["fn_strict_rate"]),
        float(args.fn_strict_stop_rate),
        "<=",
        "Strict false-negative rate must be under threshold.",
    )
    add_stop_criterion(
        criteria,
        "stop_false_negative_rate_cross_query",
        float(fn_metrics["fn_cross_query_rate"]),
        float(args.fn_cross_query_stop_rate),
        "<=",
        "Cross-query false-negative rate must be under threshold.",
    )
    add_stop_criterion(
        criteria,
        "stop_false_negative_count_any_scope",
        float(fn_metrics["fn_any_scope_count"]),
        float(args.fn_any_scope_stop_count),
        "<=",
        "Any-scope positive/negative overlap must stay under threshold.",
    )
    add_stop_criterion(
        criteria,
        "stop_batch_query_family_collision_rate",
        float(batch_audit["query_family_collision_rate"]),
        float(args.batch_query_family_stop_rate),
        "<=",
        "Query-family collision rate in batches must stay under threshold.",
    )
    add_stop_criterion(
        criteria,
        "stop_multi_positive_retention",
        float(multi_positive_retention.get("retention_rate", 1.0)),
        float(args.multi_positive_retention_stop_rate),
        ">=",
        "Retention of multi-positive queries from prepare to final pairs must meet threshold.",
    )

    add_stop_criterion(
        criteria,
        "stop_requested_hard_negative_mode_usable",
        1.0 if not train_example_generation_error else 0.0,
        1.0,
        "==",
        "Requested hard-negative mode must be executable with prepared data.",
    )

    mandatory_sections_present = all(
        [
            compatibility_report.get("loss"),
            compatibility_report.get("sampler"),
            compatibility_report.get("data_format"),
            compatibility_report.get("checked_compatibility"),
        ]
    )
    add_stop_criterion(
        criteria,
        "stop_mandatory_qa_sections",
        1.0 if mandatory_sections_present else 0.0,
        1.0,
        "==",
        "QA report must contain loss/sampler/data_format/compatibility sections.",
    )

    add_warn_criterion(
        criteria,
        "warn_queries_without_hn_rate",
        float(hn_quality["queries_without_hn_rate_requested_mode"]),
        float(args.queries_without_hn_warn_rate),
        "<=",
        "Queries without hard negatives in requested training mode should stay under warn threshold.",
    )
    add_warn_criterion(
        criteria,
        "warn_duplicate_negatives_rate",
        float(hn_quality["duplicate_negatives_rate"]),
        0.0,
        "<=",
        "Duplicate negatives rate should be zero.",
    )

    if "train_vs_eval" in near_duplicate_matrix:
        text_rate = near_duplicate_matrix["train_vs_eval"]["queries"]["text_based"]["left_to_right"]["left_rate"]
        add_warn_criterion(
            criteria,
            "warn_text_near_duplicates_train_eval_queries",
            float(text_rate),
            float(args.near_dup_text_warn_rate),
            "<=",
            "Text-based near duplicate rate Train vs Eval (queries) should stay low.",
        )

        embedding_payload = near_duplicate_matrix["train_vs_eval"]["queries"].get("embedding_based", {})
        if embedding_payload.get("status") != "skipped":
            emb_rate = embedding_payload["left_to_right"]["left_rate"]
            add_warn_criterion(
                criteria,
                "warn_embedding_near_duplicates_train_eval_queries",
                float(emb_rate),
                float(args.near_dup_emb_warn_rate),
                "<=",
                "Embedding-based near duplicate rate Train vs Eval (queries) should stay low.",
            )

    if truncation_report.get("status") != "skipped":
        for field in ("query", "positive", "negative"):
            field_report = truncation_report.get(field, {})
            add_warn_criterion(
                criteria,
                f"warn_truncation_rate_{field}",
                float(field_report.get("truncated_rate", 0.0)),
                float(args.truncation_warn_rate),
                "<=",
                f"Truncation rate for {field} should stay below warn threshold.",
            )
            add_warn_criterion(
                criteria,
                f"warn_high_risk_tail_rate_{field}",
                float(field_report.get("high_risk_rate", 0.0)),
                float(args.high_risk_tail_warn_rate),
                "<=",
                f"High-risk tail truncation rate for {field} should stay below warn threshold.",
            )

    stop_failures = [item for item in criteria if item.gate_type == "STOP" and item.status == "FAIL"]
    warnings = [item for item in criteria if item.gate_type == "WARN" and item.status == "WARN"]

    report = {
        "meta": {
            "run_id": run_id,
            "rule_hash": args.rule_hash.strip(),
            "training_path": "dense-only",
            "prefix_mode": prefix_strategy["prefix_mode"],
            "source_of_prefix_setting": prefix_strategy["source_of_prefix_setting"],
            "dense_only_bge_m3_default_applied": prefix_strategy["dense_only_bge_m3_default_applied"],
            "legacy_prefix_experiment_active": prefix_strategy["legacy_prefix_experiment_active"],
            "pairs_file": str(pairs_file),
            "prepare_pairs_file": str(prepare_pairs_file) if prepare_pairs_file else "",
            "query_file": str(query_file),
            "expected_file": str(expected_file),
            "hard_negatives_file": str(hard_negatives_file) if hard_negatives_file else "",
            "details_file": str(details_file) if details_file else "",
            "eval_query_file": str(eval_query_file) if eval_query_file else "",
            "eval_expected_file": str(eval_expected_file) if eval_expected_file else "",
            "base_model": args.base_model,
            "seed": args.seed,
            "dev_ratio": args.dev_ratio,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "hard_negative_mode": args.hard_negative_mode,
            "hard_negative_selection": args.hard_negative_selection,
            "num_hard_negatives": args.num_hard_negatives,
            "batch_audit_epochs": args.batch_audit_epochs,
            "fn_strict_stop_rate": args.fn_strict_stop_rate,
            "fn_cross_query_stop_rate": args.fn_cross_query_stop_rate,
            "fn_cross_query_scope": args.fn_cross_query_scope,
            "fn_cross_query_near_jaccard_threshold": args.fn_cross_query_near_jaccard_threshold,
            "fn_any_scope_stop_count": args.fn_any_scope_stop_count,
            "batch_query_family_stop_rate": args.batch_query_family_stop_rate,
            "multi_positive_retention_stop_rate": args.multi_positive_retention_stop_rate,
        },
        "status": {
            "stop_failures": len(stop_failures),
            "warnings": len(warnings),
            "overall": "FAIL" if stop_failures else ("WARN" if warnings else "PASS"),
        },
        "sections": {
            "prefix_strategy": prefix_strategy,
            "hard_negative_stats": hard_negative_stats,
            "overlap_matrix": overlap_matrix,
            "near_duplicate_matrix": near_duplicate_matrix,
            "hard_negative_quality": hn_quality,
            "hard_negative_viability": hn_viability,
            "multi_positive_retention": multi_positive_retention,
            "false_negative_metrics": fn_metrics,
            "balance": balance_report,
            "batch_audit": batch_audit,
            "truncation": truncation_report,
            "instruction_ab_test": instruction_ab_report,
            "compatibility": compatibility_report,
        },
        "criteria": [criterion_to_row(item) for item in criteria],
    }

    json_path = report_dir / f"qa_report_{run_id}.json"
    gate_csv_path = report_dir / f"qa_gate_{run_id}.csv"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_gate_csv(gate_csv_path, criteria)

    print(f"QA report: {json_path}")
    print(f"QA gates:  {gate_csv_path}")
    print(f"STOP failures: {len(stop_failures)}")
    print(f"WARN findings: {len(warnings)}")

    if stop_failures and args.fail_on_stop:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
