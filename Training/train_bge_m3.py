import argparse
import hashlib
import inspect
import json
import math
import random
import shutil
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, cast

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SentenceEvaluator
from sentence_transformers.util import batch_to_device
from torch.utils.data import DataLoader, Dataset

try:
    from Training import text_normalization
except ModuleNotFoundError:
    import text_normalization


class InputExampleDataset(Dataset[InputExample]):
    def __init__(self, examples: list[InputExample]) -> None:
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> InputExample:
        return self._examples[index]


@dataclass(frozen=True)
class TrainingRecord:
    query: str
    positive: str
    hard_negatives: tuple[str, ...] = ()
    preselected_hard_negatives: tuple[str, ...] = ()
    preselected_hard_negative: str = ""
    query_class: str = ""
    sample_weight: float = 1.0
    query_positives: tuple[str, ...] = ()


def to_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def to_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def metric_get_float(metrics: Mapping[str, object] | None, key: str, default: float = 0.0) -> float:
    if metrics is None:
        return default
    return to_float(metrics.get(key, default), default)


def metric_get_int(metrics: Mapping[str, object] | None, key: str, default: int = 0) -> int:
    if metrics is None:
        return default
    return to_int(metrics.get(key, default), default)


class TimedEvaluatorWrapper(SentenceEvaluator):
    def __init__(self, evaluator: SentenceEvaluator, runtime_metrics: dict[str, object]) -> None:
        self._evaluator = evaluator
        self._runtime_metrics = runtime_metrics

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> float | dict[str, float]:
        started = time.perf_counter()
        result = self._evaluator(model, output_path=output_path, epoch=epoch, steps=steps)
        elapsed = time.perf_counter() - started

        total = metric_get_float(self._runtime_metrics, "epoch_evaluation_seconds_total", 0.0)
        calls = metric_get_int(self._runtime_metrics, "epoch_evaluation_calls", 0)
        self._runtime_metrics["epoch_evaluation_seconds_total"] = total + elapsed
        self._runtime_metrics["epoch_evaluation_calls"] = calls + 1
        return result

    def __getattr__(self, name: str) -> object:
        return getattr(self._evaluator, name)


class CombinedHit5Mrr10Evaluator(SentenceEvaluator):
    """Adds a deterministic combined metric for model selection.

    Hit@5 is treated as the primary signal and MRR@10 as tie-break.
    """

    def __init__(self, evaluator: SentenceEvaluator, hit_priority_scale: float = 1_000_000.0) -> None:
        self._evaluator = evaluator
        self._hit_priority_scale = float(hit_priority_scale)
        self.primary_metric = "combined_hit5_mrr10"
        self.greater_is_better = True
        self._warned_missing_metrics = False

    @staticmethod
    def _normalize_metric_key(key: str) -> str:
        normalized = key.casefold()
        return "".join(ch for ch in normalized if ch.isalnum() or ch == "@")

    @classmethod
    def _extract_metric(cls, metrics: Mapping[str, object], aliases: list[str]) -> float:
        normalized_aliases = [cls._normalize_metric_key(alias) for alias in aliases]
        candidates: list[float] = []
        for key, value in metrics.items():
            try:
                metric_value = float(cast(Any, value))
            except (TypeError, ValueError):
                continue

            normalized_key = cls._normalize_metric_key(str(key))
            if any(alias in normalized_key for alias in normalized_aliases):
                candidates.append(metric_value)

        if not candidates:
            raise ValueError(f"Erwartete Evaluationsmetrik nicht gefunden: {aliases}")
        return max(candidates)

    @staticmethod
    def _metric_at_k(values: Mapping[object, object], target_k: int) -> float | None:
        for key, value in values.items():
            try:
                if int(str(key)) == target_k:
                    return float(cast(Any, value))
            except (TypeError, ValueError):
                continue
        return None

    @classmethod
    def _extract_hit_mrr_from_scores_dict(cls, score_payload: Mapping[str, object]) -> tuple[float, float] | None:
        # Flat dict variant: keys like "cosine_accuracy@5" / "cosine_mrr@10".
        try:
            return (
                cls._extract_metric(score_payload, ["accuracy@5", "hit@5"]),
                cls._extract_metric(score_payload, ["mrr@10"]),
            )
        except ValueError:
            pass

        # Nested variant: {"cosine": {"accuracy@k": {5: ...}, "mrr@k": {10: ...}}}
        hit_candidates: list[float] = []
        mrr_candidates: list[float] = []
        for value in score_payload.values():
            if not isinstance(value, dict):
                continue

            accuracy_block = value.get("accuracy@k")
            if isinstance(accuracy_block, dict):
                hit_at_5 = cls._metric_at_k(accuracy_block, 5)
                if hit_at_5 is not None:
                    hit_candidates.append(hit_at_5)

            mrr_block = value.get("mrr@k")
            if isinstance(mrr_block, dict):
                mrr_at_10 = cls._metric_at_k(mrr_block, 10)
                if mrr_at_10 is not None:
                    mrr_candidates.append(mrr_at_10)

        if hit_candidates and mrr_candidates:
            return max(hit_candidates), max(mrr_candidates)
        return None

    def _extract_from_latest_csv(self, output_path: str | None) -> tuple[float, float] | None:
        if not output_path:
            return None

        csv_name = str(getattr(self._evaluator, "csv_file", "") or "").strip()
        if not csv_name:
            return None

        csv_path = Path(output_path) / csv_name
        if not csv_path.is_file():
            return None

        try:
            lines = [line for line in csv_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except OSError:
            return None

        if len(lines) < 2:
            return None

        headers = lines[0].split(",")
        values = lines[-1].split(",")
        if len(headers) != len(values):
            return None

        flat_metrics: dict[str, float] = {}
        for key, value in zip(headers, values):
            try:
                flat_metrics[key] = float(value)
            except ValueError:
                continue

        return self._extract_hit_mrr_from_scores_dict(flat_metrics)

    def _extract_from_evaluator_scores(self, model: SentenceTransformer, output_path: str | None) -> tuple[float, float] | None:
        for method_name in ("compute_metrices", "compute_metrics"):
            method = getattr(self._evaluator, method_name, None)
            if not callable(method):
                continue

            for call_kwargs in ({"output_path": output_path}, {}):
                try:
                    score_payload = method(model, **call_kwargs)
                except TypeError:
                    continue
                except Exception:
                    return None

                if isinstance(score_payload, dict):
                    extracted = self._extract_hit_mrr_from_scores_dict(score_payload)
                    if extracted is not None:
                        return extracted
        return None

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> float | dict[str, float]:
        result = self._evaluator(model, output_path=output_path, epoch=epoch, steps=steps)
        metrics = result if isinstance(result, dict) else {"evaluator": float(result)}

        extracted = self._extract_hit_mrr_from_scores_dict(metrics)
        if extracted is None:
            extracted = self._extract_from_latest_csv(output_path)
        if extracted is None:
            extracted = self._extract_from_evaluator_scores(model, output_path)

        if extracted is None:
            if not self._warned_missing_metrics:
                print(
                    "Warnung: hit5_mrr10 konnte Hit@5/MRR@10 nicht aus Evaluator-Outputs extrahieren; "
                    "fallback auf Standard-Evaluator-Score."
                )
                self._warned_missing_metrics = True
            return metrics if isinstance(result, dict) else float(result)

        hit_at_5, mrr_at_10 = extracted
        combined_score = (hit_at_5 * self._hit_priority_scale) + mrr_at_10

        if not isinstance(result, dict):
            # Legacy sentence-transformers fit() expects a scalar score.
            return float(combined_score)

        enriched = dict(metrics)
        enriched[self.primary_metric] = float(combined_score)
        enriched[f"{self.primary_metric}_hit_at_5"] = float(hit_at_5)
        enriched[f"{self.primary_metric}_mrr_at_10"] = float(mrr_at_10)
        return enriched

    def __getattr__(self, name: str) -> object:
        return getattr(self._evaluator, name)


def extract_model_save_target(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> str:
    if args:
        return str(args[0])
    for key in ("path", "output_path"):
        value = kwargs.get(key)
        if value is not None:
            return str(value)
    return ""


def resolve_save_target(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(Path(text).expanduser().resolve())
    except OSError:
        return text


def normalize_text_key(value: str) -> str:
    normalize_fn = cast(Callable[[str], str], getattr(text_normalization, "normalize_text_key"))
    return normalize_fn(value)


def query_family_key(value: str) -> str:
    family_fn = cast(Callable[[str], str], getattr(text_normalization, "query_family_key"))
    return family_fn(value)


def stable_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        key = normalize_text_key(value)
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def build_record_query_positive_union(records: list[TrainingRecord]) -> dict[str, set[str]]:
    query_positive_union: dict[str, set[str]] = defaultdict(set)
    for record in records:
        query_key = normalize_text_key(record.query)
        if not query_key:
            continue

        positive_candidates = stable_unique(list(record.query_positives) + [record.positive])
        for positive in positive_candidates:
            positive_key = normalize_text_key(positive)
            if positive_key:
                query_positive_union[query_key].add(positive_key)

    return {query_key: set(values) for query_key, values in query_positive_union.items()}


def coerce_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def sync_cuda_if_needed(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def deterministic_choice(values: list[str], seed: int, record_identity: str, channel: str = "") -> str:
    if not values:
        raise ValueError("deterministic_choice erwartet mindestens einen Wert.")
    digest = hashlib.sha1(f"{seed}|{record_identity}|{channel}".encode("utf-8")).hexdigest()
    index = int(digest[:16], 16) % len(values)
    return values[index]


class UniquePositiveBatchSampler:
    """BatchSampler ensuring no positive text appears more than once per batch.

    Prevents false negatives in MultipleNegativesRankingLoss by guaranteeing
    that each in-batch negative is a genuinely different document.
    """

    def __init__(
        self,
        examples: list[InputExample],
        batch_size: int,
        seed: int,
        query_positive_union: dict[str, set[str]] | None = None,
        runtime_profile: dict[str, object] | None = None,
        enforce_query_family_separation: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size muss > 0 sein.")

        self._batch_size = batch_size
        self._seed = seed
        self._epoch = 0
        self._runtime_profile = runtime_profile
        self._enforce_query_family_separation = enforce_query_family_separation

        self._example_query_keys: list[str] = []
        self._example_query_family_keys: list[str] = []
        self._example_positive_keys: list[str] = []
        inferred_union: dict[str, set[str]] = defaultdict(set)

        for example in examples:
            texts = list(getattr(example, "texts", []))
            query = texts[0] if len(texts) > 0 else ""
            positive = texts[1] if len(texts) > 1 else ""

            query_key = normalize_text_key(query)
            query_family = query_family_key(query)
            positive_key = normalize_text_key(positive)

            self._example_query_keys.append(query_key)
            self._example_query_family_keys.append(query_family)
            self._example_positive_keys.append(positive_key)

            if query_key and positive_key:
                inferred_union[query_key].add(positive_key)

        merged_union: dict[str, set[str]] = defaultdict(set)
        for query_key, positives in inferred_union.items():
            merged_union[query_key].update(positives)

        if query_positive_union:
            for query, positives in query_positive_union.items():
                query_key = normalize_text_key(query)
                if not query_key:
                    continue
                for positive in positives:
                    positive_key = normalize_text_key(positive)
                    if positive_key:
                        merged_union[query_key].add(positive_key)

        self._query_positive_union = {query_key: set(values) for query_key, values in merged_union.items()}
        self._total = len(examples)
        self._unique_positives = len({key for key in self._example_positive_keys if key})

    def _candidate_is_safe(
        self,
        candidate_index: int,
        batch_indices: list[int],
        batch_positive_keys: set[str],
    ) -> bool:
        candidate_query = self._example_query_keys[candidate_index]
        candidate_positive = self._example_positive_keys[candidate_index]

        if candidate_positive and candidate_positive in batch_positive_keys:
            return False

        candidate_known_positives = self._query_positive_union.get(candidate_query, set())

        for batch_index in batch_indices:
            batch_query = self._example_query_keys[batch_index]
            batch_positive = self._example_positive_keys[batch_index]
            batch_known_positives = self._query_positive_union.get(batch_query, set())

            # Enforce both directions so no in-batch positive can become a known positive negative.
            if candidate_positive and candidate_positive in batch_known_positives:
                return False
            if batch_positive and batch_positive in candidate_known_positives:
                return False

        return True

    def _build_batches(self, epoch: int) -> list[list[int]]:
        started = time.perf_counter()
        rng = random.Random(self._seed + epoch)

        remaining = list(range(self._total))
        rng.shuffle(remaining)

        batches: list[list[int]] = []
        family_collision_rejects = 0

        while remaining:
            batch: list[int] = []
            batch_positive_keys: set[str] = set()
            batch_query_families: set[str] = set()
            next_remaining: list[int] = []

            for index in remaining:
                candidate_family = self._example_query_family_keys[index]
                if (
                    len(batch) < self._batch_size
                    and self._enforce_query_family_separation
                    and candidate_family
                    and candidate_family in batch_query_families
                ):
                    family_collision_rejects += 1
                    next_remaining.append(index)
                    continue

                if len(batch) < self._batch_size and self._candidate_is_safe(index, batch, batch_positive_keys):
                    batch.append(index)
                    positive_key = self._example_positive_keys[index]
                    if positive_key:
                        batch_positive_keys.add(positive_key)
                    if candidate_family:
                        batch_query_families.add(candidate_family)
                else:
                    next_remaining.append(index)

            if not batch:
                forced_index = next_remaining.pop(0)
                batch = [forced_index]

            batches.append(batch)
            remaining = next_remaining

        rng.shuffle(batches)

        if self._runtime_profile is not None:
            elapsed = time.perf_counter() - started
            calls = self._runtime_profile.setdefault("sampler_build_calls", [])
            if isinstance(calls, list):
                calls.append({
                    "epoch": int(epoch),
                    "elapsed_seconds": elapsed,
                    "batches": len(batches),
                    "family_collision_rejects": int(family_collision_rejects),
                    "query_family_separation_enabled": bool(self._enforce_query_family_separation),
                })

        return batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._build_batches(epoch=self._epoch)
        self._epoch += 1
        yield from batches

    def __len__(self) -> int:
        return len(self._build_batches(epoch=self._epoch))


def read_records(path: Path) -> list[TrainingRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"Trainingsdatei nicht gefunden: {path}")

    parsed_rows: list[
        tuple[
            str,
            str,
            list[str],
            tuple[str, ...],
            str,
            int,
            str,
            float,
            tuple[str, ...],
        ]
    ] = []
    query_positive_union: dict[str, set[str]] = defaultdict(set)
    explicit_query_positive_union: dict[str, set[str]] = defaultdict(set)
    query_positive_text_by_key: dict[str, dict[str, str]] = defaultdict(dict)
    query_anchor_payload: dict[
        str,
        tuple[
            str,
            tuple[str, ...],
            tuple[str, ...],
            str,
            int,
            str,
            float,
            tuple[str, ...],
        ],
    ] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            query = str(row.get("query", "")).strip()
            if not query:
                raise ValueError(f"Ungültiger Datensatz in Zeile {line_no}: query fehlt.")

            positives = coerce_str_list(row.get("pos"))
            legacy_positive = str(row.get("positive", "")).strip()
            if legacy_positive:
                positives.append(legacy_positive)
            positives = stable_unique(positives)
            if not positives:
                raise ValueError(f"Ungültiger Datensatz in Zeile {line_no}: positive/pos fehlt.")

            query_positives = stable_unique(coerce_str_list(row.get("query_positives")) + positives)
            query_class = str(row.get("query_class", "")).strip().lower()
            if not query_class:
                query_class = "mehrdeutig" if len(query_positives) > 1 else "eindeutig"

            sample_weight_value = row.get("sample_weight", row.get("weight", 1.0))
            sample_weight = float(to_float(sample_weight_value, default=1.0))
            if sample_weight < 0.0:
                sample_weight = 0.0

            hard_negatives_raw = coerce_str_list(row.get("hard_negatives"))
            hard_negatives_raw.extend(coerce_str_list(row.get("neg")))

            preselected_hard_negatives_raw = stable_unique(coerce_str_list(row.get("preselected_hard_negatives")))
            legacy_preselected_hard_negative = str(row.get("preselected_hard_negative", "")).strip()
            if preselected_hard_negatives_raw:
                preselected_hard_negatives = tuple(preselected_hard_negatives_raw)
                preselected_hard_negative = preselected_hard_negatives[0]
            elif legacy_preselected_hard_negative:
                preselected_hard_negatives = (legacy_preselected_hard_negative,)
                preselected_hard_negative = legacy_preselected_hard_negative
            else:
                preselected_hard_negatives = ()
                preselected_hard_negative = ""

            query_key = normalize_text_key(query)
            if not query_key:
                raise ValueError(f"Ungültiger Datensatz in Zeile {line_no}: query ist leer nach Normalisierung.")

            if query_key not in query_anchor_payload:
                query_anchor_payload[query_key] = (
                    query,
                    tuple(hard_negatives_raw),
                    tuple(preselected_hard_negatives),
                    preselected_hard_negative,
                    line_no,
                    query_class,
                    sample_weight,
                    tuple(query_positives),
                )

            for query_positive in query_positives:
                query_positive_key = normalize_text_key(query_positive)
                if not query_positive_key:
                    continue
                query_positive_union[query_key].add(query_positive_key)
                query_positive_text_by_key[query_key].setdefault(query_positive_key, query_positive)

            for positive in positives:
                positive_key = normalize_text_key(positive)
                if not positive_key:
                    continue
                explicit_query_positive_union[query_key].add(positive_key)
                query_positive_text_by_key[query_key].setdefault(positive_key, positive)
                parsed_rows.append(
                    (
                        query,
                        positive,
                        list(hard_negatives_raw),
                        tuple(preselected_hard_negatives),
                        preselected_hard_negative,
                        line_no,
                        query_class,
                        sample_weight,
                        tuple(query_positives),
                    )
                )

    for query_key, all_positive_keys in query_positive_union.items():
        explicit_positive_keys = explicit_query_positive_union.get(query_key, set())
        missing_positive_keys = sorted(all_positive_keys - explicit_positive_keys)
        if not missing_positive_keys:
            continue

        anchor_payload = query_anchor_payload.get(query_key)
        if anchor_payload is None:
            continue

        (
            anchor_query,
            anchor_hard_negatives,
            anchor_preselected_hard_negatives,
            anchor_preselected_hard_negative,
            anchor_line_no,
            anchor_query_class,
            anchor_sample_weight,
            anchor_query_positives,
        ) = anchor_payload

        for missing_positive_key in missing_positive_keys:
            missing_positive = query_positive_text_by_key.get(query_key, {}).get(missing_positive_key, missing_positive_key)
            parsed_rows.append(
                (
                    anchor_query,
                    missing_positive,
                    list(anchor_hard_negatives),
                    tuple(anchor_preselected_hard_negatives),
                    anchor_preselected_hard_negative,
                    anchor_line_no,
                    anchor_query_class,
                    anchor_sample_weight,
                    anchor_query_positives,
                )
            )

    records: list[TrainingRecord] = []
    preselected_conflicts_with_positives = 0
    for (
        query,
        positive,
        hard_negatives_raw,
        preselected_hard_negatives,
        preselected_hard_negative,
        _line_no,
        query_class,
        sample_weight,
        query_positives,
    ) in parsed_rows:
        query_key = normalize_text_key(query)
        positive_key = normalize_text_key(positive)
        query_positive_keys = query_positive_union.get(query_key, set())

        cleaned_preselected: list[str] = []
        seen_preselected: set[str] = set()
        for value in preselected_hard_negatives:
            candidate = str(value).strip()
            candidate_key = normalize_text_key(candidate)
            if not candidate_key:
                continue
            if candidate_key in query_positive_keys:
                preselected_conflicts_with_positives += 1
                continue
            if candidate_key in seen_preselected:
                continue
            seen_preselected.add(candidate_key)
            cleaned_preselected.append(candidate)

        if not cleaned_preselected and preselected_hard_negative:
            candidate_key = normalize_text_key(preselected_hard_negative)
            if candidate_key and candidate_key in query_positive_keys:
                preselected_conflicts_with_positives += 1

        resolved_preselected_hard_negative = cleaned_preselected[0] if cleaned_preselected else ""

        seen_negatives: set[str] = set()
        hard_negatives: list[str] = []
        for value in hard_negatives_raw:
            candidate = str(value).strip()
            candidate_key = normalize_text_key(candidate)
            if (
                not candidate_key
                or candidate_key == positive_key
                or candidate_key in query_positive_keys
                or candidate_key in seen_negatives
            ):
                continue
            seen_negatives.add(candidate_key)
            hard_negatives.append(candidate)

        records.append(
            TrainingRecord(
                query=query,
                positive=positive,
                hard_negatives=tuple(hard_negatives),
                preselected_hard_negatives=tuple(cleaned_preselected),
                preselected_hard_negative=resolved_preselected_hard_negative,
                query_class=query_class,
                sample_weight=sample_weight,
                query_positives=query_positives,
            )
        )

    if not records:
        raise ValueError("Keine Trainingspaare gefunden.")

    if preselected_conflicts_with_positives > 0:
        print(
            "Warnung: preselected_hard_negative Konflikte mit query_positives erkannt und entfernt: "
            f"{preselected_conflicts_with_positives}"
        )

    return records


def split_records(
    records: list[TrainingRecord],
    dev_ratio: float,
    seed: int,
) -> tuple[list[TrainingRecord], list[TrainingRecord]]:
    """Split on query level: all pairs of a query stay together in train or dev."""
    if not 0 <= dev_ratio < 1:
        raise ValueError("--dev-ratio muss im Bereich [0, 1) liegen.")

    if dev_ratio == 0:
        return list(records), []

    query_to_records: dict[str, list[TrainingRecord]] = {}
    for record in records:
        query_to_records.setdefault(record.query, []).append(record)

    unique_queries = list(query_to_records.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_queries)

    dev_query_count = int(len(unique_queries) * dev_ratio)
    if dev_query_count == 0 and len(unique_queries) > 5:
        dev_query_count = 1

    dev_queries = set(unique_queries[:dev_query_count])

    dev_records: list[TrainingRecord] = []
    train_records: list[TrainingRecord] = []
    for record in records:
        if record.query in dev_queries:
            dev_records.append(record)
        else:
            train_records.append(record)

    if not train_records:
        raise ValueError("Nach dem Split sind keine Trainingsdaten übrig. --dev-ratio verringern.")
    return train_records, dev_records


def build_ir_evaluator(dev_records: list[TrainingRecord]) -> InformationRetrievalEvaluator | None:
    if not dev_records:
        return None

    query_to_positives: dict[str, set[str]] = {}
    for record in dev_records:
        query_to_positives.setdefault(record.query, set()).add(record.positive)

    if not query_to_positives:
        return None

    corpus_docs: dict[str, str] = {}
    doc_id_by_text: dict[str, str] = {}
    queries: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}

    for q_idx, (query, positives) in enumerate(query_to_positives.items()):
        qid = f"q{q_idx}"
        queries[qid] = query
        relevant_docs[qid] = set()

        for positive in positives:
            if positive not in doc_id_by_text:
                doc_id = f"d{len(doc_id_by_text)}"
                doc_id_by_text[positive] = doc_id
                corpus_docs[doc_id] = positive
            relevant_docs[qid].add(doc_id_by_text[positive])

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus_docs,
        relevant_docs=relevant_docs,
        name="dev_ir",
    )


def build_train_examples(
    train_records: list[TrainingRecord],
    hard_negative_mode: str,
    hard_negative_selection: str,
    seed: int,
    num_hard_negatives: int,
    runtime_profile: dict[str, object] | None = None,
) -> tuple[list[InputExample], dict[str, int | float]]:
    if not train_records:
        raise ValueError("Keine Trainingsdaten vorhanden.")
    if num_hard_negatives <= 0:
        raise ValueError("num_hard_negatives muss > 0 sein.")

    weighted_records = [record for record in train_records if record.sample_weight > 0.0]
    dropped_non_positive_weight = len(train_records) - len(weighted_records)
    if not weighted_records:
        raise ValueError("Alle Trainingsdatensätze haben sample_weight <= 0.0.")

    records_with_hard_negatives = sum(
        1
        for record in weighted_records
        if record.hard_negatives or record.preselected_hard_negatives or record.preselected_hard_negative
    )
    dropped_no_hard_negatives = 0
    fallback_negatives_used = 0
    fallback_fill_count = 0
    dropped_unusable = 0
    preselected_used = 0
    preselected_missing_runtime_fallback = 0
    preselected_conflicts_with_positives = 0
    renormalized_weight_applied = 0

    selection_seconds_total = 0.0
    selection_records_timed = 0

    selected_records = list(weighted_records)
    use_hard_negatives = hard_negative_mode != "off" and records_with_hard_negatives > 0

    if hard_negative_mode == "strict":
        selected_records = [
            record
            for record in weighted_records
            if record.hard_negatives or record.preselected_hard_negatives or record.preselected_hard_negative
        ]
        dropped_no_hard_negatives = len(weighted_records) - len(selected_records)
        use_hard_negatives = True
        if not selected_records:
            raise ValueError("--hard-negative-mode strict gewählt, aber keine hard_negatives in den Trainingsdaten gefunden.")
    elif hard_negative_mode == "fallback" and not use_hard_negatives:
        print("Hinweis: Keine hard_negatives gefunden, es wird mit [query, positive] trainiert.")

    rng = random.Random(seed)
    unique_positives = stable_unique([record.positive for record in selected_records])
    query_positive_union = build_record_query_positive_union(selected_records)

    global_hard_pool: list[str] = []
    seen_global_hard: set[str] = set()
    for record in selected_records:
        for candidate in record.hard_negatives:
            candidate_key = candidate.casefold().strip()
            if candidate_key and candidate_key not in seen_global_hard:
                seen_global_hard.add(candidate_key)
                global_hard_pool.append(candidate)

    examples: list[InputExample] = []

    for record in selected_records:
        if not use_hard_negatives:
            examples.append(InputExample(texts=[record.query, record.positive], label=float(record.sample_weight)))
            continue

        selection_started = time.perf_counter()
        query_key = normalize_text_key(record.query)
        query_positive_keys = query_positive_union.get(query_key, set())
        record_hard_negatives = stable_unique([
            value
            for value in record.hard_negatives
            if normalize_text_key(value) not in query_positive_keys
        ])

        selected_hard_negatives: list[str] = []
        selected_hard_negative_keys: set[str] = set()

        def add_selected(candidate: str) -> bool:
            candidate_key = normalize_text_key(candidate)
            if not candidate_key:
                return False
            if candidate_key in query_positive_keys:
                return False
            if candidate_key in selected_hard_negative_keys:
                return False
            selected_hard_negative_keys.add(candidate_key)
            selected_hard_negatives.append(candidate)
            return True

        if hard_negative_selection == "random_preselected":
            for candidate in record.preselected_hard_negatives:
                if len(selected_hard_negatives) >= num_hard_negatives:
                    break
                candidate_key = normalize_text_key(candidate)
                if candidate_key and candidate_key in query_positive_keys:
                    preselected_conflicts_with_positives += 1
                    continue
                if add_selected(candidate):
                    preselected_used += 1

            if len(selected_hard_negatives) < num_hard_negatives and record_hard_negatives:
                available = [
                    value
                    for value in record_hard_negatives
                    if normalize_text_key(value) not in selected_hard_negative_keys
                ]
                while available and len(selected_hard_negatives) < num_hard_negatives:
                    slot_index = len(selected_hard_negatives)
                    record_identity = f"{record.query}|{record.positive}|record_hard_negatives"
                    choice = deterministic_choice(
                        list(available),
                        seed=seed,
                        record_identity=record_identity,
                        channel=f"slot_{slot_index}",
                    )
                    if add_selected(choice):
                        preselected_missing_runtime_fallback += 1
                    choice_key = normalize_text_key(choice)
                    available = [value for value in available if normalize_text_key(value) != choice_key]

        elif hard_negative_selection == "random":
            sample_count = min(num_hard_negatives, len(record_hard_negatives))
            for candidate in rng.sample(record_hard_negatives, k=sample_count):
                add_selected(candidate)

        else:
            for candidate in record_hard_negatives:
                if len(selected_hard_negatives) >= num_hard_negatives:
                    break
                add_selected(candidate)

        while len(selected_hard_negatives) < num_hard_negatives and hard_negative_mode == "fallback":
            fallback_candidates = [
                value
                for value in unique_positives
                if normalize_text_key(value) not in query_positive_keys
                and normalize_text_key(value) not in selected_hard_negative_keys
            ]

            chosen_fallback: str | None = None
            if fallback_candidates:
                if hard_negative_selection == "random_preselected":
                    slot_index = len(selected_hard_negatives)
                    record_identity = f"{record.query}|{record.positive}|fallback_positives"
                    chosen_fallback = deterministic_choice(
                        fallback_candidates,
                        seed=seed,
                        record_identity=record_identity,
                        channel=f"slot_{slot_index}",
                    )
                    preselected_missing_runtime_fallback += 1
                elif hard_negative_selection == "random":
                    chosen_fallback = rng.choice(fallback_candidates)
                else:
                    chosen_fallback = fallback_candidates[0]
            else:
                pool_candidates = [
                    value
                    for value in global_hard_pool
                    if normalize_text_key(value) not in query_positive_keys
                    and normalize_text_key(value) not in selected_hard_negative_keys
                ]
                if pool_candidates:
                    if hard_negative_selection == "random_preselected":
                        slot_index = len(selected_hard_negatives)
                        record_identity = f"{record.query}|{record.positive}|fallback_pool"
                        chosen_fallback = deterministic_choice(
                            pool_candidates,
                            seed=seed,
                            record_identity=record_identity,
                            channel=f"slot_{slot_index}",
                        )
                        preselected_missing_runtime_fallback += 1
                    elif hard_negative_selection == "random":
                        chosen_fallback = rng.choice(pool_candidates)
                    else:
                        chosen_fallback = pool_candidates[0]

            if not chosen_fallback:
                break

            if add_selected(chosen_fallback):
                fallback_negatives_used += 1
                fallback_fill_count += 1
            else:
                break

        selection_seconds_total += time.perf_counter() - selection_started
        selection_records_timed += 1

        if not selected_hard_negatives:
            if hard_negative_mode == "fallback":
                examples.append(InputExample(texts=[record.query, record.positive], label=float(record.sample_weight)))
                continue
            dropped_unusable += 1
            continue

        if len(selected_hard_negatives) > 1:
            renormalized_weight_applied += 1
        renormalized_weight = float(record.sample_weight) / float(len(selected_hard_negatives))

        for hard_negative in selected_hard_negatives:
            examples.append(InputExample(texts=[record.query, record.positive, hard_negative], label=renormalized_weight))

    if not examples:
        raise ValueError("Nach Hard-Negative-Verarbeitung sind keine Trainingsbeispiele übrig.")

    examples_per_record_avg = len(examples) / len(selected_records) if selected_records else 0.0
    preselected_used_count = preselected_used

    stats: dict[str, int | float] = {
        "records_total": len(train_records),
        "dropped_non_positive_weight": dropped_non_positive_weight,
        "records_after_mode": len(selected_records),
        "records_with_hard_negatives": records_with_hard_negatives,
        "examples_total": len(examples),
        "examples_per_record_avg": examples_per_record_avg,
        "dropped_no_hard_negatives": dropped_no_hard_negatives,
        "fallback_negatives_used": fallback_negatives_used,
        "fallback_fill_count": fallback_fill_count,
        "dropped_unusable": dropped_unusable,
        "use_hard_negatives": 1 if use_hard_negatives else 0,
        "num_hard_negatives_requested": int(num_hard_negatives),
        "renormalized_weight_applied": renormalized_weight_applied,
        "preselected_used": preselected_used,
        "preselected_used_count": preselected_used_count,
        "preselected_missing_runtime_fallback": preselected_missing_runtime_fallback,
        "preselected_conflicts_with_positives": preselected_conflicts_with_positives,
    }

    if runtime_profile is not None:
        runtime_profile["hn_selection"] = {
            "mode": hard_negative_selection,
            "records_total": len(train_records),
            "records_after_weight_filter": len(weighted_records),
            "records_after_mode": len(selected_records),
            "records_timed": selection_records_timed,
            "selection_seconds_total": selection_seconds_total,
            "selection_seconds_per_record": (
                selection_seconds_total / selection_records_timed if selection_records_timed > 0 else 0.0
            ),
            "num_hard_negatives_requested": int(num_hard_negatives),
            "examples_per_record_avg": examples_per_record_avg,
            "renormalized_weight_applied": renormalized_weight_applied,
            "preselected_used": preselected_used,
            "preselected_missing_runtime_fallback": preselected_missing_runtime_fallback,
            "preselected_conflicts_with_positives": preselected_conflicts_with_positives,
            "fallback_fill_count": fallback_fill_count,
        }

    return examples, stats


def choose_device(user_device: str) -> str:
    user_device = user_device.strip().lower()
    if user_device in {"cpu", "cuda"}:
        return user_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning von BAAI/bge-m3 als Bi-Encoder.")
    parser.add_argument("--train-file", required=True, help="Pfad zur JSONL-Datei mit query/positive Paaren.")
    parser.add_argument("--base-model", default="BAAI/bge-m3", help="Sentence-Transformer Startmodell.")
    parser.add_argument("--output-dir", required=True, help="Ausgabeverzeichnis für das trainierte Modell.")
    parser.add_argument("--epochs", type=int, default=2, help="Anzahl Epochen.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch-Size.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning Rate.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup-Anteil (0..1).")
    parser.add_argument("--max-length", type=int, default=512, help="Maximale Token-Länge.")
    parser.add_argument("--dev-ratio", type=float, default=0.1, help="Anteil für Dev-Evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed.")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
    parser.add_argument("--fp16", action="store_true", help="Mixed precision Training (use_amp).")
    parser.add_argument(
        "--run-id",
        default="",
        help="Deterministische Run-ID für nachvollziehbare Artefakte (wenn leer, wird ein Fallback verwendet).",
    )
    parser.add_argument(
        "--rule-hash",
        default="",
        help="Optionaler deterministischer Hash ueber Regel-/Policy-Konfigurationen.",
    )
    parser.add_argument(
        "--save-each-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Speichert zusätzlich zu best-model den Zustand jeder Epoche.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        default="",
        help="Optionaler Ordner für Epochen-Checkpoints (Default: <output-dir>/epochs).",
    )
    parser.add_argument(
        "--hard-negative-mode",
        choices=["off", "fallback", "strict"],
        default="fallback",
        help="off: ignorieren, fallback: nutzen + fehlende auffüllen, strict: nur Datensätze mit hard_negatives.",
    )
    parser.add_argument(
        "--hard-negative-selection",
        choices=["first", "random", "random_preselected"],
        default="random_preselected",
        help="Auswahlstrategie, wenn mehrere hard_negatives pro Datensatz vorliegen.",
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
        help=(
            "Metrik fuer best-model Auswahl: default nutzt die Evaluator-Primary-Metric, "
            "hit5_mrr10 kombiniert Hit@5 (Prioritaet) mit MRR@10 (Tie-Break)."
        ),
    )
    parser.add_argument(
        "--runtime-profile-out",
        default="",
        help="Optionales JSON-Ausgabefile fuer Runtime-Profiling (12A).",
    )
    parser.add_argument(
        "--runtime-profile-max-batches",
        type=int,
        default=0,
        help="Maximale Anzahl Batches fuer Profilmodus (0 = alle verfuegbaren Batches).",
    )
    parser.add_argument(
        "--full-runtime-profile-out",
        default="",
        help="Optionales JSON-Ausgabefile fuer Voll-Laufzeitprofil.",
    )
    return parser.parse_args()


def sanitize_label(value: str, fallback: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    safe = "".join(ch if ch in allowed else "_" for ch in value).strip("._-")
    return safe or fallback


def normalize_step_checkpoints_to_epochs(step_checkpoint_dir: Path, epoch_checkpoint_dir: Path) -> list[str]:
    epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not step_checkpoint_dir.is_dir():
        return []

    step_dirs = [path for path in step_checkpoint_dir.iterdir() if path.is_dir()]
    step_dirs.sort(key=lambda item: item.stat().st_mtime)

    saved_epoch_dirs: list[str] = []
    for index, source_dir in enumerate(step_dirs, start=1):
        target_dir = epoch_checkpoint_dir / f"epoch-{index:02d}"
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)
        saved_epoch_dirs.append(str(target_dir))

    return saved_epoch_dirs


def summarize_sampler_build_calls(calls: list[dict[str, object]]) -> dict[str, float | int]:
    elapsed_values = [
        metric_get_float(call, "elapsed_seconds", 0.0)
        for call in calls
        if isinstance(call, dict)
    ]
    if not elapsed_values:
        return {
            "calls": 0,
            "seconds_total": 0.0,
            "seconds_mean": 0.0,
            "seconds_min": 0.0,
            "seconds_max": 0.0,
        }
    return {
        "calls": len(elapsed_values),
        "seconds_total": float(sum(elapsed_values)),
        "seconds_mean": float(sum(elapsed_values) / len(elapsed_values)),
        "seconds_min": float(min(elapsed_values)),
        "seconds_max": float(max(elapsed_values)),
    }


def run_runtime_profile_loop(
    *,
    model: SentenceTransformer,
    train_dataloader: DataLoader,
    train_loss: losses.MultipleNegativesRankingLoss,
    device: str,
    max_batches: int,
    learning_rate: float,
) -> dict[str, float | int]:
    # Profile only a controlled number of optimizer steps to keep local runtime bounded.
    train_dataloader.collate_fn = model.smart_batching_collate
    target_device = torch.device(device)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    data_fetch_seconds_total = 0.0
    forward_backward_seconds_total = 0.0
    optimizer_step_seconds_total = 0.0
    losses_observed: list[float] = []
    batches_executed = 0

    iterator = iter(train_dataloader)
    while True:
        if max_batches > 0 and batches_executed >= max_batches:
            break

        fetch_started = time.perf_counter()
        try:
            sentence_features, labels = next(iterator)
        except StopIteration:
            break
        data_fetch_seconds_total += time.perf_counter() - fetch_started

        sentence_features = [batch_to_device(features, target_device) for features in sentence_features]
        if labels is not None and hasattr(labels, "to"):
            labels = labels.to(target_device)

        sync_cuda_if_needed(device)
        forward_started = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        loss_value = train_loss(sentence_features, labels)
        loss_scalar = float(loss_value.detach().cpu().item())

        loss_value.backward()
        opt_started = time.perf_counter()
        optimizer.step()
        optimizer_step_seconds_total += time.perf_counter() - opt_started

        sync_cuda_if_needed(device)
        forward_backward_seconds_total += time.perf_counter() - forward_started

        losses_observed.append(loss_scalar)
        batches_executed += 1

    mean_loss = float(sum(losses_observed) / len(losses_observed)) if losses_observed else 0.0
    return {
        "batches_executed": batches_executed,
        "data_fetch_seconds_total": data_fetch_seconds_total,
        "forward_backward_seconds_total": forward_backward_seconds_total,
        "optimizer_step_seconds_total": optimizer_step_seconds_total,
        "mean_loss": mean_loss,
    }


def main() -> None:
    args = parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs muss > 0 sein.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size muss > 0 sein.")
    if args.lr <= 0:
        raise ValueError("--lr muss > 0 sein.")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("--warmup-ratio muss im Bereich [0, 1) liegen.")
    if args.max_length <= 0:
        raise ValueError("--max-length muss > 0 sein.")
    if args.num_hard_negatives <= 0:
        raise ValueError("--num-hard-negatives muss > 0 sein.")
    if args.runtime_profile_max_batches < 0:
        raise ValueError("--runtime-profile-max-batches muss >= 0 sein.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_path = Path(args.train_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = sanitize_label(args.run_id, fallback=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    full_runtime_metrics: dict[str, object] | None = None
    if args.full_runtime_profile_out.strip():
        full_runtime_metrics = {
            "train_setup_seconds": 0.0,
            "fit_total_seconds": 0.0,
            "dataloader_batch_collation_seconds_total": 0.0,
            "dataloader_batches_collated": 0,
            "tokenization_feature_prep_seconds_total": 0.0,
            "tokenization_calls": 0,
            "forward_seconds_total": 0.0,
            "forward_calls": 0,
            "backward_seconds_total": 0.0,
            "backward_calls": 0,
            "epoch_evaluation_seconds_total": 0.0,
            "epoch_evaluation_calls": 0,
            "checkpoint_saving_seconds_total": 0.0,
            "checkpoint_save_calls": 0,
            "final_model_save_seconds_total": 0.0,
            "final_model_save_calls": 0,
            "post_fit_post_processing_seconds": 0.0,
            "model_save_events": [],
        }

    train_setup_started = time.perf_counter()

    all_records = read_records(train_path)
    train_records, dev_records = split_records(all_records, dev_ratio=args.dev_ratio, seed=args.seed)

    device = choose_device(args.device)
    print(f"Device: {device}")
    print(f"Base model: {args.base_model}")
    train_queries = len({record.query for record in train_records})
    dev_queries = len({record.query for record in dev_records})
    print(
        f"Gesamtpaare: {len(all_records)} | Train: {len(train_records)} ({train_queries} Queries) | "
        f"Dev: {len(dev_records)} ({dev_queries} Queries)"
    )

    model = SentenceTransformer(args.base_model, device=device)
    model.max_seq_length = args.max_length

    runtime_profile: dict[str, object] | None = {} if args.runtime_profile_out.strip() else None

    train_examples, hard_negative_stats = build_train_examples(
        train_records=train_records,
        hard_negative_mode=args.hard_negative_mode,
        hard_negative_selection=args.hard_negative_selection,
        seed=args.seed,
        num_hard_negatives=args.num_hard_negatives,
        runtime_profile=runtime_profile,
    )
    if hard_negative_stats["use_hard_negatives"]:
        print(
            "Hard-negatives aktiv: "
            f"records_with_hard_negatives={hard_negative_stats['records_with_hard_negatives']}, "
            f"fallback_negatives_used={hard_negative_stats['fallback_negatives_used']}, "
            f"dropped_unusable={hard_negative_stats['dropped_unusable']}"
        )
    else:
        print("Hard-negatives deaktiviert oder nicht vorhanden: Training mit [query, positive].")

    if (
        hard_negative_stats.get("use_hard_negatives")
        and args.num_hard_negatives > 1
        and float(hard_negative_stats.get("examples_per_record_avg", 0.0)) < (args.num_hard_negatives * 0.5)
    ):
        print(
            "Warnung: Effektive Hard-Negative-Abdeckung deutlich unter Ziel-K. "
            f"requested_k={args.num_hard_negatives}, "
            f"examples_per_record_avg={float(hard_negative_stats.get('examples_per_record_avg', 0.0)):.4f}"
        )

    query_positive_union = build_record_query_positive_union(train_records)

    train_dataset = InputExampleDataset(train_examples)
    batch_sampler = UniquePositiveBatchSampler(
        train_examples,
        batch_size=args.batch_size,
        seed=args.seed,
        query_positive_union=dict(query_positive_union),
        runtime_profile=runtime_profile,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        pin_memory=device == "cuda",
    )
    # sentence-transformers model.fit() reads dataloader.batch_size internally;
    # PyTorch sets it to None when batch_sampler is used, so patch it back.
    object.__setattr__(train_dataloader, "batch_size", args.batch_size)
    print(f"UniquePositiveBatchSampler: {batch_sampler._unique_positives} unique Positives, batch_size={args.batch_size}")
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    if runtime_profile is not None:
        profile_loop = run_runtime_profile_loop(
            model=model,
            train_dataloader=train_dataloader,
            train_loss=train_loss,
            device=device,
            max_batches=int(args.runtime_profile_max_batches),
            learning_rate=float(args.lr),
        )

        sampler_calls_raw = runtime_profile.get("sampler_build_calls", [])
        sampler_calls = sampler_calls_raw if isinstance(sampler_calls_raw, list) else []

        profile_payload = {
            "profile_name": "hard_negative_runtime_profile",
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "run_id": run_id,
            "train_file": str(train_path),
            "hard_negative_mode": args.hard_negative_mode,
            "hard_negative_selection": args.hard_negative_selection,
            "num_hard_negatives": int(args.num_hard_negatives),
            "seed": int(args.seed),
            "batch_size": int(args.batch_size),
            "dev_ratio": float(args.dev_ratio),
            "epochs_configured": int(args.epochs),
            "device": device,
            "runtime_profile_max_batches": int(args.runtime_profile_max_batches),
            "hard_negative_stats": hard_negative_stats,
            "hn_selection": runtime_profile.get("hn_selection", {}),
            "sampler_build": {
                "calls": sampler_calls,
                "summary": summarize_sampler_build_calls(sampler_calls),
            },
            "training_loop": profile_loop,
        }

        runtime_profile_path = Path(args.runtime_profile_out).expanduser().resolve()
        runtime_profile_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_profile_path.write_text(json.dumps(profile_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"Runtime profile gespeichert: {runtime_profile_path}")
        return

    evaluator = build_ir_evaluator(dev_records)
    if evaluator is not None and args.model_selection_metric == "hit5_mrr10":
        evaluator = CombinedHit5Mrr10Evaluator(evaluator)
        print("Model selection metric: Hit@5 (primary) + MRR@10 (tie-break).")
    warmup_steps = math.ceil(len(train_dataloader) * args.epochs * args.warmup_ratio)

    step_checkpoint_dir: Path | None = None
    epoch_checkpoint_dir: Path | None = None
    steps_per_epoch = max(1, len(train_dataloader))
    checkpoint_args_enabled = False

    if args.save_each_epoch:
        if args.checkpoints_dir:
            epoch_checkpoint_dir = Path(args.checkpoints_dir).expanduser().resolve()
        else:
            epoch_checkpoint_dir = output_dir / "epochs"
        step_checkpoint_dir = output_dir / "_checkpoints_steps"

        fit_params = inspect.signature(model.fit).parameters
        supports_checkpoints = {"checkpoint_path", "checkpoint_save_steps", "checkpoint_save_total_limit"}.issubset(
            set(fit_params.keys())
        )
        if supports_checkpoints:
            checkpoint_args_enabled = True
        else:
            print(
                "Warnung: Diese sentence-transformers-Version unterstützt keine Checkpoint-Argumente in model.fit; "
                "Epochen-Checkpoints werden übersprungen."
            )
            step_checkpoint_dir = None
            epoch_checkpoint_dir = None

    if full_runtime_metrics is not None:
        full_runtime_metrics["train_setup_seconds"] = time.perf_counter() - train_setup_started

    original_tokenize: Callable[..., dict[str, torch.Tensor]] | None = None
    original_smart_batching_collate: Callable[..., tuple[list[dict[str, torch.Tensor]], torch.Tensor]] | None = None
    original_loss_forward: Callable[..., torch.Tensor] | None = None
    original_tensor_backward: Callable[..., Any] | None = None
    original_model_save: Callable[..., None] | None = None
    checkpoint_root = ""

    timed_evaluator: SentenceEvaluator | None = evaluator

    if full_runtime_metrics is not None:
        checkpoint_root = str(step_checkpoint_dir.resolve()) if step_checkpoint_dir is not None else ""

        original_tokenize = cast(Callable[..., dict[str, torch.Tensor]], model.tokenize)

        def timed_tokenize(*tokenize_args: Any, **tokenize_kwargs: Any) -> dict[str, torch.Tensor]:
            started = time.perf_counter()
            result = original_tokenize(*tokenize_args, **tokenize_kwargs)
            elapsed = time.perf_counter() - started

            total = metric_get_float(full_runtime_metrics, "tokenization_feature_prep_seconds_total", 0.0)
            calls = metric_get_int(full_runtime_metrics, "tokenization_calls", 0)
            full_runtime_metrics["tokenization_feature_prep_seconds_total"] = total + elapsed
            full_runtime_metrics["tokenization_calls"] = calls + 1
            return result

        model.tokenize = timed_tokenize  # type: ignore[assignment]

        original_smart_batching_collate = cast(
            Callable[..., tuple[list[dict[str, torch.Tensor]], torch.Tensor]],
            model.smart_batching_collate,
        )

        def timed_smart_batching_collate(batch: list[InputExample]) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
            started = time.perf_counter()
            result = original_smart_batching_collate(batch)
            elapsed = time.perf_counter() - started

            total = metric_get_float(full_runtime_metrics, "dataloader_batch_collation_seconds_total", 0.0)
            batches = metric_get_int(full_runtime_metrics, "dataloader_batches_collated", 0)
            full_runtime_metrics["dataloader_batch_collation_seconds_total"] = total + elapsed
            full_runtime_metrics["dataloader_batches_collated"] = batches + 1
            return result

        model.smart_batching_collate = timed_smart_batching_collate  # type: ignore[assignment]

        original_loss_forward = cast(Callable[..., torch.Tensor], train_loss.forward)

        def timed_loss_forward(*forward_args: Any, **forward_kwargs: Any) -> torch.Tensor:
            started = time.perf_counter()
            result = original_loss_forward(*forward_args, **forward_kwargs)
            elapsed = time.perf_counter() - started

            total = metric_get_float(full_runtime_metrics, "forward_seconds_total", 0.0)
            calls = metric_get_int(full_runtime_metrics, "forward_calls", 0)
            full_runtime_metrics["forward_seconds_total"] = total + elapsed
            full_runtime_metrics["forward_calls"] = calls + 1
            return result

        train_loss.forward = timed_loss_forward  # type: ignore[assignment]

        original_tensor_backward = cast(Callable[..., Any], torch.Tensor.backward)

        def timed_tensor_backward(self: torch.Tensor, *backward_args: Any, **backward_kwargs: Any) -> Any:
            started = time.perf_counter()
            result = original_tensor_backward(self, *backward_args, **backward_kwargs)
            elapsed = time.perf_counter() - started

            total = metric_get_float(full_runtime_metrics, "backward_seconds_total", 0.0)
            calls = metric_get_int(full_runtime_metrics, "backward_calls", 0)
            full_runtime_metrics["backward_seconds_total"] = total + elapsed
            full_runtime_metrics["backward_calls"] = calls + 1
            return result

        torch.Tensor.backward = timed_tensor_backward  # type: ignore[assignment]

        original_model_save = cast(Callable[..., None], model.save)

        def timed_model_save(*save_args: Any, **save_kwargs: Any) -> None:
            target_raw = extract_model_save_target(save_args, save_kwargs)
            target_resolved = resolve_save_target(target_raw)

            started = time.perf_counter()
            original_model_save(*save_args, **save_kwargs)
            elapsed = time.perf_counter() - started

            is_checkpoint_save = bool(checkpoint_root and target_resolved.startswith(checkpoint_root))

            if is_checkpoint_save:
                total = metric_get_float(full_runtime_metrics, "checkpoint_saving_seconds_total", 0.0)
                calls = metric_get_int(full_runtime_metrics, "checkpoint_save_calls", 0)
                full_runtime_metrics["checkpoint_saving_seconds_total"] = total + elapsed
                full_runtime_metrics["checkpoint_save_calls"] = calls + 1
            else:
                total = metric_get_float(full_runtime_metrics, "final_model_save_seconds_total", 0.0)
                calls = metric_get_int(full_runtime_metrics, "final_model_save_calls", 0)
                full_runtime_metrics["final_model_save_seconds_total"] = total + elapsed
                full_runtime_metrics["final_model_save_calls"] = calls + 1

            events_raw = full_runtime_metrics.get("model_save_events", [])
            if isinstance(events_raw, list) and len(events_raw) < 200:
                events_raw.append(
                    {
                        "target": target_resolved,
                        "elapsed_seconds": elapsed,
                        "is_checkpoint": is_checkpoint_save,
                    }
                )

        model.save = timed_model_save  # type: ignore[assignment]

        if evaluator is not None:
            timed_evaluator = TimedEvaluatorWrapper(evaluator, full_runtime_metrics)

    fit_started = time.perf_counter()
    try:
        if checkpoint_args_enabled and step_checkpoint_dir is not None:
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                evaluator=timed_evaluator,
                epochs=args.epochs,
                warmup_steps=warmup_steps,
                optimizer_params={"lr": args.lr},
                output_path=str(output_dir),
                save_best_model=evaluator is not None,
                use_amp=args.fp16,
                show_progress_bar=True,
                checkpoint_path=str(step_checkpoint_dir),
                checkpoint_save_steps=int(steps_per_epoch),
                checkpoint_save_total_limit=int(args.epochs),
            )
        else:
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                evaluator=timed_evaluator,
                epochs=args.epochs,
                warmup_steps=warmup_steps,
                optimizer_params={"lr": args.lr},
                output_path=str(output_dir),
                save_best_model=evaluator is not None,
                use_amp=args.fp16,
                show_progress_bar=True,
            )
    finally:
        fit_elapsed = time.perf_counter() - fit_started
        if full_runtime_metrics is not None:
            full_runtime_metrics["fit_total_seconds"] = fit_elapsed

        if original_model_save is not None:
            model.save = original_model_save  # type: ignore[assignment]
        if original_tensor_backward is not None:
            torch.Tensor.backward = original_tensor_backward  # type: ignore[assignment]
        if original_loss_forward is not None:
            train_loss.forward = original_loss_forward  # type: ignore[assignment]
        if original_smart_batching_collate is not None:
            model.smart_batching_collate = original_smart_batching_collate  # type: ignore[assignment]
        if original_tokenize is not None:
            model.tokenize = original_tokenize  # type: ignore[assignment]
    post_fit_processing_started = time.perf_counter()
    epoch_checkpoint_paths: list[str] = []
    if args.save_each_epoch and step_checkpoint_dir is not None and epoch_checkpoint_dir is not None:
        epoch_checkpoint_paths = normalize_step_checkpoints_to_epochs(step_checkpoint_dir, epoch_checkpoint_dir)
        if step_checkpoint_dir.exists():
            shutil.rmtree(step_checkpoint_dir)

    metadata = {
        "run_id": run_id,
        "rule_hash": args.rule_hash.strip(),
        "train_file": str(train_path),
        "base_model": args.base_model,
        "output_dir": str(output_dir),
        "prefix_mode": "no_prefix",
        "source_of_prefix_setting": "dense_only_bge_m3_default",
        "dense_only_bge_m3_default_applied": True,
        "legacy_prefix_experiment_active": False,
        "device": device,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.lr),
        "warmup_ratio": float(args.warmup_ratio),
        "max_length": int(args.max_length),
        "dev_ratio": float(args.dev_ratio),
        "seed": int(args.seed),
        "fp16": bool(args.fp16),
        "hard_negative_mode": args.hard_negative_mode,
        "hard_negative_selection": args.hard_negative_selection,
        "num_hard_negatives": int(args.num_hard_negatives),
        "model_selection_metric": args.model_selection_metric,
        "hard_negative_stats": hard_negative_stats,
        "sampler_mode": "UniquePositiveBatchSampler",
        "sampler_query_positive_union_aware": True,
        "train_queries_with_multiple_positives": sum(1 for values in query_positive_union.values() if len(values) > 1),
        "save_each_epoch": bool(args.save_each_epoch),
        "steps_per_epoch": int(steps_per_epoch),
        "total_pairs": len(all_records),
        "train_pairs": len(train_records),
        "dev_pairs": len(dev_records),
        "epoch_checkpoints": epoch_checkpoint_paths,
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if full_runtime_metrics is not None:
        full_runtime_metrics["post_fit_post_processing_seconds"] = time.perf_counter() - post_fit_processing_started

        tokenization_total = metric_get_float(full_runtime_metrics, "tokenization_feature_prep_seconds_total", 0.0)
        dataloader_collation_total = metric_get_float(full_runtime_metrics, "dataloader_batch_collation_seconds_total", 0.0)
        dataloader_collation_exclusive = max(0.0, dataloader_collation_total - tokenization_total)
        forward_total = metric_get_float(full_runtime_metrics, "forward_seconds_total", 0.0)
        backward_total = metric_get_float(full_runtime_metrics, "backward_seconds_total", 0.0)
        forward_backward_total = forward_total + backward_total
        epoch_evaluation_total = metric_get_float(full_runtime_metrics, "epoch_evaluation_seconds_total", 0.0)
        checkpoint_saving_total = metric_get_float(full_runtime_metrics, "checkpoint_saving_seconds_total", 0.0)
        final_model_save_total = metric_get_float(full_runtime_metrics, "final_model_save_seconds_total", 0.0)
        post_fit_post_processing_total = metric_get_float(full_runtime_metrics, "post_fit_post_processing_seconds", 0.0)
        final_save_post_processing_total = final_model_save_total + post_fit_post_processing_total

        full_runtime_payload = {
            "profile_name": "train_full_runtime_profile",
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "run_id": run_id,
            "train_file": str(train_path),
            "output_dir": str(output_dir),
            "hard_negative_mode": args.hard_negative_mode,
            "hard_negative_selection": args.hard_negative_selection,
            "seed": int(args.seed),
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "device": device,
            "runtime_blocks_seconds": {
                "train_setup": metric_get_float(full_runtime_metrics, "train_setup_seconds", 0.0),
                "dataloader_batch_collation": dataloader_collation_exclusive,
                "tokenization_feature_prep": tokenization_total,
                "forward_backward": forward_backward_total,
                "epoch_evaluation": epoch_evaluation_total,
                "checkpoint_saving": checkpoint_saving_total,
                "final_save_post_processing": final_save_post_processing_total,
            },
            "raw_timing_seconds": {
                "fit_total": metric_get_float(full_runtime_metrics, "fit_total_seconds", 0.0),
                "dataloader_batch_collation_total_inclusive": dataloader_collation_total,
                "dataloader_batch_collation_exclusive": dataloader_collation_exclusive,
                "tokenization_feature_prep_total": tokenization_total,
                "forward_total": forward_total,
                "backward_total": backward_total,
                "epoch_evaluation_total": epoch_evaluation_total,
                "checkpoint_saving_total": checkpoint_saving_total,
                "final_model_save_total": final_model_save_total,
                "post_fit_post_processing_total": post_fit_post_processing_total,
            },
            "counts": {
                "dataloader_batches_collated": metric_get_int(full_runtime_metrics, "dataloader_batches_collated", 0),
                "tokenization_calls": metric_get_int(full_runtime_metrics, "tokenization_calls", 0),
                "forward_calls": metric_get_int(full_runtime_metrics, "forward_calls", 0),
                "backward_calls": metric_get_int(full_runtime_metrics, "backward_calls", 0),
                "epoch_evaluation_calls": metric_get_int(full_runtime_metrics, "epoch_evaluation_calls", 0),
                "checkpoint_save_calls": metric_get_int(full_runtime_metrics, "checkpoint_save_calls", 0),
                "final_model_save_calls": metric_get_int(full_runtime_metrics, "final_model_save_calls", 0),
            },
            "save_events": full_runtime_metrics.get("model_save_events", []),
        }

        full_runtime_profile_path = Path(args.full_runtime_profile_out).expanduser().resolve()
        full_runtime_profile_path.parent.mkdir(parents=True, exist_ok=True)
        full_runtime_profile_path.write_text(
            json.dumps(full_runtime_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Full runtime profile gespeichert: {full_runtime_profile_path}")

    print(f"Training abgeschlossen. Modell gespeichert unter: {output_dir}")
    if epoch_checkpoint_paths:
        print(f"Epochen-Checkpoints gespeichert unter: {epoch_checkpoint_dir}")
    print(f"Run-Metadaten: {metadata_path}")


if __name__ == "__main__":
    main()
