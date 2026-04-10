import argparse
import importlib.util
import json
import statistics
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

warnings.warn(
    "[DEPRECATED] run_hn_trace_report.py ist als deprecated markiert und wird in einem "
    "zukuenftigen Release entfernt.",
    DeprecationWarning,
    stacklevel=2,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT_PATH = PROJECT_ROOT / "Training" / "train_bge_m3.py"


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


def coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def load_hn_input(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"HN-Input nicht gefunden: {path}")

    by_query: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Ungueltiges JSON in HN-Input Zeile {line_no}: {exc}") from exc

            query = str(row.get("query", "")).strip()
            if not query:
                continue

            query_key = normalize_text(query)
            positives = coerce_str_list(row.get("positives", []))
            hard_negatives = coerce_str_list(row.get("hard_negatives", []))

            by_query[query_key] = {
                "query": query,
                "positives": positives,
                "hard_negatives": hard_negatives,
            }

    return by_query


def load_pairs_by_query(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Pairs-Datei nicht gefunden: {path}")

    by_query: dict[str, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Ungueltiges JSON in Pairs-Datei Zeile {line_no}: {exc}") from exc

            query = str(row.get("query", "")).strip()
            positive = str(row.get("positive", "")).strip()
            if not query or not positive:
                continue

            query_key = normalize_text(query)
            payload = by_query.setdefault(
                query_key,
                {
                    "query": query,
                    "positives": set(),
                    "hard_negatives": set(),
                    "preselected_hard_negatives": set(),
                    "rows": 0,
                },
            )

            payload["rows"] += 1
            payload["positives"].add(positive)

            for value in coerce_str_list(row.get("hard_negatives", [])):
                payload["hard_negatives"].add(value)

            for value in coerce_str_list(row.get("preselected_hard_negatives", [])):
                payload["preselected_hard_negatives"].add(value)

            preselected_legacy = str(row.get("preselected_hard_negative", "")).strip()
            if preselected_legacy:
                payload["preselected_hard_negatives"].add(preselected_legacy)

    return by_query


def ordered_query_candidates(
    query_file: Path | None,
    hn_input_by_query: dict[str, dict[str, list[str]]],
) -> list[str]:
    if query_file is None:
        return sorted(hn_input_by_query.keys())

    candidates: list[str] = []
    seen: set[str] = set()
    for query in load_non_empty_lines(query_file):
        key = normalize_text(query)
        if key in hn_input_by_query and key not in seen:
            seen.add(key)
            candidates.append(key)
    return candidates


def build_trainer_input_map(
    final_train_file: Path,
    dev_ratio: float,
    seed: int,
    hard_negative_selection: str,
    num_hard_negatives: int,
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    train_module = load_module(TRAIN_SCRIPT_PATH, "train_bge_m3_hn_trace")

    all_records = train_module.read_records(final_train_file)
    train_records, _ = train_module.split_records(all_records, dev_ratio=dev_ratio, seed=seed)

    train_query_total = len({normalize_text(record.query) for record in train_records})
    train_query_with_hn_in_records = len(
        {
            normalize_text(record.query)
            for record in train_records
            if getattr(record, "hard_negatives", ())
        }
    )

    strict_error = ""
    strict_examples: list[Any] = []
    try:
        strict_examples, strict_stats = train_module.build_train_examples(
            train_records=train_records,
            hard_negative_mode="strict",
            hard_negative_selection=hard_negative_selection,
            seed=seed,
            num_hard_negatives=num_hard_negatives,
        )
        strict_usable = True
    except ValueError as exc:
        strict_stats = {}
        strict_error = str(exc)
        strict_usable = False

    query_to_hn: dict[str, set[str]] = defaultdict(set)
    for example in strict_examples:
        texts = list(getattr(example, "texts", []))
        if len(texts) < 3:
            continue
        query = normalize_text(texts[0])
        hard_negative = str(texts[2]).strip()
        if query and hard_negative:
            query_to_hn[query].add(hard_negative)

    stats = {
        "strict_usable": strict_usable,
        "strict_error": strict_error,
        "strict_examples_total": len(strict_examples),
        "strict_stats": strict_stats,
        "train_queries_total": train_query_total,
        "train_queries_with_hn_in_records": train_query_with_hn_in_records,
        "train_queries_with_hn_in_trainer_input": len(query_to_hn),
    }
    return dict(query_to_hn), stats


def as_sorted_list(values: set[str] | list[str]) -> list[str]:
    if isinstance(values, set):
        return sorted(values)
    return sorted(set(values))


def median_int(values: list[int]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Erzeugt End-to-End Trace-Report fuer Hard-Negatives.")
    parser.add_argument("--hn-input-file", required=True, help="JSONL aus mine_hard_negatives.py")
    parser.add_argument("--prepare-pairs-file", required=True, help="Pairs-JSONL nach prepare step")
    parser.add_argument("--final-train-file", required=True, help="Finales Trainings-JSONL vor dem Trainer")
    parser.add_argument("--query-file", default="", help="Optionales Query-TXT fuer reproduzierbare Stichprobe")
    parser.add_argument("--sample-size", type=int, default=10, help="Anzahl Queries in der Stichprobe")
    parser.add_argument("--seed", type=int, default=42, help="Seed fuer Split/Trainer-Simulation")
    parser.add_argument("--dev-ratio", type=float, default=0.1, help="Dev-Ratio analog Training")
    parser.add_argument(
        "--hard-negative-selection",
        choices=["first", "random", "random_preselected"],
        default="random",
        help="Auswahlstrategie analog Trainer.",
    )
    parser.add_argument("--num-hard-negatives", type=int, default=1, help="K analog Trainer.")
    parser.add_argument("--report-file", required=True, help="Output JSON fuer Trace-Report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_hard_negatives <= 0:
        raise ValueError("--num-hard-negatives muss > 0 sein.")

    hn_input_file = resolve_path(args.hn_input_file)
    prepare_pairs_file = resolve_path(args.prepare_pairs_file)
    final_train_file = resolve_path(args.final_train_file)
    query_file = resolve_path(args.query_file) if args.query_file.strip() else None
    report_file = resolve_path(args.report_file)

    if args.sample_size <= 0:
        raise ValueError("--sample-size muss > 0 sein.")

    hn_input_by_query = load_hn_input(hn_input_file)
    prepare_by_query = load_pairs_by_query(prepare_pairs_file)
    final_by_query = load_pairs_by_query(final_train_file)

    trainer_input_map, trainer_stats = build_trainer_input_map(
        final_train_file=final_train_file,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
        hard_negative_selection=args.hard_negative_selection,
        num_hard_negatives=args.num_hard_negatives,
    )

    candidates = ordered_query_candidates(query_file=query_file, hn_input_by_query=hn_input_by_query)
    if not candidates:
        raise ValueError("Keine ueberschneidenden Queries fuer Trace gefunden.")

    selected_queries = candidates[: args.sample_size]

    trace_rows: list[dict[str, Any]] = []
    for query_key in selected_queries:
        hn_payload = hn_input_by_query.get(query_key, {"query": query_key, "positives": [], "hard_negatives": []})
        prepare_payload = prepare_by_query.get(query_key)
        final_payload = final_by_query.get(query_key)
        trainer_hn = as_sorted_list(trainer_input_map.get(query_key, set()))

        positives_after_prepare = as_sorted_list(prepare_payload["positives"]) if prepare_payload else []
        hn_after_prepare = as_sorted_list(prepare_payload["hard_negatives"]) if prepare_payload else []

        positives_final = as_sorted_list(final_payload["positives"]) if final_payload else []
        hn_final = as_sorted_list(final_payload["hard_negatives"]) if final_payload else []
        preselected_final = as_sorted_list(final_payload["preselected_hard_negatives"]) if final_payload else []

        strict_usable = bool(hn_final and trainer_hn and trainer_stats.get("strict_usable", False))
        reason = ""
        if not strict_usable:
            if not prepare_payload:
                reason = "query_missing_after_prepare"
            elif not hn_after_prepare:
                reason = "no_hard_negatives_after_prepare"
            elif not final_payload:
                reason = "query_missing_in_final_train_artifact"
            elif not hn_final:
                reason = "no_hard_negatives_in_final_train_artifact"
            elif not trainer_stats.get("strict_usable", False):
                reason = f"strict_mode_not_usable: {trainer_stats.get('strict_error', '')}".strip()
            elif not trainer_hn:
                reason = "query_not_present_in_trainer_input"
            else:
                reason = "unknown"

        trace_rows.append(
            {
                "query": hn_payload.get("query", query_key),
                "positives_in_hn_input": hn_payload.get("positives", []),
                "hard_negatives_in_hn_input": hn_payload.get("hard_negatives", []),
                "positives_after_prepare": positives_after_prepare,
                "hard_negatives_after_prepare": hn_after_prepare,
                "positives_in_final_train_record": positives_final,
                "hard_negatives_in_final_train_record": hn_final,
                "preselected_hard_negatives_in_final_train_record": preselected_final,
                "hard_negatives_in_trainerinput": trainer_hn,
                "strict_usable": strict_usable,
                "reason_if_not_usable": reason,
            }
        )

    hn_input_queries = set(hn_input_by_query.keys())
    prepare_queries_with_hn = {key for key, value in prepare_by_query.items() if value["hard_negatives"]}
    final_queries_with_hn = {key for key, value in final_by_query.items() if value["hard_negatives"]}
    trainer_queries_with_hn = set(trainer_input_map.keys())

    final_hn_counts = [len(value["hard_negatives"]) for value in final_by_query.values() if value["hard_negatives"]]
    final_preselected_counts = [
        len(value.get("preselected_hard_negatives", set()))
        for value in final_by_query.values()
        if value.get("preselected_hard_negatives")
    ]

    total_queries = len(final_by_query)
    queries_with_hn_in_final = len(final_queries_with_hn)
    queries_without_hn_rate = 1.0
    if total_queries > 0:
        queries_without_hn_rate = (total_queries - queries_with_hn_in_final) / total_queries

    report = {
        "meta": {
            "hn_input_file": str(hn_input_file),
            "prepare_pairs_file": str(prepare_pairs_file),
            "final_train_file": str(final_train_file),
            "query_file": str(query_file) if query_file else "",
            "sample_size": args.sample_size,
            "seed": args.seed,
            "dev_ratio": args.dev_ratio,
            "hard_negative_selection": args.hard_negative_selection,
            "num_hard_negatives": args.num_hard_negatives,
        },
        "metrics": {
            "total_queries": total_queries,
            "queries_with_hn_input": len(hn_input_queries),
            "queries_with_hn_after_prepare": len(prepare_queries_with_hn),
            "queries_with_hn_in_final_train_artifact": queries_with_hn_in_final,
            "queries_with_hn_in_trainer_input": len(trainer_queries_with_hn),
            "queries_without_hn_rate": queries_without_hn_rate,
            "strict_hn_usable": bool(trainer_stats.get("strict_usable", False) and queries_with_hn_in_final > 0),
            "median_hn_per_query": median_int(final_hn_counts),
            "median_preselected_per_query": median_int(final_preselected_counts),
            "min_hn_per_query": min(final_hn_counts) if final_hn_counts else 0,
            "max_hn_per_query": max(final_hn_counts) if final_hn_counts else 0,
            "max_preselected_per_query": max(final_preselected_counts) if final_preselected_counts else 0,
            "queries_with_hn_lost_after_prepare": len(hn_input_queries - prepare_queries_with_hn),
            "queries_with_hn_lost_before_final_artifact": len(prepare_queries_with_hn - final_queries_with_hn),
            "queries_with_hn_lost_before_trainer_input": len(final_queries_with_hn - trainer_queries_with_hn),
        },
        "trainer_stage": trainer_stats,
        "trace_queries": trace_rows,
    }

    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"HN trace report: {report_file}")
    print(f"Traced queries: {len(trace_rows)}")
    print(f"strict_hn_usable: {report['metrics']['strict_hn_usable']}")


if __name__ == "__main__":
    main()
