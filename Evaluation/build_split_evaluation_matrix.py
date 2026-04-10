from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "Evaluation" / "outputs" / "results"

VALID_QUERY_CLASSES = {"eindeutig", "mehrdeutig", "ungenau", "fraglich"}

STRENGTH_PATTERN = re.compile(r"\b(?:c\d{1,2}/\d{1,2}|s\d{3}(?:j[0-9r]+)?)\b", re.IGNORECASE)
RULE_NPK_PATTERN = re.compile(r"\bnpk\b", re.IGNORECASE)
RULE_STEEL_ALIAS_PATTERN = re.compile(r"\bs(?:235|355)(?:jr|j0)\b", re.IGNORECASE)
RULE_PTFE_PATTERN = re.compile(r"\b(?:ptfe|teflon|polytetrafluoroethylene)\b", re.IGNORECASE)
RULE_PAVEMENT_PATTERN = re.compile(r"\b(?:asphaltbelag|bitumenbelag|pavement|wearing)\b", re.IGNORECASE)
RULE_AGGREGATE_PATTERN = re.compile(r"\b(?:aggregate|kies|schotter|naturstein)\b", re.IGNORECASE)


@dataclass
class Aggregation:
    cases: int = 0
    hit1_sum: float = 0.0
    hit10_sum: float = 0.0
    hit20_sum: float = 0.0
    hit30_sum: float = 0.0
    hit50_sum: float = 0.0
    mrr10_sum: float = 0.0
    map10_sum: float = 0.0
    ndcg10_sum: float = 0.0
    recall10_sum: float = 0.0
    expected_rank_sum: float = 0.0
    top1_error_count: int = 0


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_bool(value: Any) -> bool:
    return normalize_text(str(value)) in {"1", "true", "yes", "y"}


def parse_pipe_list(value: Any) -> list[str]:
    raw = str(value).strip()
    if not raw:
        return []
    return [token.strip() for token in raw.split("|") if token.strip()]


def resolve_query_label_from_env() -> str:
    query_file_env = os.environ.get("SBERT_QUERY_FILE", "").strip()
    if not query_file_env:
        return "latest"

    base_name = Path(query_file_env).stem or "latest"
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", base_name).strip("._-")
    return safe or "latest"


def resolve_cross_encoder_label_from_env() -> str:
    model_id = os.environ.get("SBERT_CROSS_ENCODER_MODEL", "").strip()
    if not model_id:
        return "no-reranker"
    short = model_id.split("/")[-1]
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", short).strip("._-")
    return safe or "reranker"


def resolve_details_file(details_file_arg: str) -> Path:
    if details_file_arg.strip():
        path = Path(details_file_arg.strip())
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Details CSV nicht gefunden: {path}")
        return path

    query_label = resolve_query_label_from_env()
    ce_label = resolve_cross_encoder_label_from_env()
    label = f"{query_label}_{ce_label}"

    candidates = [
        RESULTS_DIR / f"details_{label}.csv",
        RESULTS_DIR / "details.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Keine Details CSV gefunden. Gesucht in:\n{searched}")


def infer_query_class(query: str, relevant_count: int, query_class_map: dict[str, str]) -> str:
    key = normalize_text(query)
    mapped = query_class_map.get(key, "")
    if mapped in VALID_QUERY_CLASSES:
        return mapped

    query_norm = normalize_text(query)
    if "?" in query or " fraglich" in query_norm or " uncertain" in query_norm:
        return "fraglich"
    if "ca." in query_norm or " circa " in query_norm or "ungefaehr" in query_norm:
        return "ungenau"
    if relevant_count > 1:
        return "mehrdeutig"
    return "eindeutig"


def infer_source_rule(query: str) -> str:
    if RULE_NPK_PATTERN.search(query):
        return "rule_npk"
    if RULE_STEEL_ALIAS_PATTERN.search(query):
        return "rule_steel_grade_alias"
    if RULE_PTFE_PATTERN.search(query):
        return "rule_ptfe_alias"
    if RULE_PAVEMENT_PATTERN.search(query):
        return "rule_pavement_alias"
    if RULE_AGGREGATE_PATTERN.search(query):
        return "rule_aggregate_alias"
    return "rule_base"


def infer_material_family(query: str) -> str:
    text = normalize_text(query)
    if "stahlbeton" in text:
        return "stahlbeton"
    if "beton" in text:
        return "beton"
    if "stahl" in text:
        return "stahl"
    if "holz" in text:
        return "holz"
    if "asphalt" in text or "bitumen" in text:
        return "asphalt_bitumen"
    if "kunststoff" in text or "ptfe" in text or "teflon" in text:
        return "kunststoff"
    if "mauerwerk" in text or "stein" in text:
        return "stein_mauerwerk"
    return "other"


def infer_casting_method(query: str) -> str:
    text = normalize_text(query)
    has_insitu = "insitu" in text
    has_precast = "precast" in text
    if has_insitu and has_precast:
        return "BOTH"
    if has_insitu:
        return "INSITU"
    if has_precast:
        return "PRECAST"
    return "UNKNOWN"


def infer_strength_presence(query: str) -> str:
    return "present" if STRENGTH_PATTERN.search(query or "") else "absent"


def safe_average(total: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return total / count


def load_query_class_map(path_value: str) -> dict[str, str]:
    if not path_value.strip():
        return {}

    path = Path(path_value.strip())
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Query-Class-Map nicht gefunden: {path}")

    mapping: dict[str, str] = {}
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                query = str(row.get("query", "")).strip()
                query_class = normalize_text(str(row.get("query_class", "")))
                if query and query_class in VALID_QUERY_CLASSES:
                    mapping[normalize_text(query)] = query_class
                elif query and query_class:
                    raise ValueError(f"Ungueltige query_class in {path} Zeile {line_no}: {query_class}")
        return mapping

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "query" not in reader.fieldnames or "query_class" not in reader.fieldnames:
            raise ValueError("Query-Class-CSV muss 'query' und 'query_class' Spalten enthalten.")

        for row in reader:
            query = str(row.get("query", "")).strip()
            query_class = normalize_text(str(row.get("query_class", "")))
            if not query:
                continue
            if query_class not in VALID_QUERY_CLASSES:
                raise ValueError(f"Ungueltige query_class in {path}: {query_class}")
            mapping[normalize_text(query)] = query_class

    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Erzeugt Split-Metrikmatrix aus details_*.csv nach Query-Facetten."
    )
    parser.add_argument("--details-file", default="", help="Details CSV (default: auto via Env/Latest).")
    parser.add_argument(
        "--query-class-map-file",
        default="",
        help="Optionales CSV/JSONL fuer query->query_class Zuordnung.",
    )
    parser.add_argument("--out-csv", default="", help="Output CSV fuer Split-Matrix.")
    parser.add_argument("--out-json", default="", help="Output JSON fuer Split-Matrix-Meta.")
    parser.add_argument("--annotated-details-csv", default="", help="Optional: details inkl. Split-Spalten.")
    parser.add_argument("--min-cases", type=int, default=1, help="Mindestanzahl Cases pro Split-Zeile.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_cases <= 0:
        raise ValueError("--min-cases muss > 0 sein.")

    details_file = resolve_details_file(args.details_file)
    query_class_map = load_query_class_map(args.query_class_map_file)

    label = details_file.stem
    if label.startswith("details_"):
        label = label[len("details_") :]
    if not label:
        label = "latest"

    out_csv = Path(args.out_csv) if args.out_csv.strip() else details_file.parent / f"split_eval_matrix_{label}.csv"
    if not out_csv.is_absolute():
        out_csv = PROJECT_ROOT / out_csv
    out_csv = out_csv.resolve()

    out_json = Path(args.out_json) if args.out_json.strip() else details_file.parent / f"split_eval_matrix_{label}.json"
    if not out_json.is_absolute():
        out_json = PROJECT_ROOT / out_json
    out_json = out_json.resolve()

    annotated_details_csv: Path | None = None
    if args.annotated_details_csv.strip():
        annotated_details_csv = Path(args.annotated_details_csv)
        if not annotated_details_csv.is_absolute():
            annotated_details_csv = PROJECT_ROOT / annotated_details_csv
        annotated_details_csv = annotated_details_csv.resolve()

    with details_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Details CSV enthaelt keine Zeilen: {details_file}")

    axis_names = [
        "query_class",
        "source_rule",
        "material_family",
        "casting_method",
        "strength_presence",
    ]

    aggregations: dict[tuple[str, str, str, str, str], Aggregation] = defaultdict(Aggregation)
    annotated_rows: list[dict[str, Any]] = []

    for row in rows:
        query = str(row.get("query", ""))
        relevant_tokens = parse_pipe_list(row.get("relevant_resolved", ""))
        relevant_count = parse_int(row.get("relevant_count", len(relevant_tokens)), default=len(relevant_tokens))

        split_values = {
            "query_class": infer_query_class(query, relevant_count, query_class_map),
            "source_rule": infer_source_rule(query),
            "material_family": infer_material_family(query),
            "casting_method": infer_casting_method(query),
            "strength_presence": infer_strength_presence(query),
        }

        model_name = str(row.get("model", ""))
        pipeline_variant = str(row.get("pipeline_variant", "baseline") or "baseline")
        cross_encoder_model = str(row.get("cross_encoder_model", "-") or "-")

        hit1 = parse_float(row.get("hit@1", 0.0), default=0.0)
        hit10 = parse_float(row.get("hit@10", 0.0), default=0.0)
        hit20 = parse_float(row.get("hit@20", 0.0), default=0.0)
        hit30 = parse_float(row.get("hit@30", 0.0), default=0.0)
        hit50 = parse_float(row.get("hit@50", 0.0), default=0.0)
        mrr10 = parse_float(row.get("mrr@10", 0.0), default=0.0)
        map10 = parse_float(row.get("map@10", 0.0), default=0.0)
        ndcg10 = parse_float(row.get("ndcg@10", 0.0), default=0.0)
        recall10 = parse_float(row.get("recall@10", 0.0), default=0.0)
        expected_rank = parse_float(row.get("expected_rank", 0.0), default=0.0)
        top1_correct = parse_bool(row.get("top1_correct", False))

        for axis_name in axis_names:
            axis_value = split_values[axis_name]
            key = (model_name, pipeline_variant, cross_encoder_model, axis_name, axis_value)
            agg = aggregations[key]
            agg.cases += 1
            agg.hit1_sum += hit1
            agg.hit10_sum += hit10
            agg.hit20_sum += hit20
            agg.hit30_sum += hit30
            agg.hit50_sum += hit50
            agg.mrr10_sum += mrr10
            agg.map10_sum += map10
            agg.ndcg10_sum += ndcg10
            agg.recall10_sum += recall10
            agg.expected_rank_sum += expected_rank
            if not top1_correct:
                agg.top1_error_count += 1

        annotated_row = dict(row)
        annotated_row.update(split_values)
        annotated_rows.append(annotated_row)

    matrix_rows: list[dict[str, Any]] = []
    for (model_name, pipeline_variant, cross_encoder_model, axis_name, axis_value), agg in aggregations.items():
        if agg.cases < args.min_cases:
            continue
        matrix_rows.append(
            {
                "model": model_name,
                "pipeline_variant": pipeline_variant,
                "cross_encoder_model": cross_encoder_model,
                "split_axis": axis_name,
                "split_value": axis_value,
                "cases": agg.cases,
                "hit@1": f"{safe_average(agg.hit1_sum, agg.cases):.6f}",
                "hit@10": f"{safe_average(agg.hit10_sum, agg.cases):.6f}",
                "hit@20": f"{safe_average(agg.hit20_sum, agg.cases):.6f}",
                "hit@30": f"{safe_average(agg.hit30_sum, agg.cases):.6f}",
                "hit@50": f"{safe_average(agg.hit50_sum, agg.cases):.6f}",
                "mrr@10": f"{safe_average(agg.mrr10_sum, agg.cases):.6f}",
                "map@10": f"{safe_average(agg.map10_sum, agg.cases):.6f}",
                "ndcg@10": f"{safe_average(agg.ndcg10_sum, agg.cases):.6f}",
                "recall@10": f"{safe_average(agg.recall10_sum, agg.cases):.6f}",
                "avg_expected_rank": f"{safe_average(agg.expected_rank_sum, agg.cases):.6f}",
                "top1_error_count": agg.top1_error_count,
                "top1_error_rate": f"{safe_average(float(agg.top1_error_count), agg.cases):.6f}",
            }
        )

    matrix_rows.sort(
        key=lambda item: (
            item["split_axis"],
            item["pipeline_variant"],
            item["model"],
            -int(item["cases"]),
            item["split_value"],
        )
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "model",
            "pipeline_variant",
            "cross_encoder_model",
            "split_axis",
            "split_value",
            "cases",
            "hit@1",
            "hit@10",
            "hit@20",
            "hit@30",
            "hit@50",
            "mrr@10",
            "map@10",
            "ndcg@10",
            "recall@10",
            "avg_expected_rank",
            "top1_error_count",
            "top1_error_rate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matrix_rows)

    if annotated_details_csv is not None:
        annotated_details_csv.parent.mkdir(parents=True, exist_ok=True)
        with annotated_details_csv.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = list(annotated_rows[0].keys()) if annotated_rows else []
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(annotated_rows)

    split_counts: dict[str, int] = defaultdict(int)
    for row in matrix_rows:
        split_counts[str(row["split_axis"])] += 1

    payload = {
        "details_file": str(details_file),
        "out_csv": str(out_csv),
        "annotated_details_csv": str(annotated_details_csv) if annotated_details_csv else "",
        "rows_input": len(rows),
        "rows_output": len(matrix_rows),
        "min_cases": int(args.min_cases),
        "split_rows_by_axis": dict(split_counts),
        "query_class_map_file": args.query_class_map_file,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Split matrix CSV: {out_csv}")
    print(f"Split matrix JSON: {out_json}")


if __name__ == "__main__":
    main()
