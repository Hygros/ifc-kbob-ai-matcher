import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


def normalize(value: str) -> str:
    return value.strip().casefold()


def parse_pipe_tokens(value: str) -> list[str]:
    return [token.strip() for token in value.split("|") if token.strip()]


def parse_bool(value: str) -> bool:
    return normalize(value) in {"1", "true", "yes", "y"}


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class QueryHardNegatives:
    query: str
    positives: list[str]
    hard_negatives: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract hard negatives from evaluation details CSV and write JSONL for training prep."
    )
    parser.add_argument("--details-file", required=True, help="Path to details_*.csv from evaluation output.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--max-expected-rank",
        type=int,
        default=10,
        help="Keep only top1-wrong rows with expected_rank <= this value (0 keeps all).",
    )
    parser.add_argument(
        "--max-negatives-per-query",
        type=int,
        default=3,
        help="Maximum number of mined hard negatives per query.",
    )
    parser.add_argument(
        "--min-predicted-score",
        type=float,
        default=0.0,
        help="Minimum predicted_top1_score for a row to be considered.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Optional cap on number of output queries (0 means no cap).",
    )
    return parser.parse_args()


def mine_hard_negatives(
    details_file: Path,
    max_expected_rank: int,
    max_negatives_per_query: int,
    min_predicted_score: float,
) -> tuple[list[QueryHardNegatives], dict[str, int]]:
    if not details_file.is_file():
        raise FileNotFoundError(f"Details CSV not found: {details_file}")
    if max_negatives_per_query <= 0:
        raise ValueError("--max-negatives-per-query must be > 0.")

    rows_total = 0
    wrong_rows = 0
    kept_rows = 0

    by_query: dict[str, QueryHardNegatives] = {}
    negatives_seen: dict[str, set[str]] = {}
    positives_seen: dict[str, set[str]] = {}

    with details_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "query",
            "relevant_resolved",
            "predicted_top1_score",
            "expected_rank",
            "top1_correct",
            "top10_materials",
        }
        missing = required_columns.difference(set(reader.fieldnames or []))
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Details CSV missing required columns: {missing_text}")

        for row in reader:
            rows_total += 1

            if parse_bool(row.get("top1_correct", "False")):
                continue
            wrong_rows += 1

            score = parse_float(row.get("predicted_top1_score", "0"), default=0.0)
            if score < min_predicted_score:
                continue

            expected_rank = parse_int(row.get("expected_rank", "0"), default=0)
            if max_expected_rank > 0 and expected_rank > max_expected_rank:
                continue

            query = str(row.get("query", "")).strip()
            if not query:
                continue

            relevant_tokens = parse_pipe_tokens(str(row.get("relevant_resolved", "")))
            relevant_set = {normalize(token) for token in relevant_tokens}

            top10_materials = parse_pipe_tokens(str(row.get("top10_materials", "")))
            if not top10_materials:
                continue

            query_entry = by_query.get(query)
            if query_entry is None:
                query_entry = QueryHardNegatives(query=query, positives=[], hard_negatives=[])
                by_query[query] = query_entry
                negatives_seen[query] = set()
                positives_seen[query] = set()

            for positive in relevant_tokens:
                positive_key = normalize(positive)
                if positive_key and positive_key not in positives_seen[query]:
                    positives_seen[query].add(positive_key)
                    query_entry.positives.append(positive)

            for candidate in top10_materials:
                if len(query_entry.hard_negatives) >= max_negatives_per_query:
                    break
                candidate_key = normalize(candidate)
                if not candidate_key:
                    continue
                if candidate_key in relevant_set:
                    continue
                if candidate_key in negatives_seen[query]:
                    continue
                negatives_seen[query].add(candidate_key)
                query_entry.hard_negatives.append(candidate)

            if query_entry.hard_negatives:
                kept_rows += 1

    records = [record for record in by_query.values() if record.hard_negatives]
    records.sort(key=lambda record: record.query.casefold())

    stats = {
        "rows_total": rows_total,
        "wrong_rows": wrong_rows,
        "kept_rows": kept_rows,
        "queries_with_hard_negatives": len(records),
    }
    return records, stats


def write_jsonl(records: list[QueryHardNegatives], out_file: Path, source_file_name: str) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as handle:
        for record in records:
            row = {
                "query": record.query,
                "positives": record.positives,
                "hard_negatives": record.hard_negatives,
                "source_details_file": source_file_name,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    details_file = Path(args.details_file).expanduser().resolve()
    out_file = Path(args.out).expanduser().resolve()

    records, stats = mine_hard_negatives(
        details_file=details_file,
        max_expected_rank=args.max_expected_rank,
        max_negatives_per_query=args.max_negatives_per_query,
        min_predicted_score=args.min_predicted_score,
    )

    if args.max_queries > 0:
        records = records[: args.max_queries]

    write_jsonl(records=records, out_file=out_file, source_file_name=details_file.name)

    print(f"Rows total: {stats['rows_total']}")
    print(f"Rows top1-wrong: {stats['wrong_rows']}")
    print(f"Rows kept after filters: {stats['kept_rows']}")
    print(f"Queries with hard negatives: {len(records)}")
    print(f"Output: {out_file}")


if __name__ == "__main__":
    main()
