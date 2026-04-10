import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from mine_family_hard_negatives import load_material_families
from mine_family_hard_negatives import mine_family_hard_negatives
from text_normalization import jaccard_similarity
from text_normalization import normalize_material_key
from text_normalization import normalize_text_key
from text_normalization import query_family_key
from text_normalization import query_semantic_tokens


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_QUERY_FILE = (
    PROJECT_ROOT / "Training" / "query_generation" / "generated_queries" / "generated_queries.txt"
)
DEFAULT_MAPPING_FILE = (
    PROJECT_ROOT / "Training" / "query_generation" / "generated_queries" / "mapping_generated_queries.txt"
)
DEFAULT_TAXONOMY = (
    PROJECT_ROOT / "Training" / "query_generation" / "sources" / "material_ökobilanz.txt"
)


def normalize(value: str) -> str:
    return normalize_material_key(value)


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
    parser.add_argument(
        "--cross-query-positive-protection",
        choices=["off", "family", "global"],
        default="family",
        help="Schutzstufe gegen Cross-Query-False-Negatives beim Mining.",
    )
    parser.add_argument(
        "--query-near-jaccard-threshold",
        type=float,
        default=0.60,
        help="Jaccard-Schwelle fuer Query-Naehe in Modus 'family'.",
    )
    parser.add_argument(
        "--family-hard-negatives",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Intra-Family Hard-Negatives automatisch minen und an die Ausgabe anhaengen.",
    )
    parser.add_argument(
        "--query-file",
        default=str(DEFAULT_QUERY_FILE),
        help="Query-TXT fuer Family-HN Mining (Default: generated_queries.txt).",
    )
    parser.add_argument(
        "--mapping-file",
        default=str(DEFAULT_MAPPING_FILE),
        help="Mapping-TXT fuer Family-HN Mining (Default: mapping_generated_queries.txt).",
    )
    parser.add_argument(
        "--material-taxonomy",
        default=str(DEFAULT_TAXONOMY),
        help="material_oekobilanz.txt mit #Family-Headern (Default: %(default)s).",
    )
    parser.add_argument(
        "--family-hn-max-negatives",
        type=int,
        default=5,
        help="Maximale Anzahl Intra-Family HN pro Query (Default: 5).",
    )
    parser.add_argument(
        "--cross-family-extras",
        default=None,
        help=(
            "JSON-String: Positive-Material \u2192 extra HN-Liste aus anderen Familien. "
            "Beispiel: '{\"Gussasphalt\": [\"Dichtungsbahn bitumin\u00f6s\"]}'"
        ),
    )
    parser.add_argument(
        "--include-narrow-wins",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mine HN auch aus knappen Siegen (top1 korrekt, Score-Gap < threshold).",
    )
    parser.add_argument(
        "--score-gap-threshold",
        type=float,
        default=0.03,
        help="Score-Gap Top1-Top2 unter dem Narrow-Win HN extrahiert werden (Default: 0.03).",
    )
    return parser.parse_args()


def mine_hard_negatives(
    details_file: Path,
    max_expected_rank: int,
    max_negatives_per_query: int,
    min_predicted_score: float,
    cross_query_positive_protection: str,
    query_near_jaccard_threshold: float,
    include_narrow_wins: bool = False,
    score_gap_threshold: float = 0.03,
) -> tuple[list[QueryHardNegatives], dict[str, int | float | str]]:
    if not details_file.is_file():
        raise FileNotFoundError(f"Details CSV not found: {details_file}")
    if max_negatives_per_query <= 0:
        raise ValueError("--max-negatives-per-query must be > 0.")

    rows_total = 0
    wrong_rows = 0
    narrow_win_rows = 0
    kept_rows = 0
    dropped_cross_query_positive = 0
    dropped_same_query_positive = 0

    all_rows: list[dict[str, str]] = []

    by_query: dict[str, QueryHardNegatives] = {}
    negatives_seen: dict[str, set[str]] = {}
    positives_seen: dict[str, set[str]] = {}
    query_text_by_key: dict[str, str] = {}
    positives_by_query_key: dict[str, set[str]] = {}

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
        if include_narrow_wins:
            required_columns.add("top10_scores")
        missing = required_columns.difference(set(reader.fieldnames or []))
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Details CSV missing required columns: {missing_text}")

        for row in reader:
            all_rows.append(dict(row))

    for row in all_rows:
        query = str(row.get("query", "")).strip()
        if not query:
            continue
        query_key = normalize_text_key(query)
        if not query_key:
            continue
        query_text_by_key.setdefault(query_key, query)

        relevant_tokens = parse_pipe_tokens(str(row.get("relevant_resolved", "")))
        positives_for_query = positives_by_query_key.setdefault(query_key, set())
        for token in relevant_tokens:
            token_key = normalize(token)
            if token_key:
                positives_for_query.add(token_key)

    all_positive_keys: set[str] = set()
    for values in positives_by_query_key.values():
        all_positive_keys.update(values)

    family_by_query = {query_key: query_family_key(query_text) for query_key, query_text in query_text_by_key.items()}
    semantic_tokens_by_query = {
        query_key: set(query_semantic_tokens(query_text)) for query_key, query_text in query_text_by_key.items()
    }

    blocked_positive_keys_by_query: dict[str, set[str]] = {}
    query_keys = list(query_text_by_key.keys())
    for query_key in query_keys:
        blocked = set(positives_by_query_key.get(query_key, set()))

        if cross_query_positive_protection == "global":
            blocked.update(all_positive_keys)
        elif cross_query_positive_protection == "family":
            query_family = family_by_query.get(query_key, "")
            query_tokens = semantic_tokens_by_query.get(query_key, set())
            for other_query_key in query_keys:
                if other_query_key == query_key:
                    continue

                same_family = bool(query_family) and query_family == family_by_query.get(other_query_key, "")
                is_near_query = False
                if not same_family:
                    other_tokens = semantic_tokens_by_query.get(other_query_key, set())
                    is_near_query = jaccard_similarity(query_tokens, other_tokens) >= query_near_jaccard_threshold

                if same_family or is_near_query:
                    blocked.update(positives_by_query_key.get(other_query_key, set()))

        blocked_positive_keys_by_query[query_key] = blocked

    for row in all_rows:
            rows_total += 1

            is_correct = parse_bool(row.get("top1_correct", "False"))
            is_narrow_win = False

            if is_correct:
                if not include_narrow_wins:
                    continue
                # Check score gap for narrow-win mining
                top10_scores_raw = parse_pipe_tokens(str(row.get("top10_scores", "")))
                if len(top10_scores_raw) >= 2:
                    top1_score = parse_float(top10_scores_raw[0], default=0.0)
                    top2_score = parse_float(top10_scores_raw[1], default=0.0)
                    gap = top1_score - top2_score
                    if gap >= score_gap_threshold or gap < 0:
                        continue
                    is_narrow_win = True
                else:
                    continue

            if not is_narrow_win:
                wrong_rows += 1
            else:
                narrow_win_rows += 1

            score = parse_float(row.get("predicted_top1_score", "0"), default=0.0)
            if score < min_predicted_score:
                continue

            expected_rank = parse_int(row.get("expected_rank", "0"), default=0)
            if max_expected_rank > 0 and expected_rank > max_expected_rank:
                continue

            query = str(row.get("query", "")).strip()
            if not query:
                continue
            query_key = normalize_text_key(query)
            if not query_key:
                continue

            relevant_tokens = parse_pipe_tokens(str(row.get("relevant_resolved", "")))
            relevant_set = {normalize(token) for token in relevant_tokens}
            blocked_set = blocked_positive_keys_by_query.get(query_key, set())

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
                    dropped_same_query_positive += 1
                    continue
                if (
                    cross_query_positive_protection != "off"
                    and candidate_key in blocked_set
                    and candidate_key not in relevant_set
                ):
                    dropped_cross_query_positive += 1
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
        "narrow_win_rows": narrow_win_rows,
        "kept_rows": kept_rows,
        "queries_with_hard_negatives": len(records),
        "dropped_same_query_positive": dropped_same_query_positive,
        "dropped_cross_query_positive": dropped_cross_query_positive,
        "cross_query_positive_protection": cross_query_positive_protection,
        "query_near_jaccard_threshold": query_near_jaccard_threshold,
        "include_narrow_wins": include_narrow_wins,
        "score_gap_threshold": score_gap_threshold,
    }
    return records, stats


def write_jsonl(
    records: list[QueryHardNegatives],
    out_file: Path,
    source_file_name: str,
    extra_records: list[dict] | None = None,
) -> None:
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
        if extra_records:
            for row in extra_records:
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
        cross_query_positive_protection=args.cross_query_positive_protection,
        query_near_jaccard_threshold=args.query_near_jaccard_threshold,
        include_narrow_wins=args.include_narrow_wins,
        score_gap_threshold=args.score_gap_threshold,
    )

    if args.max_queries > 0:
        records = records[: args.max_queries]

    # --- Intra-Family Hard-Negatives ---
    family_records: list[dict] = []
    if args.family_hard_negatives:
        query_file = Path(args.query_file).expanduser().resolve()
        mapping_file = Path(args.mapping_file).expanduser().resolve()
        taxonomy_file = Path(args.material_taxonomy).expanduser().resolve()

        if not query_file.is_file():
            print(f"WARN: Query-Datei fuer Family-HN nicht gefunden, uebersprungen: {query_file}")
        elif not mapping_file.is_file():
            print(f"WARN: Mapping-Datei fuer Family-HN nicht gefunden, uebersprungen: {mapping_file}")
        elif not taxonomy_file.is_file():
            print(f"WARN: Taxonomie-Datei fuer Family-HN nicht gefunden, uebersprungen: {taxonomy_file}")
        else:
            query_lines = query_file.read_text(encoding="utf-8").splitlines()
            mapping_lines = mapping_file.read_text(encoding="utf-8").splitlines()
            if len(query_lines) != len(mapping_lines):
                print(
                    f"WARN: Zeilenanzahl Query ({len(query_lines)}) != Mapping ({len(mapping_lines)}), "
                    "Family-HN uebersprungen."
                )
            else:
                mat_to_family, family_to_mats = load_material_families(taxonomy_file)
                cross_family_extras: dict | None = None
                if getattr(args, "cross_family_extras", None):
                    import json as _json
                    cross_family_extras = _json.loads(args.cross_family_extras)
                    print(f"Cross-family extras: {cross_family_extras}")
                family_records = mine_family_hard_negatives(
                    query_lines=query_lines,
                    mapping_lines=mapping_lines,
                    mat_to_family=mat_to_family,
                    family_to_mats=family_to_mats,
                    max_negatives_per_query=args.family_hn_max_negatives,
                    cross_family_extras=cross_family_extras,
                )
                family_hn_total = sum(len(r["hard_negatives"]) for r in family_records)
                print(f"\n--- Intra-Family Hard-Negatives ---")
                print(f"Taxonomy: {len(mat_to_family)} materials in {len(family_to_mats)} families")
                print(f"Queries with family HN: {len(family_records)}")
                print(f"Total family HN: {family_hn_total}")

    write_jsonl(
        records=records,
        out_file=out_file,
        source_file_name=details_file.name,
        extra_records=family_records,
    )

    print(f"\n--- Eval-basierte Hard-Negatives ---")
    print(f"Rows total: {stats['rows_total']}")
    print(f"Rows top1-wrong: {stats['wrong_rows']}")
    if stats.get("include_narrow_wins"):
        print(f"Rows narrow-win (gap < {stats['score_gap_threshold']}): {stats['narrow_win_rows']}")
    print(f"Rows kept after filters: {stats['kept_rows']}")
    print(f"Dropped same-query positives: {stats['dropped_same_query_positive']}")
    print(f"Dropped cross-query positives: {stats['dropped_cross_query_positive']}")
    print(f"Cross-query protection: {stats['cross_query_positive_protection']}")
    print(f"Queries with eval HN: {len(records)}")
    print(f"\n--- Zusammenfassung ---")
    print(f"Eval-HN records: {len(records)}, Family-HN records: {len(family_records)}")
    print(f"Total records in output: {len(records) + len(family_records)}")
    print(f"Output: {out_file}")


if __name__ == "__main__":
    main()
