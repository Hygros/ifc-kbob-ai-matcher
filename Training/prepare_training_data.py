import argparse
import json
import random
import re
from pathlib import Path

from text_normalization import normalize_text_key


VALID_QUERY_CLASSES = {"eindeutig", "mehrdeutig"}


def load_non_empty_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")

    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                lines.append(line)
    return lines


def parse_expected_tokens_line(line: str) -> list[tuple[str, float | None]]:
    raw = line.strip()
    if not raw:
        return []

    parts = re.split(r"[|;]", raw)
    tokens: list[tuple[str, float | None]] = []
    for part in parts:
        token = part.strip()
        if not token:
            continue

        weight: float | None = None
        if "::" in token:
            token_raw, weight_raw = token.rsplit("::", 1)
            token = token_raw.strip()
            weight_raw = weight_raw.strip()
            if weight_raw:
                try:
                    weight = float(weight_raw)
                except ValueError:
                    weight = None

        if token:
            tokens.append((token, weight))

    return tokens


def normalize_key(value: str) -> str:
    return normalize_text_key(value)


def stable_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        key = normalize_key(value)
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def infer_query_class(query_positives: list[str]) -> str:
    return "mehrdeutig" if len(query_positives) > 1 else "eindeutig"


def load_hard_negatives_jsonl(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Hard-negatives Datei nicht gefunden: {path}")

    by_query: dict[str, list[str]] = {}
    seen_by_query: dict[str, set[str]] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Ungültiges JSONL in hard-negatives Zeile {line_no}: {exc}") from exc

            query = str(row.get("query", "")).strip()
            if not query:
                continue

            raw_negatives = row.get("hard_negatives", [])
            if isinstance(raw_negatives, str):
                raw_negatives = [raw_negatives]
            if not isinstance(raw_negatives, list):
                continue

            query_key = normalize_key(query)
            if query_key not in by_query:
                by_query[query_key] = []
                seen_by_query[query_key] = set()

            for value in raw_negatives:
                candidate = str(value).strip()
                candidate_key = normalize_key(candidate)
                if not candidate_key or candidate_key in seen_by_query[query_key]:
                    continue
                seen_by_query[query_key].add(candidate_key)
                by_query[query_key].append(candidate)

    return by_query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Erzeugt Trainingspaare (query, positive) aus Query- und Expected-TXT."
    )
    parser.add_argument("--query-file", required=True, help="Pfad zur Query-TXT (eine Query pro Zeile).")
    parser.add_argument(
        "--expected-file",
        required=True,
        help="Pfad zur Expected-TXT (eine Zeile pro Query, mehrere Relevante via ';' oder '|').",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Ausgabe-JSONL mit je einer Zeile pro (query, positive)-Paar.",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Dedupliziert identische (query, positive)-Paare.",
    )
    parser.add_argument(
        "--max-per-positive",
        type=int,
        default=0,
        help="Maximale Anzahl Paare pro unique Positive (0 = unbegrenzt). Reduziert Überrepräsentation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random Seed für reproduzierbares Sampling bei --max-per-positive.",
    )
    parser.add_argument(
        "--hard-negatives-file",
        default="",
        help="Optionales JSONL mit query + hard_negatives (z. B. aus mine_hard_negatives.py).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query_file = Path(args.query_file).expanduser().resolve()
    expected_file = Path(args.expected_file).expanduser().resolve()
    out_file = Path(args.out).expanduser().resolve()

    hard_negatives_by_query: dict[str, list[str]] = {}
    if args.hard_negatives_file.strip():
        hard_negatives_file = Path(args.hard_negatives_file).expanduser().resolve()
        hard_negatives_by_query = load_hard_negatives_jsonl(hard_negatives_file)
        print(f"Hard-negatives geladen für {len(hard_negatives_by_query)} Queries: {hard_negatives_file}")



    queries = load_non_empty_lines(query_file)
    expected_lines = load_non_empty_lines(expected_file)

    if len(queries) != len(expected_lines):
        raise ValueError(
            "Anzahl Query-Zeilen passt nicht zur Anzahl Expected-Zeilen: "
            f"{len(queries)} != {len(expected_lines)}"
        )

    records: list[dict[str, object]] = []
    seen_pairs: set[tuple[str, str]] = set()
    skipped_empty_expected = 0
    class_counter: dict[str, int] = {label: 0 for label in sorted(VALID_QUERY_CLASSES)}
    parsed_expected_per_query: list[list[tuple[str, float | None]]] = []
    query_positive_key_union: dict[str, set[str]] = {}

    for idx, query in enumerate(queries):
        parsed_tokens = parse_expected_tokens_line(expected_lines[idx])
        parsed_expected_per_query.append(parsed_tokens)
        if not parsed_tokens:
            continue

        query_key = normalize_key(query)
        query_positive_key_union.setdefault(query_key, set())
        for positive, _weight in parsed_tokens:
            positive_key = normalize_key(positive)
            if positive_key:
                query_positive_key_union[query_key].add(positive_key)

    for idx, query in enumerate(queries):
        parsed_tokens = parsed_expected_per_query[idx]
        if not parsed_tokens:
            skipped_empty_expected += 1
            continue

        query_positive_labels = stable_unique([positive for positive, _weight in parsed_tokens])
        query_key = normalize_key(query)
        query_class = infer_query_class(query_positive_labels)
        class_counter[query_class] += 1

        for positive, weight in parsed_tokens:
            key = (query.casefold().strip(), positive.casefold().strip())
            if args.deduplicate and key in seen_pairs:
                continue

            seen_pairs.add(key)
            record = {
                "query": query,
                "positive": positive,
                "query_index": idx,
                "query_class": query_class,
                "query_positives": list(query_positive_labels),
            }
            if weight is not None:
                record["weight"] = weight

            query_key = normalize_key(query)
            query_positive_keys = query_positive_key_union.get(query_key, set())
            hard_negatives = [
                negative
                for negative in hard_negatives_by_query.get(query_key, [])
                if normalize_key(negative) not in query_positive_keys
            ]
            if hard_negatives:
                record["hard_negatives"] = hard_negatives

            records.append(record)

    if args.max_per_positive > 0:
        by_positive: dict[str, list[dict[str, object]]] = {}
        for record in records:
            key = str(record["positive"]).casefold()
            by_positive.setdefault(key, []).append(record)

        capped_count = 0
        balanced: list[dict[str, object]] = []
        rng = random.Random(args.seed)
        for group in by_positive.values():
            if len(group) > args.max_per_positive:
                rng.shuffle(group)
                balanced.extend(group[: args.max_per_positive])
                capped_count += len(group) - args.max_per_positive
            else:
                balanced.extend(group)

        records = balanced
        if capped_count:
            print(f"Balancing: {capped_count} Paare entfernt (max {args.max_per_positive} pro Positive)")

    if not records:
        raise ValueError("Keine Trainingspaare erzeugt. Bitte Eingabedateien prüfen.")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Queries: {len(queries)}")
    print(f"Expected-Zeilen: {len(expected_lines)}")
    print(f"Trainingspaare: {len(records)}")
    print(f"Leere Expected-Zeilen übersprungen: {skipped_empty_expected}")
    print("Query-Klassen im Haupttraining: " + ", ".join(f"{key}={class_counter[key]}" for key in sorted(class_counter)))
    print(f"Output: {out_file}")


if __name__ == "__main__":
    main()
