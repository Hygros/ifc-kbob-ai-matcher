import argparse
import json
import random
import re
from pathlib import Path


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query_file = Path(args.query_file).expanduser().resolve()
    expected_file = Path(args.expected_file).expanduser().resolve()
    out_file = Path(args.out).expanduser().resolve()

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

    for idx, query in enumerate(queries):
        parsed_tokens = parse_expected_tokens_line(expected_lines[idx])
        if not parsed_tokens:
            skipped_empty_expected += 1
            continue

        for positive, weight in parsed_tokens:
            key = (query.casefold().strip(), positive.casefold().strip())
            if args.deduplicate and key in seen_pairs:
                continue

            seen_pairs.add(key)
            record = {
                "query": query,
                "positive": positive,
                "query_index": idx,
            }
            if weight is not None:
                record["weight"] = weight
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
    print(f"Output: {out_file}")


if __name__ == "__main__":
    main()
