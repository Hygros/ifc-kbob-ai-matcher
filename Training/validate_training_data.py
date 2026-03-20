import argparse
import json
import re
from pathlib import Path


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


def parse_expected_tokens_line(line: str) -> list[str]:
    raw = line.strip()
    if not raw:
        return []

    parts = re.split(r"[|;]", raw)
    tokens: list[str] = []
    for part in parts:
        token = part.strip()
        if not token:
            continue
        if "::" in token:
            token = token.rsplit("::", 1)[0].strip()
        if token:
            tokens.append(token)
    return tokens


def validate_raw_files(query_file: Path, expected_file: Path) -> tuple[int, int]:
    queries = load_non_empty_lines(query_file)
    expected_lines = load_non_empty_lines(expected_file)

    if not queries:
        raise ValueError("Query-Datei enthält keine nicht-leeren Zeilen.")
    if not expected_lines:
        raise ValueError("Expected-Datei enthält keine nicht-leeren Zeilen.")
    if len(queries) != len(expected_lines):
        raise ValueError(
            "Anzahl Query-Zeilen passt nicht zur Anzahl Expected-Zeilen: "
            f"{len(queries)} != {len(expected_lines)}"
        )

    empty_query_indices = [idx for idx, q in enumerate(queries) if not q.strip()]
    if empty_query_indices:
        raise ValueError(f"Leere Querys gefunden (Indices): {empty_query_indices[:10]}")

    empty_expected_indices: list[int] = []
    total_expected_tokens = 0
    for idx, line in enumerate(expected_lines):
        tokens = parse_expected_tokens_line(line)
        if not tokens:
            empty_expected_indices.append(idx)
        total_expected_tokens += len(tokens)

    if empty_expected_indices:
        raise ValueError(
            "Expected-Zeilen ohne verwertbare Tokens gefunden (Indices): "
            f"{empty_expected_indices[:10]}"
        )

    return len(queries), total_expected_tokens


def validate_pairs_file(pairs_file: Path) -> tuple[int, int, int, int, int, int]:
    if not pairs_file.is_file():
        raise FileNotFoundError(f"Pairs-Datei nicht gefunden: {pairs_file}")

    row_count = 0
    unique_pairs: set[tuple[str, str]] = set()
    unique_queries: set[str] = set()
    rows_with_hard_negatives = 0
    hard_negative_total = 0

    with pairs_file.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Ungültiges JSONL in Zeile {line_no}: {exc}") from exc

            query = str(row.get("query", "")).strip()
            positive = str(row.get("positive", "")).strip()
            if not query or not positive:
                raise ValueError(
                    f"Ungültiger Eintrag in Zeile {line_no}: 'query' und 'positive' müssen gesetzt sein."
                )

            hard_negatives_raw = row.get("hard_negatives", [])
            if isinstance(hard_negatives_raw, str):
                hard_negatives = [hard_negatives_raw] if hard_negatives_raw.strip() else []
            elif isinstance(hard_negatives_raw, list):
                hard_negatives = hard_negatives_raw
            elif hard_negatives_raw is None:
                hard_negatives = []
            else:
                raise ValueError(
                    f"Ungültiger Eintrag in Zeile {line_no}: 'hard_negatives' muss Liste oder String sein."
                )

            row_count += 1
            query_key = query.casefold()
            positive_key = positive.casefold()
            unique_queries.add(query_key)
            unique_pairs.add((query_key, positive_key))

            if hard_negatives:
                rows_with_hard_negatives += 1
                seen_negatives: set[str] = set()
                for value in hard_negatives:
                    candidate = str(value).strip()
                    candidate_key = candidate.casefold()
                    if not candidate_key:
                        raise ValueError(
                            f"Ungültiger Eintrag in Zeile {line_no}: leeres hard negative gefunden."
                        )
                    if candidate_key == positive_key:
                        raise ValueError(
                            f"Ungültiger Eintrag in Zeile {line_no}: positive darf nicht in hard_negatives enthalten sein."
                        )
                    if candidate_key in seen_negatives:
                        continue
                    seen_negatives.add(candidate_key)
                    hard_negative_total += 1

    if row_count == 0:
        raise ValueError("Pairs-Datei enthält keine gültigen Trainingszeilen.")

    unique_pair_count = len(unique_pairs)
    duplicate_pair_count = row_count - unique_pair_count
    return (
        row_count,
        unique_pair_count,
        duplicate_pair_count,
        len(unique_queries),
        rows_with_hard_negatives,
        hard_negative_total,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validiert Trainingsdaten vor dem Fine-Tuning.")
    parser.add_argument("--query-file", help="Pfad zur Query-TXT.")
    parser.add_argument("--expected-file", help="Pfad zur Expected-TXT.")
    parser.add_argument("--pairs-file", help="Pfad zur JSONL-Pairs-Datei.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.pairs_file and not (args.query_file and args.expected_file):
        raise ValueError("Entweder --pairs-file oder beide --query-file und --expected-file angeben.")

    if args.query_file and args.expected_file:
        query_count, token_count = validate_raw_files(
            Path(args.query_file).expanduser().resolve(),
            Path(args.expected_file).expanduser().resolve(),
        )
        print(f"Raw-Validierung OK: queries={query_count}, expected_tokens={token_count}")

    if args.pairs_file:
        (
            pair_count,
            unique_pair_count,
            duplicate_pair_count,
            query_count,
            rows_with_hard_negatives,
            hard_negative_total,
        ) = validate_pairs_file(
            Path(args.pairs_file).expanduser().resolve()
        )
        print(
            "Pairs-Validierung OK: "
            f"pairs_total={pair_count}, "
            f"unique_pairs={unique_pair_count}, "
            f"duplicate_pairs={duplicate_pair_count}, "
            f"unique_queries={query_count}, "
            f"rows_with_hard_negatives={rows_with_hard_negatives}, "
            f"hard_negatives_total={hard_negative_total}"
        )
        print("Für das Training werden die unique-pairs verwendet.")


if __name__ == "__main__":
    main()
