# usage:
# python.exe Evaluation/export_sbert_queries_to_txt.py Evaluation/test_data/Bohrpfahl_4.3.jsonl
# or
# python.exe Evaluation/export_sbert_queries_to_txt.py IFC-Modelle\Tekla\2026\test-2026-4.3.ifc

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IFC_EXPORT_FIELDS = [
    "IfcEntity",
    "PredefinedType",
    "Name",
    "Material",
    "Description",
    "Durchmesser",
    "CastingMethod",
]


QUERY_EXPORT_DIR = PROJECT_ROOT / "Evaluation" / "exports" / "queries"


def load_ifc_jsonl_entries(jsonl_path: str) -> list[dict]:
    entries: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def ifc_entry_to_string(entry: dict) -> str:
    values: list[str] = []
    for field in IFC_EXPORT_FIELDS:
        raw_value = entry.get(field, "")
        if isinstance(raw_value, list):
            raw_value = ", ".join(str(value) for value in raw_value if value)

        text = str(raw_value).strip() if raw_value is not None else ""
        if text and text not in {"NOTDEFINED", "Undefined"}:
            values.append(text)

    return " ".join(values)


def run_ifc_export(ifc_path: Path) -> Path:
    result = subprocess.run(
        [sys.executable, "-m", "core.ifc_extraction.ifc_extraction_main", str(ifc_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(PROJECT_ROOT),
    )

    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")

    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise RuntimeError("IFC-Export fehlgeschlagen.")

    return ifc_path.with_suffix(".jsonl")


def resolve_paths(input_path: Path, output_path: Path | None, skip_ifc_export: bool) -> tuple[Path, Path]:
    suffix = input_path.suffix.lower()

    if suffix == ".ifc":
        if skip_ifc_export:
            jsonl_path = input_path.with_suffix(".jsonl")
        else:
            jsonl_path = run_ifc_export(input_path)
        default_output = QUERY_EXPORT_DIR / f"{input_path.stem}_sbert_queries.txt"
    elif suffix == ".jsonl":
        jsonl_path = input_path
        default_output = QUERY_EXPORT_DIR / f"{input_path.stem}_sbert_queries.txt"
    else:
        raise ValueError("Input muss eine .ifc- oder .jsonl-Datei sein.")

    final_output = output_path if output_path is not None else default_output
    return jsonl_path, final_output


def build_queries_from_jsonl(jsonl_path: Path) -> list[str]:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL-Datei nicht gefunden: {jsonl_path}")

    entries = load_ifc_jsonl_entries(str(jsonl_path))
    queries = [ifc_entry_to_string(entry).strip() for entry in entries]
    return [query for query in queries if query]


def write_queries_to_txt(queries: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(queries))
        if queries:
            file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Erzeugt SBERT-Queries aus bestehendem IFC/JSONL-Workflow und speichert sie als TXT.",
    )
    parser.add_argument(
        "input_path",
        help="Pfad zu .ifc (führt IFC-Export aus) oder .jsonl",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optionaler Ausgabepfad für die TXT-Datei",
    )
    parser.add_argument(
        "--skip-ifc-export",
        action="store_true",
        help="Bei .ifc nicht neu exportieren, sondern vorhandene .jsonl neben der IFC verwenden.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    output_path = Path(args.output).resolve() if args.output else None

    jsonl_path, target_txt = resolve_paths(
        input_path=input_path,
        output_path=output_path,
        skip_ifc_export=args.skip_ifc_export,
    )

    queries = build_queries_from_jsonl(jsonl_path)
    write_queries_to_txt(queries, target_txt)

    print(f"JSONL-Quelle: {jsonl_path}")
    print(f"Queries geschrieben: {len(queries)}")
    print(f"TXT-Ausgabe: {target_txt}")


if __name__ == "__main__":
    main()
