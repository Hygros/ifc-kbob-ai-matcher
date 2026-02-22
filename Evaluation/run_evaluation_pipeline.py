"""
Evaluation-Pipeline (Kurzüberblick)

Dieses Skript führt die komplette Evaluation aus:
1) optional Query-Export aus .ifc/.jsonl nach .txt,
2) Modell-Evaluation,
3) Report-Generierung.

Nutzung:
- Interaktiv (Dateiauswahl im Terminal):
    python Evaluation/run_evaluation_pipeline.py

- Mit fixer Query-Quelle:
    python Evaluation/run_evaluation_pipeline.py --query-source <pfad_zu_.ifc|.jsonl|.txt>

- Optional mit Expected-Datei (eine Zeile pro Query):
    python Evaluation/run_evaluation_pipeline.py --query-source <...> --expected-file <pfad_zu_.txt>

Ausgaben:
- Queries: Evaluation/exports/queries/
- Evaluation + Report: Evaluation/exports/model_evaluation/
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = PROJECT_ROOT / "Evaluation"
EXPORT_SCRIPT = EVALUATION_DIR / "export_sbert_queries_to_txt.py"
EVALUATE_SCRIPT = EVALUATION_DIR / "evaluate_material_models.py"
REPORT_SCRIPT = EVALUATION_DIR / "build_evaluation_report.py"
QUERIES_EXPORT_DIR = EVALUATION_DIR / "exports" / "queries"
EXPECTED_MATERIAL_DIR = EVALUATION_DIR / "expected_material"
SUPPORTED_QUERY_SUFFIXES = {".ifc", ".jsonl", ".txt"}
IGNORED_DIRS = {".git", "__pycache__", "models", ".venv", "venv", ".pytest_cache", ".mypy_cache", "node_modules"}


def run_command(command: list[str], env: dict[str, str] | None = None) -> None:
    print(f"\n> {' '.join(command)}")
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Befehl fehlgeschlagen mit Exit-Code {result.returncode}: {' '.join(command)}")


def ensure_evaluation_dependencies(env: dict[str, str]) -> None:
    check_code = "import sentence_transformers, torch; print('ok')"
    result = subprocess.run(
        [sys.executable, "-c", check_code],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode == 0:
        return

    raise RuntimeError(
        "Fehlende Python-Pakete für Evaluation (mindestens 'sentence-transformers' und 'torch'). "
        "Installiere sie z.B. mit: python -m pip install -r requirements.txt"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Führt die komplette Evaluation aus: optional Query-Export, "
            "danach Modell-Evaluation und Report-Generierung."
        )
    )
    parser.add_argument(
        "--query-source",
        help="Optional: Quelle für Query-Export (.ifc, .jsonl oder .txt). Bei .txt wird kein Export ausgeführt.",
    )
    parser.add_argument(
        "--expected-file",
        help="Optional: TXT-Datei mit Expected-Materialien (eine Zeile pro Query).",
    )
    parser.add_argument(
        "--skip-ifc-export",
        action="store_true",
        help="Bei .ifc beim Query-Export keinen IFC-Export ausführen, sondern vorhandene .jsonl verwenden.",
    )
    return parser.parse_args()


def iter_project_files() -> list[Path]:
    files: list[Path] = []
    for root, dirnames, filenames in os.walk(PROJECT_ROOT):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        root_path = Path(root)
        for filename in filenames:
            files.append(root_path / filename)
    return files


def find_query_source_candidates() -> list[Path]:
    candidates: list[Path] = []
    if not QUERIES_EXPORT_DIR.is_dir():
        return candidates

    for root, _, filenames in os.walk(QUERIES_EXPORT_DIR):
        root_path = Path(root)
        for filename in filenames:
            path = root_path / filename
            if path.suffix.lower() in SUPPORTED_QUERY_SUFFIXES:
                candidates.append(path)

    return sorted(candidates)


def find_txt_candidates() -> list[Path]:
    candidates: list[Path] = []
    if not EXPECTED_MATERIAL_DIR.is_dir():
        return candidates

    for root, _, filenames in os.walk(EXPECTED_MATERIAL_DIR):
        root_path = Path(root)
        for filename in filenames:
            path = root_path / filename
            if path.suffix.lower() == ".txt":
                candidates.append(path)

    return sorted(candidates)


def select_query_source_interactively() -> Path | None:
    print(f"Suche Query-Dateien in {QUERIES_EXPORT_DIR.relative_to(PROJECT_ROOT)} ...", flush=True)
    candidates = find_query_source_candidates()
    if not candidates:
        print(f"Keine .ifc/.jsonl/.txt Dateien in {QUERIES_EXPORT_DIR.relative_to(PROJECT_ROOT)} gefunden.")
        return None

    print("\nWähle eine Query-Quelle (.ifc/.jsonl/.txt):")
    for idx, candidate in enumerate(candidates, start=1):
        rel = candidate.relative_to(PROJECT_ROOT)
        print(f"  {idx:>3}: {rel}")

    while True:
        user_input = input("Nummer eingeben (Enter = ohne Auswahl): ").strip()
        if not user_input:
            return None
        if not user_input.isdigit():
            print("Ungültige Eingabe. Bitte nur eine Nummer eingeben.")
            continue

        selected_index = int(user_input)
        if 1 <= selected_index <= len(candidates):
            return candidates[selected_index - 1]

        print(f"Bitte eine Nummer zwischen 1 und {len(candidates)} wählen.")


def select_expected_file_interactively() -> Path | None:
    ask = input("Expected-Material-Datei (.txt) auswählen? [j/N]: ").strip().lower()
    if ask not in {"j", "ja", "y", "yes"}:
        return None

    print(f"Suche TXT-Dateien in {EXPECTED_MATERIAL_DIR.relative_to(PROJECT_ROOT)} ...", flush=True)
    candidates = find_txt_candidates()
    if not candidates:
        print(f"Keine .txt Dateien in {EXPECTED_MATERIAL_DIR.relative_to(PROJECT_ROOT)} gefunden.")
        return None

    print("\nWähle eine Expected-Material-Datei (.txt):")
    for idx, candidate in enumerate(candidates, start=1):
        rel = candidate.relative_to(PROJECT_ROOT)
        print(f"  {idx:>3}: {rel}")

    while True:
        user_input = input("Nummer eingeben (Enter = ohne Auswahl): ").strip()
        if not user_input:
            return None
        if not user_input.isdigit():
            print("Ungültige Eingabe. Bitte nur eine Nummer eingeben.")
            continue

        selected_index = int(user_input)
        if 1 <= selected_index <= len(candidates):
            return candidates[selected_index - 1]

        print(f"Bitte eine Nummer zwischen 1 und {len(candidates)} wählen.")


def resolve_query_file(args: argparse.Namespace) -> Path | None:
    if args.query_source:
        source = Path(args.query_source)
        if not source.is_absolute():
            source = PROJECT_ROOT / source
        source = source.resolve()
    else:
        source = select_query_source_interactively()
        if source is None:
            return None

    suffix = source.suffix.lower()
    if suffix == ".txt":
        if not source.is_file():
            raise FileNotFoundError(f"Query-TXT nicht gefunden: {source}")
        return source

    if suffix not in {".ifc", ".jsonl"}:
        raise ValueError("--query-source muss .ifc, .jsonl oder .txt sein.")

    output_txt = (QUERIES_EXPORT_DIR / f"{source.stem}_sbert_queries.txt").resolve()
    command = [
        sys.executable,
        str(EXPORT_SCRIPT),
        str(source),
        "-o",
        str(output_txt),
    ]
    if args.skip_ifc_export:
        command.append("--skip-ifc-export")
    run_command(command)
    return output_txt


def main() -> None:
    args = parse_args()

    for script in (EXPORT_SCRIPT, EVALUATE_SCRIPT, REPORT_SCRIPT):
        if not script.is_file():
            raise FileNotFoundError(f"Skript nicht gefunden: {script}")

    query_file = resolve_query_file(args)

    env = os.environ.copy()
    if query_file is not None:
        env["SBERT_QUERY_FILE"] = str(query_file)
        print(f"Using SBERT_QUERY_FILE={query_file}")

    if args.expected_file:
        expected_file = Path(args.expected_file)
        if not expected_file.is_absolute():
            expected_file = PROJECT_ROOT / expected_file
        expected_file = expected_file.resolve()
        if not expected_file.is_file():
            raise FileNotFoundError(f"Expected-Datei nicht gefunden: {expected_file}")
    else:
        expected_file = select_expected_file_interactively()

    if expected_file is not None:
        env["SBERT_EXPECTED_FILE"] = str(expected_file)
        print(f"Using SBERT_EXPECTED_FILE={expected_file}")

    ensure_evaluation_dependencies(env)
    run_command([sys.executable, str(EVALUATE_SCRIPT)], env=env)
    run_command([sys.executable, str(REPORT_SCRIPT)], env=env)

    print("\nEvaluation-Pipeline abgeschlossen.")


if __name__ == "__main__":
    main()
