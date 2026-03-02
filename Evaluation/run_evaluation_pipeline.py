"""
Evaluation-Pipeline (How-To)
============================

Was dieses Skript macht
-----------------------
Dieses Skript orchestriert den kompletten Evaluationsablauf für das
Sentence-Embedding-Retrieval:

1) Optionaler Query-Export aus `.ifc` oder `.jsonl` nach `.txt`
   (bei `.txt` als Quelle wird dieser Schritt übersprungen).
2) Modell-Evaluation über `Evaluation/evaluate_material_models.py`.
3) Report-Erstellung über `Evaluation/build_evaluation_report.py`.

Die eigentliche Modellbewertung passiert im Evaluator-Skript; dieses
Pipeline-Skript kümmert sich um Dateiauswahl, Umgebungsvariablen,
Abhängigkeits-Check und Reihenfolge der Schritte.

Evaluiert werden die Ranking-Metriken: Hit@K, MRR@10, MAP@10, nDCG@10, Recall@10.

Eingaben
--------
- Query-Quelle (`--query-source`): `.ifc`, `.jsonl` oder `.txt`.
- Optional `--expected-file`: TXT mit Expected/Relevant-Einträgen
  (eine Zeile pro Query).

Wenn keine Parameter übergeben werden, fragt das Skript interaktiv
im Terminal nach Query- und optional Expected-Datei.

Beispiele
---------
- Interaktiv starten:
    python Evaluation/run_evaluation_pipeline.py

- Mit fixer Query-Quelle (TXT):
    python Evaluation/run_evaluation_pipeline.py --query-source Evaluation/exports/queries/meine_queries.txt

- Vollständig mit Expected-Datei:
    python Evaluation/run_evaluation_pipeline.py --query-source <pfad> --expected-file <pfad>

- Mit Cross-Encoder-Re-Ranking konfigurieren:
    python Evaluation/run_evaluation_pipeline.py --cross-encoder-model BAAI/bge-reranker-v2-m3 --rerank-top-n 30

Ausgaben
--------
- Query-Export: `Evaluation/exports/queries/`
- Evaluation + Report: `Evaluation/exports/model_evaluation/`
  (u. a. `summary*.csv`, `details*.csv`, `evaluation_report*.md`, `overview*.svg`)
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
    parser.add_argument(
        "--cross-encoder-model",
        default=None,
        help=(
            "Cross-Encoder-Modell für Re-Ranking in der Evaluation "
            "(Default: BAAI/bge-reranker-v2-m3). Wenn nicht angegeben, wird interaktiv gefragt."
        ),
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=None,
        help="Anzahl Top-Kandidaten pro Query, die per Cross-Encoder neu sortiert werden (Default: 30). Wenn nicht angegeben, wird interaktiv gefragt.",
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


def select_expected_file_interactively() -> Path:
    print(f"Suche TXT-Dateien in {EXPECTED_MATERIAL_DIR.relative_to(PROJECT_ROOT)} ...", flush=True)
    candidates = find_txt_candidates()
    if not candidates:
        raise FileNotFoundError(
            f"Keine .txt Dateien in {EXPECTED_MATERIAL_DIR.relative_to(PROJECT_ROOT)} gefunden. "
            "Eine Expected-Material-Datei ist erforderlich."
        )

    print("\nWähle eine Expected-Material-Datei (.txt):")
    for idx, candidate in enumerate(candidates, start=1):
        rel = candidate.relative_to(PROJECT_ROOT)
        print(f"  {idx:>3}: {rel}")

    while True:
        user_input = input("Nummer eingeben: ").strip()
        if not user_input.isdigit():
            print("Ungültige Eingabe. Bitte nur eine Nummer eingeben.")
            continue

        selected_index = int(user_input)
        if 1 <= selected_index <= len(candidates):
            return candidates[selected_index - 1]

        print(f"Bitte eine Nummer zwischen 1 und {len(candidates)} wählen.")


MULTILINGUAL_CROSS_ENCODER_MODELS: list[tuple[str, str]] = [
    (
        "BAAI/bge-reranker-v2-m3",
        "BGE-M3-basiert · 100+ Sprachen · 0.6B · Apache-2.0 · empfohlen für multilingualen Einsatz",
    ),
    (
        "Alibaba-NLP/gte-multilingual-reranker-base",
        "GTE-basiert · 75 Sprachen · 0.3B · Apache-2.0 · hohe Leistung, schnelle Inferenz, max 8192 Tokens",
    ),
    (
        "jinaai/jina-reranker-v2-base-multilingual",
        "Jina v2 · 100+ Sprachen · 0.28B · CC-BY-NC-4.0 (nur nicht-kommerziell) · max 8192 Tokens",
    ),
    (
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "mMiniLMv2-basiert · 15 Sprachen (inkl. DE+EN) · 0.1B · Apache-2.0 · sehr leichtgewichtig",
    ),
]


def select_cross_encoder_settings_interactively() -> tuple[str | None, int | None]:
    """Fragt interaktiv, ob Cross-Encoder genutzt werden soll und wie viele Kandidaten re-ranked werden."""
    print("\nCross-Encoder Re-Ranking:")
    while True:
        use_ce = input("  Mit Cross-Encoder re-ranken? [j/n]: ").strip().lower()
        if use_ce in ("", "j", "ja", "y", "yes"):
            break
        if use_ce in ("n", "nein", "no"):
            print("  Cross-Encoder wird übersprungen.")
            return None, None
        print("  Ungültige Eingabe. Bitte 'j' oder 'n' eingeben.")

    print("\n  Verfügbare Cross-Encoder-Modelle (multilingual empfohlen):")
    for idx, (model_id, description) in enumerate(MULTILINGUAL_CROSS_ENCODER_MODELS, start=1):
        print(f"  {idx:>3}: {model_id}")
        print(f"       {description}")

    default_model = MULTILINGUAL_CROSS_ENCODER_MODELS[0][0]
    while True:
        model_input = input(
            f"\n  Nummer wählen oder Modell-ID eingeben [Default: 1 · {default_model}]: "
        ).strip()
        if not model_input:
            model = default_model
            break
        if model_input.isdigit():
            idx = int(model_input)
            if 1 <= idx <= len(MULTILINGUAL_CROSS_ENCODER_MODELS):
                model = MULTILINGUAL_CROSS_ENCODER_MODELS[idx - 1][0]
                break
            print(f"  Bitte eine Nummer zwischen 1 und {len(MULTILINGUAL_CROSS_ENCODER_MODELS)} wählen.")
        else:
            # Freitext-Eingabe einer beliebigen Modell-ID
            model = model_input
            break

    default_top_n = 30
    while True:
        top_n_input = input(f"  Anzahl Kandidaten für Re-Ranking [Default: {default_top_n}]: ").strip()
        if not top_n_input:
            top_n = default_top_n
            break
        if top_n_input.isdigit() and int(top_n_input) > 0:
            top_n = int(top_n_input)
            break
        print("  Ungültige Eingabe. Bitte eine positive ganze Zahl eingeben.")

    return model, top_n


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

    if args.rerank_top_n is not None and args.rerank_top_n <= 0:
        raise ValueError("--rerank-top-n muss > 0 sein.")

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

    env["SBERT_EXPECTED_FILE"] = str(expected_file)
    print(f"Using SBERT_EXPECTED_FILE={expected_file}")

    # Cross-Encoder: interaktiv fragen wenn nicht per CLI angegeben
    if args.cross_encoder_model is None and args.rerank_top_n is None:
        cross_encoder_model, rerank_top_n = select_cross_encoder_settings_interactively()
    else:
        cross_encoder_model = args.cross_encoder_model or "BAAI/bge-reranker-v2-m3"
        rerank_top_n = args.rerank_top_n if args.rerank_top_n is not None else 30

    if cross_encoder_model is not None and rerank_top_n is not None:
        env["SBERT_CROSS_ENCODER_MODEL"] = str(cross_encoder_model).strip()
        env["SBERT_RERANK_TOP_N"] = str(int(rerank_top_n))
        print(f"Using SBERT_CROSS_ENCODER_MODEL={env['SBERT_CROSS_ENCODER_MODEL']}")
        print(f"Using SBERT_RERANK_TOP_N={env['SBERT_RERANK_TOP_N']}")
    else:
        env.pop("SBERT_CROSS_ENCODER_MODEL", None)
        env.pop("SBERT_RERANK_TOP_N", None)
        print("Cross-Encoder Re-Ranking deaktiviert.")

    ensure_evaluation_dependencies(env)
    run_command([sys.executable, str(EVALUATE_SCRIPT)], env=env)
    run_command([sys.executable, str(REPORT_SCRIPT)], env=env)

    print("\nEvaluation-Pipeline abgeschlossen.")


if __name__ == "__main__":
    main()
