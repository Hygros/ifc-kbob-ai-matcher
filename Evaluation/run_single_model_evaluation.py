"""
Beispiel:

python Evaluation/run_single_model_evaluation.py \
    --model Training/artifacts/models/baseline_clean_run \
    --query-file Training/query_generation/generated_queries/generated_queries_without_exposure.txt \
    --expected-file Training/query_generation/generated_queries/mapping_generated_queries_without_exposure.txt \
    --run-label baseline_clean_run \
    --output-dir Evaluation/outputs/single_model

Optional mit Re-Ranking:
--cross-encoder-model BAAI/bge-reranker-v2-m3 --rerank-top-n 30

"""

# python Evaluation/run_single_model_evaluation.py --model models\Hygroskopisch\bge-m3-ifc-kbob-finetuned --query-file Evaluation\ground_truth\queries.txt --expected-file Evaluation\ground_truth\expected.txt --run-label eval-bge-m3-ifc-kbob-finetuned --output-dir Evaluation\outputs\single_model\bge-m3-ifc-kbob-finetuned\normal_queries

# python Evaluation/run_single_model_evaluation.py --model models\Hygroskopisch\bge-m3-ifc-kbob-finetuned --query-file Evaluation\ground_truth\queries_typos.txt --expected-file Evaluation\ground_truth\expected.txt --run-label eval-bge-m3-ifc-kbob-finetuned-typos --output-dir Evaluation\outputs\single_model\bge-m3-ifc-kbob-finetuned\queries_typos

# python Evaluation/run_single_model_evaluation.py --model models\Hygroskopisch\bge-m3-ifc-kbob-finetuned --query-file Evaluation\ground_truth\queries_missing.txt --expected-file Evaluation\ground_truth\expected.txt --run-label eval-bge-m3-ifc-kbob-finetuned-missing-queries --output-dir Evaluation\outputs\single_model\bge-m3-ifc-kbob-finetuned\queries_missing

# python Evaluation/run_single_model_evaluation.py --model models\Hygroskopisch\bge-m3-ifc-kbob-finetuned --query-file Evaluation\ground_truth\queries_missing+typos.txt --expected-file Evaluation\ground_truth\expected.txt --run-label eval-bge-m3-ifc-kbob-finetuned-missing-queries+typos --output-dir Evaluation\outputs\single_model\bge-m3-ifc-kbob-finetuned\queries_missing+typos


import argparse
import hashlib
import importlib.util
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_SCRIPT = PROJECT_ROOT / "Evaluation" / "evaluate_material_models.py"
REPORT_SCRIPT = PROJECT_ROOT / "Evaluation" / "build_evaluation_report.py"
SPLIT_MATRIX_SCRIPT = PROJECT_ROOT / "Evaluation" / "build_split_evaluation_matrix.py"
EXCLUDED_MATERIALS = {"Verzinken"}


def load_eval_module(script_path: Path) -> ModuleType:
    if not script_path.is_file():
        raise FileNotFoundError(f"Evaluator-Skript nicht gefunden: {script_path}")

    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    spec = importlib.util.spec_from_file_location("eval_models_single", str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Konnte Modul nicht laden: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_module(script_path: Path, module_name: str) -> ModuleType:
    if not script_path.is_file():
        raise FileNotFoundError(f"Skript nicht gefunden: {script_path}")

    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Konnte Modul nicht laden: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Führt die bestehende Evaluation nur für ein einzelnes Modell aus."
    )
    parser.add_argument("--model", default="BAAI/bge-m3", help="Zu evaluierendes Bi-Encoder-Modell.")
    parser.add_argument("--query-file", required=True, help="Query-TXT (eine Query pro Zeile).")
    parser.add_argument("--expected-file", required=True, help="Expected-TXT (eine Zeile pro Query).")
    parser.add_argument(
        "--cross-encoder-model",
        default="",
        help="Cross-Encoder fürs Re-Ranking (leer lassen für kein Re-Ranking).",
    )
    parser.add_argument("--rerank-top-n", type=int, default=30, help="Top-N Kandidaten fürs Re-Ranking.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Rechen-Device.")
    parser.add_argument(
        "--output-dir",
        default="Evaluation/outputs/single_model",
        help="Ausgabeverzeichnis für summary/details CSV.",
    )
    parser.add_argument(
        "--run-label",
        required=True,
        help="Pflichtlabel für den Lauf (z. B. baseline oder finetuned), wird in Dateinamen aufgenommen.",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Kompatibilitätsflag ohne Wirkung (Dateinamen sind deterministisch ohne Zeitstempel).",
    )
    parser.add_argument(
        "--split-eval-matrix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Erzeugt zusätzlich eine Split-Metrikmatrix aus dem geschriebenen Details-CSV.",
    )
    parser.add_argument(
        "--split-eval-query-class-map-file",
        default="",
        help="Optionales CSV/JSONL mit query->query_class für Split-Matrix.",
    )
    parser.add_argument(
        "--split-eval-min-cases",
        type=int,
        default=1,
        help="Mindestanzahl Cases pro Split-Zeile.",
    )
    return parser.parse_args()


def make_safe_label(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    safe = "".join(ch if ch in allowed else "_" for ch in value).strip("._-")
    if not safe:
        raise ValueError("--run-label enthält keine gültigen Zeichen.")
    return safe


def compact_token(value: str, max_len: int = 24) -> str:
    safe = make_safe_label(value)
    short = safe[:max_len]
    digest = hashlib.sha1(safe.encode("utf-8")).hexdigest()[:8]
    return f"{short}-{digest}"


def compact_model_label(model_value: str) -> str:
    raw = str(model_value).strip().replace("\\", "/")
    if "/" in raw:
        raw = raw.split("/")[-1]
    return compact_token(raw, max_len=28)


def run_command(command: list[str]) -> None:
    print(f"\n> {' '.join(command)}")
    result = subprocess.run(command, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Befehl fehlgeschlagen mit Exit-Code {result.returncode}: {' '.join(command)}")


def main() -> None:
    args = parse_args()
    eval_module = load_eval_module(EVALUATION_SCRIPT)
    report_module = load_module(REPORT_SCRIPT, "build_eval_report_single")
    if args.split_eval_matrix and not SPLIT_MATRIX_SCRIPT.is_file():
        raise FileNotFoundError(f"Split-Matrix-Skript nicht gefunden: {SPLIT_MATRIX_SCRIPT}")

    if args.split_eval_min_cases <= 0:
        raise ValueError("--split-eval-min-cases muss > 0 sein.")

    cross_encoder_model = (args.cross_encoder_model or "").strip()

    if cross_encoder_model and args.rerank_top_n <= 0:
        raise ValueError("--rerank-top-n muss > 0 sein.")

    query_file = Path(args.query_file)
    if not query_file.is_absolute():
        query_file = PROJECT_ROOT / query_file
    query_file = query_file.resolve()

    expected_file = Path(args.expected_file)
    if not expected_file.is_absolute():
        expected_file = PROJECT_ROOT / expected_file
    expected_file = expected_file.resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    setattr(eval_module, "SBERT_DEVICE", "" if args.device == "auto" else args.device)
    setattr(eval_module, "SBERT_CROSS_ENCODER_MODEL", cross_encoder_model)
    setattr(eval_module, "SBERT_RERANK_TOP_N", int(args.rerank_top_n))

    print(f"Using model: {args.model}")
    print(f"Using query file: {query_file}")
    print(f"Using expected file: {expected_file}")
    print(f"Using cross-encoder: {cross_encoder_model or '-'}")
    if cross_encoder_model:
        print(f"Using rerank top-n: {args.rerank_top_n}")
    else:
        print("Using rerank top-n: - (disabled)")

    cases = eval_module.build_evaluation_cases(query_file=query_file, expected_file=expected_file)

    database_path = eval_module.resolve_database_path(PROJECT_ROOT)
    print(f"Using database: {database_path}")

    with sqlite3.connect(str(database_path)) as connection:
        materials = eval_module.fetch_materials_from_db(connection)

    excluded_norm = {eval_module.normalize(material) for material in EXCLUDED_MATERIALS}
    before_count = len(materials)
    materials = [
        material
        for material in materials
        if eval_module.normalize(material) not in excluded_norm
    ]
    excluded_count = before_count - len(materials)
    if excluded_count > 0:
        print(f"Excluded materials from DB: {', '.join(sorted(EXCLUDED_MATERIALS))} ({excluded_count})")

    if not materials:
        raise RuntimeError("Keine Materialien aus der DB geladen.")

    exact_index: dict[str, list[int]] = {}
    normalized_index: dict[str, list[int]] = {}
    for idx, material in enumerate(materials):
        exact_index.setdefault(material, []).append(idx)
        normalized_index.setdefault(eval_module.normalize(material), []).append(idx)

    print(f"\nEvaluating model: {args.model}")
    result = eval_module.evaluate_model(
        model_name=args.model,
        materials=materials,
        cases=cases,
        exact_index=exact_index,
        normalized_index=normalized_index,
        project_root=PROJECT_ROOT,
        cross_encoder_model=cross_encoder_model,
        rerank_top_n=args.rerank_top_n,
    )

    summary_rows = list(result.summaries)
    detail_rows = list(result.details)

    for summary in summary_rows:
        print(
            f"  [{summary['pipeline_variant']}] "
            f"Hit@1: {float(summary['hit@1']):.2%} | "
            f"Hit@10: {float(summary['hit@10']):.2%} | "
            f"Hit@20: {float(summary['hit@20']):.2%} | "
            f"Hit@30: {float(summary['hit@30']):.2%} | "
            f"Hit@50: {float(summary['hit@50']):.2%} | "
            f"MRR@10: {float(summary['mrr']):.4f} | "
            f"MAP@10: {float(summary['map@10']):.4f} | "
            f"nDCG@10: {float(summary['ndcg@10']):.4f} | "
            f"Recall@10: {float(summary['recall@10']):.4f}"
        )

    query_label = compact_token(eval_module.make_query_label(query_file), max_len=20)
    ce_label = compact_token(eval_module.make_cross_encoder_label(cross_encoder_model), max_len=20)
    model_label = compact_model_label(args.model)
    run_label = make_safe_label(args.run_label)

    file_stem = f"{run_label}_{model_label}_{query_label}_{ce_label}"
    summary_path = output_dir / f"summary_{file_stem}.csv"
    details_path = output_dir / f"details_{file_stem}.csv"

    summary_fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    details_fieldnames = list(detail_rows[0].keys()) if detail_rows else []

    eval_module.write_csv(summary_path, summary_rows, summary_fieldnames)
    eval_module.write_csv(details_path, detail_rows, details_fieldnames)

    report_summary_rows = report_module.load_summary(summary_path)
    report_details_rows = report_module.load_details(details_path)

    ce_label = compact_token(report_module.resolve_cross_encoder_label(report_summary_rows), max_len=20)
    report_label = f"{run_label}_{model_label}_{query_label}_{ce_label}"

    chart_path = output_dir / f"overview_{report_label}.svg"
    report_path = output_dir / f"evaluation_report_{report_label}.md"
    latest_chart = output_dir / "overview_single_latest.svg"
    latest_report = output_dir / "evaluation_report_single_latest.md"

    report_module.render_svg_chart(report_summary_rows, chart_path)
    report_module.render_markdown_report(
        report_summary_rows,
        report_details_rows,
        summary_path,
        details_path,
        chart_path,
        report_path,
    )

    latest_chart.write_text(chart_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_report.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    split_csv_path = output_dir / f"split_eval_matrix_{file_stem}.csv"
    split_json_path = output_dir / f"split_eval_matrix_{file_stem}.json"
    if args.split_eval_matrix:
        split_command = [
            sys.executable,
            str(SPLIT_MATRIX_SCRIPT),
            "--details-file",
            str(details_path),
            "--out-csv",
            str(split_csv_path),
            "--out-json",
            str(split_json_path),
            "--min-cases",
            str(args.split_eval_min_cases),
        ]
        if args.split_eval_query_class_map_file.strip():
            split_command.extend([
                "--query-class-map-file",
                args.split_eval_query_class_map_file,
            ])
        run_command(split_command)

    print("\nDone.")
    print(f"Summary: {summary_path}")
    print(f"Details: {details_path}")
    print(f"Overview SVG: {chart_path}")
    print(f"Report MD: {report_path}")
    print(f"Latest SVG: {latest_chart}")
    print(f"Latest Report: {latest_report}")
    if args.split_eval_matrix:
        print(f"Split matrix CSV: {split_csv_path}")
        print(f"Split matrix JSON: {split_json_path}")


if __name__ == "__main__":
    main()
