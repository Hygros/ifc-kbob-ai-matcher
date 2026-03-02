"""

Training/run_single_model_evaluation.py --model Training/artifacts/models/bge-m3-finetuned-ifcentity-material-strength-3-epochs --query-file Evaluation/exports/queries/ifcentity_material_strength.txt --expected-file Evaluation/expected_material/expected.txt --cross-encoder-model "" --device auto --run-label finetuned --output-dir Training/outputs

additional options:
--cross-encoder-model BAAI/bge-reranker-v2-m3 --rerank-top-n 30


usage: run_single_model_evaluation.py [-h] [--model MODEL] --query-file QUERY_FILE --expected-file EXPECTED_FILE [--cross-encoder-model CROSS_ENCODER_MODEL]
                                      [--rerank-top-n RERANK_TOP_N] [--device {auto,cpu,cuda}] [--output-dir OUTPUT_DIR] --run-label RUN_LABEL [--no-timestamp]
run_single_model_evaluation.py: error: the following arguments are required: --query-file, --expected-file, --run-label
"""



import argparse
import importlib.util
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_SCRIPT = PROJECT_ROOT / "Evaluation" / "evaluate_material_models.py"
REPORT_SCRIPT = PROJECT_ROOT / "Evaluation" / "build_evaluation_report.py"


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
        default="BAAI/bge-reranker-v2-m3",
        help="Cross-Encoder fürs Re-Ranking (leer lassen für kein Re-Ranking).",
    )
    parser.add_argument("--rerank-top-n", type=int, default=30, help="Top-N Kandidaten fürs Re-Ranking.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Rechen-Device.")
    parser.add_argument(
        "--output-dir",
        default="Training/outputs",
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
        help="Wenn gesetzt, wird kein Zeitstempel an den Dateinamen angehängt.",
    )
    return parser.parse_args()


def make_safe_label(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    safe = "".join(ch if ch in allowed else "_" for ch in value).strip("._-")
    if not safe:
        raise ValueError("--run-label enthält keine gültigen Zeichen.")
    return safe


def main() -> None:
    args = parse_args()
    eval_module = load_eval_module(EVALUATION_SCRIPT)
    report_module = load_module(REPORT_SCRIPT, "build_eval_report_single")

    if args.rerank_top_n <= 0:
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

    cross_encoder_model = (args.cross_encoder_model or "").strip()

    setattr(eval_module, "SBERT_DEVICE", "" if args.device == "auto" else args.device)
    setattr(eval_module, "SBERT_CROSS_ENCODER_MODEL", cross_encoder_model)
    setattr(eval_module, "SBERT_RERANK_TOP_N", int(args.rerank_top_n))

    print(f"Using model: {args.model}")
    print(f"Using query file: {query_file}")
    print(f"Using expected file: {expected_file}")
    print(f"Using cross-encoder: {cross_encoder_model or '-'}")
    print(f"Using rerank top-n: {args.rerank_top_n}")

    cases = eval_module.build_evaluation_cases(query_file=query_file, expected_file=expected_file)

    database_path = eval_module.resolve_database_path(PROJECT_ROOT)
    print(f"Using database: {database_path}")

    with sqlite3.connect(str(database_path)) as connection:
        materials = eval_module.fetch_materials_from_db(connection)

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

    query_label = eval_module.make_query_label(query_file)
    ce_label = eval_module.make_cross_encoder_label(cross_encoder_model)
    model_label = args.model.replace("/", "_")
    run_label = make_safe_label(args.run_label)

    summary_path = output_dir / f"summary_{run_label}_{model_label}_{query_label}_{ce_label}.csv"
    details_path = output_dir / f"details_{run_label}_{model_label}_{query_label}_{ce_label}.csv"

    summary_fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    details_fieldnames = list(detail_rows[0].keys()) if detail_rows else []

    eval_module.write_csv(summary_path, summary_rows, summary_fieldnames)
    eval_module.write_csv(details_path, detail_rows, details_fieldnames)

    report_summary_rows = report_module.load_summary(summary_path)
    report_details_rows = report_module.load_details(details_path)

    ce_label = report_module.resolve_cross_encoder_label(report_summary_rows)
    report_label = f"{run_label}_{model_label}_{query_label}_{ce_label}"

    chart_path = output_dir / f"overview_{report_label}.svg"
    report_path = output_dir / f"evaluation_report_{report_label}.md"
    latest_chart = output_dir / "overview_latest.svg"
    latest_report = output_dir / "evaluation_report_latest.md"

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

    print("\nDone.")
    print(f"Summary: {summary_path}")
    print(f"Details: {details_path}")
    print(f"Overview SVG: {chart_path}")
    print(f"Report MD: {report_path}")
    print(f"Latest SVG: {latest_chart}")
    print(f"Latest Report: {latest_report}")


if __name__ == "__main__":
    main()
