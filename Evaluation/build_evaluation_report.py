import csv
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "Evaluation" / "exports" / "model_evaluation"
METRIC_EXPLANATIONS_FILE = PROJECT_ROOT / "Evaluation" / "metric_explanations.md"


@dataclass
class SummaryRow:
    model: str
    pipeline_variant: str
    cross_encoder_model: str
    cases: int
    hit1: float
    hit5: float
    hit10: float
    mrr: float
    map10: float
    ndcg10: float
    recall10: float
    avg_expected_score: float
    hit1_ci_low: float
    hit1_ci_high: float
    hit10_ci_low: float
    hit10_ci_high: float
    mrr10_ci_low: float
    mrr10_ci_high: float
    ndcg10_ci_low: float
    ndcg10_ci_high: float


def parse_optional_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    raw = str(value).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def parse_timestamp_from_filename(name: str, prefix: str) -> str:
    pattern = rf"^{prefix}_(\d{{8}}_\d{{6}})\.csv$"
    match = re.match(pattern, name)
    if not match:
        return ""
    return match.group(1)


def resolve_query_label() -> str:
    query_file_env = os.environ.get("SBERT_QUERY_FILE", "").strip()
    if not query_file_env:
        return "latest"

    query_path = Path(query_file_env)
    base_name = query_path.stem or "latest"
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", base_name).strip("._-")
    return safe_name or "latest"


def find_latest_pair(results_dir: Path) -> Tuple[Path, Path, str]:
    query_label = resolve_query_label()
    labeled_summary = results_dir / f"summary_{query_label}.csv"
    labeled_details = results_dir / f"details_{query_label}.csv"
    if labeled_summary.is_file() and labeled_details.is_file():
        return labeled_summary, labeled_details, query_label

    summary_by_stamp: Dict[str, Path] = {}
    details_by_stamp: Dict[str, Path] = {}

    for path in results_dir.glob("summary_*.csv"):
        stamp = parse_timestamp_from_filename(path.name, "summary")
        if stamp:
            summary_by_stamp[stamp] = path

    for path in results_dir.glob("details_*.csv"):
        stamp = parse_timestamp_from_filename(path.name, "details")
        if stamp:
            details_by_stamp[stamp] = path

    common = sorted(set(summary_by_stamp) & set(details_by_stamp))
    if not common:
        raise FileNotFoundError("Keine passenden summary/details CSV-Dateien gefunden.")

    latest_stamp = common[-1]
    return summary_by_stamp[latest_stamp], details_by_stamp[latest_stamp], latest_stamp


def load_summary(summary_file: Path) -> List[SummaryRow]:
    rows: List[SummaryRow] = []
    with summary_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            hit1 = parse_optional_float(raw.get("hit@1"), default=parse_optional_float(raw.get("top1_accuracy")))
            hit5 = parse_optional_float(raw.get("hit@5"), default=parse_optional_float(raw.get("top5_accuracy", raw.get("topk_accuracy"))))
            hit10 = parse_optional_float(raw.get("hit@10"), default=parse_optional_float(raw.get("top10_accuracy")))
            map10 = parse_optional_float(raw.get("map@10"), default=parse_optional_float(raw.get("map10")))
            ndcg10 = parse_optional_float(raw.get("ndcg@10"), default=parse_optional_float(raw.get("ndcg10")))
            recall10 = parse_optional_float(raw.get("recall@10"), default=parse_optional_float(raw.get("recall10")))
            hit1_ci_low = parse_optional_float(raw.get("hit@1_ci_low"))
            hit1_ci_high = parse_optional_float(raw.get("hit@1_ci_high"))
            hit10_ci_low = parse_optional_float(raw.get("hit@10_ci_low"))
            hit10_ci_high = parse_optional_float(raw.get("hit@10_ci_high"))
            mrr10_ci_low = parse_optional_float(raw.get("mrr@10_ci_low"))
            mrr10_ci_high = parse_optional_float(raw.get("mrr@10_ci_high"))
            ndcg10_ci_low = parse_optional_float(raw.get("ndcg@10_ci_low"))
            ndcg10_ci_high = parse_optional_float(raw.get("ndcg@10_ci_high"))
            rows.append(
                SummaryRow(
                    model=raw["model"],
                    pipeline_variant=(raw.get("pipeline_variant") or "baseline").strip() or "baseline",
                    cross_encoder_model=(raw.get("cross_encoder_model") or "-").strip() or "-",
                    cases=int(raw["cases"]),
                    hit1=hit1,
                    hit5=hit5,
                    hit10=hit10,
                    mrr=float(raw["mrr"]),
                    map10=map10,
                    ndcg10=ndcg10,
                    recall10=recall10,
                    avg_expected_score=float(raw["avg_expected_score"]),
                    hit1_ci_low=hit1_ci_low,
                    hit1_ci_high=hit1_ci_high,
                    hit10_ci_low=hit10_ci_low,
                    hit10_ci_high=hit10_ci_high,
                    mrr10_ci_low=mrr10_ci_low,
                    mrr10_ci_high=mrr10_ci_high,
                    ndcg10_ci_low=ndcg10_ci_low,
                    ndcg10_ci_high=ndcg10_ci_high,
                )
            )

    rows.sort(
        key=lambda r: (
            r.pipeline_variant,
            -r.hit1,
            -r.mrr,
            -r.map10,
            -r.ndcg10,
            -r.recall10,
            -r.hit5,
            -r.hit10,
            -r.avg_expected_score,
        ),
    )
    return rows


def load_details(details_file: Path) -> List[dict]:
    with details_file.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def short_model_name(model: str) -> str:
    return model.split("/")[-1]


def compute_error_stats(details: List[dict]) -> Dict[Tuple[str, str], int]:
    errors_by_variant_model: Dict[Tuple[str, str], int] = {}
    for row in details:
        model = row["model"]
        variant = (row.get("pipeline_variant") or "baseline").strip() or "baseline"
        top1 = row.get("top1_correct", "False").strip().lower() == "true"
        if not top1:
            key = (variant, model)
            errors_by_variant_model[key] = errors_by_variant_model.get(key, 0) + 1
    return errors_by_variant_model


def compute_hard_queries(details: List[dict], pipeline_variant: str, top_n: int = 5) -> List[Tuple[str, int]]:
    wrong_counts: Dict[str, int] = {}
    for row in details:
        variant = (row.get("pipeline_variant") or "baseline").strip() or "baseline"
        if variant != pipeline_variant:
            continue
        query = row["query"]
        top1 = row.get("top1_correct", "False").strip().lower() == "true"
        if not top1:
            wrong_counts[query] = wrong_counts.get(query, 0) + 1
    return sorted(wrong_counts.items(), key=lambda item: item[1], reverse=True)[:top_n]


def to_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_svg_chart(rows: List[SummaryRow], output_file: Path) -> None:
    if not rows:
        return

    line_height = 68
    top_margin = 60
    chart_top = 60
    left_margin = 290
    chart_width = 800
    width = left_margin + chart_width + 180
    height = top_margin + len(rows) * line_height + 40

    def bar_width(value: float) -> float:
        return max(0.0, min(1.0, value)) * chart_width

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text { font-family: Arial, sans-serif; fill: #1f2937; } .label { font-size: 13px; } .title { font-size: 18px; font-weight: bold; } .axis { font-size: 12px; fill: #4b5563; } .value { font-size: 12px; font-weight: bold; }</style>',
        '<text x="20" y="28" class="title">Model Comparison (Top1 / Top5 / Top10 / MRR)</text>',
        f'<line x1="{left_margin}" y1="{chart_top}" x2="{left_margin + chart_width}" y2="{chart_top}" stroke="#d1d5db" stroke-width="1"/>',
        f'<line x1="{left_margin}" y1="{height - 20}" x2="{left_margin + chart_width}" y2="{height - 20}" stroke="#d1d5db" stroke-width="1"/>',
    ]

    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    for t in ticks:
        x = left_margin + bar_width(t)
        svg_lines.append(f'<line x1="{x}" y1="{chart_top}" x2="{x}" y2="{height - 20}" stroke="#e5e7eb" stroke-width="1"/>')
        svg_lines.append(f'<text x="{x - 10}" y="{chart_top - 4}" class="axis">{int(t * 100)}%</text>')

    legend_y = height - 4
    svg_lines.append(f'<rect x="20" y="{legend_y - 12}" width="12" height="12" fill="#2563eb"/>')
    svg_lines.append(f'<text x="38" y="{legend_y - 2}" class="axis">Top1</text>')
    svg_lines.append(f'<rect x="90" y="{legend_y - 12}" width="12" height="12" fill="#059669"/>')
    svg_lines.append(f'<text x="108" y="{legend_y - 2}" class="axis">Top5</text>')
    svg_lines.append(f'<rect x="160" y="{legend_y - 12}" width="12" height="12" fill="#7c3aed"/>')
    svg_lines.append(f'<text x="178" y="{legend_y - 2}" class="axis">Top10</text>')
    svg_lines.append(f'<rect x="235" y="{legend_y - 12}" width="12" height="12" fill="#f59e0b"/>')
    svg_lines.append(f'<text x="253" y="{legend_y - 2}" class="axis">MRR</text>')

    for idx, row in enumerate(rows):
        y_base = top_margin + idx * line_height
        label = short_model_name(row.model)
        svg_lines.append(f'<text x="20" y="{y_base + 18}" class="label">{label}</text>')

        y1 = y_base + 6
        y2 = y_base + 18
        y3 = y_base + 30
        y4 = y_base + 42

        w_top1 = bar_width(row.hit1)
        w_top5 = bar_width(row.hit5)
        w_top10 = bar_width(row.hit10)
        w_mrr = bar_width(row.mrr)

        svg_lines.append(f'<rect x="{left_margin}" y="{y1}" width="{w_top1}" height="8" fill="#2563eb" rx="2"/>')
        svg_lines.append(f'<rect x="{left_margin}" y="{y2}" width="{w_top5}" height="8" fill="#059669" rx="2"/>')
        svg_lines.append(f'<rect x="{left_margin}" y="{y3}" width="{w_top10}" height="8" fill="#7c3aed" rx="2"/>')
        svg_lines.append(f'<rect x="{left_margin}" y="{y4}" width="{w_mrr}" height="8" fill="#f59e0b" rx="2"/>')
        svg_lines.append(
            f'<text x="{left_margin + chart_width + 8}" y="{y_base + 31}" class="value">'
            f'{to_percent(row.hit1)} / {to_percent(row.hit5)} / {to_percent(row.hit10)} / {row.mrr:.3f}'
            '</text>'
        )

    svg_lines.append("</svg>")
    output_file.write_text("\n".join(svg_lines), encoding="utf-8")


def render_markdown_report(
    summary_rows: List[SummaryRow],
    details: List[dict],
    summary_file: Path,
    details_file: Path,
    chart_file: Path,
    report_file: Path,
) -> None:
    error_stats = compute_error_stats(details)
    baseline_rows = [row for row in summary_rows if row.pipeline_variant == "baseline"]
    reranked_rows = [row for row in summary_rows if row.pipeline_variant == "reranked"]
    baseline_rows.sort(
        key=lambda r: (r.hit1, r.mrr, r.map10, r.ndcg10, r.recall10, r.hit5, r.hit10, r.avg_expected_score),
        reverse=True,
    )
    reranked_rows.sort(
        key=lambda r: (r.hit1, r.mrr, r.map10, r.ndcg10, r.recall10, r.hit5, r.hit10, r.avg_expected_score),
        reverse=True,
    )
    baseline_hard_queries = compute_hard_queries(details, pipeline_variant="baseline")
    reranked_hard_queries = compute_hard_queries(details, pipeline_variant="reranked")

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = []
    lines.append("## Evaluation Report")
    lines.append("")
    lines.append(f"Generated: {generated_at}")
    lines.append("")
    lines.append("### Inputs")
    lines.append(f"- Summary CSV: `{summary_file.name}`")
    lines.append(f"- Details CSV: `{details_file.name}`")
    # lines.append("")
    # lines.append("### Metric Meaning")
    # relative_metric_file = os.path.relpath(METRIC_EXPLANATIONS_FILE, report_file.parent).replace("\\", "/")
    # lines.append(f"- Siehe vollständige Erklärung in: [{METRIC_EXPLANATIONS_FILE.name}]({relative_metric_file})")
    lines.append("")
    lines.append("### Overview")
    lines.append(f"![Model overview]({chart_file.name})")
    lines.append("")
    lines.append("### Leaderboard")
    lines.append("")
    lines.append("#### Baseline (Bi-Encoder)")
    lines.append("")
    lines.append("| Rank | Model | Hit@1 | Hit@5 | Hit@10 | MRR@10 | MAP@10 | nDCG@10 | Recall@10 | Avg expected score | Hit@1 95% CI | Hit@10 95% CI | MRR@10 95% CI | nDCG@10 95% CI | Top1 errors |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|")
    for rank, row in enumerate(baseline_rows, start=1):
        lines.append(
            "| "
            f"{rank} | {row.model} | {to_percent(row.hit1)} | {to_percent(row.hit5)} | {to_percent(row.hit10)} | {row.mrr:.3f} | {row.map10:.3f} | {row.ndcg10:.3f} | {row.recall10:.3f} | {row.avg_expected_score:.3f} | [{row.hit1_ci_low:.3f}, {row.hit1_ci_high:.3f}] | [{row.hit10_ci_low:.3f}, {row.hit10_ci_high:.3f}] | [{row.mrr10_ci_low:.3f}, {row.mrr10_ci_high:.3f}] | [{row.ndcg10_ci_low:.3f}, {row.ndcg10_ci_high:.3f}] | {error_stats.get((row.pipeline_variant, row.model), 0)}"
            " |"
        )

    lines.append("")
    lines.append("#### Reranked (Bi-Encoder + Cross-Encoder)")
    lines.append("")
    lines.append("| Rank | Model | Cross-Encoder | Hit@1 | Hit@5 | Hit@10 | MRR@10 | MAP@10 | nDCG@10 | Recall@10 | Avg expected score | Hit@1 95% CI | Hit@10 95% CI | MRR@10 95% CI | nDCG@10 95% CI | Top1 errors |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|")
    for rank, row in enumerate(reranked_rows, start=1):
        lines.append(
            "| "
            f"{rank} | {row.model} | {row.cross_encoder_model} | {to_percent(row.hit1)} | {to_percent(row.hit5)} | {to_percent(row.hit10)} | {row.mrr:.3f} | {row.map10:.3f} | {row.ndcg10:.3f} | {row.recall10:.3f} | {row.avg_expected_score:.3f} | [{row.hit1_ci_low:.3f}, {row.hit1_ci_high:.3f}] | [{row.hit10_ci_low:.3f}, {row.hit10_ci_high:.3f}] | [{row.mrr10_ci_low:.3f}, {row.mrr10_ci_high:.3f}] | [{row.ndcg10_ci_low:.3f}, {row.ndcg10_ci_high:.3f}] | {error_stats.get((row.pipeline_variant, row.model), 0)}"
            " |"
        )

    query_count = summary_rows[0].cases if summary_rows else 0
    lines.append("")
    lines.append(f"Anzahl Queries: {query_count}")

    if baseline_hard_queries:
        lines.append("")
        lines.append("### Hardest Queries (Baseline)")
        lines.append("Queries mit den meisten Top1-Fehlern in der Baseline:")
        lines.append("")
        for query, count in baseline_hard_queries:
            lines.append(f"- ({count} Fehler) {query}")

    if reranked_hard_queries:
        lines.append("")
        lines.append("### Hardest Queries (Reranked)")
        lines.append("Queries mit den meisten Top1-Fehlern nach Re-Ranking:")
        lines.append("")
        for query, count in reranked_hard_queries:
            lines.append(f"- ({count} Fehler) {query}")

    report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Ordner nicht gefunden: {RESULTS_DIR}")

    summary_file, details_file, query_label = find_latest_pair(RESULTS_DIR)
    summary_rows = load_summary(summary_file)
    details_rows = load_details(details_file)

    report_file = RESULTS_DIR / f"evaluation_report_{query_label}.md"
    chart_file = RESULTS_DIR / f"overview_{query_label}.svg"
    latest_report = RESULTS_DIR / "evaluation_report_latest.md"
    latest_chart = RESULTS_DIR / "overview_latest.svg"

    chart_rows = [row for row in summary_rows if row.pipeline_variant == "reranked"] or summary_rows
    render_svg_chart(chart_rows, chart_file)
    render_markdown_report(summary_rows, details_rows, summary_file, details_file, chart_file, report_file)

    latest_report.write_text(report_file.read_text(encoding="utf-8"), encoding="utf-8")
    latest_chart.write_text(chart_file.read_text(encoding="utf-8"), encoding="utf-8")

    print("Report erstellt:")
    print(f"- {report_file}")
    print(f"- {chart_file}")
    print("Latest links:")
    print(f"- {latest_report}")
    print(f"- {latest_chart}")


if __name__ == "__main__":
    main()
