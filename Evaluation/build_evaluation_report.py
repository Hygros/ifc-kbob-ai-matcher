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
    hit10: float
    hit20: float
    hit30: float
    hit50: float
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


def resolve_ce_label_from_env() -> str:
    """Leitet den CE-Label aus der Umgebungsvariable SBERT_CROSS_ENCODER_MODEL ab."""
    model_id = os.environ.get("SBERT_CROSS_ENCODER_MODEL", "").strip()
    if not model_id:
        return "no-reranker"
    short = model_id.split("/")[-1]
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", short).strip("._-")
    return safe or "reranker"


def resolve_cross_encoder_label(summary_rows: List["SummaryRow"]) -> str:
    """Leitet einen dateisicheren Kürzel aus dem verwendeten Cross-Encoder ab."""
    reranked = [r for r in summary_rows if r.pipeline_variant == "reranked" and r.cross_encoder_model not in ("", "-")]
    if not reranked:
        # Kein Re-Ranking durchgeführt
        return "no-reranker"
    model_id = reranked[0].cross_encoder_model
    # Nur den Modellnamen ohne Org-Prefix verwenden (z.B. "BAAI/bge-reranker-v2-m3" → "bge-reranker-v2-m3")
    short = model_id.split("/")[-1]
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", short).strip("._-")
    return safe or "reranker"


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
            hit10 = parse_optional_float(raw.get("hit@10"), default=parse_optional_float(raw.get("top10_accuracy")))
            hit20 = parse_optional_float(raw.get("hit@20"))
            hit30 = parse_optional_float(raw.get("hit@30"))
            hit50 = parse_optional_float(raw.get("hit@50"))
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
                    hit10=hit10,
                    hit20=hit20,
                    hit30=hit30,
                    hit50=hit50,
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
            -r.hit50,
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


def render_svg_chart(all_rows: List[SummaryRow], output_file: Path) -> None:
    if not all_rows:
        return

    # Group rows by bi-encoder model into baseline and reranked buckets
    baseline_by_model: Dict[str, SummaryRow] = {}
    reranked_by_model: Dict[str, SummaryRow] = {}
    model_order: List[str] = []
    for row in all_rows:
        model = row.model
        if row.pipeline_variant == "reranked":
            reranked_by_model[model] = row
        else:
            baseline_by_model[model] = row
            if model not in model_order:
                model_order.append(model)
    for model in reranked_by_model:
        if model not in model_order:
            model_order.append(model)

    # Sort by baseline Hit@1 descending
    model_order.sort(
        key=lambda m: -(
            (baseline_by_model.get(m) or reranked_by_model[m]).hit1
        )
    )

    has_reranked = bool(reranked_by_model)

    # Layout constants
    bar_h = 8
    pair_gap = 3       # gap between baseline and reranked bar of the same metric
    metric_gap = 10    # gap between metric groups
    label_height = 16
    bottom_pad = 14
    pair_height = bar_h + pair_gap + bar_h  # 19 px per metric pair
    model_content_height = pair_height * 3 + metric_gap * 2  # 77 px
    line_height = label_height + model_content_height + bottom_pad  # 107 px

    top_margin = 50
    chart_top = 60
    left_margin = 300
    chart_width = 760
    legend_height = 42
    value_col_width = 50
    width = left_margin + chart_width + value_col_width
    height = top_margin + len(model_order) * line_height + legend_height + 16

    def bar_width(value: float) -> float:
        return max(0.0, min(1.0, value)) * chart_width

    # Colour palette
    C_HIT1_BASE  = "#1d4ed8"  # dark blue
    C_HIT1_RE    = "#93c5fd"  # light blue
    C_HIT10_BASE = "#6d28d9"  # dark purple
    C_HIT10_RE   = "#c4b5fd"  # light purple
    C_MRR_BASE   = "#b45309"  # dark amber
    C_MRR_RE     = "#fcd34d"  # light amber

    sep_y = height - legend_height - 8

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>'
        'text { font-family: Arial, sans-serif; fill: #1f2937; } '
        '.label { font-size: 13px; font-weight: bold; } '
        '.title { font-size: 17px; font-weight: bold; } '
        '.axis { font-size: 11px; fill: #4b5563; } '
        '.value { font-size: 10px; font-weight: bold; } '
        '.mlabel { font-size: 10px; fill: #6b7280; }'
        '</style>',
        '<text x="20" y="28" class="title">Model Comparison — Bi-Encoder vs. Cross-Encoder (Hit@1 / Hit@10 / MRR)</text>',
        f'<line x1="{left_margin}" y1="{chart_top}" x2="{left_margin + chart_width}" y2="{chart_top}" stroke="#d1d5db" stroke-width="1"/>',
        f'<line x1="{left_margin}" y1="{sep_y}" x2="{left_margin + chart_width}" y2="{sep_y}" stroke="#d1d5db" stroke-width="1"/>',
    ]

    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    for t in ticks:
        x = left_margin + bar_width(t)
        svg_lines.append(f'<line x1="{x}" y1="{chart_top}" x2="{x}" y2="{sep_y}" stroke="#e5e7eb" stroke-width="1"/>')
        svg_lines.append(f'<text x="{x - 10}" y="{chart_top - 4}" class="axis">{int(t * 100)}%</text>')

    # Per-model bars
    for idx, model in enumerate(model_order):
        base = baseline_by_model.get(model)
        re = reranked_by_model.get(model)
        y0 = top_margin + idx * line_height

        label = short_model_name(model)
        svg_lines.append(f'<text x="20" y="{y0 + 12}" class="label">{label}</text>')

        y_cursor = y0 + label_height

        # Helper: render one metric pair (baseline + optional reranked)
        def _pair(
            y: int,
            base_val: float,
            re_val: float | None,
            c_base: str,
            c_re: str,
            metric_name: str,
        ) -> None:
            def _label_x(val: float) -> float:
                return left_margin + bar_width(val) + 4

            def _label_y(bar_y: int) -> int:
                return bar_y + bar_h - 1

            svg_lines.append(
                f'<text x="{left_margin - 4}" y="{y + bar_h}" '
                f'class="mlabel" text-anchor="end">{metric_name}</text>'
            )
            svg_lines.append(
                f'<rect x="{left_margin}" y="{y}" width="{bar_width(base_val)}" '
                f'height="{bar_h}" fill="{c_base}" rx="2"/>'
            )
            svg_lines.append(
                f'<text x="{_label_x(base_val)}" y="{_label_y(y)}" class="value">{to_percent(base_val)}</text>'
            )
            if re_val is not None:
                svg_lines.append(
                    f'<rect x="{left_margin}" y="{y + bar_h + pair_gap}" '
                    f'width="{bar_width(re_val)}" height="{bar_h}" fill="{c_re}" rx="2"/>'
                )
                svg_lines.append(
                    f'<text x="{_label_x(re_val)}" y="{_label_y(y + bar_h + pair_gap)}" class="value">{to_percent(re_val)}</text>'
                )

        _pair(y_cursor, base.hit1 if base else 0.0,  re.hit1  if re else None, C_HIT1_BASE,  C_HIT1_RE,  "Hit@1")
        y_cursor += pair_height + metric_gap
        _pair(y_cursor, base.hit10 if base else 0.0, re.hit10 if re else None, C_HIT10_BASE, C_HIT10_RE, "Hit@10")
        y_cursor += pair_height + metric_gap
        _pair(y_cursor, base.mrr   if base else 0.0, re.mrr   if re else None, C_MRR_BASE,   C_MRR_RE,   "MRR")

    # Legend
    legend_entries = [
        (C_HIT1_BASE,  "Hit@1 Bi-Enc"),
        (C_HIT1_RE,    "Hit@1 Cross-Enc"),
        (C_HIT10_BASE, "Hit@10 Bi-Enc"),
        (C_HIT10_RE,   "Hit@10 Cross-Enc"),
        (C_MRR_BASE,   "MRR Bi-Enc"),
        (C_MRR_RE,     "MRR Cross-Enc"),
    ]
    legend_y = sep_y + 20
    x_leg = 20
    for color, leg_label in legend_entries:
        svg_lines.append(f'<rect x="{x_leg}" y="{legend_y - 9}" width="12" height="9" fill="{color}" rx="2"/>')
        svg_lines.append(f'<text x="{x_leg + 16}" y="{legend_y}" class="axis">{leg_label}</text>')
        x_leg += 140

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
        key=lambda r: (r.hit1, r.mrr, r.map10, r.ndcg10, r.recall10, r.hit50, r.hit10, r.avg_expected_score),
        reverse=True,
    )
    reranked_rows.sort(
        key=lambda r: (r.hit1, r.mrr, r.map10, r.ndcg10, r.recall10, r.hit50, r.hit10, r.avg_expected_score),
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
    lines.append("| Rank | Model | Hit@1 | Hit@10 | Hit@20 | Hit@30 | Hit@50 | MRR@10 | MAP@10 | nDCG@10 | Recall@10 | Avg expected score | Hit@1 95% CI | Hit@10 95% CI | MRR@10 95% CI | nDCG@10 95% CI | Top1 errors |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|")
    for rank, row in enumerate(baseline_rows, start=1):
        lines.append(
            "| "
            f"{rank} | {row.model} | {to_percent(row.hit1)} | {to_percent(row.hit10)} | {to_percent(row.hit20)} | {to_percent(row.hit30)} | {to_percent(row.hit50)} | {row.mrr:.3f} | {row.map10:.3f} | {row.ndcg10:.3f} | {row.recall10:.3f} | {row.avg_expected_score:.3f} | [{row.hit1_ci_low:.3f}, {row.hit1_ci_high:.3f}] | [{row.hit10_ci_low:.3f}, {row.hit10_ci_high:.3f}] | [{row.mrr10_ci_low:.3f}, {row.mrr10_ci_high:.3f}] | [{row.ndcg10_ci_low:.3f}, {row.ndcg10_ci_high:.3f}] | {error_stats.get((row.pipeline_variant, row.model), 0)}"
            " |"
        )

    lines.append("")
    lines.append("#### Reranked (Bi-Encoder + Cross-Encoder)")
    lines.append("")
    lines.append("| Rank | Model | Cross-Encoder | Hit@1 | Hit@10 | Hit@20 | Hit@30 | Hit@50 | MRR@10 | MAP@10 | nDCG@10 | Recall@10 | Avg expected score | Hit@1 95% CI | Hit@10 95% CI | MRR@10 95% CI | nDCG@10 95% CI | Top1 errors |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|")
    for rank, row in enumerate(reranked_rows, start=1):
        lines.append(
            "| "
            f"{rank} | {row.model} | {row.cross_encoder_model} | {to_percent(row.hit1)} | {to_percent(row.hit10)} | {to_percent(row.hit20)} | {to_percent(row.hit30)} | {to_percent(row.hit50)} | {row.mrr:.3f} | {row.map10:.3f} | {row.ndcg10:.3f} | {row.recall10:.3f} | {row.avg_expected_score:.3f} | [{row.hit1_ci_low:.3f}, {row.hit1_ci_high:.3f}] | [{row.hit10_ci_low:.3f}, {row.hit10_ci_high:.3f}] | [{row.mrr10_ci_low:.3f}, {row.mrr10_ci_high:.3f}] | [{row.ndcg10_ci_low:.3f}, {row.ndcg10_ci_high:.3f}] | {error_stats.get((row.pipeline_variant, row.model), 0)}"
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


def select_summary_file_interactively(results_dir: Path) -> Tuple[Path, Path, str]:
    """Wählt summary/details-CSV aus.

    Wenn SBERT_QUERY_FILE gesetzt ist (Aufruf aus Pipeline), wird automatisch
    die passende Datei per Umgebungsvariablen abgeleitet, ohne zu fragen.
    Sonst erscheint eine interaktive Auswahlliste.
    """
    # Auto-Auswahl wenn Env-Var gesetzt (Pipeline-Modus)
    if os.environ.get("SBERT_QUERY_FILE", "").strip():
        query_label = resolve_query_label()
        ce_label = resolve_ce_label_from_env()
        file_label = f"{query_label}_{ce_label}"
        summary_file = results_dir / f"summary_{file_label}.csv"
        details_file = results_dir / f"details_{file_label}.csv"
        if summary_file.is_file() and details_file.is_file():
            print(f"Verwende summary: {summary_file.name}")
            return summary_file, details_file, file_label
        # Fallback: summary.csv / details.csv
        summary_file = results_dir / "summary.csv"
        details_file = results_dir / "details.csv"
        if summary_file.is_file() and details_file.is_file():
            return summary_file, details_file, "latest"

    # Interaktive Auswahl (standalone-Modus)
    candidates = sorted(
        [p for p in results_dir.glob("summary_*.csv") if not p.stem.endswith("_latest")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    plain = results_dir / "summary.csv"
    if plain.is_file() and plain not in candidates:
        candidates.insert(0, plain)

    if not candidates:
        raise FileNotFoundError(f"Keine summary-CSV-Dateien in {results_dir} gefunden.")

    if len(candidates) == 1:
        summary_file = candidates[0]
        print(f"Verwende einzige verfügbare Datei: {summary_file.name}")
    else:
        print("\nVerfügbare summary-CSV-Dateien:")
        for idx, path in enumerate(candidates, start=1):
            print(f"  {idx:>3}: {path.name}")
        while True:
            user_input = input(f"Nummer wählen [Default: 1 · {candidates[0].name}]: ").strip()
            if not user_input:
                summary_file = candidates[0]
                break
            if user_input.isdigit() and 1 <= int(user_input) <= len(candidates):
                summary_file = candidates[int(user_input) - 1]
                break
            print(f"  Bitte eine Nummer zwischen 1 und {len(candidates)} eingeben.")

    stem = summary_file.stem
    label = stem[len("summary_"):] if stem.startswith("summary_") else stem
    details_file = results_dir / f"details_{label}.csv"
    if not details_file.is_file():
        details_file = results_dir / "details.csv"
    if not details_file.is_file():
        raise FileNotFoundError(f"Keine passende details-CSV für {summary_file.name} gefunden.")

    return summary_file, details_file, label


def main() -> None:
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Ordner nicht gefunden: {RESULTS_DIR}")

    summary_file, details_file, query_label = select_summary_file_interactively(RESULTS_DIR)
    summary_rows = load_summary(summary_file)
    details_rows = load_details(details_file)

    ce_label = resolve_cross_encoder_label(summary_rows)
    ce_suffix = f"_{ce_label}" if ce_label else ""
    final_label = query_label if ce_suffix and query_label.endswith(ce_suffix) else f"{query_label}{ce_suffix}"

    report_file = RESULTS_DIR / f"evaluation_report_{final_label}.md"
    chart_file = RESULTS_DIR / f"overview_{final_label}.svg"
    latest_report = RESULTS_DIR / "evaluation_report_latest.md"
    latest_chart = RESULTS_DIR / "overview_latest.svg"

    render_svg_chart(summary_rows, chart_file)
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
