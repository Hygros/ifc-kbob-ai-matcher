import csv
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from retrieval_metrics import binary_ranking_metrics_at_10


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Leicht anpassbar: Bi-Encoder-Kandidaten pro Query ────────────────────────
# Wie viele Kandidaten der Bi-Encoder maximal zurückgibt.
# Der Wert sollte >= SBERT_RERANK_TOP_N sein; er wird weiter unten übermax(BI_ENCODER_TOP_K, SBERT_RERANK_TOP_N) erzwungen.
# 
BI_ENCODER_TOP_K = 50


def resolve_database_path(project_root: Path) -> Path:
    """Resolve the KBOB/Ökobilanz SQLite DB path.

    Azure/CI/Linux runs must not depend on a developer-specific absolute path.

    Resolution order:
    1) Env var `KBOB_DB_PATH` (or `ECOBILANZ_DB_PATH`)
    2) `<project_root>/Ökobilanzdaten.sqlite3`
    3) `<project_root>/../Ökobilanzdaten.sqlite3` (legacy layout)
    """

    for env_var in ("KBOB_DB_PATH", "ECOBILANZ_DB_PATH"):
        raw = os.environ.get(env_var, "").strip()
        if raw:
            candidate = Path(raw).expanduser().resolve()
            if candidate.is_file():
                return candidate

    candidates = [
        (project_root / "Ökobilanzdaten.sqlite3").resolve(),
        (project_root.parent / "Ökobilanzdaten.sqlite3").resolve(),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(
        "Ökobilanzdaten.sqlite3 nicht gefunden. "
        "Lege die DB ins Projektverzeichnis oder setze die Umgebungsvariable KBOB_DB_PATH.\n"
        "Gesucht in:\n"
        f"{searched}"
    )
TABLE_NAME = "Oekobilanzdaten"
COLUMN_MATERIAL = "Material"

MODEL_NAMES = [
    "google/embeddinggemma-300m",
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-large",
    "intfloat/multilingual-e5-base",
    "sentence-transformers/LaBSE",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "kforth/IfcMaterial2MP",
    "kforth/IfcElement2ConstructionSets",
    "google-bert/bert-base-german-cased",
    "google-bert/bert-base-multilingual-uncased",
    "google-bert/bert-base-multilingual-cased",
]

SBERT_DEVICE = os.environ.get("SBERT_DEVICE", "").strip().lower()
SBERT_QUERY_FILE = os.environ.get("SBERT_QUERY_FILE", "").strip()
SBERT_EXPECTED_FILE = os.environ.get("SBERT_EXPECTED_FILE", "").strip()
SBERT_CROSS_ENCODER_MODEL = os.environ.get("SBERT_CROSS_ENCODER_MODEL", "").strip()
SBERT_RERANK_TOP_N = int(os.environ.get("SBERT_RERANK_TOP_N", "30"))
SBERT_CROSS_ENCODER_REVISION = os.environ.get("SBERT_CROSS_ENCODER_REVISION", "").strip() or None

# Bi-Encoder retrieval pool: mindestens 10, mindestens so gross wie der Re-Rank-Pool
# Standardwert kommt von BI_ENCODER_TOP_K (oben im Skript leicht änderbar).
TOP_K = max(BI_ENCODER_TOP_K, SBERT_RERANK_TOP_N)

BOOTSTRAP_SAMPLES = int(os.environ.get("EVAL_BOOTSTRAP_SAMPLES", "300"))
BOOTSTRAP_SEED = 42

DEFAULT_QUERY_FILE_RELATIVE = Path("static") / "Bohrpfahl_4.3_sbert_queries.txt"
DEFAULT_EVALUATION_OUTPUT_RELATIVE = Path("Evaluation") / "exports" / "model_evaluation"

EXPECTED_BY_QUERY_OVERRIDES: Dict[str, str] = {}

_global_cross_encoder_models: dict[tuple[str, str], CrossEncoder] = {}


@dataclass
class EvaluationCase:
    query: str
    relevant_tokens: List[str]


@dataclass
class PerModelResult:
    summaries: List[Dict[str, str]]
    details: List[Dict[str, str]]


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def fetch_materials_from_db(connection: sqlite3.Connection) -> List[str]:
    cursor = connection.cursor()
    cursor.execute(f"SELECT {COLUMN_MATERIAL} FROM {TABLE_NAME}")
    return [row[0] for row in cursor.fetchall() if row[0]]


def load_queries_from_txt(query_file: Path) -> List[str]:
    if not query_file.is_file():
        raise FileNotFoundError(f"Query-Datei nicht gefunden: {query_file}")

    queries: List[str] = []
    with query_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            query = line.strip()
            if query:
                queries.append(query)

    if not queries:
        raise ValueError(f"Keine Queries in Datei gefunden: {query_file}")

    return queries


def load_expected_materials_from_txt(expected_file: Path, expected_count: int) -> List[str]:
    if not expected_file.is_file():
        raise FileNotFoundError(f"Expected-Datei nicht gefunden: {expected_file}")

    lines: List[str] = []
    with expected_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            token = line.strip()
            if token:
                lines.append(token)

    if len(lines) != expected_count:
        raise ValueError(
            "Anzahl Expected-Zeilen passt nicht zur Anzahl Queries: "
            f"{len(lines)} != {expected_count}"
        )

    return lines


def parse_expected_tokens_line(line: str) -> List[str]:
    raw = line.strip()
    if not raw:
        return []

    parts = re.split(r"[|;]", raw)
    tokens: List[str] = []
    for part in parts:
        token = part.strip()
        if not token:
            continue
        if "::" in token:
            token = token.rsplit("::", 1)[0].strip()
        if token:
            tokens.append(token)
    return tokens


def build_evaluation_cases(query_file: Path, expected_file: Path | None = None) -> List[EvaluationCase]:
    queries = load_queries_from_txt(query_file)
    expected_from_file: List[str] | None = None
    if expected_file is not None:
        expected_from_file = load_expected_materials_from_txt(expected_file, expected_count=len(queries))

    cases: List[EvaluationCase] = []
    for index, query in enumerate(queries):
        expected_default = expected_from_file[index] if expected_from_file is not None else ""
        expected_raw = EXPECTED_BY_QUERY_OVERRIDES.get(query, expected_default).strip()
        tokens = parse_expected_tokens_line(expected_raw)
        if not tokens:
            raise ValueError(
                "Leere Relevant-Menge für Query gefunden. "
                "Bitte Expected-Datei/Overrides prüfen."
            )
        cases.append(EvaluationCase(query=query, relevant_tokens=tokens))
    return cases


def resolve_relevant_indices(
    relevant_tokens: Sequence[str],
    materials: Sequence[str],
    exact_index: Dict[str, List[int]],
    normalized_index: Dict[str, List[int]],
) -> Tuple[List[str], List[int]]:
    resolved_materials: List[str] = []
    resolved_indices: List[int] = []

    missing: List[str] = []
    for token in relevant_tokens:
        if token.isdigit():
            candidate_id = int(token)
            if 0 <= candidate_id < len(materials):
                resolved_indices.append(candidate_id)
                resolved_materials.append(materials[candidate_id])
                continue

        if token in exact_index:
            indices = exact_index[token]
            resolved_indices.extend(indices)
            resolved_materials.append(token)
            continue

        normalized_token = normalize(token)
        if normalized_token in normalized_index:
            indices = normalized_index[normalized_token]
            resolved_indices.extend(indices)
            resolved_materials.append(materials[indices[0]])
            continue

        missing.append(token)

    if missing:
        last = missing[0]
        normalized_last = normalize(last)
        suggestions = [m for m in materials if normalized_last in normalize(m)][:5]
        suggestion_text = "; ".join(suggestions) if suggestions else "keine"
        raise ValueError(
            "Relevant-ID/Material nicht in Kandidaten gefunden: "
            f"{', '.join(missing)}. Vorschläge (für '{last}'): {suggestion_text}"
        )

    unique_indices: List[int] = []
    seen_idx: set[int] = set()
    for idx in resolved_indices:
        if idx not in seen_idx:
            unique_indices.append(idx)
            seen_idx.add(idx)

    unique_materials: List[str] = []
    seen_materials: set[str] = set()
    for material in resolved_materials:
        if material not in seen_materials:
            unique_materials.append(material)
            seen_materials.add(material)

    return unique_materials, unique_indices


def choose_device() -> str:
    if SBERT_DEVICE in {"cpu", "cuda"}:
        return SBERT_DEVICE
    return "cuda" if torch.cuda.is_available() else "cpu"


def local_model_dir(project_root: Path, model_name: str) -> Path:
    return project_root / "models" / model_name


def load_or_save_model(model_name: str, project_root: Path, device: str) -> SentenceTransformer:
    model_dir = local_model_dir(project_root, model_name)
    if model_dir.is_dir() and any(model_dir.iterdir()):
        return SentenceTransformer(str(model_dir), device=device)

    model = SentenceTransformer(model_name, device=device)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir))
    return model


def _normalize_cross_encoder_scores(raw_scores: np.ndarray) -> list[float]:
    scores = np.array(raw_scores)

    if scores.ndim == 1:
        return (1.0 / (1.0 + np.exp(-scores))).tolist()

    if scores.ndim == 2:
        n_classes = scores.shape[1]
        exp_s = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs = exp_s / exp_s.sum(axis=1, keepdims=True)
        return probs[:, n_classes - 1].tolist()

    s_min, s_max = scores.min(), scores.max()
    return ((scores - s_min) / (s_max - s_min + 1e-8)).tolist()


def load_or_get_cross_encoder(model_name: str, project_root: Path, device: str) -> CrossEncoder:
    key = (model_name, device)
    if key in _global_cross_encoder_models:
        return _global_cross_encoder_models[key]

    save_path = project_root / "models" / "cross-encoder" / model_name.replace("/", "_")
    if save_path.exists() and any(save_path.iterdir()):
        ce = CrossEncoder(
            str(save_path),
            device=device,
            trust_remote_code=True,
            revision=SBERT_CROSS_ENCODER_REVISION,
        )
    else:
        ce = CrossEncoder(
            model_name,
            device=device,
            trust_remote_code=True,
            revision=SBERT_CROSS_ENCODER_REVISION,
        )
        save_path.mkdir(parents=True, exist_ok=True)
        ce.save(str(save_path))

    _global_cross_encoder_models[key] = ce
    return ce


def rerank_query_indices(
    query: str,
    ranked_indices: Sequence[int],
    materials: Sequence[str],
    rerank_top_n: int,
    cross_encoder: CrossEncoder,
) -> Tuple[List[int], Dict[int, float]]:
    prefix_n = min(rerank_top_n, len(ranked_indices))
    if prefix_n <= 0:
        return list(ranked_indices), {}

    prefix_indices = list(ranked_indices[:prefix_n])
    cross_input = [[query, materials[int(idx)]] for idx in prefix_indices]
    raw_scores = cross_encoder.predict(cross_input, apply_softmax=False)
    normalized_scores = _normalize_cross_encoder_scores(np.array(raw_scores))

    scored = list(zip(prefix_indices, normalized_scores))
    scored.sort(key=lambda item: item[1], reverse=True)

    reranked_prefix = [idx for idx, _ in scored]
    reranked_full = reranked_prefix + list(ranked_indices[prefix_n:])
    score_overrides = {int(idx): float(score) for idx, score in zip(prefix_indices, normalized_scores)}
    return reranked_full, score_overrides



def bootstrap_metric_ci(
    frame: pd.DataFrame,
    metric_name: str,
    bootstrap_samples: int,
    random_state: int,
) -> Tuple[float, float]:
    n = len(frame)
    if n == 0:
        return 0.0, 0.0

    rng = np.random.default_rng(random_state)
    estimates: List[float] = []

    for _ in range(bootstrap_samples):
        indices = rng.integers(0, n, size=n)
        sample = frame.iloc[indices]
        value = float(sample[metric_name].mean())
        estimates.append(value)

    estimates_arr = np.array(estimates, dtype=float)
    low = float(np.quantile(estimates_arr, 0.025))
    high = float(np.quantile(estimates_arr, 0.975))
    return low, high


def build_model_dataframe(
    similarities: np.ndarray,
    materials: Sequence[str],
    cases: Sequence[EvaluationCase],
    exact_index: Dict[str, List[int]],
    normalized_index: Dict[str, List[int]],
    ranked_indices_per_query: Sequence[Sequence[int]],
    pipeline_variant: str,
    cross_encoder_model: str,
    score_overrides_per_query: Sequence[Dict[int, float]] | None = None,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for query_index, case in enumerate(cases):
        resolved_relevant_materials, relevant_indices = resolve_relevant_indices(
            relevant_tokens=case.relevant_tokens,
            materials=materials,
            exact_index=exact_index,
            normalized_index=normalized_index,
        )
        relevant_set = set(relevant_indices)

        scores = similarities[query_index]
        ranked_indices = list(ranked_indices_per_query[query_index])
        score_overrides: Dict[int, float]
        if score_overrides_per_query is None:
            score_overrides = {}
        else:
            score_overrides = dict(score_overrides_per_query[query_index])
        metrics = binary_ranking_metrics_at_10(ranked_indices, relevant_set)

        rank_by_index = {material_idx: rank for rank, material_idx in enumerate(ranked_indices, start=1)}
        best_rank = min(rank_by_index[idx] for idx in relevant_indices)

        predicted_index = ranked_indices[0]
        predicted_material = materials[predicted_index]
        if predicted_index in score_overrides:
            predicted_score = float(score_overrides[predicted_index])
        else:
            predicted_score = float(scores[predicted_index])

        top10_indices = ranked_indices[:10]
        top10_materials = [str(materials[int(i)]) for i in top10_indices]
        top10_scores: List[float] = []
        for material_idx in top10_indices:
            idx = int(material_idx)
            if idx in score_overrides:
                top10_scores.append(float(score_overrides[idx]))
            else:
                top10_scores.append(float(scores[idx]))

        rows.append(
            {
                "query_id": query_index,
                "query": case.query,
                "relevant_input": " | ".join(case.relevant_tokens),
                "relevant_resolved": " | ".join(resolved_relevant_materials),
                "relevant_count": len(relevant_set),
                "pipeline_variant": pipeline_variant,
                "cross_encoder_model": cross_encoder_model,
                "predicted_top1": predicted_material,
                "predicted_top1_score": predicted_score,
                "expected_rank": best_rank,
                "top1_correct": bool(metrics["hit@1"] > 0.0),
                "hit@1": float(metrics["hit@1"]),
                "hit@10": float(metrics["hit@10"]),
                "hit@20": float(metrics["hit@20"]),
                "hit@30": float(metrics["hit@30"]),
                "hit@50": float(metrics["hit@50"]),
                "mrr@10": float(metrics["mrr@10"]),
                "ndcg@10": float(metrics["ndcg@10"]),
                "map@10": float(metrics["map@10"]),
                "recall@10": float(metrics["recall@10"]),
                "top10_materials": " | ".join(top10_materials),
                "top10_scores": " | ".join(f"{s:.6f}" for s in top10_scores),
            }
        )

    return pd.DataFrame(rows)


def summarize_and_detail_rows(
    frame: pd.DataFrame,
    model_name: str,
    pipeline_variant: str,
    cross_encoder_model: str,
) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    ci_hit1 = bootstrap_metric_ci(frame, "hit@1", BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
    ci_hit10 = bootstrap_metric_ci(frame, "hit@10", BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
    ci_mrr10 = bootstrap_metric_ci(frame, "mrr@10", BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
    ci_ndcg10 = bootstrap_metric_ci(frame, "ndcg@10", BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)

    summary = {
        "model": model_name,
        "pipeline_variant": pipeline_variant,
        "cross_encoder_model": cross_encoder_model,
        "cases": str(len(frame)),
        "hit@1": f"{float(frame['hit@1'].mean()):.6f}",
        "hit@10": f"{float(frame['hit@10'].mean()):.6f}",
        "hit@20": f"{float(frame['hit@20'].mean()):.6f}",
        "hit@30": f"{float(frame['hit@30'].mean()):.6f}",
        "hit@50": f"{float(frame['hit@50'].mean()):.6f}",
        "mrr": f"{float(frame['mrr@10'].mean()):.6f}",
        "map@10": f"{float(frame['map@10'].mean()):.6f}",
        "ndcg@10": f"{float(frame['ndcg@10'].mean()):.6f}",
        "recall@10": f"{float(frame['recall@10'].mean()):.6f}",
        "avg_expected_score": f"{float(frame['predicted_top1_score'].mean()):.6f}",
        "hit@1_ci_low": f"{ci_hit1[0]:.6f}",
        "hit@1_ci_high": f"{ci_hit1[1]:.6f}",
        "hit@10_ci_low": f"{ci_hit10[0]:.6f}",
        "hit@10_ci_high": f"{ci_hit10[1]:.6f}",
        "mrr@10_ci_low": f"{ci_mrr10[0]:.6f}",
        "mrr@10_ci_high": f"{ci_mrr10[1]:.6f}",
        "ndcg@10_ci_low": f"{ci_ndcg10[0]:.6f}",
        "ndcg@10_ci_high": f"{ci_ndcg10[1]:.6f}",
    }

    details: List[Dict[str, str]] = []
    for _, row in frame.iterrows():
        details.append(
            {
                "model": model_name,
                "pipeline_variant": pipeline_variant,
                "cross_encoder_model": cross_encoder_model,
                "query_id": str(int(row["query_id"])),
                "query": str(row["query"]),
                "relevant_input": str(row["relevant_input"]),
                "relevant_resolved": str(row["relevant_resolved"]),
                "relevant_count": str(int(row["relevant_count"])),
                "predicted_top1": str(row["predicted_top1"]),
                "predicted_top1_score": f"{float(row['predicted_top1_score']):.6f}",
                "expected_rank": str(int(row["expected_rank"])),
                "top1_correct": str(bool(row["top1_correct"])),
                "hit@1": f"{float(row['hit@1']):.6f}",
                "hit@10": f"{float(row['hit@10']):.6f}",
                "hit@20": f"{float(row['hit@20']):.6f}",
                "hit@30": f"{float(row['hit@30']):.6f}",
                "hit@50": f"{float(row['hit@50']):.6f}",
                "mrr@10": f"{float(row['mrr@10']):.6f}",
                "ndcg@10": f"{float(row['ndcg@10']):.6f}",
                "map@10": f"{float(row['map@10']):.6f}",
                "recall@10": f"{float(row['recall@10']):.6f}",
                "top10_materials": str(row["top10_materials"]),
                "top10_scores": str(row["top10_scores"]),
            }
        )

    return summary, details


def evaluate_model(
    model_name: str,
    materials: Sequence[str],
    cases: Sequence[EvaluationCase],
    exact_index: Dict[str, List[int]],
    normalized_index: Dict[str, List[int]],
    project_root: Path,
    cross_encoder_model: str,
    rerank_top_n: int,
) -> PerModelResult:
    device = choose_device()
    print(f"  Loading model ({device})...", flush=True)
    model = load_or_save_model(model_name=model_name, project_root=project_root, device=device)

    queries = [case.query for case in cases]
    print(f"  Encoding {len(queries)} queries...", flush=True)
    query_embeddings = model.encode(queries, normalize_embeddings=True, convert_to_numpy=True)
    print(f"  Encoding {len(materials)} materials...", flush=True)
    material_embeddings = model.encode(list(materials), normalize_embeddings=True, convert_to_numpy=True)
    print("  Computing similarities...", flush=True)
    similarities = np.matmul(query_embeddings, material_embeddings.T)

    # Bi-Encoder nach dem Encoding vom GPU entladen, damit der Cross-Encoder
    # (der ggf. noch im VRAM ist) ausreichend Speicher hat.
    if device == "cuda":
        model.cpu()
        torch.cuda.empty_cache()

    baseline_rankings = [np.argsort(-similarities[query_index]).tolist() for query_index in range(len(cases))]

    baseline_frame = build_model_dataframe(
        similarities=similarities,
        materials=materials,
        cases=cases,
        exact_index=exact_index,
        normalized_index=normalized_index,
        ranked_indices_per_query=baseline_rankings,
        pipeline_variant="baseline",
        cross_encoder_model="-",
    )

    summaries: List[Dict[str, str]] = []
    details: List[Dict[str, str]] = []

    baseline_summary, baseline_details = summarize_and_detail_rows(
        frame=baseline_frame,
        model_name=model_name,
        pipeline_variant="baseline",
        cross_encoder_model="-",
    )
    summaries.append(baseline_summary)
    details.extend(baseline_details)

    if cross_encoder_model:
        print(f"  Re-ranking top {rerank_top_n} with Cross-Encoder: {cross_encoder_model}", flush=True)
        cross_encoder = load_or_get_cross_encoder(cross_encoder_model, project_root=project_root, device=device)
        reranked_rankings: List[List[int]] = []
        reranked_score_overrides: List[Dict[int, float]] = []

        # ── Batched Cross-Encoder Inference ──────────────────────────────────
        # Alle Query-Kandidaten-Paare über alle Queries hinweg in einem einzigen
        # cross_encoder.predict()-Aufruf bündeln.  Das vermeidet N_queries
        # separate Vorwärtspässe (mit je Python- und CUDA-Overhead) und ist
        # typischerweise 5–20× schneller als eine Query-by-Query-Schleife.
        all_pairs: List[List[str]] = []
        prefix_indices_per_query: List[List[int]] = []
        pair_counts: List[int] = []

        for query_index, case in enumerate(cases):
            ranked = baseline_rankings[query_index]
            prefix_n = min(rerank_top_n, len(ranked))
            prefix_indices = list(ranked[:prefix_n])
            prefix_indices_per_query.append(prefix_indices)
            pairs = [[case.query, materials[int(idx)]] for idx in prefix_indices]
            all_pairs.extend(pairs)
            pair_counts.append(len(pairs))

        if all_pairs:
            raw_scores_all = cross_encoder.predict(all_pairs, apply_softmax=False, batch_size=64)
            normalized_all = _normalize_cross_encoder_scores(np.array(raw_scores_all))
        else:
            normalized_all = []

        offset = 0
        for query_index in range(len(cases)):
            ranked = baseline_rankings[query_index]
            prefix_indices = prefix_indices_per_query[query_index]
            count = pair_counts[query_index]
            normalized_scores = list(normalized_all[offset : offset + count])
            offset += count

            scored = list(zip(prefix_indices, normalized_scores))
            scored.sort(key=lambda item: item[1], reverse=True)

            reranked_prefix = [idx for idx, _ in scored]
            reranked_full = reranked_prefix + list(ranked[count:])
            reranked_rankings.append(reranked_full)
            reranked_score_overrides.append(
                {int(idx): float(score) for idx, score in zip(prefix_indices, normalized_scores)}
            )
        # ─────────────────────────────────────────────────────────────────────

        reranked_frame = build_model_dataframe(
            similarities=similarities,
            materials=materials,
            cases=cases,
            exact_index=exact_index,
            normalized_index=normalized_index,
            ranked_indices_per_query=reranked_rankings,
            pipeline_variant="reranked",
            cross_encoder_model=cross_encoder_model,
            score_overrides_per_query=reranked_score_overrides,
        )
        reranked_summary, reranked_details = summarize_and_detail_rows(
            frame=reranked_frame,
            model_name=model_name,
            pipeline_variant="reranked",
            cross_encoder_model=cross_encoder_model,
        )
        summaries.append(reranked_summary)
        details.extend(reranked_details)

    return PerModelResult(summaries=summaries, details=details)


def write_csv(file_path: Path, rows: List[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_query_label(query_file: Path) -> str:
    base_name = query_file.stem or "latest"
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", base_name).strip("._-")
    return safe_name or "latest"


def make_cross_encoder_label(model_id: str) -> str:
    """Leitet einen dateisicheren Kürzel aus dem Cross-Encoder-Modell-Namen ab."""
    if not model_id:
        return "no-reranker"
    short = model_id.split("/")[-1]
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", short).strip("._-")
    return safe or "reranker"


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    if SBERT_QUERY_FILE:
        query_file = Path(SBERT_QUERY_FILE)
        if not query_file.is_absolute():
            query_file = project_root / query_file
    else:
        query_file = project_root / DEFAULT_QUERY_FILE_RELATIVE

    expected_file: Path | None = None
    if SBERT_EXPECTED_FILE:
        expected_file = Path(SBERT_EXPECTED_FILE)
        if not expected_file.is_absolute():
            expected_file = project_root / expected_file

    print(f"Using query file: {query_file}")
    if expected_file is not None:
        print(f"Using expected file: {expected_file}")
    print(f"Using cross-encoder model: {SBERT_CROSS_ENCODER_MODEL or '-'}")
    print(f"Using rerank top-n: {SBERT_RERANK_TOP_N}")

    if SBERT_RERANK_TOP_N <= 0:
        raise ValueError("SBERT_RERANK_TOP_N muss > 0 sein.")

    cases = build_evaluation_cases(query_file=query_file, expected_file=expected_file)

    database_path = resolve_database_path(project_root)
    print(f"Using database: {database_path}")

    with sqlite3.connect(str(database_path)) as connection:
        materials = fetch_materials_from_db(connection)

    if not materials:
        raise RuntimeError("Keine Materialien aus der DB geladen.")

    exact_index: Dict[str, List[int]] = {}
    normalized_index: Dict[str, List[int]] = {}
    for idx, material in enumerate(materials):
        exact_index.setdefault(material, []).append(idx)
        normalized_index.setdefault(normalize(material), []).append(idx)

    output_dir = project_root / DEFAULT_EVALUATION_OUTPUT_RELATIVE
    output_dir.mkdir(parents=True, exist_ok=True)

    query_label = make_query_label(query_file)

    summary_rows: List[Dict[str, str]] = []
    detail_rows: List[Dict[str, str]] = []

    for model_name in MODEL_NAMES:
        print(f"\nEvaluating model: {model_name}")
        result = evaluate_model(
            model_name=model_name,
            materials=materials,
            cases=cases,
            exact_index=exact_index,
            normalized_index=normalized_index,
            project_root=project_root,
            cross_encoder_model=SBERT_CROSS_ENCODER_MODEL,
            rerank_top_n=SBERT_RERANK_TOP_N,
        )

        summary_rows.extend(result.summaries)
        detail_rows.extend(result.details)

        for summary in result.summaries:
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

    summary_rows.sort(key=lambda r: (float(r["hit@1"]), float(r["mrr"]), float(r["ndcg@10"])), reverse=True)

    summary_fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    details_fieldnames = list(detail_rows[0].keys()) if detail_rows else []

    ce_label = make_cross_encoder_label(SBERT_CROSS_ENCODER_MODEL)
    file_label = f"{query_label}_{ce_label}"

    summary_labeled = output_dir / f"summary_{file_label}.csv"
    details_labeled = output_dir / f"details_{file_label}.csv"
    summary_latest = output_dir / "summary.csv"
    details_latest = output_dir / "details.csv"

    write_csv(summary_labeled, summary_rows, summary_fieldnames)
    write_csv(details_labeled, detail_rows, details_fieldnames)
    write_csv(summary_latest, summary_rows, summary_fieldnames)
    write_csv(details_latest, detail_rows, details_fieldnames)

    print("\nDone.")
    print(f"Summary: {summary_labeled}")
    print(f"Details: {details_labeled}")
    print(f"Latest Summary: {summary_latest}")
    print(f"Latest Details: {details_latest}")


if __name__ == "__main__":
    main()
