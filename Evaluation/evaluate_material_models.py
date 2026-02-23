import argparse
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
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from retrieval_metrics import (
    best_threshold_for_target_accuracy,
    binary_ranking_metrics_at_10,
    coverage_at_target_accuracy,
    risk_coverage_curve,
)


DATABASE_PATH = r"C:\Users\wpx619\AAA_Python_MTH\Ökobilanzdaten.sqlite3"
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

TOP_K = 10
SBERT_DEVICE = os.environ.get("SBERT_DEVICE", "").strip().lower()
SBERT_QUERY_FILE = os.environ.get("SBERT_QUERY_FILE", "").strip()
SBERT_EXPECTED_FILE = os.environ.get("SBERT_EXPECTED_FILE", "").strip()
EVAL_MODE_ENV = os.environ.get("EVAL_MODE", "decision").strip().lower()

SPLIT_RANDOM_SEED = int(os.environ.get("EVAL_SPLIT_SEED", "42"))
BOOTSTRAP_SAMPLES = int(os.environ.get("EVAL_BOOTSTRAP_SAMPLES", "300"))
DEFAULT_ACCEPTANCE_TARGET = float(os.environ.get("EVAL_ACCEPT_TARGET", "0.95"))
COVERAGE_TARGETS = [0.90, 0.95, 0.97, 0.99]

DEFAULT_QUERY_FILE_RELATIVE = Path("static") / "Bohrpfahl_4.3_sbert_queries.txt"
DEFAULT_EVALUATION_OUTPUT_RELATIVE = Path("Evaluation") / "exports" / "model_evaluation"

DEFAULT_EXPECTED_MATERIAL = "Tiefgründung Ortbetonbohrpfahl 900"
EXPECTED_BY_QUERY_OVERRIDES: Dict[str, str] = {}
EVAL_MODES = {"ranking-only", "decision"}


@dataclass
class EvaluationCase:
    query: str
    relevant_tokens: List[str]


@dataclass
class PerModelResult:
    summary: Dict[str, str]
    details: List[Dict[str, str]]


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def parse_args() -> argparse.Namespace:
    default_mode = EVAL_MODE_ENV if EVAL_MODE_ENV in EVAL_MODES else "decision"
    parser = argparse.ArgumentParser(description="Evaluate embedding retrieval models.")
    parser.add_argument(
        "--mode",
        choices=sorted(EVAL_MODES),
        default=default_mode,
        help="ranking-only: nur Rankingmetriken; decision: zusätzlich Kalibrierung/Split/Auto-Decision.",
    )
    return parser.parse_args()


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
        expected_default = expected_from_file[index] if expected_from_file is not None else DEFAULT_EXPECTED_MATERIAL
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


def safe_std(values: np.ndarray) -> float:
    std = float(np.std(values))
    return std if std > 1e-12 else 1e-12


def entropy_from_top_scores(top_scores: np.ndarray) -> float:
    shifted = top_scores - np.max(top_scores)
    probs = np.exp(shifted)
    probs_sum = float(np.sum(probs))
    if probs_sum <= 0:
        return 0.0
    probs = probs / probs_sum
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    return entropy


def extract_confidence_features(scores: np.ndarray, ranked_indices: Sequence[int]) -> Dict[str, float]:
    s1 = float(scores[ranked_indices[0]])
    s2 = float(scores[ranked_indices[1]]) if len(ranked_indices) > 1 else s1
    s5 = float(scores[ranked_indices[4]]) if len(ranked_indices) > 4 else float(scores[ranked_indices[-1]])

    gap12 = s1 - s2
    gap1_5 = s1 - s5
    mean_all = float(np.mean(scores))
    std_all = safe_std(scores)
    z1 = (s1 - mean_all) / std_all
    z_gap = gap12 / std_all

    top10_indices = list(ranked_indices[:10])
    top10_scores = np.array([scores[i] for i in top10_indices], dtype=float)
    entropy_top10 = entropy_from_top_scores(top10_scores)

    return {
        "s1": s1,
        "s2": s2,
        "gap12": gap12,
        "gap1_5": gap1_5,
        "mean90": mean_all,
        "std90": std_all,
        "z1": z1,
        "z_gap": z_gap,
        "entropy_top10": entropy_top10,
    }


def split_indices_train_val_test(y: np.ndarray, random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    def can_use_stratify(labels: np.ndarray) -> bool:
        if len(labels) < 4:
            return False
        values, counts = np.unique(labels, return_counts=True)
        if len(values) < 2:
            return False
        return int(np.min(counts)) >= 2

    def safe_split(
        indices: np.ndarray,
        labels: np.ndarray,
        test_size: float,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        stratify = labels if can_use_stratify(labels) else None
        try:
            first, second = train_test_split(
                indices,
                test_size=test_size,
                random_state=seed,
                stratify=stratify,
            )
            return np.array(first), np.array(second)
        except ValueError:
            first, second = train_test_split(
                indices,
                test_size=test_size,
                random_state=seed,
                stratify=None,
            )
            return np.array(first), np.array(second)

    n = len(y)
    all_idx = np.arange(n)

    if n < 10:
        train_end = max(1, int(0.6 * n))
        val_end = max(train_end + 1, int(0.8 * n)) if n > 2 else n
        train_idx = all_idx[:train_end]
        val_idx = all_idx[train_end:val_end]
        test_idx = all_idx[val_end:]
        if len(test_idx) == 0:
            test_idx = val_idx
        if len(val_idx) == 0:
            val_idx = train_idx
        return train_idx, val_idx, test_idx

    train_idx, temp_idx = safe_split(
        indices=all_idx,
        labels=y,
        test_size=0.4,
        seed=random_state,
    )

    y_temp = y[temp_idx]
    val_idx, test_idx = safe_split(
        indices=temp_idx,
        labels=y_temp,
        test_size=0.5,
        seed=random_state + 1,
    )

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def fit_logistic_calibrator(
    feature_frame: pd.DataFrame,
    labels: np.ndarray,
    train_idx: np.ndarray,
) -> Pipeline | None:
    if len(train_idx) < 4:
        return None

    y_train = labels[train_idx]
    if len(np.unique(y_train)) < 2:
        return None

    feature_columns = ["s1", "s2", "gap12", "gap1_5", "mean90", "std90", "z1", "z_gap", "entropy_top10"]
    x_train = feature_frame.iloc[train_idx][feature_columns].to_numpy(dtype=float)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SPLIT_RANDOM_SEED)),
        ]
    )
    model.fit(x_train, y_train)
    return model


def predict_probabilities(feature_frame: pd.DataFrame, calibrator: Pipeline | None, labels: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    feature_columns = ["s1", "s2", "gap12", "gap1_5", "mean90", "std90", "z1", "z_gap", "entropy_top10"]

    if calibrator is None:
        if len(train_idx) == 0:
            base_rate = float(np.mean(labels)) if len(labels) > 0 else 0.5
        else:
            base_rate = float(np.mean(labels[train_idx]))
        return np.full(shape=(len(feature_frame),), fill_value=base_rate, dtype=float)

    x_all = feature_frame[feature_columns].to_numpy(dtype=float)
    return calibrator.predict_proba(x_all)[:, 1]


def bootstrap_metric_ci(
    test_frame: pd.DataFrame,
    metric_name: str,
    bootstrap_samples: int,
    random_state: int,
    target_for_coverage: float = 0.95,
) -> Tuple[float, float]:
    n = len(test_frame)
    if n == 0:
        return 0.0, 0.0

    rng = np.random.default_rng(random_state)
    estimates: List[float] = []

    for _ in range(bootstrap_samples):
        indices = rng.integers(0, n, size=n)
        sample = test_frame.iloc[indices]

        if metric_name == "coverage@95":
            value = coverage_at_target_accuracy(
                confidences=sample["p_correct"].astype(float).tolist(),
                correct=sample["top1_correct"].astype(bool).tolist(),
                target_accuracy=target_for_coverage,
            )
        else:
            value = float(sample[metric_name].mean())
        estimates.append(value)

    low = float(np.quantile(estimates, 0.025))
    high = float(np.quantile(estimates, 0.975))
    return low, high


def build_model_dataframe(
    similarities: np.ndarray,
    materials: Sequence[str],
    cases: Sequence[EvaluationCase],
    exact_index: Dict[str, List[int]],
    normalized_index: Dict[str, List[int]],
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
        ranked_indices = np.argsort(-scores).tolist()
        metrics = binary_ranking_metrics_at_10(ranked_indices, relevant_set)

        rank_by_index = {material_idx: rank for rank, material_idx in enumerate(ranked_indices, start=1)}
        best_rank = min(rank_by_index[idx] for idx in relevant_indices)

        predicted_index = ranked_indices[0]
        predicted_material = materials[predicted_index]

        top10_indices = ranked_indices[:10]
        top10_materials = [str(materials[int(i)]) for i in top10_indices]
        top10_scores = [float(scores[int(i)]) for i in top10_indices]

        features = extract_confidence_features(scores=scores, ranked_indices=ranked_indices)

        rows.append(
            {
                "query_id": query_index,
                "query": case.query,
                "relevant_input": " | ".join(case.relevant_tokens),
                "relevant_resolved": " | ".join(resolved_relevant_materials),
                "relevant_count": len(relevant_set),
                "predicted_top1": predicted_material,
                "predicted_top1_score": float(scores[predicted_index]),
                "expected_rank": best_rank,
                "top1_correct": bool(metrics["hit@1"] > 0.0),
                "hit@1": float(metrics["hit@1"]),
                "hit@5": float(metrics["hit@5"]),
                "hit@10": float(metrics["hit@10"]),
                "mrr@10": float(metrics["mrr@10"]),
                "ndcg@10": float(metrics["ndcg@10"]),
                "map@10": float(metrics["map@10"]),
                "recall@10": float(metrics["recall@10"]),
                "top10_materials": " | ".join(top10_materials),
                "top10_scores": " | ".join(f"{s:.6f}" for s in top10_scores),
                **features,
            }
        )

    return pd.DataFrame(rows)


def evaluate_model(
    model_name: str,
    materials: Sequence[str],
    cases: Sequence[EvaluationCase],
    exact_index: Dict[str, List[int]],
    normalized_index: Dict[str, List[int]],
    project_root: Path,
    mode: str,
) -> PerModelResult:
    device = choose_device()
    model = load_or_save_model(model_name=model_name, project_root=project_root, device=device)

    queries = [case.query for case in cases]
    query_embeddings = model.encode(queries, normalize_embeddings=True, convert_to_numpy=True)
    material_embeddings = model.encode(list(materials), normalize_embeddings=True, convert_to_numpy=True)

    similarities = np.matmul(query_embeddings, material_embeddings.T)

    frame = build_model_dataframe(
        similarities=similarities,
        materials=materials,
        cases=cases,
        exact_index=exact_index,
        normalized_index=normalized_index,
    )

    labels = frame["top1_correct"].astype(int).to_numpy()
    if mode == "decision":
        train_idx, val_idx, test_idx = split_indices_train_val_test(labels, random_state=SPLIT_RANDOM_SEED)

        frame["split"] = "train"
        frame.loc[val_idx, "split"] = "validation"
        frame.loc[test_idx, "split"] = "test"

        calibrator = fit_logistic_calibrator(frame, labels=labels, train_idx=train_idx)
        frame["p_correct"] = predict_probabilities(frame, calibrator=calibrator, labels=labels, train_idx=train_idx)

        val_probs = frame.iloc[val_idx]["p_correct"].astype(float).tolist()
        val_correct = frame.iloc[val_idx]["top1_correct"].astype(bool).tolist()
        threshold, val_cov, val_acc = best_threshold_for_target_accuracy(
            probabilities=val_probs,
            correct=val_correct,
            target_accuracy=DEFAULT_ACCEPTANCE_TARGET,
        )

        test_frame = frame.iloc[test_idx].copy()
        if len(test_frame) == 0:
            test_frame = frame.copy()
    else:
        frame["split"] = "test"
        frame["p_correct"] = frame["predicted_top1_score"].astype(float)
        threshold = 2.0
        val_cov = 0.0
        val_acc = 0.0
        test_frame = frame.copy()

    test_probs = test_frame["p_correct"].astype(float).tolist()
    test_correct = test_frame["top1_correct"].astype(bool).tolist()

    coverages_by_target: Dict[float, float] = {}
    for target in COVERAGE_TARGETS:
        coverages_by_target[target] = coverage_at_target_accuracy(
            confidences=test_probs,
            correct=test_correct,
            target_accuracy=target,
        )

    auto_mask = test_frame["p_correct"] >= threshold
    auto_frame = test_frame[auto_mask]
    manual_frame = test_frame[~auto_mask]

    auto_coverage = float(len(auto_frame) / len(test_frame)) if len(test_frame) > 0 else 0.0
    auto_accuracy = float(auto_frame["top1_correct"].mean()) if len(auto_frame) > 0 else 0.0
    manual_hit10 = float(manual_frame["hit@10"].mean()) if len(manual_frame) > 0 else 0.0

    _, _, aurc = risk_coverage_curve(probabilities=test_probs, correct=test_correct)

    ci_hit1 = bootstrap_metric_ci(test_frame, metric_name="hit@1", bootstrap_samples=BOOTSTRAP_SAMPLES, random_state=SPLIT_RANDOM_SEED)
    ci_hit10 = bootstrap_metric_ci(test_frame, metric_name="hit@10", bootstrap_samples=BOOTSTRAP_SAMPLES, random_state=SPLIT_RANDOM_SEED)
    ci_mrr10 = bootstrap_metric_ci(test_frame, metric_name="mrr@10", bootstrap_samples=BOOTSTRAP_SAMPLES, random_state=SPLIT_RANDOM_SEED)
    ci_ndcg10 = bootstrap_metric_ci(test_frame, metric_name="ndcg@10", bootstrap_samples=BOOTSTRAP_SAMPLES, random_state=SPLIT_RANDOM_SEED)
    ci_cov95 = bootstrap_metric_ci(
        test_frame,
        metric_name="coverage@95",
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        random_state=SPLIT_RANDOM_SEED,
        target_for_coverage=0.95,
    )

    summary = {
        "model": model_name,
        "mode": mode,
        "cases": str(len(test_frame)),
        "hit@1": f"{float(test_frame['hit@1'].mean()):.6f}",
        "hit@5": f"{float(test_frame['hit@5'].mean()):.6f}",
        "hit@10": f"{float(test_frame['hit@10'].mean()):.6f}",
        "mrr": f"{float(test_frame['mrr@10'].mean()):.6f}",
        "map@10": f"{float(test_frame['map@10'].mean()):.6f}",
        "ndcg@10": f"{float(test_frame['ndcg@10'].mean()):.6f}",
        "recall@10": f"{float(test_frame['recall@10'].mean()):.6f}",
        "coverage_at_90acc": f"{coverages_by_target[0.90]:.6f}",
        "coverage_at_95acc": f"{coverages_by_target[0.95]:.6f}",
        "coverage_at_97acc": f"{coverages_by_target[0.97]:.6f}",
        "coverage_at_99acc": f"{coverages_by_target[0.99]:.6f}",
        "coverage_at_95acc_margin": f"{coverages_by_target[0.95]:.6f}",
        "coverage_at_90acc_margin": f"{coverages_by_target[0.90]:.6f}",
        "auto_threshold": f"{threshold:.6f}",
        "auto_coverage": f"{auto_coverage:.6f}",
        "auto_accuracy": f"{auto_accuracy:.6f}",
        "manual_hit@10": f"{manual_hit10:.6f}",
        "aurc": f"{aurc:.6f}",
        "val_coverage_at_target": f"{val_cov:.6f}",
        "val_accuracy_at_target": f"{val_acc:.6f}",
        "avg_expected_score": f"{float(test_frame['predicted_top1_score'].mean()):.6f}",
        "hit@1_ci_low": f"{ci_hit1[0]:.6f}",
        "hit@1_ci_high": f"{ci_hit1[1]:.6f}",
        "hit@10_ci_low": f"{ci_hit10[0]:.6f}",
        "hit@10_ci_high": f"{ci_hit10[1]:.6f}",
        "mrr@10_ci_low": f"{ci_mrr10[0]:.6f}",
        "mrr@10_ci_high": f"{ci_mrr10[1]:.6f}",
        "ndcg@10_ci_low": f"{ci_ndcg10[0]:.6f}",
        "ndcg@10_ci_high": f"{ci_ndcg10[1]:.6f}",
        "coverage@95_ci_low": f"{ci_cov95[0]:.6f}",
        "coverage@95_ci_high": f"{ci_cov95[1]:.6f}",
    }

    details = []
    for _, row in frame.iterrows():
        details.append(
            {
                "model": model_name,
                "query_id": str(int(row["query_id"])),
                "split": str(row["split"]),
                "query": str(row["query"]),
                "relevant_input": str(row["relevant_input"]),
                "relevant_resolved": str(row["relevant_resolved"]),
                "relevant_count": str(int(row["relevant_count"])),
                "predicted_top1": str(row["predicted_top1"]),
                "predicted_top1_score": f"{float(row['predicted_top1_score']):.6f}",
                "p_correct": f"{float(row['p_correct']):.6f}",
                "auto_accept": str(bool(float(row["p_correct"]) >= threshold)),
                "expected_rank": str(int(row["expected_rank"])),
                "top1_correct": str(bool(row["top1_correct"])),
                "hit@1": f"{float(row['hit@1']):.6f}",
                "hit@5": f"{float(row['hit@5']):.6f}",
                "hit@10": f"{float(row['hit@10']):.6f}",
                "mrr@10": f"{float(row['mrr@10']):.6f}",
                "ndcg@10": f"{float(row['ndcg@10']):.6f}",
                "map@10": f"{float(row['map@10']):.6f}",
                "recall@10": f"{float(row['recall@10']):.6f}",
                "s1": f"{float(row['s1']):.6f}",
                "s2": f"{float(row['s2']):.6f}",
                "gap12": f"{float(row['gap12']):.6f}",
                "gap1_5": f"{float(row['gap1_5']):.6f}",
                "mean90": f"{float(row['mean90']):.6f}",
                "std90": f"{float(row['std90']):.6f}",
                "z1": f"{float(row['z1']):.6f}",
                "z_gap": f"{float(row['z_gap']):.6f}",
                "entropy_top10": f"{float(row['entropy_top10']):.6f}",
                "top10_materials": str(row["top10_materials"]),
                "top10_scores": str(row["top10_scores"]),
            }
        )

    return PerModelResult(summary=summary, details=details)


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


def main() -> None:
    args = parse_args()
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
    print(f"Using eval mode: {args.mode}")

    cases = build_evaluation_cases(query_file=query_file, expected_file=expected_file)

    with sqlite3.connect(DATABASE_PATH) as connection:
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
            mode=args.mode,
        )

        summary_rows.append(result.summary)
        detail_rows.extend(result.details)

        print(
            f"  Hit@1: {float(result.summary['hit@1']):.2%} | "
            f"Hit@10: {float(result.summary['hit@10']):.2%} | "
            f"MRR@10: {float(result.summary['mrr']):.4f} | "
            f"nDCG@10: {float(result.summary['ndcg@10']):.4f} | "
            f"Cov@95: {float(result.summary['coverage_at_95acc']):.2%}"
        )

    summary_rows.sort(key=lambda r: (float(r["hit@1"]), float(r["mrr"]), float(r["ndcg@10"])), reverse=True)

    summary_fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    details_fieldnames = list(detail_rows[0].keys()) if detail_rows else []

    summary_labeled = output_dir / f"summary_{query_label}.csv"
    details_labeled = output_dir / f"details_{query_label}.csv"
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
