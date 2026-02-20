import csv
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import torch
from sentence_transformers import SentenceTransformer, SimilarityFunction


# Top1 Accuracy: Anteil Queries, bei denen das richtige Material auf Rang 1 steht. Beispiel: 0.8 bedeutet 8 von 10 direkt korrekt.
# Top5 Accuracy: Anteil Queries, bei denen das richtige Material irgendwo in den Top 5 steht.
# MRR: bewertet den Rang des richtigen Treffers (MRR = (1/N)*∑ (1/Rang)). Rang 1 zählt voll, Rang 2 nur 0.5, Rang 3 nur 0.33.
# Avg expected score: mittlerer Similarity-Score des korrekten Materials (nur als internes Vertrauenssignal pro Modell, nicht perfekt modellübergreifend vergleichbar).
# Die Score-Höhe allein ist nicht das wichtigste Kriterium; Ranking-Metriken (Top1/Top5/MRR) sind für Zuordnung robuster.



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
]

TOP_K = 5
SIMILARITY_FUNCTION = SimilarityFunction.COSINE
SBERT_DEVICE = os.environ.get("SBERT_DEVICE", "").strip().lower()
SBERT_QUERY_FILE = os.environ.get("SBERT_QUERY_FILE", "").strip()
SBERT_EXPECTED_FILE = os.environ.get("SBERT_EXPECTED_FILE", "").strip()

DEFAULT_QUERY_FILE_RELATIVE = Path("static") / "Bohrpfahl_4.3_sbert_queries.txt"
DEFAULT_EVALUATION_OUTPUT_RELATIVE = Path("Evaluation") / "exports" / "model_evaluation"

DEFAULT_EXPECTED_MATERIAL = "Tiefgründung Ortbetonbohrpfahl 900"
EXPECTED_BY_QUERY_OVERRIDES: Dict[str, str] = {}


@dataclass
class EvaluationCase:
    query: str
    expected_material: str


@dataclass
class PerModelSummary:
    model_name: str
    top1_accuracy: float
    topk_accuracy: float
    mrr: float
    avg_expected_score: float
    cases: int


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
        raise FileNotFoundError(f"Expected-Material-Datei nicht gefunden: {expected_file}")

    expected_materials: List[str] = []
    with expected_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            material = line.strip()
            if material:
                expected_materials.append(material)

    if len(expected_materials) != expected_count:
        raise ValueError(
            "Anzahl Expected-Material-Zeilen passt nicht zur Anzahl Queries: "
            f"{len(expected_materials)} != {expected_count}"
        )

    return expected_materials


def build_evaluation_cases(query_file: Path, expected_file: Path | None = None) -> List[EvaluationCase]:
    queries = load_queries_from_txt(query_file)
    expected_from_file: List[str] | None = None
    if expected_file is not None:
        expected_from_file = load_expected_materials_from_txt(expected_file, expected_count=len(queries))

    cases: List[EvaluationCase] = []
    for index, query in enumerate(queries):
        expected_default = expected_from_file[index] if expected_from_file is not None else DEFAULT_EXPECTED_MATERIAL
        expected = EXPECTED_BY_QUERY_OVERRIDES.get(query, expected_default).strip()
        if not expected:
            raise ValueError(
                "Setze DEFAULT_EXPECTED_MATERIAL oder EXPECTED_BY_QUERY_OVERRIDES, "
                "damit jede Query ein korrektes Zielmaterial hat."
            )
        cases.append(EvaluationCase(query=query, expected_material=expected))
    return cases


def resolve_expected_material(
    expected: str,
    materials: Sequence[str],
    exact_index: Dict[str, List[int]],
    normalized_index: Dict[str, List[int]],
) -> Tuple[str, List[int]]:
    if expected in exact_index:
        return expected, exact_index[expected]

    normalized_expected = normalize(expected)
    if normalized_expected in normalized_index:
        first_index = normalized_index[normalized_expected][0]
        canonical = materials[first_index]
        return canonical, normalized_index[normalized_expected]

    suggestions = [m for m in materials if normalized_expected in normalize(m)][:5]
    suggestion_text = "; ".join(suggestions) if suggestions else "keine"
    raise ValueError(
        f"Expected Material '{expected}' wurde in der DB nicht gefunden. "
        f"Vorschläge: {suggestion_text}"
    )


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


def evaluate_model(
    model_name: str,
    materials: Sequence[str],
    cases: Sequence[EvaluationCase],
    exact_index: Dict[str, List[int]],
    normalized_index: Dict[str, List[int]],
    project_root: Path,
    top_k: int,
) -> Tuple[PerModelSummary, List[Dict[str, str]]]:
    device = choose_device()
    model = load_or_save_model(model_name=model_name, project_root=project_root, device=device)
    model.similarity_fn_name = SIMILARITY_FUNCTION

    queries = [case.query for case in cases]
    query_embeddings = model.encode(queries)
    material_embeddings = model.encode(list(materials))
    similarities = model.similarity(query_embeddings, material_embeddings)

    details: List[Dict[str, str]] = []
    top1_hits = 0
    topk_hits = 0
    reciprocal_rank_sum = 0.0
    expected_score_sum = 0.0

    for index, case in enumerate(cases):
        canonical_expected, expected_indices = resolve_expected_material(
            expected=case.expected_material,
            materials=materials,
            exact_index=exact_index,
            normalized_index=normalized_index,
        )

        scores = similarities[index].tolist()
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        rank_by_index = {material_index: rank for rank, material_index in enumerate(ranked_indices, start=1)}
        expected_rank = min(rank_by_index[i] for i in expected_indices)
        predicted_index = ranked_indices[0]
        predicted_material = materials[predicted_index]

        topk_indices = ranked_indices[:top_k]
        topk_materials = [materials[i] for i in topk_indices]
        topk_scores = [scores[i] for i in topk_indices]

        expected_score = max(scores[i] for i in expected_indices)
        top1_correct = expected_rank == 1
        topk_correct = expected_rank <= top_k

        if top1_correct:
            top1_hits += 1
        if topk_correct:
            topk_hits += 1

        reciprocal_rank_sum += 1.0 / expected_rank
        expected_score_sum += expected_score

        details.append(
            {
                "model": model_name,
                "query": case.query,
                "expected_material_input": case.expected_material,
                "expected_material_resolved": canonical_expected,
                "predicted_top1": predicted_material,
                "predicted_top1_score": f"{scores[predicted_index]:.6f}",
                "expected_rank": str(expected_rank),
                "expected_score": f"{expected_score:.6f}",
                "top1_correct": str(top1_correct),
                f"top{top_k}_correct": str(topk_correct),
                "topk_materials": " | ".join(topk_materials),
                "topk_scores": " | ".join(f"{s:.6f}" for s in topk_scores),
            }
        )

    n = len(cases)
    summary = PerModelSummary(
        model_name=model_name,
        top1_accuracy=top1_hits / n,
        topk_accuracy=topk_hits / n,
        mrr=reciprocal_rank_sum / n,
        avg_expected_score=expected_score_sum / n,
        cases=n,
    )
    return summary, details


def write_csv(file_path: Path, rows: List[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        print(f"Using expected materials file: {expected_file}")

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / DEFAULT_EVALUATION_OUTPUT_RELATIVE

    summary_rows: List[Dict[str, str]] = []
    all_detail_rows: List[Dict[str, str]] = []

    for model_name in MODEL_NAMES:
        print(f"\nEvaluating model: {model_name}")
        summary, details = evaluate_model(
            model_name=model_name,
            materials=materials,
            cases=cases,
            exact_index=exact_index,
            normalized_index=normalized_index,
            project_root=project_root,
            top_k=TOP_K,
        )

        print(
            f"  Top-1: {summary.top1_accuracy:.2%} | "
            f"Top-{TOP_K}: {summary.topk_accuracy:.2%} | "
            f"MRR: {summary.mrr:.4f} | "
            f"Avg expected score: {summary.avg_expected_score:.4f}"
        )

        summary_rows.append(
            {
                "model": summary.model_name,
                "cases": str(summary.cases),
                "top1_accuracy": f"{summary.top1_accuracy:.6f}",
                f"top{TOP_K}_accuracy": f"{summary.topk_accuracy:.6f}",
                "mrr": f"{summary.mrr:.6f}",
                "avg_expected_score": f"{summary.avg_expected_score:.6f}",
            }
        )
        all_detail_rows.extend(details)

    summary_file = output_dir / f"summary_{timestamp}.csv"
    details_file = output_dir / f"details_{timestamp}.csv"

    write_csv(
        file_path=summary_file,
        rows=summary_rows,
        fieldnames=["model", "cases", "top1_accuracy", f"top{TOP_K}_accuracy", "mrr", "avg_expected_score"],
    )

    detail_fields = [
        "model",
        "query",
        "expected_material_input",
        "expected_material_resolved",
        "predicted_top1",
        "predicted_top1_score",
        "expected_rank",
        "expected_score",
        "top1_correct",
        f"top{TOP_K}_correct",
        "topk_materials",
        "topk_scores",
    ]
    write_csv(file_path=details_file, rows=all_detail_rows, fieldnames=detail_fields)

    print("\nDone.")
    print(f"Summary: {summary_file}")
    print(f"Details: {details_file}")


if __name__ == "__main__":
    main()
