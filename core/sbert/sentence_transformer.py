# 



import sqlite3
import difflib
from typing import List
import numpy as np
import torch
import os
from sentence_transformers import SentenceTransformer, SimilarityFunction, util
from sentence_transformers import CrossEncoder
import sys
import json
import re
import unicodedata
from pathlib import Path
from functools import lru_cache

try:
    from .batch_benchmark import recommend_batch_size
except ImportError:
    from batch_benchmark import recommend_batch_size


# --- Configuration ---
_BASE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BASE_DIR.parent.parent
SBERT_MODELS_DIR = _PROJECT_ROOT / "models"

DATABASE_PATH = os.environ.get("KBOB_DATABASE_PATH", str(_PROJECT_ROOT / "Ökobilanzdaten.sqlite3"))
TABLE_NAME = "Oekobilanzdaten"
COLUMN_MATERIAL = "Material"
MODEL_NAME = "BAAI/bge-m3"  # Change model name here

TOP_K_RESULTS = 30
SIMILARITY_FUNCTION = SimilarityFunction.COSINE  # Alternatives: DOT_PRODUCT, EUCLIDEAN, MANHATTAN

# --- Cross-Encoder Reranking ---
CROSS_ENCODER_MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"
RERANK_TOP_N = 30  # how many bi-encoder hits to re-rank (all by default)
CROSS_ENCODER_REVISION = os.environ.get("SBERT_CROSS_ENCODER_REVISION", "").strip() or None
CROSS_ENCODER_ALLOW_REMOTE_CODE_UPDATES = os.environ.get("SBERT_CROSS_ENCODER_ALLOW_UPDATES", "0").strip().lower() in {
    "1", "true", "yes", "on"
}

# For tiny workloads (e.g., 90 short strings), CPU is often faster overall due to CUDA init overhead.
# Override via environment variable: SBERT_DEVICE=cpu|cuda
SBERT_DEVICE = os.environ.get("SBERT_DEVICE", "").strip().lower()
CUDA_QUERY_THRESHOLD = int(os.environ.get("SBERT_CUDA_QUERY_THRESHOLD", "500"))
ENCODE_BATCH_SIZE = int(os.environ.get("SBERT_BATCH_SIZE", "64"))
NORMALIZE_EMBEDDINGS = True
AUTO_BENCH_BATCH = os.environ.get("SBERT_AUTO_BENCH_BATCH", "").strip().lower() in {"1", "true", "yes", "on"}
AUTO_HEURISTIC_BATCH = os.environ.get("SBERT_AUTO_HEURISTIC_BATCH", "1").strip().lower() in {"1", "true", "yes", "on"}
HAS_MANUAL_BATCH_SIZE = "SBERT_BATCH_SIZE" in os.environ


def resolve_runtime_device(query_count: int) -> str:
    if SBERT_DEVICE in {"cpu", "cuda"}:
        return SBERT_DEVICE
    if query_count >= CUDA_QUERY_THRESHOLD and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_heuristic_batch_size(query_count: int, device: str) -> int:
    if device == "cuda" and query_count >= 500:
        return 64
    if device == "cpu" and query_count <= 200:
        return 16
    return ENCODE_BATCH_SIZE


# --- Database Query ---
# KBOB materials excluded from SBERT matching — these are used only for
# synthetic add-on rows (e.g. galvanization) and should not appear in the
# AI-Mapping selectbox as top matches.
SBERT_EXCLUDED_MATERIALS: set[str] = {"Verzinken"}


def fetch_materials_from_db(connection: sqlite3.Connection) -> List[str]:
    cursor = connection.cursor()
    cursor.execute(f"SELECT {COLUMN_MATERIAL} FROM {TABLE_NAME} WHERE {COLUMN_MATERIAL} IS NOT NULL")
    materials = [
        row[0] for row in cursor.fetchall()
        if str(row[0]).strip() != "" and row[0] not in SBERT_EXCLUDED_MATERIALS
    ]
    # Deduplicate while preserving order (reduces corpus size and speeds up encoding/search)
    return list(dict.fromkeys(materials))

# --- Load or Save Model ---
def _resolve_model_name(model_name: str | None = None) -> str:
    if model_name is None:
        return MODEL_NAME
    normalized = str(model_name).strip()
    if not normalized:
        return MODEL_NAME
    local_model_path = _resolve_local_model_path(normalized)
    return str(local_model_path) if local_model_path is not None else normalized


def _resolve_local_model_path(model_name: str) -> Path | None:
    candidate = Path(model_name)
    candidates = [candidate] if candidate.is_absolute() else [_PROJECT_ROOT / candidate, candidate]
    for local_candidate in candidates:
        try:
            if local_candidate.is_dir():
                return local_candidate.resolve()
        except OSError:
            continue
    return None


def _is_sentence_transformer_directory(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    has_modules = (directory / "modules.json").is_file()
    has_sbert_config = (directory / "config_sentence_transformers.json").is_file() or (
        directory / "sentence_bert_config.json"
    ).is_file()
    return has_modules and has_sbert_config


def _model_directory_for(model_name: str) -> str:
    model_path = Path(model_name)
    if model_path.is_absolute():
        return str(model_path)
    return str(SBERT_MODELS_DIR / model_name)


def load_or_save_model(model_name: str | None = None, device: str = "cpu") -> SentenceTransformer:
    resolved_model_name = _resolve_model_name(model_name)
    resolved_model_path = Path(resolved_model_name)
    if _is_sentence_transformer_directory(resolved_model_path):
        return SentenceTransformer(str(resolved_model_path), device=device)

    model_directory = Path(_model_directory_for(resolved_model_name))
    if _is_sentence_transformer_directory(model_directory):
        return SentenceTransformer(str(model_directory), device=device)

    model = SentenceTransformer(resolved_model_name, device=device)
    if not resolved_model_path.is_absolute():
        model_directory.mkdir(parents=True, exist_ok=True)
        model.save(str(model_directory))
    return model

# --- JSONL Processing ---
IFC_EXPORT_FIELDS = [
    "IfcEntity",
    "PredefinedType",
    "Material",
    "Durchmesser",
    "CastingMethod",
    "StrengthClass",
]

# "Name"
# "ReinforcementStrengthClass",
# "Description",
# "StructuralClass",
# "ExposureClass",
# "comment", (nur für Tekla 2025)


_BAUGRUBENSICHERUNG_WALL_TYPE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "Bohrpfahlwand": (
        "bohrpfahlwand",
        "bohrpfahl",
        "bohrpfahlw",
        "paroi de pieux fores",
        "pieux fores",
        "parete di pali trivellati",
        "pali trivellati",
        "bored pile wall",
        "bored pile",
    ),
    "Nagelwand": (
        "nagelwand",
        "bodennagelung",
        "paroi clouee",
        "clouage",
        "parete chiodata",
        "soil nail wall",
        "soil nailing",
    ),
    "Rühlwand": (
        "ruhlwand",
        "berliner verbau",
        "berliner wand",
        "paroi berlinoise",
        "berlinese",
        "soldier pile wall",
        "berlin wall",
    ),
    "Schlitzwand": (
        "schlitzwand",
        "paroi moulee",
        "parete a diaframma",
        "diaframma",
        "diaphragm wall",
    ),
    "Spundwand": (
        "spundwand",
        "palplanche",
        "rideau de palplanches",
        "palancola",
        "palancolata",
        "sheet pile wall",
        "sheet pile",
    ),
}

_BAUGRUBENSICHERUNG_VARIANT_SYNONYMS: dict[str, tuple[str, ...]] = {
    "auskragend": (
        "auskragend",
        "freitragend",
        "en console",
        "a sbalzo",
        "cantilever",
        "cantilevered",
    ),
    "gespriesst": (
        "gespriesst",
        "gespriest",
        "gespriesst",
        "spriess",
        "spriessung",
        "butonne",
        "puntellato",
        "braced",
        "strutted",
    ),
    "unverankert": (
        "unverankert",
        "nicht verankert",
        "non ancre",
        "non ancorato",
        "unanchored",
    ),
    "verankert": (
        "verankert",
        "anker",
        "ancre",
        "ancorato",
        "anchored",
    ),
}

_BAUGRUBENSICHERUNG_SIZE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "400": (re.compile(r"\b(?:d\s*)?400\b"),),
    "800": (re.compile(r"\b(?:d\s*)?800\b"),),
}


def _normalize_disambiguation_text(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = text.replace("ß", "ss")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _best_group_match(
    normalized_name: str,
    words: list[str],
    synonyms_by_token: dict[str, tuple[str, ...]],
    cutoff: float,
) -> str | None:
    best_token: str | None = None
    best_score = 0.0

    for token, raw_synonyms in synonyms_by_token.items():
        for synonym in raw_synonyms:
            normalized_synonym = _normalize_disambiguation_text(synonym)
            if not normalized_synonym:
                continue

            score = 0.0
            if " " in normalized_synonym:
                if normalized_synonym in normalized_name:
                    score = 1.0
                else:
                    synonym_word_count = len(normalized_synonym.split())
                    if synonym_word_count <= len(words):
                        for index in range(len(words) - synonym_word_count + 1):
                            ngram = " ".join(words[index : index + synonym_word_count])
                            score = max(score, difflib.SequenceMatcher(None, normalized_synonym, ngram).ratio())
            else:
                if normalized_synonym in words:
                    score = 1.0
                else:
                    candidates = difflib.get_close_matches(
                        normalized_synonym,
                        words,
                        n=1,
                        cutoff=cutoff,
                    )
                    if candidates:
                        score = difflib.SequenceMatcher(None, normalized_synonym, candidates[0]).ratio()

            if score >= cutoff and score > best_score:
                best_score = score
                best_token = token

    return best_token


def extract_disambiguation_tokens(
    name: str | None,
    ifc_entity: str | None = None,
) -> list[str]:
    entity_upper = str(ifc_entity or "").strip().upper()
    if entity_upper and not entity_upper.startswith("IFCWALL"):
        return []

    normalized_name = _normalize_disambiguation_text(name)
    if not normalized_name:
        return []

    words = normalized_name.split()
    wall_type = _best_group_match(
        normalized_name,
        words,
        _BAUGRUBENSICHERUNG_WALL_TYPE_SYNONYMS,
        cutoff=0.78,
    )
    if wall_type is None:
        return []

    tokens: list[str] = [wall_type]
    variant = _best_group_match(
        normalized_name,
        words,
        _BAUGRUBENSICHERUNG_VARIANT_SYNONYMS,
        cutoff=0.76,
    )
    if variant is not None and variant not in tokens:
        tokens.append(variant)

    for size, patterns in _BAUGRUBENSICHERUNG_SIZE_PATTERNS.items():
        if any(pattern.search(normalized_name) for pattern in patterns):
            tokens.append(size)

    return tokens



# Bi-encoder global cache
_global_sbert_models: dict[tuple[str, str], SentenceTransformer] = {}

# Cross-encoder global cache
_global_cross_encoder_models: dict[tuple[str, str], CrossEncoder] = {}


def get_global_sbert_model(model_name: str | None = None, device: str = "cpu"):
    resolved_model_name = _resolve_model_name(model_name)
    key = (resolved_model_name, device)
    if key not in _global_sbert_models:
        _global_sbert_models[key] = load_or_save_model(model_name=resolved_model_name, device=device)
        # Warmup to reduce first-call latency in interactive runs (Streamlit dev)
        with torch.inference_mode():
            _ = _global_sbert_models[key].encode(["warmup"], batch_size=1, show_progress_bar=False)
    return _global_sbert_models[key]


def _normalize_biencoder_score(score: float) -> float:
    """Cosine-Similarity [-1, 1] → [0, 1] via linearer Skalierung."""
    return (score + 1.0) / 2.0


def _normalize_cross_encoder_scores(raw_scores: np.ndarray) -> list[float]:
    """
    Normalisiert Cross-Encoder Rohscores auf [0, 1].

    - 1D-Output (MS MARCO / Jina / BGE): Sigmoid
    - 2D-Output mit 2 Klassen:           Softmax → Positive-Spalte (Index 1)
    """
    scores = np.array(raw_scores)

    if scores.ndim == 1:
        # MS MARCO, Jina, BGE: einzelner Relevanz-Logit → Sigmoid
        return (1.0 / (1.0 + np.exp(-scores))).tolist()

    elif scores.ndim == 2:
        n_classes = scores.shape[1]
        exp_s = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs = exp_s / exp_s.sum(axis=1, keepdims=True)
        # positive Klasse = letzte Spalte
        return probs[:, n_classes - 1].tolist()

    else:
        # Fallback: Min-Max Normalisierung
        s_min, s_max = scores.min(), scores.max()
        return ((scores - s_min) / (s_max - s_min + 1e-8)).tolist()


def load_or_get_cross_encoder(model_name: str, device: str) -> CrossEncoder:
    """
    Lädt ein CrossEncoder-Modell und cacht es im globalen Dict.
    Speichert das Modell lokal unter models/cross-encoder/<model_name>/
    trust_remote_code=True ist generisch gesetzt (zwingend für Jina, ignoriert von BAAI/mmarco).
    """
    key = (model_name, device)
    if key not in _global_cross_encoder_models:
        save_path = SBERT_MODELS_DIR / "cross-encoder" / model_name.replace("/", "_")
        ce_kwargs = {
            "device": device,
            "trust_remote_code": True,
        }
        if CROSS_ENCODER_REVISION:
            ce_kwargs["revision"] = CROSS_ENCODER_REVISION

        if save_path.exists():
            print(f"[CrossEncoder] Lade lokal: {save_path}")
            ce = CrossEncoder(
                str(save_path),
                **ce_kwargs,
            )
        else:
            print(f"[CrossEncoder] Lade von HuggingFace: {model_name}")
            if not CROSS_ENCODER_ALLOW_REMOTE_CODE_UPDATES:
                print("[CrossEncoder] Hinweis: SBERT_CROSS_ENCODER_ALLOW_UPDATES=0 (Default).")
                if CROSS_ENCODER_REVISION:
                    print(f"[CrossEncoder] Nutze fixierte Revision: {CROSS_ENCODER_REVISION}")
                else:
                    print("[CrossEncoder] Keine Revision fixiert. Für reproduzierbare Läufe SBERT_CROSS_ENCODER_REVISION setzen.")
            ce = CrossEncoder(
                model_name,
                **ce_kwargs,
            )
            save_path.mkdir(parents=True, exist_ok=True)
            ce.save(str(save_path))
            print(f"[CrossEncoder] Gespeichert unter: {save_path}")
        _global_cross_encoder_models[key] = ce
    return _global_cross_encoder_models[key]


def rerank_with_cross_encoder(
    query_texts: list[str],
    top_k_results_per_query: list[list[dict]],
    ce_model_name: str,
    device: str,
) -> list[list[dict]]:
    """
    Re-rankt Bi-Encoder Treffer mit einem Cross-Encoder.
    Folgt dem sbert.net Retrieve & Re-Rank Muster:
    https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html

    Args:
        query_texts:              Liste der Query-Texte (IFC-Materialnamen)
        top_k_results_per_query:  Bi-Encoder top_k_matches pro Query
        ce_model_name:            HuggingFace Modell-ID oder lokaler Pfad
        device:                   "cpu" oder "cuda"

    Returns:
        Neue top_k_matches-Liste — nach Cross-Encoder-Score absteigend sortiert.
        Jeder Treffer enthält zusätzlich 'biencoder_score' (originaler Bi-Encoder Score).
    """
    ce_model = load_or_get_cross_encoder(ce_model_name, device)
    reranked_results: list[list[dict]] = []

    for query, hits in zip(query_texts, top_k_results_per_query):
        if not hits:
            reranked_results.append(hits)
            continue

        # sbert.net Muster: [[query, candidate], ...] Paare
        cross_inp = [[query, hit["material"]] for hit in hits]

        # Rohe Logits — Normalisierung via _normalize_cross_encoder_scores
        raw_scores = ce_model.predict(cross_inp, apply_softmax=False)
        normalized_scores = _normalize_cross_encoder_scores(np.array(raw_scores))

        # biencoder_score sichern, Cross-Encoder Score setzen
        for hit, ce_score in zip(hits, normalized_scores):
            hit["biencoder_score"] = hit.get("score", 0.0)
            hit["score"] = float(ce_score)

        # Absteigend nach Cross-Encoder Score sortieren
        reranked = sorted(hits, key=lambda h: h["score"], reverse=True)
        reranked_results.append(reranked)

    return reranked_results


@lru_cache(maxsize=32)
def get_cached_corpus(
    model_name: str = MODEL_NAME,
    device: str = "cpu",
    batch_size: int = ENCODE_BATCH_SIZE,
) -> tuple[list[str], torch.Tensor]:
    """Load materials from DB and precompute embeddings once per process."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        materials = fetch_materials_from_db(connection)

    model = get_global_sbert_model(model_name=model_name, device=device)
    model.similarity_fn_name = SIMILARITY_FUNCTION

    with torch.inference_mode():
        embeddings = model.encode(
            materials,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            show_progress_bar=False,
        )
    return materials, embeddings

def load_ifc_jsonl_entries(jsonl_path: str) -> list:
    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries

def ifc_entry_to_string(entry: dict) -> str:
    # Erzeuge String wie TXT-Export: Werte mit Leerzeichen trennen, leere Felder überspringen
    values = []
    for field in IFC_EXPORT_FIELDS:
        val = entry.get(field, "")
        if isinstance(val, list):
            val = ", ".join(str(v) for v in val if v)
        if val is not None and str(val).strip() != "" and str(val).strip() != "NOTDEFINED" and str(val).strip() != "Undefined":
            values.append(str(val).strip())

    for token in extract_disambiguation_tokens(
        entry.get("Name", ""),
        ifc_entity=entry.get("IfcEntity", ""),
    ):
        if token not in values:
            values.append(token)
    return " ".join(values)


def benchmark_recommend_batch_size(jsonl_path: str, model_name: str | None = None, verbose: bool = True) -> int:
    resolved_model_name = _resolve_model_name(model_name)
    entries = load_ifc_jsonl_entries(jsonl_path)
    queries = [ifc_entry_to_string(e) for e in entries]
    device = resolve_runtime_device(len(queries))
    model = get_global_sbert_model(model_name=resolved_model_name, device=device)
    return recommend_batch_size(
        queries=queries,
        device=device,
        model=model,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        default_batch_size=ENCODE_BATCH_SIZE,
        verbose=verbose,
    )


def find_most_similar_db_entries(jsonl_path: str, model_name: str | None = None, cross_encoder_model_name: str | None = None):
    resolved_model_name = _resolve_model_name(model_name)
    print(f"Using model: {resolved_model_name}")
    # IFC-Export als Queries laden
    entries = load_ifc_jsonl_entries(jsonl_path)
    queries = [ifc_entry_to_string(e) for e in entries]
    query_count = len(queries)
    device = resolve_runtime_device(query_count)
    if HAS_MANUAL_BATCH_SIZE:
        selected_batch_size = ENCODE_BATCH_SIZE
        batch_mode = "manual-env"
    elif AUTO_BENCH_BATCH:
        selected_batch_size = benchmark_recommend_batch_size(jsonl_path, model_name=resolved_model_name)
        batch_mode = "auto-bench"
    elif AUTO_HEURISTIC_BATCH:
        selected_batch_size = resolve_heuristic_batch_size(query_count, device)
        batch_mode = "heuristic"
    else:
        selected_batch_size = ENCODE_BATCH_SIZE
        batch_mode = "default"
    print(f"Number of IFC queries: {query_count}")
    # Zeige zusätzlich die Anzahl deduplizierter Query-Strings (Anzeige only)
    unique_queries = list(dict.fromkeys(queries))
    print(f"Number of IFC queries deduplicated: {len(unique_queries)}")
    print(f"Runtime device: {device} (threshold={CUDA_QUERY_THRESHOLD})")
    print(f"SBERT batch size: {selected_batch_size} ({batch_mode})")

    # Load corpus (DB) once per process
    materials, material_embeddings = get_cached_corpus(
        model_name=resolved_model_name,
        device=device,
        batch_size=selected_batch_size,
    )
    print(f"Number of KBOB entries: {len(materials)}")

    model = get_global_sbert_model(model_name=resolved_model_name, device=device)
    model.similarity_fn_name = SIMILARITY_FUNCTION

    with torch.inference_mode():
        query_embeddings = model.encode(
            queries,
            batch_size=selected_batch_size,
            convert_to_tensor=True,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            show_progress_bar=False,
        )

    hits = util.semantic_search(
        query_embeddings,
        material_embeddings,
        top_k=min(TOP_K_RESULTS, len(materials)),
    )

    # Build initial top_k_matches from bi-encoder hits
    top_k_per_query: list[list[dict]] = []
    for query_index in range(len(queries)):
        top = hits[query_index]
        matches = [
            {
                "material": materials[int(h["corpus_id"])],
                "score": round(_normalize_biencoder_score(float(h["score"])), 3),
            }
            for h in top
        ]
        top_k_per_query.append(matches)

    # --- Cross-Encoder Re-Ranking (Retrieve & Re-Rank) ---
    # https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html
    if cross_encoder_model_name is not None:
        rerank_n = min(RERANK_TOP_N, len(top_k_per_query[0]) if top_k_per_query else 0)
        print(f"Cross-Encoder re-ranking top {rerank_n} hits with: {cross_encoder_model_name}")
        candidates_per_query = [m[:rerank_n] for m in top_k_per_query]
        reranked = rerank_with_cross_encoder(
            query_texts=queries,
            top_k_results_per_query=candidates_per_query,
            ce_model_name=cross_encoder_model_name,
            device=device,
        )
        # Merge: reranked hits first, append any remaining bi-encoder hits not in top rerank_n
        for qi in range(len(queries)):
            top_k_per_query[qi] = reranked[qi] + top_k_per_query[qi][rerank_n:]

    for query_index, query_text in enumerate(queries):
        entries[query_index]["top_k_matches"] = top_k_per_query[query_index]

    print(f"Query: {query_text}")
    for rank, m in enumerate(top_k_per_query[-1][:10], 1):
        print(f"  {rank}. {m['material']} (Score={m['score']:.4f})")
    print()

    # Überschreibe die JSONL-Datei mit den neuen Feldern
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def run_sbert_matching(jsonl_path: str, model_name: str | None = None, cross_encoder_model_name: str | None = None):
    find_most_similar_db_entries(jsonl_path, model_name=model_name, cross_encoder_model_name=cross_encoder_model_name)


if __name__ == "__main__":
    # Erlaube Übergabe des JSONL-Pfads als Argument
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python Sentence_Transformer_V00.py <path-to-jsonl> [model-name]")
        print("  python Sentence_Transformer_V00.py --benchmark-batch <path-to-jsonl> [model-name]")
        raise SystemExit(1)

    if sys.argv[1] == "--benchmark-batch":
        if len(sys.argv) < 3:
            print("Usage: python Sentence_Transformer_V00.py --benchmark-batch <path-to-jsonl> [model-name]")
            raise SystemExit(1)
        JSONL_PATH = sys.argv[2]
        CLI_MODEL_NAME = sys.argv[3] if len(sys.argv) > 3 else None
        benchmark_recommend_batch_size(JSONL_PATH, model_name=CLI_MODEL_NAME, verbose=True)
        raise SystemExit(0)

    JSONL_PATH = sys.argv[1]
    CLI_MODEL_NAME = sys.argv[2] if len(sys.argv) > 2 else None
    run_sbert_matching(JSONL_PATH, model_name=CLI_MODEL_NAME)