from __future__ import annotations

import re
from typing import Iterable

STRENGTH_TOKEN_PATTERN = re.compile(r"^c\d{1,2}/\d{1,2}$", re.IGNORECASE)
NUMERIC_TOKEN_PATTERN = re.compile(r"^\d+(?:[.,]\d+)?$")
NPK_TOKEN_PATTERN = re.compile(r"^npk$|^[a-z]\d(?:[+-]\d)?$", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[a-z0-9/._+-]+", re.IGNORECASE)

QUERY_FAMILY_STOPWORDS = {
    "insitu",
    "precast",
    "material",
    "klasse",
    "class",
    "grade",
}


def normalize_text_key(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def normalize_material_key(value: str) -> str:
    return normalize_text_key(value)


def tokenize_query(value: str) -> list[str]:
    return [token for token in TOKEN_PATTERN.findall(normalize_text_key(value)) if token]


def is_strength_token(token: str) -> bool:
    return bool(STRENGTH_TOKEN_PATTERN.fullmatch(token))


def _drop_variable_token(token: str) -> bool:
    if not token:
        return True
    if token in QUERY_FAMILY_STOPWORDS:
        return True
    if is_strength_token(token):
        return True
    if NUMERIC_TOKEN_PATTERN.fullmatch(token):
        return True
    if NPK_TOKEN_PATTERN.fullmatch(token):
        return True
    return False


def query_semantic_tokens(query: str) -> list[str]:
    tokens = tokenize_query(query)
    filtered = [token for token in tokens if not _drop_variable_token(token)]
    if not filtered:
        return tokens
    return filtered


def query_family_key(query: str, max_tokens: int = 4) -> str:
    tokens = query_semantic_tokens(query)
    if not tokens:
        return normalize_text_key(query)

    if tokens[0].startswith("ifc") and len(tokens) >= 3:
        return " ".join(tokens[: min(3, len(tokens))])

    return " ".join(tokens[:max_tokens])


def jaccard_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = {token for token in left if token}
    right_set = {token for token in right if token}
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def are_queries_semantically_close(query_a: str, query_b: str, threshold: float = 0.60) -> bool:
    tokens_a = query_semantic_tokens(query_a)
    tokens_b = query_semantic_tokens(query_b)
    return jaccard_similarity(tokens_a, tokens_b) >= threshold
