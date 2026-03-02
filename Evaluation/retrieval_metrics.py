import math
from typing import Dict, Sequence


def hit_at_k(ranked_indices: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    return 1.0 if any(idx in relevant for idx in ranked_indices[:k]) else 0.0


def recall_at_k(ranked_indices: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    hits = sum(1 for idx in ranked_indices[:k] if idx in relevant)
    return hits / len(relevant)


def mrr_at_k(ranked_indices: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    for rank, idx in enumerate(ranked_indices[:k], start=1):
        if idx in relevant:
            return 1.0 / rank
    return 0.0


def average_precision_at_k_binary(ranked_indices: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, idx in enumerate(ranked_indices[:k], start=1):
        if idx in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant)


def ndcg_at_k_binary(ranked_indices: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0

    dcg = 0.0
    for rank, idx in enumerate(ranked_indices[:k], start=1):
        if idx in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def binary_ranking_metrics_at_10(ranked_indices: Sequence[int], relevant: set[int]) -> Dict[str, float]:
    return {
        "hit@1": hit_at_k(ranked_indices, relevant, 1),
        "hit@50": hit_at_k(ranked_indices, relevant, 50),
        "hit@10": hit_at_k(ranked_indices, relevant, 10),
        "hit@20": hit_at_k(ranked_indices, relevant, 20),
        "hit@30": hit_at_k(ranked_indices, relevant, 30),
        "mrr@10": mrr_at_k(ranked_indices, relevant, 10),
        "ndcg@10": ndcg_at_k_binary(ranked_indices, relevant, 10),
        "map@10": average_precision_at_k_binary(ranked_indices, relevant, 10),
        "recall@10": recall_at_k(ranked_indices, relevant, 10),
    }