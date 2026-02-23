import math
from typing import Dict, Sequence, Tuple


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


def coverage_at_target_accuracy(confidences: Sequence[float], correct: Sequence[bool], target_accuracy: float) -> float:
    if not confidences:
        return 0.0

    pairs = sorted(zip(confidences, correct), key=lambda p: p[0], reverse=True)
    total = len(pairs)

    correct_so_far = 0
    best_coverage = 0.0
    for i, (_, is_correct) in enumerate(pairs, start=1):
        if is_correct:
            correct_so_far += 1
        accuracy = correct_so_far / i
        if accuracy >= target_accuracy:
            best_coverage = i / total
    return best_coverage


def best_threshold_for_target_accuracy(
    probabilities: Sequence[float],
    correct: Sequence[bool],
    target_accuracy: float,
) -> Tuple[float, float, float]:
    if not probabilities:
        return 1.0, 0.0, 0.0

    pairs = sorted(zip(probabilities, correct), key=lambda p: p[0], reverse=True)
    total = len(pairs)

    best_coverage = 0.0
    best_accuracy = 0.0
    best_threshold = 1.0

    accepted = 0
    accepted_correct = 0
    for prob, is_correct in pairs:
        accepted += 1
        if is_correct:
            accepted_correct += 1

        accuracy = accepted_correct / accepted
        coverage = accepted / total
        if accuracy >= target_accuracy and coverage >= best_coverage:
            best_coverage = coverage
            best_accuracy = accuracy
            best_threshold = prob

    return best_threshold, best_coverage, best_accuracy


def risk_coverage_curve(probabilities: Sequence[float], correct: Sequence[bool]) -> Tuple[list[float], list[float], float]:
    if not probabilities:
        return [0.0], [0.0], 0.0

    pairs = sorted(zip(probabilities, correct), key=lambda p: p[0], reverse=True)
    total = len(pairs)

    coverages = [0.0]
    risks = [0.0]

    accepted = 0
    accepted_correct = 0
    for _, is_correct in pairs:
        accepted += 1
        if is_correct:
            accepted_correct += 1

        coverage = accepted / total
        accuracy = accepted_correct / accepted
        risk = 1.0 - accuracy
        coverages.append(coverage)
        risks.append(risk)

    aurc = 0.0
    for i in range(1, len(coverages)):
        dx = coverages[i] - coverages[i - 1]
        y_avg = 0.5 * (risks[i] + risks[i - 1])
        aurc += dx * y_avg

    return coverages, risks, aurc


def binary_ranking_metrics_at_10(ranked_indices: Sequence[int], relevant: set[int]) -> Dict[str, float]:
    return {
        "hit@1": hit_at_k(ranked_indices, relevant, 1),
        "hit@5": hit_at_k(ranked_indices, relevant, 5),
        "hit@10": hit_at_k(ranked_indices, relevant, 10),
        "mrr@10": mrr_at_k(ranked_indices, relevant, 10),
        "ndcg@10": ndcg_at_k_binary(ranked_indices, relevant, 10),
        "map@10": average_precision_at_k_binary(ranked_indices, relevant, 10),
        "recall@10": recall_at_k(ranked_indices, relevant, 10),
    }