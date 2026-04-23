from __future__ import annotations
from math import log2


def hit_rate_at_k(recommended: list[int], ground_truth: int, k: int) -> float:
    """Leave-one-out: 1 se l'unico item rilevante è nei top-k raccomandati, 0 altrimenti."""
    rec_k = recommended[:k]
    return 1.0 if ground_truth in rec_k else 0.0


def ndcg_at_k_single(recommended: list[int], ground_truth: int, k: int) -> float:
    """Leave-one-out: NDCG con un solo item rilevante. DCG = 1/log2(rank+1), IDCG = 1."""
    rec_k = recommended[:k]
    if ground_truth in rec_k:
        rank = rec_k.index(ground_truth) + 1
        return 1.0 / log2(rank + 1)
    return 0.0


def mrr_at_k_single(recommended: list[int], ground_truth: int, k: int) -> float:
    """Leave-one-out: reciproco del rank del singolo item rilevante, 0 se fuori top-k."""
    rec_k = recommended[:k]
    if ground_truth in rec_k:
        rank = rec_k.index(ground_truth) + 1
        return 1.0 / rank
    return 0.0


def precision_at_k(recommended: list[int], ground_truth: int, k: int) -> float:
    """Leave-one-out: #rilevanti in top-k / k. Con 1 ground truth coincide con HR@k / k."""
    rec_k = recommended[:k]
    return (1.0 / k) if ground_truth in rec_k else 0.0


def recall_at_k(recommended: list[int], ground_truth: int, k: int) -> float:
    """Leave-one-out: #rilevanti in top-k / #rilevanti totali.
    Con 1 ground truth equivale a hit_rate_at_k."""
    rec_k = recommended[:k]
    return 1.0 if ground_truth in rec_k else 0.0


def average_precision_at_k(recommended: list[int], ground_truth: int, k: int) -> float:
    """AP@K per singolo utente con un solo item rilevante.
    AP = precision al rank in cui compare l'item, 0 se non compare."""
    rec_k = recommended[:k]
    if ground_truth not in rec_k:
        return 0.0
    rank = rec_k.index(ground_truth) + 1
    return 1.0 / rank
