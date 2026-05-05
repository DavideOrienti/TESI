from __future__ import annotations
from math import log2
import numpy as np


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


def compute_catalog_coverage(
    recommendations: dict[int, list[int]],
    catalog_size: int,
    k: int = 10,
) -> float:
    """
    Calcola la Catalog Coverage@K:
    percentuale di item distinti nelle top-K di tutti gli utenti.

    Un valore basso indica popularity bias (filter bubble).

    Args:
        recommendations: {user_id: [item_id, ...]} lista ordinata top-K
        catalog_size: numero totale di item nel catalogo
        k: cutoff

    Returns:
        coverage in [0, 1]
    """
    if not recommendations or catalog_size == 0:
        return 0.0
    recommended_items: set[int] = set()
    for items in recommendations.values():
        recommended_items.update(items[:k])
    return len(recommended_items) / catalog_size


def compute_novelty(
    recommendations: dict[int, list[int]],
    item_popularity: dict[int, int],
    total_ratings: int,
    k: int = 10,
) -> float:
    """
    Calcola la Novelty@K media.

    Formula: -log2(pop(i) / total_ratings)
    Item popolare → bassa novelty; item raro → alta novelty.

    Args:
        recommendations: {user_id: [item_id, ...]}
        item_popularity: {item_id: n_ratings}
        total_ratings: numero totale di rating nel training
        k: cutoff

    Returns:
        novelty media in [0, inf)
    """
    import math
    if not recommendations or total_ratings == 0:
        return 0.0
    novelties = []
    for items in recommendations.values():
        for item_id in items[:k]:
            pop = item_popularity.get(item_id, 1)
            prob = pop / total_ratings
            novelties.append(-math.log2(prob + 1e-10))
    return float(sum(novelties) / len(novelties)) if novelties else 0.0


def compute_rmse(predictions: list[tuple[float, float]]) -> float:
    """
    Calcola RMSE tra rating predetti e reali.

    Args:
        predictions: lista di tuple (rating_reale, rating_predetto)

    Returns:
        RMSE (float), nan se lista vuota
    """
    if not predictions:
        return float("nan")
    errors = [(true - pred) ** 2 for true, pred in predictions]
    return float(np.sqrt(np.mean(errors)))


def compute_mae(predictions: list[tuple[float, float]]) -> float:
    """
    Calcola MAE tra rating predetti e reali.

    Args:
        predictions: lista di tuple (rating_reale, rating_predetto)

    Returns:
        MAE (float), nan se lista vuota
    """
    if not predictions:
        return float("nan")
    errors = [abs(true - pred) for true, pred in predictions]
    return float(np.mean(errors))
