from __future__ import annotations

from math import log2

import numpy as np


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be > 0.")


def hit_rate_at_k(recommended: list[int], ground_truth: int, k: int) -> float:
    """Leave-one-out HR@K: 1 if the relevant item is in top-k, otherwise 0."""
    _validate_k(k)
    return 1.0 if ground_truth in recommended[:k] else 0.0


def ndcg_at_k_single(recommended: list[int], ground_truth: int, k: int) -> float:
    """Leave-one-out NDCG@K with one relevant item."""
    _validate_k(k)
    rec_k = recommended[:k]
    if ground_truth not in rec_k:
        return 0.0
    rank = rec_k.index(ground_truth) + 1
    return 1.0 / log2(rank + 1)


def mrr_at_k_single(recommended: list[int], ground_truth: int, k: int) -> float:
    """Leave-one-out MRR@K with one relevant item."""
    _validate_k(k)
    rec_k = recommended[:k]
    if ground_truth not in rec_k:
        return 0.0
    rank = rec_k.index(ground_truth) + 1
    return 1.0 / rank


def precision_at_k(recommended: list[int], ground_truth: int, k: int) -> float:
    """Leave-one-out Precision@K. With one relevant item this equals HR@K / K."""
    _validate_k(k)
    return (1.0 / k) if ground_truth in recommended[:k] else 0.0


def recall_at_k(recommended: list[int], ground_truth: int, k: int) -> float:
    """Leave-one-out Recall@K. With one relevant item this equals HR@K."""
    _validate_k(k)
    return 1.0 if ground_truth in recommended[:k] else 0.0


def average_precision_at_k(recommended: list[int], ground_truth: int, k: int) -> float:
    """Leave-one-out AP@K: precision at the rank where the item appears."""
    _validate_k(k)
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
    """Catalog Coverage@K: fraction of catalog items appearing in top-k lists."""
    _validate_k(k)
    if not recommendations or catalog_size <= 0:
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
    Mean Novelty@K based on self-information: -log2(popularity / total_ratings).
    """
    _validate_k(k)
    if not recommendations or total_ratings <= 0:
        return 0.0
    novelties: list[float] = []
    for items in recommendations.values():
        for item_id in items[:k]:
            popularity = item_popularity.get(item_id, 1)
            probability = popularity / total_ratings
            novelties.append(-log2(probability + 1e-10))
    return float(sum(novelties) / len(novelties)) if novelties else 0.0


def compute_rmse(predictions: list[tuple[float, float]]) -> float:
    """RMSE over `(true_rating, predicted_rating)` pairs."""
    if not predictions:
        return float("nan")
    errors = [(true - pred) ** 2 for true, pred in predictions]
    return float(np.sqrt(np.mean(errors)))


def compute_mae(predictions: list[tuple[float, float]]) -> float:
    """MAE over `(true_rating, predicted_rating)` pairs."""
    if not predictions:
        return float("nan")
    errors = [abs(true - pred) for true, pred in predictions]
    return float(np.mean(errors))

