from __future__ import annotations
import numpy as np
import pandas as pd


def build_user_seen_ratings(train: pd.DataFrame) -> dict[int, dict[int, float]]:
    """Returns {userId: {movieId: rating}} for all training ratings."""
    mapping = {}
    for user_id, g in train.groupby("userId"):
        mapping[int(user_id)] = {
            int(mid): float(r)
            for mid, r in zip(g["movieId"], g["rating"])
        }
    return mapping


def build_user_mean_ratings(train: pd.DataFrame) -> dict[int, float]:
    """Returns {userId: mean_rating} computed on training data."""
    return {
        int(user_id): float(g["rating"].mean())
        for user_id, g in train.groupby("userId")
    }


def minmax_normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    """
    Normalizza gli score in [0, 1] con Min-Max scaling.

    In caso di range degenere (tutti gli score uguali o dict vuoto),
    restituisce score uniformi a 0.5 invece di 0 silenzioso, e logga un warning.

    Args:
        scores: dict {item_id: score}

    Returns:
        dict {item_id: score_normalizzato in [0,1]}
    """
    if not scores:
        return {}
    min_s = min(scores.values())
    max_s = max(scores.values())
    range_s = max_s - min_s
    if range_s < 1e-12:
        import logging
        logging.warning(
            f"[minmax_normalize] degenerate range "
            f"({range_s:.2e}) for {len(scores)} items. "
            f"Returning uniform 0.5."
        )
        return {k: 0.5 for k in scores}
    return {k: (v - min_s) / range_s for k, v in scores.items()}


def score_candidate_cf(
    candidate_movie_id: int,
    seen_ratings: dict[int, float],
    user_mean_rating: float,
    cf_neighbors_dict: dict[int, list[tuple[int, float]]]
) -> float:
    """Item-KNN CF score: mean-centered weighted sum over seen CF neighbors.
    Returns user_mean_rating when no rated CF neighbor exists."""
    neighbors = cf_neighbors_dict.get(candidate_movie_id, [])
    num = 0.0
    den = 0.0
    for neighbor_movie_id, sim in neighbors:
        if neighbor_movie_id not in seen_ratings:
            continue
        centered = seen_ratings[neighbor_movie_id] - user_mean_rating
        num += sim * centered
        den += abs(sim)
    if den == 0.0:
        return user_mean_rating
    return user_mean_rating + (num / den)


def score_candidate_content(
    candidate_movie_id: int,
    seen_ratings: dict[int, float],
    user_mean_rating: float,
    content_neighbors_dict: dict[int, list[tuple[int, float]]]
) -> float:
    """Content-based score: embedding-neighbor weighted sum, only positive sims.
    Returns user_mean_rating when no rated content neighbor exists."""
    neighbors = content_neighbors_dict.get(candidate_movie_id, [])
    num = 0.0
    den = 0.0
    for neighbor_movie_id, sim in neighbors:
        if neighbor_movie_id not in seen_ratings or sim <= 0:
            continue
        centered = seen_ratings[neighbor_movie_id] - user_mean_rating
        num += sim * centered
        den += abs(sim)
    if den == 0.0:
        return user_mean_rating
    return user_mean_rating + (num / den)
