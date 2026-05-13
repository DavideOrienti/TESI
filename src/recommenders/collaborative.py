from __future__ import annotations

import pandas as pd

from src.recommenders.popularity import recommend_popular
from src.recommenders.scoring import score_candidate_cf


NeighborMap = dict[int, list[tuple[int, float]]]


def build_cf_neighbors_dict(
    neighbors_df: pd.DataFrame,
    movie_id_col: str = "movieId",
    neighbor_id_col: str = "neighbor_movieId",
    similarity_col: str = "similarity",
) -> NeighborMap:
    """Convert an ItemKNN neighbors dataframe to `movieId -> [(neighborId, sim)]`."""
    neighbors: NeighborMap = {}
    for movie_id, group in neighbors_df.groupby(movie_id_col):
        rows = [
            (int(row[neighbor_id_col]), float(row[similarity_col]))
            for _, row in group.iterrows()
        ]
        rows.sort(key=lambda item: (-item[1], item[0]))
        neighbors[int(movie_id)] = rows
    return neighbors


def get_svd_scores_for_user(
    user_id: int,
    candidate_items: list[int],
    svd_matrix: pd.DataFrame | None,
    fallback_score: float,
) -> dict[int, float]:
    """
    Return SVD scores for candidate items, using fallback for cold-start/missing items.

    The matrix is expected to have user IDs as index and movie IDs as columns.
    """
    if svd_matrix is None or user_id not in svd_matrix.index:
        return {int(movie_id): float(fallback_score) for movie_id in candidate_items}

    user_row = svd_matrix.loc[user_id]
    svd_cols = set(user_row.index)
    return {
        int(movie_id): float(user_row[movie_id]) if movie_id in svd_cols else float(fallback_score)
        for movie_id in candidate_items
    }


def score_cf_candidates(
    candidate_items: list[int],
    seen_ratings: dict[int, float],
    user_mean_rating: float,
    cf_neighbors_dict: NeighborMap,
    exclude_seen: bool = True,
) -> dict[int, float]:
    """Score candidate movies with an item-item collaborative neighborhood model."""
    seen_items = set(seen_ratings)
    scores: dict[int, float] = {}
    for movie_id in candidate_items:
        movie_id = int(movie_id)
        if exclude_seen and movie_id in seen_items:
            continue
        scores[movie_id] = score_candidate_cf(
            candidate_movie_id=movie_id,
            seen_ratings=seen_ratings,
            user_mean_rating=user_mean_rating,
            cf_neighbors_dict=cf_neighbors_dict,
        )
    return scores


def recommend_itemknn_top_k(
    candidate_items: list[int],
    seen_ratings: dict[int, float],
    user_mean_rating: float,
    cf_neighbors_dict: NeighborMap,
    popularity_ranking: list[int] | None,
    k: int,
    min_neighbors_for_personalized: int = 1,
) -> list[int]:
    """
    Return ItemKNN top-k recommendations with optional popularity fallback.

    Personalized candidates are ranked by predicted score, number of used
    neighbors, then movieId for deterministic output.
    """
    if k < 0:
        raise ValueError("k must be >= 0.")

    seen_items = set(seen_ratings)
    scored: list[tuple[int, float, int]] = []
    for movie_id in candidate_items:
        movie_id = int(movie_id)
        if movie_id in seen_items:
            continue

        neighbors = cf_neighbors_dict.get(movie_id, [])
        used_neighbors = sum(1 for neighbor_id, _ in neighbors if neighbor_id in seen_ratings)
        if used_neighbors < min_neighbors_for_personalized:
            continue

        score = score_candidate_cf(
            candidate_movie_id=movie_id,
            seen_ratings=seen_ratings,
            user_mean_rating=user_mean_rating,
            cf_neighbors_dict=cf_neighbors_dict,
        )
        scored.append((movie_id, score, used_neighbors))

    scored.sort(key=lambda item: (-item[1], -item[2], item[0]))

    recommendations: list[int] = []
    used = set(seen_items)
    for movie_id, _, _ in scored:
        if movie_id not in used:
            recommendations.append(movie_id)
            used.add(movie_id)
        if len(recommendations) >= k:
            return recommendations

    recommendations.extend(recommend_popular(popularity_ranking or [], used, k - len(recommendations)))

    return recommendations[:k]


def recommend_svd_top_k(
    user_id: int,
    svd_matrix: pd.DataFrame | None,
    seen_items: set[int],
    fallback_score: float,
    popularity_ranking: list[int] | None,
    k: int,
    score_offset: float = 0.0,
) -> list[int]:
    """Return SVD top-k recommendations with popularity fallback."""
    if k < 0:
        raise ValueError("k must be >= 0.")

    if svd_matrix is None or user_id not in svd_matrix.index:
        return recommend_popular(popularity_ranking or [], seen_items, k)

    scores = {
        int(movie_id): float(score) + float(score_offset)
        for movie_id, score in svd_matrix.loc[user_id].items()
        if int(movie_id) not in seen_items
    }
    if not scores:
        return recommend_popular(popularity_ranking or [], seen_items, k)

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    recommendations = [movie_id for movie_id, _ in ranked[:k]]

    if len(recommendations) < k:
        used = set(seen_items).union(recommendations)
        recommendations.extend(recommend_popular(popularity_ranking or [], used, k - len(recommendations)))

    return recommendations[:k]
