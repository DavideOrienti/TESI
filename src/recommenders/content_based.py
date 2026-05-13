from __future__ import annotations

import pandas as pd

from src.recommenders.scoring import score_candidate_content


NeighborMap = dict[int, list[tuple[int, float]]]


def build_movieid_to_index(index_df: pd.DataFrame, movie_id_col: str = "movieId") -> dict[int, int]:
    """Build a `movieId -> row index` mapping from an embedding index dataframe."""
    return {
        int(movie_id): int(idx)
        for idx, movie_id in enumerate(index_df[movie_id_col].tolist())
    }


def build_index_to_movieid(index_df: pd.DataFrame, movie_id_col: str = "movieId") -> dict[int, int]:
    """Build a `row index -> movieId` mapping from an embedding index dataframe."""
    return {
        int(idx): int(movie_id)
        for idx, movie_id in enumerate(index_df[movie_id_col].tolist())
    }


def build_content_neighbors_dict(
    neighbors_df: pd.DataFrame,
    index_to_movieid: dict[int, int],
    movie_idx_col: str = "movie_idx",
    neighbor_idx_col: str = "neighbor_idx",
    similarity_col: str = "similarity",
) -> NeighborMap:
    """
    Convert content-neighbor rows based on embedding indices to movieId neighbors.

    Input rows usually have `(movie_idx, neighbor_idx, similarity)`. The returned
    mapping uses MovieLens IDs: `candidate_movieId -> [(neighbor_movieId, sim)]`.
    """
    neighbors: NeighborMap = {}
    for movie_idx, group in neighbors_df.groupby(movie_idx_col):
        candidate_movie_id = index_to_movieid.get(int(movie_idx))
        if candidate_movie_id is None:
            continue

        rows: list[tuple[int, float]] = []
        for _, row in group.iterrows():
            neighbor_movie_id = index_to_movieid.get(int(row[neighbor_idx_col]))
            if neighbor_movie_id is None:
                continue
            rows.append((neighbor_movie_id, float(row[similarity_col])))

        rows.sort(key=lambda item: (-item[1], item[0]))
        neighbors[candidate_movie_id] = rows

    return neighbors


def count_supported_content_candidates(
    candidate_items: list[int],
    seen_ratings: dict[int, float],
    content_neighbors_dict: NeighborMap,
) -> int:
    """Count unseen candidates that have at least one positive rated content neighbor."""
    supported = 0
    for movie_id in candidate_items:
        neighbors = content_neighbors_dict.get(movie_id, [])
        if any(neighbor_id in seen_ratings and similarity > 0 for neighbor_id, similarity in neighbors):
            supported += 1
    return supported


def score_content_candidates(
    candidate_items: list[int],
    seen_ratings: dict[int, float],
    user_mean_rating: float,
    content_neighbors_dict: NeighborMap,
    exclude_seen: bool = True,
) -> dict[int, float]:
    """Score candidate movies with the content-based neighborhood model."""
    seen_items = set(seen_ratings)
    scores: dict[int, float] = {}
    for movie_id in candidate_items:
        if exclude_seen and movie_id in seen_items:
            continue
        scores[int(movie_id)] = score_candidate_content(
            candidate_movie_id=int(movie_id),
            seen_ratings=seen_ratings,
            user_mean_rating=user_mean_rating,
            content_neighbors_dict=content_neighbors_dict,
        )
    return scores


def recommend_content_top_k(
    candidate_items: list[int],
    seen_ratings: dict[int, float],
    user_mean_rating: float,
    content_neighbors_dict: NeighborMap,
    k: int,
) -> tuple[list[int], int]:
    """
    Return content-based top-k recommendations and support-count audit value.

    The ranking tie-breaks by score, then number of rated positive neighbors,
    then movieId for deterministic output.
    """
    if k < 0:
        raise ValueError("k must be >= 0.")

    seen_items = set(seen_ratings)
    scored: list[tuple[int, float, int]] = []
    candidate_pool = [int(movie_id) for movie_id in candidate_items if int(movie_id) not in seen_items]

    for movie_id in candidate_pool:
        neighbors = content_neighbors_dict.get(movie_id, [])
        support = sum(
            1
            for neighbor_id, similarity in neighbors
            if neighbor_id in seen_ratings and similarity > 0
        )
        score = score_candidate_content(
            candidate_movie_id=movie_id,
            seen_ratings=seen_ratings,
            user_mean_rating=user_mean_rating,
            content_neighbors_dict=content_neighbors_dict,
        )
        scored.append((movie_id, score, support))

    scored.sort(key=lambda item: (-item[1], -item[2], item[0]))
    supported_count = sum(1 for _, _, support in scored if support > 0)
    return [movie_id for movie_id, _, _ in scored[:k]], supported_count


def get_similar_movies(
    movie_id: int,
    content_neighbors_dict: NeighborMap,
    top_k: int = 10,
) -> list[dict]:
    """Return the top content neighbors for a movieId."""
    if top_k < 0:
        raise ValueError("top_k must be >= 0.")
    return [
        {"movie_id": int(neighbor_id), "similarity": round(float(similarity), 4)}
        for neighbor_id, similarity in content_neighbors_dict.get(int(movie_id), [])[:top_k]
    ]
