from __future__ import annotations

import pandas as pd


def build_popularity_ranking(
    ratings: pd.DataFrame,
    min_votes: int,
    movie_id_col: str = "movieId",
    rating_col: str = "rating",
) -> pd.DataFrame:
    """
    Build an IMDb-style weighted popularity ranking from explicit ratings.

    The score shrinks each item mean toward the global mean using `min_votes`.
    """
    if min_votes < 0:
        raise ValueError("min_votes must be >= 0.")
    if ratings.empty:
        return pd.DataFrame(columns=[movie_id_col, "rating_count", "rating_mean", "pop_score"])

    item_stats = (
        ratings.groupby(movie_id_col)
        .agg(
            rating_count=(rating_col, "count"),
            rating_mean=(rating_col, "mean"),
        )
        .reset_index()
    )

    global_mean = float(ratings[rating_col].mean())
    denominator = item_stats["rating_count"] + min_votes
    item_stats["pop_score"] = (
        (item_stats["rating_count"] / denominator) * item_stats["rating_mean"]
        + (min_votes / denominator) * global_mean
    )

    return item_stats.sort_values(
        ["pop_score", "rating_count", movie_id_col],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_popularity_scores(
    ratings: pd.DataFrame,
    min_votes: int,
    movie_id_col: str = "movieId",
    rating_col: str = "rating",
) -> dict[int, float]:
    """Return `{movieId: pop_score}` from explicit ratings."""
    ranking = build_popularity_ranking(ratings, min_votes, movie_id_col, rating_col)
    return {
        int(row[movie_id_col]): float(row["pop_score"])
        for _, row in ranking.iterrows()
    }


def recommend_popular(
    popularity_ranking: list[int],
    excluded: set[int] | None = None,
    k: int = 10,
) -> list[int]:
    """Return top-k popular movie IDs excluding already seen or otherwise blocked items."""
    if k < 0:
        raise ValueError("k must be >= 0.")
    excluded = excluded or set()
    return [int(movie_id) for movie_id in popularity_ranking if int(movie_id) not in excluded][:k]


def ranking_from_movies_metadata(
    movies: pd.DataFrame,
    candidate_items: list[int],
    popularity_col: str = "popularity",
    movie_id_col: str = "movieId",
) -> list[int]:
    """
    Build a ranking from movie metadata popularity, limited to candidate items.

    If the popularity column is missing, candidates are returned in deterministic
    movieId order.
    """
    candidate_set = {int(movie_id) for movie_id in candidate_items}
    if movies.empty or popularity_col not in movies.columns:
        return sorted(candidate_set)

    ranked = (
        movies[movies[movie_id_col].isin(candidate_set)]
        .sort_values([popularity_col, movie_id_col], ascending=[False, True], na_position="last")
    )
    ranked_ids = [int(movie_id) for movie_id in ranked[movie_id_col].tolist()]

    missing_ids = sorted(candidate_set.difference(ranked_ids))
    return ranked_ids + missing_ids

