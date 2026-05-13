from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from src.recommenders.scoring import minmax_normalize_scores


@dataclass(frozen=True)
class HybridScore:
    """Final hybrid score for a candidate with normalized contributions."""

    movie_id: int
    score: float
    collaborative_score: float
    content_score: float


def merge_explicit_and_implicit_feedback(
    user_ratings: dict[int, float],
    favorite_movie_ids: list[int] | None = None,
    implicit_favorite_rating: float = 4.5,
    explicit_favorite_boost: float = 1.1,
    max_rating: float = 5.0,
) -> dict[int, float]:
    """
    Merge explicit ratings and favorites into a single user profile.

    Unrated favorites become positive implicit feedback. Favorites that already
    have a rating receive a small reinforcement capped at `max_rating`.
    """
    merged = {int(movie_id): float(rating) for movie_id, rating in user_ratings.items()}
    for movie_id in favorite_movie_ids or []:
        movie_id = int(movie_id)
        if movie_id not in merged:
            merged[movie_id] = float(implicit_favorite_rating)
        else:
            merged[movie_id] = min(float(max_rating), merged[movie_id] * explicit_favorite_boost)
    return merged


def user_mean_or_default(ratings: dict[int, float], default: float = 3.5) -> float:
    """Return the user feedback mean, or a cold-start default."""
    return float(fmean(ratings.values())) if ratings else float(default)


def rank_hybrid_scores(
    collaborative_scores: dict[int, float],
    content_scores: dict[int, float],
    collaborative_weight: float,
    top_k: int,
    content_only: bool = False,
) -> list[HybridScore]:
    """
    Normalize and combine collaborative and content scores into a top-k ranking.

    `collaborative_weight` is the collaborative component weight. The content
    component receives `1 - collaborative_weight`.
    """
    if not 0.0 <= collaborative_weight <= 1.0:
        raise ValueError("collaborative_weight must be in [0, 1].")
    if top_k < 0:
        raise ValueError("top_k must be >= 0.")

    collaborative_norm = minmax_normalize_scores(collaborative_scores)
    content_norm = minmax_normalize_scores(content_scores)
    candidate_ids = set(collaborative_norm).union(content_norm)

    ranked: list[HybridScore] = []
    for movie_id in candidate_ids:
        collaborative = float(collaborative_norm.get(movie_id, 0.0))
        content = float(content_norm.get(movie_id, 0.0))
        if content_only:
            final = content
        else:
            final = collaborative_weight * collaborative + (1.0 - collaborative_weight) * content
        ranked.append(
            HybridScore(
                movie_id=int(movie_id),
                score=float(final),
                collaborative_score=collaborative,
                content_score=content,
            )
        )

    ranked.sort(key=lambda item: (-item.score, item.movie_id))
    return ranked[:top_k]
