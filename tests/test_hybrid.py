from __future__ import annotations

import unittest

import pandas as pd

from src.recommenders.hybrid import (
    merge_explicit_and_implicit_feedback,
    rank_hybrid_scores,
    user_mean_or_default,
)
from src.webapp.backend import recommender_service


class HybridRecommenderTests(unittest.TestCase):
    def test_merge_explicit_and_implicit_feedback(self) -> None:
        merged = merge_explicit_and_implicit_feedback(
            user_ratings={1: 4.0, 2: 3.0},
            favorite_movie_ids=[1, 3],
        )

        self.assertEqual(merged[1], 4.4)
        self.assertEqual(merged[2], 3.0)
        self.assertEqual(merged[3], 4.5)

    def test_merge_feedback_caps_boosted_rating(self) -> None:
        merged = merge_explicit_and_implicit_feedback(
            user_ratings={1: 4.8},
            favorite_movie_ids=[1],
        )

        self.assertEqual(merged[1], 5.0)

    def test_user_mean_or_default(self) -> None:
        self.assertEqual(user_mean_or_default({1: 4.0, 2: 2.0}), 3.0)
        self.assertEqual(user_mean_or_default({}), 3.5)

    def test_rank_hybrid_scores_combines_normalized_scores(self) -> None:
        ranked = rank_hybrid_scores(
            collaborative_scores={10: 5.0, 20: 1.0},
            content_scores={10: 1.0, 20: 5.0},
            collaborative_weight=0.75,
            top_k=2,
        )

        self.assertEqual([item.movie_id for item in ranked], [10, 20])
        self.assertAlmostEqual(ranked[0].score, 0.75)
        self.assertAlmostEqual(ranked[1].score, 0.25)

    def test_rank_hybrid_scores_content_only(self) -> None:
        ranked = rank_hybrid_scores(
            collaborative_scores={10: 5.0, 20: 1.0},
            content_scores={10: 1.0, 20: 5.0},
            collaborative_weight=0.75,
            top_k=1,
            content_only=True,
        )

        self.assertEqual(ranked[0].movie_id, 20)
        self.assertAlmostEqual(ranked[0].score, 1.0)

    def test_rank_hybrid_scores_validates_weight(self) -> None:
        with self.assertRaises(ValueError):
            rank_hybrid_scores({}, {}, collaborative_weight=1.2, top_k=10)

    def test_sparse_positive_profile_uses_neutral_content_baseline(self) -> None:
        baseline = recommender_service._content_profile_baseline({1: 5.0, 2: 4.5})

        self.assertEqual(baseline, 3.5)

    def test_genre_affinity_rewards_liked_genres(self) -> None:
        original_genres = recommender_service._movie_genres
        recommender_service._movie_genres = {
            1: {"documentary"},
            2: {"documentary"},
            3: {"action"},
        }
        try:
            scores = recommender_service._score_genre_affinity(
                candidate_items=[2, 3],
                seen_ratings={1: 5.0},
                baseline=3.5,
            )
        finally:
            recommender_service._movie_genres = original_genres

        self.assertGreater(scores[2], 3.5)
        self.assertNotIn(3, scores)

    def test_webapp_recommendations_do_not_use_colliding_movielens_user_id(self) -> None:
        original_state = {
            "svd": recommender_service._svd_matrix,
            "neighbors": recommender_service._content_neighbors_dict,
            "candidates": recommender_service._candidate_items,
            "titles": recommender_service._movie_titles,
            "genres": recommender_service._movie_genres,
            "popularity": recommender_service._popularity_ranking,
        }
        recommender_service._svd_matrix = pd.DataFrame(
            {2: [1.0], 3: [5.0]},
            index=[1],
        )
        recommender_service._content_neighbors_dict = {
            2: [(1, 0.9)],
            3: [],
        }
        recommender_service._candidate_items = [1, 2, 3]
        recommender_service._movie_titles = {1: "Doc seed", 2: "Doc candidate", 3: "Action candidate"}
        recommender_service._movie_genres = {
            1: {"documentary"},
            2: {"documentary"},
            3: {"action"},
        }
        recommender_service._popularity_ranking = [3, 2]
        try:
            recs = recommender_service.get_recommendations(
                user_id=1,
                seen_movie_ids=[1],
                user_ratings={1: 5.0},
                favorite_movie_ids=[],
                top_k=1,
            )
        finally:
            recommender_service._svd_matrix = original_state["svd"]
            recommender_service._content_neighbors_dict = original_state["neighbors"]
            recommender_service._candidate_items = original_state["candidates"]
            recommender_service._movie_titles = original_state["titles"]
            recommender_service._movie_genres = original_state["genres"]
            recommender_service._popularity_ranking = original_state["popularity"]

        self.assertEqual(recs[0]["movie_id"], 2)
        self.assertEqual(recs[0]["explanation"]["type"], "content")


if __name__ == "__main__":
    unittest.main()
