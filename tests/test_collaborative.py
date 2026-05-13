from __future__ import annotations

import unittest

import pandas as pd

from src.recommenders.collaborative import (
    build_cf_neighbors_dict,
    get_svd_scores_for_user,
    recommend_itemknn_top_k,
    recommend_svd_top_k,
    score_cf_candidates,
)


class CollaborativeRecommenderTests(unittest.TestCase):
    def test_build_cf_neighbors_dict(self) -> None:
        neighbors_df = pd.DataFrame(
            [
                {"movieId": 20, "neighbor_movieId": 10, "similarity": 0.4},
                {"movieId": 20, "neighbor_movieId": 30, "similarity": 0.9},
            ]
        )

        neighbors = build_cf_neighbors_dict(neighbors_df)

        self.assertEqual(neighbors[20], [(30, 0.9), (10, 0.4)])

    def test_get_svd_scores_for_user(self) -> None:
        svd = pd.DataFrame(
            data=[[4.5, 3.0]],
            index=[1],
            columns=[10, 20],
        )

        scores = get_svd_scores_for_user(
            user_id=1,
            candidate_items=[10, 20, 30],
            svd_matrix=svd,
            fallback_score=3.5,
        )

        self.assertEqual(scores, {10: 4.5, 20: 3.0, 30: 3.5})

    def test_get_svd_scores_for_cold_start_user(self) -> None:
        svd = pd.DataFrame(data=[[4.5]], index=[1], columns=[10])

        scores = get_svd_scores_for_user(
            user_id=99,
            candidate_items=[10, 20],
            svd_matrix=svd,
            fallback_score=3.5,
        )

        self.assertEqual(scores, {10: 3.5, 20: 3.5})

    def test_score_cf_candidates_excludes_seen_items(self) -> None:
        neighbors = {
            20: [(10, 1.0)],
            30: [(10, 0.5)],
        }

        scores = score_cf_candidates(
            candidate_items=[10, 20, 30],
            seen_ratings={10: 5.0},
            user_mean_rating=3.0,
            cf_neighbors_dict=neighbors,
        )

        self.assertNotIn(10, scores)
        self.assertAlmostEqual(scores[20], 5.0)
        self.assertAlmostEqual(scores[30], 5.0)

    def test_recommend_itemknn_top_k_with_popularity_fallback(self) -> None:
        neighbors = {
            20: [(10, 1.0)],
            30: [(10, 0.5)],
        }

        recs = recommend_itemknn_top_k(
            candidate_items=[10, 20, 30, 40],
            seen_ratings={10: 5.0},
            user_mean_rating=3.0,
            cf_neighbors_dict=neighbors,
            popularity_ranking=[10, 40, 50],
            k=4,
        )

        self.assertEqual(recs, [20, 30, 40, 50])

    def test_recommend_svd_top_k_with_fallback(self) -> None:
        svd = pd.DataFrame(
            data=[[4.0, 5.0]],
            index=[1],
            columns=[10, 20],
        )

        recs = recommend_svd_top_k(
            user_id=1,
            svd_matrix=svd,
            seen_items={10},
            fallback_score=3.5,
            popularity_ranking=[10, 30],
            k=2,
        )

        self.assertEqual(recs, [20, 30])

    def test_recommend_svd_top_k_applies_score_offset(self) -> None:
        svd = pd.DataFrame(
            data=[[0.2, 0.4]],
            index=[1],
            columns=[10, 20],
        )

        recs = recommend_svd_top_k(
            user_id=1,
            svd_matrix=svd,
            seen_items=set(),
            fallback_score=3.5,
            popularity_ranking=[],
            k=2,
            score_offset=3.0,
        )

        self.assertEqual(recs, [20, 10])


if __name__ == "__main__":
    unittest.main()
