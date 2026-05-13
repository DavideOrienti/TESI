from __future__ import annotations

import unittest

import pandas as pd

from src.recommenders.content_based import (
    build_content_neighbors_dict,
    build_index_to_movieid,
    build_movieid_to_index,
    count_supported_content_candidates,
    get_similar_movies,
    recommend_content_top_k,
    score_content_candidates,
)


class ContentBasedRecommenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index_df = pd.DataFrame({"movieId": [10, 20, 30, 40]})
        self.neighbors_df = pd.DataFrame(
            [
                {"movie_idx": 1, "neighbor_idx": 0, "similarity": 0.9},
                {"movie_idx": 1, "neighbor_idx": 2, "similarity": 0.4},
                {"movie_idx": 2, "neighbor_idx": 0, "similarity": 0.8},
                {"movie_idx": 3, "neighbor_idx": 0, "similarity": -0.5},
            ]
        )
        self.index_to_movieid = build_index_to_movieid(self.index_df)
        self.neighbors = build_content_neighbors_dict(self.neighbors_df, self.index_to_movieid)

    def test_build_index_mappings(self) -> None:
        self.assertEqual(build_movieid_to_index(self.index_df), {10: 0, 20: 1, 30: 2, 40: 3})
        self.assertEqual(self.index_to_movieid, {0: 10, 1: 20, 2: 30, 3: 40})

    def test_build_content_neighbors_dict_uses_movie_ids(self) -> None:
        self.assertEqual(self.neighbors[20], [(10, 0.9), (30, 0.4)])
        self.assertEqual(self.neighbors[30], [(10, 0.8)])

    def test_score_content_candidates_excludes_seen_items(self) -> None:
        scores = score_content_candidates(
            candidate_items=[10, 20, 30],
            seen_ratings={10: 5.0},
            user_mean_rating=3.0,
            content_neighbors_dict=self.neighbors,
        )

        self.assertNotIn(10, scores)
        self.assertAlmostEqual(scores[20], 5.0)
        self.assertAlmostEqual(scores[30], 5.0)

    def test_recommend_content_top_k_orders_by_score_support_and_id(self) -> None:
        recommendations, support_count = recommend_content_top_k(
            candidate_items=[10, 20, 30, 40],
            seen_ratings={10: 5.0},
            user_mean_rating=3.0,
            content_neighbors_dict=self.neighbors,
            k=3,
        )

        self.assertEqual(recommendations, [20, 30, 40])
        self.assertEqual(support_count, 2)

    def test_count_supported_content_candidates(self) -> None:
        supported = count_supported_content_candidates(
            candidate_items=[20, 30, 40],
            seen_ratings={10: 5.0},
            content_neighbors_dict=self.neighbors,
        )

        self.assertEqual(supported, 2)

    def test_get_similar_movies(self) -> None:
        similar = get_similar_movies(20, self.neighbors, top_k=1)

        self.assertEqual(similar, [{"movie_id": 10, "similarity": 0.9}])


if __name__ == "__main__":
    unittest.main()
