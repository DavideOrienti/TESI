from __future__ import annotations

import unittest

import pandas as pd

from src.recommenders.popularity import (
    build_popularity_ranking,
    build_popularity_scores,
    ranking_from_movies_metadata,
    recommend_popular,
)


class PopularityRecommenderTests(unittest.TestCase):
    def test_build_popularity_ranking_orders_by_weighted_score(self) -> None:
        ratings = pd.DataFrame(
            [
                {"movieId": 1, "rating": 5.0},
                {"movieId": 2, "rating": 4.0},
                {"movieId": 2, "rating": 4.0},
                {"movieId": 3, "rating": 3.0},
                {"movieId": 3, "rating": 3.0},
                {"movieId": 3, "rating": 3.0},
            ]
        )

        ranking = build_popularity_ranking(ratings, min_votes=1)

        self.assertEqual(ranking["movieId"].tolist(), [1, 2, 3])
        self.assertIn("pop_score", ranking.columns)

    def test_build_popularity_scores(self) -> None:
        ratings = pd.DataFrame(
            [
                {"movieId": 1, "rating": 5.0},
                {"movieId": 2, "rating": 3.0},
            ]
        )

        scores = build_popularity_scores(ratings, min_votes=1)

        self.assertEqual(set(scores), {1, 2})
        self.assertGreater(scores[1], scores[2])

    def test_recommend_popular_excludes_items(self) -> None:
        recs = recommend_popular([10, 20, 30], excluded={10}, k=2)

        self.assertEqual(recs, [20, 30])

    def test_recommend_popular_validates_k(self) -> None:
        with self.assertRaises(ValueError):
            recommend_popular([1], k=-1)

    def test_ranking_from_movies_metadata(self) -> None:
        movies = pd.DataFrame(
            [
                {"movieId": 10, "popularity": 2.0},
                {"movieId": 20, "popularity": 5.0},
                {"movieId": 30, "popularity": None},
            ]
        )

        ranking = ranking_from_movies_metadata(movies, candidate_items=[10, 20, 30, 40])

        self.assertEqual(ranking, [20, 10, 30, 40])

    def test_ranking_from_movies_metadata_without_popularity_column(self) -> None:
        movies = pd.DataFrame([{"movieId": 20}, {"movieId": 10}])

        ranking = ranking_from_movies_metadata(movies, candidate_items=[20, 10])

        self.assertEqual(ranking, [10, 20])


if __name__ == "__main__":
    unittest.main()
