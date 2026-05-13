from __future__ import annotations

import unittest

from src.recommenders.hybrid import (
    merge_explicit_and_implicit_feedback,
    rank_hybrid_scores,
    user_mean_or_default,
)


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


if __name__ == "__main__":
    unittest.main()
