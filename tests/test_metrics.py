from __future__ import annotations

import math
import unittest

from src.evaluation.metrics import (
    average_precision_at_k,
    compute_catalog_coverage,
    compute_mae,
    compute_novelty,
    compute_rmse,
    hit_rate_at_k,
    mrr_at_k_single,
    ndcg_at_k_single,
    precision_at_k,
    recall_at_k,
)


class EvaluationMetricsTests(unittest.TestCase):
    def test_leave_one_out_metrics_when_hit(self) -> None:
        recommended = [10, 20, 30]
        ground_truth = 20

        self.assertEqual(hit_rate_at_k(recommended, ground_truth, 3), 1.0)
        self.assertEqual(precision_at_k(recommended, ground_truth, 3), 1.0 / 3)
        self.assertEqual(recall_at_k(recommended, ground_truth, 3), 1.0)
        self.assertEqual(mrr_at_k_single(recommended, ground_truth, 3), 1.0 / 2)
        self.assertEqual(average_precision_at_k(recommended, ground_truth, 3), 1.0 / 2)
        self.assertAlmostEqual(ndcg_at_k_single(recommended, ground_truth, 3), 1.0 / math.log2(3))

    def test_leave_one_out_metrics_when_miss(self) -> None:
        recommended = [10, 20, 30]
        ground_truth = 40

        self.assertEqual(hit_rate_at_k(recommended, ground_truth, 3), 0.0)
        self.assertEqual(precision_at_k(recommended, ground_truth, 3), 0.0)
        self.assertEqual(recall_at_k(recommended, ground_truth, 3), 0.0)
        self.assertEqual(mrr_at_k_single(recommended, ground_truth, 3), 0.0)
        self.assertEqual(average_precision_at_k(recommended, ground_truth, 3), 0.0)
        self.assertEqual(ndcg_at_k_single(recommended, ground_truth, 3), 0.0)

    def test_metrics_validate_k(self) -> None:
        with self.assertRaises(ValueError):
            hit_rate_at_k([1], 1, 0)

    def test_catalog_coverage(self) -> None:
        recommendations = {
            1: [10, 20, 30],
            2: [20, 40, 50],
        }

        self.assertEqual(compute_catalog_coverage(recommendations, catalog_size=10, k=2), 0.3)

    def test_novelty(self) -> None:
        recommendations = {1: [10, 20]}
        item_popularity = {10: 50, 20: 5}

        novelty = compute_novelty(recommendations, item_popularity, total_ratings=100, k=2)
        expected = (-math.log2(0.5 + 1e-10) + -math.log2(0.05 + 1e-10)) / 2
        self.assertAlmostEqual(novelty, expected)

    def test_rating_error_metrics(self) -> None:
        predictions = [(4.0, 3.0), (2.0, 4.0)]

        self.assertAlmostEqual(compute_rmse(predictions), math.sqrt(2.5))
        self.assertEqual(compute_mae(predictions), 1.5)
        self.assertTrue(math.isnan(compute_rmse([])))
        self.assertTrue(math.isnan(compute_mae([])))


if __name__ == "__main__":
    unittest.main()
