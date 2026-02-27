import unittest

from Evaluation.retrieval_metrics import (
    average_precision_at_k_binary,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k_binary,
    recall_at_k,
)


class TestRetrievalMetrics(unittest.TestCase):
    def test_hit_at_k(self):
        ranked = [0, 1, 2, 3]
        relevant = {2}
        self.assertEqual(hit_at_k(ranked, relevant, 1), 0.0)
        self.assertEqual(hit_at_k(ranked, relevant, 3), 1.0)

    def test_mrr_at_k(self):
        ranked = [5, 4, 3, 2]
        relevant = {3}
        self.assertAlmostEqual(mrr_at_k(ranked, relevant, 10), 1 / 3)

    def test_average_precision_binary(self):
        ranked = [1, 2, 3, 4, 5]
        relevant = {2, 4}
        # hits at rank2 (1/2) and rank4 (2/4=1/2), average over 2 relevant docs
        self.assertAlmostEqual(average_precision_at_k_binary(ranked, relevant, 10), 0.5)

    def test_ndcg_binary(self):
        ranked = [1, 2, 3, 4]
        relevant = {1, 3}
        value = ndcg_at_k_binary(ranked, relevant, 10)
        self.assertGreater(value, 0.9)
        self.assertLessEqual(value, 1.0)

    def test_recall_at_k(self):
        ranked = [10, 11, 12, 13, 14]
        relevant = {11, 14, 20}
        self.assertAlmostEqual(recall_at_k(ranked, relevant, 5), 2 / 3)


if __name__ == "__main__":
    unittest.main()
