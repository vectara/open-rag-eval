import unittest
from unittest.mock import Mock
from open_rag_eval.metrics.umbrela_metric import UMBRELAMetric
from open_rag_eval.data_classes.rag_results import RetrievalResult


class TestUMBRELAMetric(unittest.TestCase):
    def setUp(self):
        self.model = Mock()
        self.metric = UMBRELAMetric(model=self.model)
        self.k_values = [1, 3, 5]

    def test_compute_successful(self):
        query = "test query"
        passages = {"1": "passage 1", "2": "passage 2"}
        retrieval_result = RetrievalResult(query=query, retrieved_passages=passages)

        mock_response1 = Mock(score=Mock(value="3"))
        mock_response2 = Mock(score=Mock(value="2"))
        self.model.parse.side_effect = [mock_response1, mock_response2]

        scores = self.metric.compute(retrieval_result, self.k_values)

        expected_umbrela_scores = {"1": 3, "2": 2}
        self.assertEqual(scores["umbrela_scores"], expected_umbrela_scores)
        self.assertIn("retrieval_scores", scores)
        self.assertEqual(self.model.parse.call_count, 2)

    def test_compute_empty_passages(self):
        query = "test query"
        passages = {}
        retrieval_result = RetrievalResult(query=query, retrieved_passages=passages)

        scores = self.metric.compute(retrieval_result, self.k_values)

        self.assertEqual(scores["umbrela_scores"], {})
        self.assertEqual(self.model.parse.call_count, 0)

    def test_compute_parse_error(self):
        query = "test query"
        passages = {"1": "passage 1"}
        retrieval_result = RetrievalResult(query=query, retrieved_passages=passages)

        self.model.parse.side_effect = Exception("Parse error")

        with self.assertRaises(Exception) as context:
            self.metric.compute(retrieval_result, self.k_values)

        self.assertTrue(
            "Error computing UMBRELA score: Parse error" in str(context.exception)
        )

    def test_compute_no_score(self):
        query = "test query"
        passages = {"1": "passage 1"}
        retrieval_result = RetrievalResult(query=query, retrieved_passages=passages)

        mock_response = Mock(score=None, refusal="Refused to score")
        self.model.parse.return_value = mock_response

        with self.assertRaises(Exception) as context:
            self.metric.compute(retrieval_result, self.k_values)

        self.assertTrue(
            "Failed to parse response: Refused to score" in str(context.exception)
        )

    def test_add_retrieval_metrics_empty_scores(self):
        scores = {"umbrela_scores": {}}
        self.metric.add_retrieval_metrics(scores, self.k_values)
        self.assertEqual(scores["umbrela_scores"], {})

    def test_add_retrieval_metrics_all_relevant(self):
        scores = {"umbrela_scores": {"1": 3, "2": 2, "3": 3}}
        self.metric.add_retrieval_metrics(scores, self.k_values)

        self.assertEqual(scores["retrieval_scores"]["precision@"]["3"], 1.0)
        self.assertEqual(scores["retrieval_scores"]["AP@"]["3"], 1.0)
        self.assertEqual(scores["retrieval_scores"]["MRR"], 1.0)

    def test_add_retrieval_metrics_mixed_scores(self):
        scores = {"umbrela_scores": {"1": 1, "2": 3, "3": 0}}
        self.metric.add_retrieval_metrics(scores, self.k_values)

        self.assertAlmostEqual(scores["retrieval_scores"]["precision@"]["3"], 1 / 3)
        self.assertAlmostEqual(scores["retrieval_scores"]["AP@"]["3"], 1 / 2)
        self.assertEqual(scores["retrieval_scores"]["MRR"], 0.5)

    def test_add_retrieval_metrics_no_relevant(self):
        scores = {"umbrela_scores": {"1": 1, "2": 1, "3": 0}}
        self.metric.add_retrieval_metrics(scores, self.k_values)

        self.assertEqual(scores["retrieval_scores"]["precision@"]["3"], 0.0)
        self.assertEqual(scores["retrieval_scores"]["AP@"]["3"], 0.0)
        self.assertEqual(scores["retrieval_scores"]["MRR"], 0.0)


if __name__ == "__main__":
    unittest.main()
