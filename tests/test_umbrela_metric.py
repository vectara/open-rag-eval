import unittest
from unittest.mock import Mock
from open_rag_eval.metrics.umbrela_metric import UMBRELAMetric
from open_rag_eval.data_classes.rag_results import RetrievalResult


# pylint: disable=protected-access
# Accessing _calculate_average_precision is intentional for unit testing


class TestUMBRELAMetric(unittest.TestCase):
    def setUp(self):
        self.model = Mock()
        self.model.model_name = "test-model"
        self.metric = UMBRELAMetric(model=self.model)
        self.k_values = [1, 3, 5]

    def test_compute_successful(self):
        query = "test query"
        passages = {"1": "passage 1", "2": "passage 2"}
        retrieval_result = RetrievalResult(query=query, retrieved_passages=passages)

        mock_response1 = {
            "response": Mock(score=Mock(value="3")),
            "metadata": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        }
        mock_response2 = {
            "response": Mock(score=Mock(value="2")),
            "metadata": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        }
        self.model.parse.side_effect = [mock_response1, mock_response2]

        scores = self.metric.compute(retrieval_result, self.k_values)

        expected_umbrela_scores = {"1": 3, "2": 2}
        self.assertEqual(scores["umbrela_scores"], expected_umbrela_scores)
        self.assertIn("retrieval_scores", scores)
        self.assertEqual(scores["token_usage"]["total_tokens"], 300)
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

        mock_response = {
            "response": Mock(score=None, refusal="Refused to score"),
            "metadata": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        }
        self.model.parse.return_value = mock_response

        with self.assertRaises(Exception) as context:
            self.metric.compute(retrieval_result, self.k_values)

        self.assertTrue(
            "Error computing UMBRELA score: Failed to parse response: Refused to score" in str(context.exception)
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

    def test_calculate_average_precision_empty_precision_list(self):
        """Test that _calculate_average_precision handles edge case of empty precision_at_k list."""
        # This is a defensive test - in normal operation, if total_relevant > 0,
        # there should be at least one relevant item. However, this ensures
        # the method is robust to unexpected inputs.
        binary_relevance = [0, 0, 0]  # All non-relevant
        total_relevant = 0  # Correctly indicates no relevant items

        # Should return 0.0 for no relevant items
        result = self.metric._calculate_average_precision(binary_relevance, total_relevant)
        self.assertEqual(result, 0.0)

    def test_calculate_average_precision_all_relevant(self):
        """Test _calculate_average_precision with all relevant documents."""
        binary_relevance = [1, 1, 1]
        total_relevant = 3

        # AP should be perfect 1.0
        result = self.metric._calculate_average_precision(binary_relevance, total_relevant)
        self.assertAlmostEqual(result, 1.0)

    def test_calculate_average_precision_mixed_relevance(self):
        """Test _calculate_average_precision with mixed relevant/non-relevant documents."""
        binary_relevance = [1, 0, 1, 0, 1]  # Relevant at positions 1, 3, 5
        total_relevant = 3

        # Precision at relevant positions: 1/1, 2/3, 3/5
        # AP = (1.0 + 0.667 + 0.6) / 3 = 0.755...
        result = self.metric._calculate_average_precision(binary_relevance, total_relevant)
        expected = (1.0 + 2/3 + 3/5) / 3
        self.assertAlmostEqual(result, expected, places=4)


if __name__ == "__main__":
    unittest.main()
