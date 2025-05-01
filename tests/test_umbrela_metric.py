import unittest
from unittest.mock import Mock
from open_rag_eval.metrics.umbrela_metric import UMBRELAMetric    
from open_rag_eval.data_classes.rag_results import RetrievalResult


class TestUMBRELAMetric(unittest.TestCase):

    def setUp(self):
        self.model = Mock()
        self.metric = UMBRELAMetric(model=self.model)

    def test_compute_successful(self):
        # Mock data
        query = "test query"
        passages = {"1": "passage 1", "2": "passage 2"}
        retrieval_result = RetrievalResult(query=query, retrieved_passages=passages)

        # Mock responses with string values that match score_map keys
        mock_response1 = Mock(score=Mock(value="3"))  # EXACT_ANSWER
        mock_response2 = Mock(score=Mock(value="2"))  # PARTIAL_ANSWER

        self.model.parse.side_effect = [mock_response1, mock_response2]

        # Mock add_retrieval_metrics
        self.metric.add_retrieval_metrics = Mock()

        # Execute
        scores = self.metric.compute(retrieval_result)

        # Assert
        expected_scores = {"1": 3, "2": 2}
        self.assertEqual(scores, expected_scores)
        self.assertEqual(self.model.parse.call_count, 2)
        self.metric.add_retrieval_metrics.assert_called_once_with(expected_scores)

    def test_compute_empty_passages(self):
        # Test with empty passages
        query = "test query"
        passages = {}
        retrieval_result = RetrievalResult(query=query, retrieved_passages=passages)

        # Mock add_retrieval_metrics
        self.metric.add_retrieval_metrics = Mock()

        # Execute
        scores = self.metric.compute(retrieval_result)

        # Assert
        self.assertEqual(scores, {})
        self.assertEqual(self.model.parse.call_count, 0)
        self.metric.add_retrieval_metrics.assert_called_once_with({})

    def test_compute_parse_error(self):
        # Mock data
        query = "test query"
        passages = {"1": "passage 1"}
        retrieval_result = RetrievalResult(query=query, retrieved_passages=passages)

        # Mock parse to raise exception
        self.model.parse.side_effect = Exception("Parse error")

        # Assert raises exception
        with self.assertRaises(Exception) as context:
            self.metric.compute(retrieval_result)

        self.assertTrue(
            "Error computing UMBRELA score: Parse error" in str(context.exception)
        )

    def test_compute_no_score(self):
        # Mock data
        query = "test query"
        passages = {"1": "passage 1"}
        retrieval_result = RetrievalResult(query=query, retrieved_passages=passages)

        # Mock response with no score
        mock_response = Mock(score=None, refusal="Refused to score")
        self.model.parse.return_value = mock_response

        # Assert raises ValueError
        with self.assertRaises(Exception) as context:
            self.metric.compute(retrieval_result)

        self.assertTrue(
            "Failed to parse response: Refused to score" in str(context.exception)
        )

    def test_add_retrieval_metrics_empty_scores(self):
        scores = {}
        self.metric.add_retrieval_metrics(scores)
        self.assertEqual(scores, {})

    def test_add_retrieval_metrics_all_relevant(self):
        scores = {"1": 3, "2": 2, "3": 3}  # All scores >= threshold (2)
        self.metric.add_retrieval_metrics(scores)

        self.assertEqual(scores["precision_at_3"], 1.0)  # All passages are relevant
        self.assertEqual(scores["ap_at_3"], 1.0)  # Perfect ranking
        self.assertEqual(scores["MRR"], 1.0)  # First result is relevant

    def test_add_retrieval_metrics_mixed_scores(self):
        scores = {"1": 1, "2": 3, "3": 0}  # Only one score >= threshold (2)
        self.metric.add_retrieval_metrics(scores)

        self.assertEqual(scores["precision_at_3"], 1 / 3)  # One out of three relevant
        self.assertEqual(scores["ap_at_3"], 1 / 2)  # One relevant at position 2
        self.assertEqual(scores["MRR"], 1 / 2)  # First relevant at position 2

    def test_add_retrieval_metrics_no_relevant(self):
        scores = {"1": 1, "2": 1, "3": 0}  # No scores >= threshold (2)
        self.metric.add_retrieval_metrics(scores)

        self.assertEqual(scores["precision_at_3"], 0.0)
        self.assertEqual(scores["ap_at_3"], 0.0)
        self.assertEqual(scores["MRR"], 0.0)


if __name__ == "__main__":
    unittest.main()
