import unittest
from unittest.mock import Mock

from open_rag_eval.metrics.no_answer_metric import (
    NoAnswerMetric,
    QueryAnswered,
    QueryAnsweredValues,
)
from open_rag_eval.data_classes.rag_results import (
    AugmentedGenerationResult,
    GeneratedAnswerPart,
)


class TestNoAnswerMetric(unittest.TestCase):
    def setUp(self):
        self.mock_model = Mock()
        self.metric = NoAnswerMetric(self.mock_model)

    def test_compute_negative_answer(self):
        # Setup
        query = "What is the capital of France?"
        answer = "I don't have enough information to answer this question."
        self.mock_model.parse.return_value = {
            "response": QueryAnswered(answered=QueryAnsweredValues.NO),
            "metadata": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        }

        result = AugmentedGenerationResult(
            query=query,
            generated_answer=[
                GeneratedAnswerPart(text=answer, citations=["1", "2"]),
            ],
        )

        # Execute
        scores = self.metric.compute(result)

        # Assert
        self.assertEqual(scores["query_answered"], "no")
        self.assertEqual(scores["token_usage"]["total_tokens"], 150)
        self.mock_model.parse.assert_called_once()

    def test_compute_positive_answer(self):
        # Setup
        query = "What is the capital of France?"
        self.mock_model.parse.return_value = {
            "response": QueryAnswered(answered=QueryAnsweredValues.YES),
            "metadata": {"input_tokens": 80, "output_tokens": 20, "total_tokens": 100}
        }

        result = AugmentedGenerationResult(
            query=query,
            generated_answer=[
                GeneratedAnswerPart(
                    text="The capital of France is Paris.", citations=["[1]", "[2]"]
                ),
                GeneratedAnswerPart(
                    text="Paris has the eiffel tower", citations=["[2]"]
                ),
            ],
        )

        # Execute
        scores = self.metric.compute(result)

        # Assert
        self.assertEqual(scores["query_answered"], "yes")
        self.assertEqual(scores["token_usage"]["total_tokens"], 100)
        self.mock_model.parse.assert_called_once()

    def test_compute_model_error(self):
        # Setup
        query = "What is the capital of France?"
        answer = "The capital of France is Paris."
        self.mock_model.parse.side_effect = Exception("Model error")

        result = AugmentedGenerationResult(
            query=query,
            generated_answer=[GeneratedAnswerPart(text=answer, citations=[])],
        )

        # Execute and Assert
        with self.assertRaises(Exception) as context:
            self.metric.compute(result)
        self.assertIn("Error computing NoAnswer metric", str(context.exception))


if __name__ == "__main__":
    unittest.main()
