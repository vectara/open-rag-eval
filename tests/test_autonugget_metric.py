import unittest
from unittest.mock import MagicMock, Mock

from open_rag_eval.metrics.autonugget_metric import (
    AutoNuggetMetric,
    Nuggets,
    NuggetImportance,
    NuggetAssignment,
    NuggetImportanceValues,
    NuggetAssignmentValues
)
from open_rag_eval.models.llm_judges import OpenAIModel
from open_rag_eval.data_classes.rag_results import GeneratedAnswerPart

class TestAutoNuggetMetric(unittest.TestCase):

    def setUp(self):
        self.model = MagicMock(spec=OpenAIModel)
        self.metric = AutoNuggetMetric(model=self.model)

    def test_create_nuggets(self):
        query = "test query"
        retrieved_passages = {"1": "passage one", "2": "passage two", "3": "passage 3"}
        umbrela_scores = {"1": 2, "2": 1 ,"3": 0}

        mock_response = Mock()
        mock_response = Nuggets(nuggets=["nugget1", "nugget2"])
        self.model.parse.return_value = mock_response

        nuggets = self.metric._create_nuggets(query, retrieved_passages, umbrela_scores)
        self.assertEqual(nuggets, ["nugget1", "nugget2"])

    def test_score_and_sort_nuggets(self):
        query = "test query"
        nuggets = ["nugget1", "nugget2", "nugget3"]

        mock_response = Mock()
        mock_response = NuggetImportance(importance=[
            NuggetImportanceValues.VITAL,
            NuggetImportanceValues.OKAY,
            NuggetImportanceValues.VITAL
        ])
        self.model.parse.return_value = mock_response

        sorted_nuggets, sorted_labels = self.metric._score_and_sort_nuggets(query, nuggets)
        self.assertEqual(sorted_nuggets, ["nugget1", "nugget3", "nugget2"])
        self.assertEqual(sorted_labels, ["vital", "vital", "okay"])

    def test_assign_nuggets(self):
        query = "test query"
        generated_answer = [GeneratedAnswerPart(text="generated passage", citations=["1"])]
        nuggets = ["nugget1", "nugget2"]

        mock_response = Mock()
        mock_response = NuggetAssignment(assignment=[
            NuggetAssignmentValues.SUPPORT,
            NuggetAssignmentValues.PARTIAL_SUPPORT
        ])
        self.model.parse.return_value = mock_response

        assignments = self.metric._assign_nuggets(query, generated_answer, nuggets)
        self.assertEqual(assignments, ["support", "partial_support"])

    def test_evaluate_answer(self):
        nuggets = ["nugget1", "nugget2"]
        labels = ["vital", "okay"]
        nugget_assignments = ["support", "partial_support"]

        scores = self.metric._evaluate_answer(nuggets, labels, nugget_assignments)
        expected_scores = {
            "All": 0.75,
            "All Strict": 0.5,
            "Vital": 1.0,
            "Vital Strict": 1.0,
            "Weighted": 0.83333,
            "Weighted Strict": 0.66666,
        }

        for key in expected_scores:
            self.assertAlmostEqual(
                scores[key],
                expected_scores[key],
                places=3,  # Number of decimal places to check
                msg=f"Mismatch for {key}"
            )


if __name__ == "__main__":
    unittest.main()
