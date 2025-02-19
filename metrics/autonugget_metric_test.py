import unittest
from unittest.mock import MagicMock

from metrics.autonugget_metric import AutoNuggetMetric
from models.llm_judges import OpenAIModel

class TestAutoNuggetMetric(unittest.TestCase):

    def setUp(self):
        self.model = MagicMock(spec=OpenAIModel)
        self.metric = AutoNuggetMetric(model=self.model)

    def test_create_nuggets(self):
        query = "test query"
        retrieved_passages = {"1": "passage one", "2": "passage two", "3": "passage 3"}
        umbrela_scores = {"1": 2, "2": 1 ,"3": 0}
        self.model.call.return_value = '["nugget1", "nugget2"]'

        nuggets = self.metric._create_nuggets(query, retrieved_passages, umbrela_scores)
        self.assertEqual(nuggets, ["nugget1", "nugget2"])

    def test_score_and_sort_nuggets(self):
        query = "test query"
        nuggets = ["nugget1", "nugget2", "nugget3"]
        self.model.call.return_value = '["vital", "okay", "vital"]'

        sorted_nuggets, sorted_labels = self.metric._score_and_sort_nuggets(query, nuggets)
        self.assertEqual(sorted_nuggets, ["nugget1", "nugget3", "nugget2"])
        self.assertEqual(sorted_labels, ["vital", "vital", "okay"])

    def test_assign_nuggets(self):
        query = "test query"
        generated_passage = "generated passage"
        nuggets = ["nugget1", "nugget2"]
        self.model.call.return_value = '["support", "partial_support"]'

        assignments = self.metric._assign_nuggets(query, generated_passage, nuggets)
        self.assertEqual(assignments, ["support", "partial_support"])

    def test_evaluate_answer(self):
        nuggets = ["nugget1", "nugget2"]
        labels = ["vital", "okay"]
        nugget_assignments = ["support", "partial_support"]

        scores = self.metric._evaluate_answer(nuggets, labels, nugget_assignments)
        print(scores)
        expected_scores = {
            "All": 0.75,
            "All Strict": 0.5,
            "Vital": 1.0,
            "Vital Strict": 1.0,
            "Weighted": 0.75,
            "Weighted Strict": 0.5
        }
        self.assertEqual(scores, expected_scores)

if __name__ == "__main__":
    unittest.main()