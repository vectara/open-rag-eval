import unittest

import pandas as pd

from open_rag_eval.utils.eval_plotting import get_answered_percentage


class TestAnsweredPercentage(unittest.TestCase):

    def test_falls_back_to_run_one_when_consolidated_value_is_missing(self):
        for missing in [float("nan"), None, pd.NA, ""]:
            with self.subTest(missing=missing):
                results = pd.DataFrame({
                    "generation_score_no_answer_score": [missing, '{"query_answered": "no"}'],
                    "run_1_generation_score_no_answer_score": [
                        '{"query_answered": "yes"}',
                        '{"query_answered": "no"}',
                    ],
                })
                self.assertEqual(get_answered_percentage(results), 50.0)

    def test_ignores_rows_without_either_score(self):
        results = pd.DataFrame({
            "generation_score_no_answer_score": [float("nan")],
            "run_1_generation_score_no_answer_score": [float("nan")],
        })
        self.assertEqual(get_answered_percentage(results), 0)

    def test_consolidated_score_takes_precedence(self):
        results = pd.DataFrame({
            "generation_score_no_answer_score": ['{"query_answered": "no"}'],
            "run_1_generation_score_no_answer_score": ['{"query_answered": "yes"}'],
        })
        self.assertEqual(get_answered_percentage(results), 0)

    def test_accepts_either_score_column_alone(self):
        for column in ["generation_score_no_answer_score", "run_1_generation_score_no_answer_score"]:
            with self.subTest(column=column):
                results = pd.DataFrame({column: ['{"query_answered": "yes"}']})
                self.assertEqual(get_answered_percentage(results), 100.0)

    def test_empty_results_have_zero_answered_percentage(self):
        self.assertEqual(get_answered_percentage(pd.DataFrame()), 0)

    def test_invalid_json_remains_excluded(self):
        results = pd.DataFrame({
            "generation_score_no_answer_score": ["not JSON", '{"query_answered": "yes"}'],
        })
        with self.assertLogs(level="ERROR"):
            self.assertEqual(get_answered_percentage(results), 100.0)


if __name__ == "__main__":
    unittest.main()
