import unittest
from unittest.mock import patch, MagicMock

from open_rag_eval.evaluators.consistency_evaluator import ConsistencyEvaluator
from open_rag_eval.data_classes.rag_results import (
    RAGResult,
    RetrievalResult,
    AugmentedGenerationResult,
    GeneratedAnswerPart,
    MultiRAGResult,
)
from open_rag_eval.data_classes.eval_scores import ConsistencyResult

# Mock Metrics
class MockBERTMetric:
    def __init__(self, model_type="xlm-roberta-large", lang="en", rescale_with_baseline=True):
        self.name = "bert_score"
        self.model_type = model_type
        self.lang = lang
        self.rescale_with_baseline = rescale_with_baseline

    def compute(self, multi_rag_result):
        return [0.85, 0.76, 0.91]


class MockROUGEMetric:
    def __init__(self):
        self.name = "rouge_score"

    def compute(self, multi_rag_result):
        return [0.65, 0.72, 0.59]


# Helper Methods
def create_mock_rag_result(query="test query", passage="test passage", answer="test answer") -> RAGResult:
    return RAGResult(
        retrieval_result=RetrievalResult(query=query, retrieved_passages={"doc1": passage}),
        generation_result=AugmentedGenerationResult(
            query=query,
            generated_answer=[GeneratedAnswerPart(text=answer, citations=["doc1"])]
        )
    )


def create_mock_multi_rag_result(query="test query", query_id="test_id", num_results=3) -> MultiRAGResult:
    multi_result = MultiRAGResult(query=query, query_id=query_id)
    for i in range(num_results):
        rag_result = create_mock_rag_result(
            query=query,
            passage=f"test passage {i}",
            answer=f"test answer version {i}"
        )
        multi_result.add_result(rag_result)
    return multi_result


# Test Suite
class TestConsistencyEvaluator(unittest.TestCase):

    def setUp(self):
        # Patch metric classes with mocks
        self.bert_patch = patch('open_rag_eval.evaluators.consistency_evaluator.BERTScoreSimilarityMetric', MockBERTMetric)
        self.rouge_patch = patch('open_rag_eval.evaluators.consistency_evaluator.ROUGEScoreSimilarityMetric', MockROUGEMetric)
        self.bert_patch.start()
        self.rouge_patch.start()
        self.addCleanup(self.bert_patch.stop)
        self.addCleanup(self.rouge_patch.stop)

        self.default_options = {
            "metrics": [
                {"bert_score": {"model_type": "xlm-roberta-large"}},
                {"rouge_score": {}}
            ]
        }
        self.evaluator = ConsistencyEvaluator(options=self.default_options)

    def test_init_with_default_options(self):
        evaluator = ConsistencyEvaluator()
        metric_names = [metric.name for metric in evaluator.metric_calculators]
        self.assertIn("bert_score", metric_names)
        self.assertIn("rouge_score", metric_names)

    def test_init_with_custom_options(self):
        evaluator = ConsistencyEvaluator(options=self.default_options)
        metric_names = [metric.name for metric in evaluator.metric_calculators]
        self.assertEqual(len(metric_names), 2)
        self.assertIn("bert_score", metric_names)
        self.assertIn("rouge_score", metric_names)

    def test_evaluate_with_multiple_results(self):
        multi_rag_result = create_mock_multi_rag_result(num_results=3)
        result = self.evaluator.evaluate(multi_rag_result)

        self.assertIsInstance(result, ConsistencyResult)
        self.assertEqual(result.query, multi_rag_result.query)
        self.assertEqual(result.query_id, multi_rag_result.query_id)

        self.assertIn("bert_score", result.consistency_scores)
        self.assertIn("rouge_score", result.consistency_scores)

        bert_score = result.consistency_scores["bert_score"]
        self.assertEqual(len(bert_score.values), 3)
        self.assertTrue(all(isinstance(v, float) for v in bert_score.values))

        self.assertIn("mean", bert_score.stats)
        self.assertIn("median", bert_score.stats)
        self.assertIn("std", bert_score.stats)
        self.assertIn("iqr", bert_score.stats)

        expected_mean = sum(bert_score.values) / len(bert_score.values)
        self.assertAlmostEqual(bert_score.stats["mean"], expected_mean, places=2)

    def test_evaluate_with_single_result(self):
        multi_rag_result = create_mock_multi_rag_result(num_results=1)
        result = self.evaluator.evaluate(multi_rag_result)
        self.assertEqual(len(result.consistency_scores), 0)

    def test_missing_metric(self):
        options = {
            "metrics": [
                {"unknown_metric": {}}
            ]
        }
        evaluator = ConsistencyEvaluator(options=options)
        self.assertEqual(len(evaluator.metric_calculators), 2)

    def test_bert_score_params_are_used(self):
        options = {
            "metrics": [
                {"bert_score": {
                    "model_type": "mock-model",
                    "lang": "de",
                    "rescale_with_baseline": False
                }}
            ]
        }
        evaluator = ConsistencyEvaluator(options=options)
        metric = evaluator.metric_calculators[0]
        self.assertEqual(metric.model_type, "mock-model")
        self.assertEqual(metric.lang, "de")
        self.assertFalse(metric.rescale_with_baseline)

    def test_evaluate_calls_compute_once_per_metric(self):
        mock_bert = MagicMock()
        mock_bert.name = "bert_score"
        mock_bert.compute.return_value = [0.5, 0.6, 0.7]

        mock_rouge = MagicMock()
        mock_rouge.name = "rouge_score"
        mock_rouge.compute.return_value = [0.3, 0.4, 0.5]

        evaluator = ConsistencyEvaluator()
        evaluator.metric_calculators = [mock_bert, mock_rouge]

        multi_rag_result = create_mock_multi_rag_result(num_results=3)
        _ = evaluator.evaluate(multi_rag_result)

        mock_bert.compute.assert_called_once_with(multi_rag_result)
        mock_rouge.compute.assert_called_once_with(multi_rag_result)

    def test_evaluate_batch_with_precomputed_scores_stats(self):
        evaluator = ConsistencyEvaluator()

        mock_result1 = create_mock_multi_rag_result(3)
        mock_result2 = create_mock_multi_rag_result(3)
        mock_result1.query_id = "q1"
        mock_result2.query_id = "q2"

        precomputed = {
            "q1": {
                "hhem_score": [0.5, 0.5, 0.5],  # mean = 0.5, std = 0.0
                "umbrela_score": [0.3, 0.3, 0.3]  # mean = 0.3, std = 0.0
            },
            "q2": {
                "hhem_score": [0.7, 0.6, 0.6],  # mean = 0.633, std ~0.047
                "umbrela_score": [0.4, 0.4, 0.4]  # mean = 0.4, std = 0.0
            }
        }

        results = evaluator.evaluate_batch(
            [mock_result1, mock_result2],
            precomputed_metric_scores_by_query=precomputed
        )

        self.assertEqual(len(results), 2)

        q1 = next(r for r in results if r.query_id == "q1")
        q2 = next(r for r in results if r.query_id == "q2")

        # ---- q1 assertions ----
        hhem_stats_q1 = q1.consistency_scores["hhem_score"].stats
        self.assertAlmostEqual(hhem_stats_q1["mean"], 0.5, places=6)
        self.assertAlmostEqual(hhem_stats_q1["std"], 0.0, places=6)
        self.assertAlmostEqual(hhem_stats_q1["median"], 0.5, places=6)
        self.assertAlmostEqual(hhem_stats_q1["iqr"], 0.0, places=6)

        umbrela_stats_q1 = q1.consistency_scores["umbrela_score"].stats
        self.assertAlmostEqual(umbrela_stats_q1["mean"], 0.3, places=6)
        self.assertAlmostEqual(umbrela_stats_q1["std"], 0.0, places=6)
        self.assertAlmostEqual(umbrela_stats_q1["median"], 0.3, places=6)
        self.assertAlmostEqual(umbrela_stats_q1["iqr"], 0.0, places=6)

        # ---- q2 assertions ----
        hhem_values_q2 = [0.7, 0.6, 0.6]
        hhem_stats_q2 = q2.consistency_scores["hhem_score"].stats
        self.assertAlmostEqual(hhem_stats_q2["mean"], sum(hhem_values_q2) / 3, places=6)
        self.assertAlmostEqual(hhem_stats_q2["std"], 0.0471, places=3)
        self.assertAlmostEqual(hhem_stats_q2["median"], 0.6, places=6)
        self.assertAlmostEqual(hhem_stats_q2["iqr"], 0.05, places=6)

        umbrela_stats_q2 = q2.consistency_scores["umbrela_score"].stats
        self.assertAlmostEqual(umbrela_stats_q2["mean"], 0.4, places=6)
        self.assertAlmostEqual(umbrela_stats_q2["std"], 0.0, places=6)
        self.assertAlmostEqual(umbrela_stats_q2["median"], 0.4, places=6)
        self.assertAlmostEqual(umbrela_stats_q2["iqr"], 0.0, places=6)

    def test_evaluate_adds_hallucination_if_missing(self):
        evaluator = ConsistencyEvaluator()
        evaluator.hallucination_metric.compute = MagicMock(return_value={"hhem_score": 0.9})

        multi_result = create_mock_multi_rag_result(3)
        result = evaluator.evaluate(multi_result, precomputed_metric_scores={"bert_score": [0.5, 0.6, 0.7]})

        self.assertIn("hallucination_score", result.consistency_scores)


if __name__ == "__main__":
    unittest.main()
