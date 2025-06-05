import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from open_rag_eval.evaluators.consistency_evaluator import ConsistencyEvaluator
from open_rag_eval.data_classes.rag_results import (
    RAGResult,
    RetrievalResult,
    AugmentedGenerationResult,
    GeneratedAnswerPart,
    MultiRAGResult,
)
from open_rag_eval.data_classes.eval_scores import (
    ConsistencyResult,
)


class MockBERTMetric:
    def __init__(self, *args, **kwargs):
        self.name = "bert"

    def compute(self, multi_rag_result):
        return [0.85, 0.76, 0.91]


class MockROUGEMetric:
    def __init__(self):
        self.name = "rouge"

    def compute(self, multi_rag_result):
        return [0.65, 0.72, 0.59]


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


class TestConsistencyEvaluator(unittest.TestCase):

    def setUp(self):
        self.bert_metric = MockBERTMetric()
        self.rouge_metric = MockROUGEMetric()

        # Inject mock metrics directly
        self.evaluator = ConsistencyEvaluator()
        self.evaluator.metric_calculators = [self.bert_metric, self.rouge_metric]
        self.evaluator.metrics_list = ["bert", "rouge"]


    def test_init_with_default_options(self):
        evaluator = ConsistencyEvaluator()
        self.assertEqual(len(evaluator.metric_calculators), 2)
        self.assertEqual(evaluator.metrics_list, ["bert", "rouge"])

    def test_init_with_custom_options(self):
        evaluator = ConsistencyEvaluator(options={"metrics": ["bert"]})
        self.assertEqual(len(evaluator.metric_calculators), 1)
        self.assertEqual(evaluator.metrics_list, ["bert"])

    def test_evaluate_with_multiple_results(self):
        multi_rag_result = create_mock_multi_rag_result(num_results=3)
        consistency_result = self.evaluator.evaluate(multi_rag_result)
        self.assertIsInstance(consistency_result, ConsistencyResult)
        self.assertEqual(consistency_result.query, "test query")
        self.assertEqual(consistency_result.query_id, "test_id")
        self.assertIsNotNone(consistency_result)
        self.assertEqual(len(consistency_result.consistency_scores), 2)

        bert_scores = consistency_result.consistency_scores["bert"]
        self.assertEqual(bert_scores.values, [0.85, 0.76, 0.91])
        self.assertAlmostEqual(bert_scores.stats["mean"], np.mean([0.85, 0.76, 0.91]), places=4)
        self.assertAlmostEqual(bert_scores.stats["std"], np.std([0.85, 0.76, 0.91]), places=4)

        rouge_scores = consistency_result.consistency_scores["rouge"]
        self.assertEqual(rouge_scores.values, [0.65, 0.72, 0.59])
        self.assertIn("mean", rouge_scores.stats)
        self.assertIn("std", rouge_scores.stats)

    def test_evaluate_with_single_result(self):
        multi_rag_result = create_mock_multi_rag_result(num_results=1)
        consistency_result = self.evaluator.evaluate(multi_rag_result)
        self.assertEqual(len(consistency_result.consistency_scores), 0)

    def test_evaluate_batch(self):
        multi_rag_results = [create_mock_multi_rag_result(f"query_{i}", f"id_{i}", num_results=3) for i in range(3)]
        consistency_results = self.evaluator.evaluate_batch(multi_rag_results)
        self.assertEqual(len(consistency_results), 3)
        for i, consistency_result in enumerate(consistency_results):
            self.assertEqual(consistency_result.query, f"query_{i}")
            self.assertEqual(consistency_result.query_id, f"id_{i}")
            self.assertEqual(len(consistency_result.consistency_scores), 2)

    def test_missing_metric(self):
        evaluator = ConsistencyEvaluator(options={"metrics": ["unknown"]})
        consistency_result = evaluator.evaluate(create_mock_multi_rag_result())
        self.assertEqual(len(consistency_result.consistency_scores), 0)


if __name__ == "__main__":
    unittest.main()
