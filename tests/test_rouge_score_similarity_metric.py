"""Deterministic unit tests for ROUGEScoreSimilarityMetric (no LLM / API keys)."""
import unittest

from open_rag_eval.metrics.rouge_score_similarity_metric import ROUGEScoreSimilarityMetric
from open_rag_eval.data_classes.rag_results import (
    MultiRAGResult,
    RAGResult,
    RetrievalResult,
    AugmentedGenerationResult,
    GeneratedAnswerPart,
)
from open_rag_eval.utils.constants import ROUGE_SCORE


def _rag_result(query: str, answer: str) -> RAGResult:
    return RAGResult(
        retrieval_result=RetrievalResult(
            query=query, retrieved_passages={"1": "passage"}
        ),
        generation_result=AugmentedGenerationResult(
            query=query,
            generated_answer=[GeneratedAnswerPart(text=answer, citations=["1"])],
        ),
    )


def _multi(answers, query="q", query_id="qid") -> MultiRAGResult:
    multi = MultiRAGResult(query=query, query_id=query_id)
    for answer in answers:
        multi.add_result(_rag_result(query, answer))
    return multi


class TestROUGEScoreSimilarityMetric(unittest.TestCase):
    def setUp(self):
        self.metric = ROUGEScoreSimilarityMetric()

    def test_name(self):
        self.assertEqual(self.metric.name, ROUGE_SCORE)

    def test_too_few_answers_returns_empty(self):
        # Metric requires more than 2 non-empty answers.
        self.assertEqual(self.metric.compute(_multi(["a", "b"])), [])
        self.assertEqual(self.metric.compute(_multi(["only one"])), [])

    def test_identical_answers_high_scores(self):
        text = "The capital of France is Paris."
        scores = self.metric.compute(_multi([text, text, text]))
        # C(3,2) = 3 pairs
        self.assertEqual(len(scores), 3)
        for score in scores:
            self.assertGreater(score, 0.99)
            self.assertLessEqual(score, 1.0)

    def test_diverse_answers_lower_than_identical(self):
        identical = self.metric.compute(
            _multi(
                [
                    "The cat sat on the mat.",
                    "The cat sat on the mat.",
                    "The cat sat on the mat.",
                ]
            )
        )
        diverse = self.metric.compute(
            _multi(
                [
                    "The cat sat on the mat.",
                    "Quantum mechanics is fascinating.",
                    "Baking bread requires flour and water.",
                ]
            )
        )
        self.assertEqual(len(identical), 3)
        self.assertEqual(len(diverse), 3)
        self.assertGreater(sum(identical) / len(identical), sum(diverse) / len(diverse))

    def test_strips_empty_answers(self):
        # Two empty + two real -> after strip only 2 remain -> empty result
        scores = self.metric.compute(_multi(["", "  ", "hello world", "hello world"]))
        self.assertEqual(scores, [])


if __name__ == "__main__":
    unittest.main()
