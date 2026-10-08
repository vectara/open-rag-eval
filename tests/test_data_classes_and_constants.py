"""Lightweight unit tests for data classes and constants (no external services)."""
import unittest

from open_rag_eval.utils import constants
from open_rag_eval.data_classes.rag_results import (
    MultiRAGResult,
    RAGResult,
    RetrievalResult,
    AugmentedGenerationResult,
    GeneratedAnswerPart,
)


class TestConstants(unittest.TestCase):
    def test_metric_name_constants(self):
        self.assertEqual(constants.ROUGE_SCORE, "rouge_score")
        self.assertEqual(constants.BERT_SCORE, "bert_score")
        self.assertEqual(constants.NO_ANSWER, "NO_ANSWER")


class TestMultiRAGResult(unittest.TestCase):
    def test_add_result(self):
        multi = MultiRAGResult(query="q", query_id="1")
        self.assertEqual(multi.rag_results, [])
        rag = RAGResult(
            retrieval_result=RetrievalResult(query="q", retrieved_passages={}),
            generation_result=AugmentedGenerationResult(
                query="q",
                generated_answer=[GeneratedAnswerPart(text="a", citations=[])],
            ),
        )
        multi.add_result(rag)
        self.assertEqual(len(multi.rag_results), 1)
        self.assertIsNone(multi.expected_answer)


if __name__ == "__main__":
    unittest.main()
