"""Unit tests for CitationMetric using a mocked judge (no LLM / API keys)."""
import unittest
from unittest.mock import Mock

from open_rag_eval.metrics.citation_metric import (
    CitationMetric,
    CitationSupport,
    CitationSupportValues,
)
from open_rag_eval.data_classes.rag_results import (
    RAGResult,
    RetrievalResult,
    AugmentedGenerationResult,
    GeneratedAnswerPart,
)


def _rag(answer_parts, passages):
    return RAGResult(
        retrieval_result=RetrievalResult(query="q", retrieved_passages=passages),
        generation_result=AugmentedGenerationResult(
            query="q", generated_answer=answer_parts
        ),
    )


class TestCitationMetric(unittest.TestCase):
    def setUp(self):
        self.model = Mock()
        self.metric = CitationMetric(self.model)

    def test_full_support_scores(self):
        self.model.parse.return_value = {
            "response": CitationSupport(support=CitationSupportValues.FULL),
            "metadata": {"input_tokens": 10, "output_tokens": 2},
        }
        result = _rag(
            [GeneratedAnswerPart(text="Paris is the capital.", citations=["1"])],
            {"1": "Paris is the capital of France."},
        )
        scores = self.metric.compute(result)
        self.assertEqual(scores["weighted_precision"], 1.0)
        self.assertEqual(scores["weighted_recall"], 1.0)
        self.assertEqual(scores["f1"], 1.0)
        self.assertEqual(scores["citation_score_1"], 1.0)
        self.assertEqual(scores["part_score_1"], 1.0)
        self.assertEqual(scores["token_usage"]["total_tokens"], 12)
        self.model.parse.assert_called_once()

    def test_partial_and_none_support(self):
        responses = [
            {
                "response": CitationSupport(support=CitationSupportValues.PARTIAL),
                "metadata": {"input_tokens": 5, "output_tokens": 1},
            },
            {
                "response": CitationSupport(support=CitationSupportValues.NONE),
                "metadata": {"input_tokens": 5, "output_tokens": 1},
            },
        ]
        self.model.parse.side_effect = responses
        result = _rag(
            [
                GeneratedAnswerPart(
                    text="Statement one.", citations=["a", "b"]
                )
            ],
            {"a": "partially related", "b": "unrelated"},
        )
        scores = self.metric.compute(result)
        # citation averages: a=0.5, b=0.0 -> precision 0.25
        self.assertAlmostEqual(scores["citation_score_a"], 0.5)
        self.assertAlmostEqual(scores["citation_score_b"], 0.0)
        self.assertAlmostEqual(scores["weighted_precision"], 0.25)
        self.assertAlmostEqual(scores["part_score_1"], 0.25)
        self.assertAlmostEqual(scores["weighted_recall"], 0.25)
        self.assertEqual(scores["token_usage"]["total_tokens"], 12)

    def test_missing_citations_zero_part_score(self):
        result = _rag(
            [GeneratedAnswerPart(text="No citations here.", citations=[])],
            {"1": "unused"},
        )
        scores = self.metric.compute(result)
        self.assertEqual(scores["part_score_1"], 0.0)
        self.assertEqual(scores["weighted_precision"], 0.0)
        self.assertEqual(scores["weighted_recall"], 0.0)
        self.assertEqual(scores["f1"], 0.0)
        self.model.parse.assert_not_called()

    def test_missing_passage_skipped(self):
        self.model.parse.return_value = {
            "response": CitationSupport(support=CitationSupportValues.FULL),
            "metadata": {"input_tokens": 1, "output_tokens": 1},
        }
        result = _rag(
            [GeneratedAnswerPart(text="Claim.", citations=["missing", "1"])],
            {"1": "supports claim"},
        )
        scores = self.metric.compute(result)
        self.assertIn("citation_score_1", scores)
        self.assertNotIn("citation_score_missing", scores)
        self.assertEqual(self.model.parse.call_count, 1)


if __name__ == "__main__":
    unittest.main()
