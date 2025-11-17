"""Tests for BERTScore similarity metric using torchmetrics."""
import unittest
from open_rag_eval.metrics.bert_score_similarity_metric import BERTScoreSimilarityMetric
from open_rag_eval.data_classes.rag_results import (
    MultiRAGResult, RAGResult, AugmentedGenerationResult,
    GeneratedAnswerPart, RetrievalResult
)


class TestBERTScoreSimilarityMetric(unittest.TestCase):
    """Test BERTScore computation with torchmetrics library."""

    def setUp(self):
        """Initialize BERTScore metric for testing."""
        # Use a smaller, faster model for tests
        self.metric = BERTScoreSimilarityMetric(
            model_type="distilbert-base-uncased",
            rescale_with_baseline=False,
            device="cpu"
        )

    def test_identical_strings_high_score(self):
        """Test that identical strings get a high BERTScore (close to 1.0)."""
        text = "The cat sat on the mat."
        score = self.metric._get_bert_score(text, text)  # pylint: disable=protected-access

        self.assertIsNotNone(score, "BERTScore should not return None for valid inputs")
        self.assertGreater(score, 0.95, f"Identical strings should score > 0.95, got {score}")
        self.assertLessEqual(score, 1.01, f"Score should be ~1.0, got {score}")  # Allow small FP error

    def test_similar_strings_good_score(self):
        """Test that similar strings get a good BERTScore."""
        text1 = "The cat sat on the mat."
        text2 = "A cat is sitting on a mat."
        score = self.metric._get_bert_score(text1, text2)  # pylint: disable=protected-access

        self.assertIsNotNone(score, "BERTScore should not return None for valid inputs")
        self.assertGreater(score, 0.7, f"Similar strings should score > 0.7, got {score}")
        self.assertLess(score, 1.0, f"Different strings should score < 1.0, got {score}")

    def test_different_strings_lower_score(self):
        """Test that very different strings get a lower BERTScore."""
        text1 = "The cat sat on the mat."
        text2 = "Quantum mechanics is fascinating."
        score = self.metric._get_bert_score(text1, text2)  # pylint: disable=protected-access

        self.assertIsNotNone(score, "BERTScore should not return None for valid inputs")
        self.assertGreater(score, 0.0, f"Score should be > 0.0, got {score}")
        self.assertLess(score, 0.75, f"Very different strings should score < 0.75, got {score}")

    def test_score_not_null_or_zero(self):
        """Test that BERTScore never returns None or exactly 0.0 for valid text."""
        test_pairs = [
            ("Hello world", "Hi there"),
            ("Machine learning", "Deep learning"),
            ("Python programming", "Java development"),
        ]

        for text1, text2 in test_pairs:
            score = self.metric._get_bert_score(text1, text2)  # pylint: disable=protected-access
            self.assertIsNotNone(score, f"Score should not be None for '{text1}' vs '{text2}'")
            self.assertNotEqual(score, 0.0, f"Score should not be 0.0 for '{text1}' vs '{text2}'")
            self.assertGreater(score, 0.0, f"Score should be positive for '{text1}' vs '{text2}'")

    def test_with_baseline_rescaling(self):
        """Test that baseline rescaling works (when enabled)."""
        # Note: This test may take longer as it downloads baseline data
        metric_with_baseline = BERTScoreSimilarityMetric(
            model_type="distilbert-base-uncased",
            rescale_with_baseline=True,
            device="cpu"
        )

        text1 = "The cat sat on the mat."
        text2 = "The cat is on the mat."
        score = metric_with_baseline._get_bert_score(text1, text2)  # pylint: disable=protected-access

        self.assertIsNotNone(score, "BERTScore with baseline should not return None")
        self.assertGreater(score, 0.0, f"Score with baseline should be > 0.0, got {score}")

    def test_compute_with_multiple_answers(self):
        """Test compute method with multiple RAG results."""
        # Create MultiRAGResult with 3 answers
        rag_results = [
            RAGResult(
                retrieval_result=RetrievalResult(query="Where did the cat sit?", retrieved_passages={}),
                generation_result=AugmentedGenerationResult(
                    query="Where did the cat sit?",
                    generated_answer=[GeneratedAnswerPart(text="The cat sat on the mat.", citations=[])]
                )
            ),
            RAGResult(
                retrieval_result=RetrievalResult(query="Where did the cat sit?", retrieved_passages={}),
                generation_result=AugmentedGenerationResult(
                    query="Where did the cat sit?",
                    generated_answer=[GeneratedAnswerPart(text="A cat is sitting on a mat.", citations=[])]
                )
            ),
            RAGResult(
                retrieval_result=RetrievalResult(query="Where did the cat sit?", retrieved_passages={}),
                generation_result=AugmentedGenerationResult(
                    query="Where did the cat sit?",
                    generated_answer=[GeneratedAnswerPart(text="The feline rested on the rug.", citations=[])]
                )
            ),
        ]

        multi_rag_result = MultiRAGResult(
            query="Where did the cat sit?",
            query_id="test_query_1",
            rag_results=rag_results
        )

        scores = self.metric.compute(multi_rag_result)

        # With 3 answers, we should get 3 pairwise comparisons
        self.assertEqual(len(scores), 3, f"Expected 3 pairwise scores, got {len(scores)}")

        # All scores should be valid
        for i, score in enumerate(scores):
            self.assertIsNotNone(score, f"Score {i} should not be None")
            self.assertGreater(score, 0.0, f"Score {i} should be > 0.0, got {score}")
            self.assertLessEqual(score, 1.0, f"Score {i} should be <= 1.0, got {score}")

    def test_compute_with_too_few_answers(self):
        """Test that compute returns empty list with <= 2 answers."""
        # Test with 2 answers
        rag_results = [
            RAGResult(
                retrieval_result=RetrievalResult(query="Test query", retrieved_passages={}),
                generation_result=AugmentedGenerationResult(
                    query="Test query",
                    generated_answer=[GeneratedAnswerPart(text="Answer 1", citations=[])]
                )
            ),
            RAGResult(
                retrieval_result=RetrievalResult(query="Test query", retrieved_passages={}),
                generation_result=AugmentedGenerationResult(
                    query="Test query",
                    generated_answer=[GeneratedAnswerPart(text="Answer 2", citations=[])]
                )
            ),
        ]

        multi_rag_result = MultiRAGResult(
            query="Test query",
            query_id="test_query_2",
            rag_results=rag_results
        )

        scores = self.metric.compute(multi_rag_result)
        self.assertEqual(len(scores), 0, "Should return empty list with only 2 answers")

    def test_truncate_long_text(self):
        """Test that long text is properly truncated to max_length."""
        # Create a metric with small max_length for testing
        metric = BERTScoreSimilarityMetric(
            model_type="distilbert-base-uncased",
            max_length=50,
            device="cpu"
        )

        # Create a very long text (more than 50 tokens)
        long_text = " ".join(["word"] * 200)  # 200 words will definitely exceed 50 tokens

        # Truncate the text
        truncated = metric._truncate_text(long_text)  # pylint: disable=protected-access

        # Verify truncated text is shorter
        self.assertLess(len(truncated), len(long_text),
                        "Truncated text should be shorter than original")

        # Verify we can compute BERTScore without errors
        score = metric._get_bert_score(long_text, long_text)  # pylint: disable=protected-access
        self.assertIsNotNone(score, "BERTScore should not fail with long text")
        self.assertGreater(score, 0.95, "Identical long texts should have high score")

    def test_bert_score_with_very_long_texts(self):
        """Test that BERTScore handles very long texts without errors."""
        # Create texts that exceed typical model max length (512 tokens)
        # Average English word is ~5 characters, so 1000 words ≈ 1000+ tokens
        long_text1 = " ".join([f"word{i}" for i in range(1000)])
        long_text2 = " ".join([f"word{i}" for i in range(1000)])

        # This should not raise an error due to truncation
        score = self.metric._get_bert_score(long_text1, long_text2)  # pylint: disable=protected-access

        self.assertIsNotNone(score, "BERTScore should handle very long texts")
        self.assertGreater(score, 0.95, "Identical long texts should score high even after truncation")


if __name__ == "__main__":
    unittest.main()
