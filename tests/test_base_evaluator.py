import unittest

from open_rag_eval.evaluators.base_evaluator import Evaluator
from open_rag_eval.data_classes.rag_results import (
    RAGResult,
    RetrievalResult,
    AugmentedGenerationResult,
    GeneratedAnswerPart,
)
from open_rag_eval.data_classes.eval_scores import (
    ScoredRAGResult,
    RAGScores,
    RetrievalScores,
    AugmentedGenerationScores,
)


class MockEvaluator(Evaluator):
    """Mock evaluator class for testing the abstract base class"""

    def __init__(self):
        self.evaluate_called = 0

    def evaluate(self, rag_result: RAGResult) -> ScoredRAGResult:
        self.evaluate_called += 1
        return ScoredRAGResult(
            rag_result=rag_result,
            scores=RAGScores(
                retrieval_score=RetrievalScores(scores={"mock_score": 0.5}),
                generation_score=AugmentedGenerationScores(scores={"mock_score": 0.7}),
            ),
        )

    # Make plot_metrics abstract by removing implementation
    @classmethod
    def plot_metrics(cls, csv_files, output_file="metrics_comparison.png"):
        raise NotImplementedError(
            "plot_metrics must be implemented by concrete evaluators"
        )


def create_mock_rag_result(query: str = "test query") -> RAGResult:
    """Helper function to create a mock RAG result for testing"""
    return RAGResult(
        retrieval_result=RetrievalResult(
            query=query, retrieved_passages={"doc1": "test passage"}
        ),
        generation_result=AugmentedGenerationResult(
            query=query,
            generated_answer=[
                GeneratedAnswerPart(text="test answer", citations=["doc1"])
            ],
        ),
    )


class TestBaseEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = MockEvaluator()

    def test_evaluate_batch(self):
        """Test that evaluate_batch processes all results and uses ThreadPoolExecutor"""
        rag_results = [create_mock_rag_result(f"query_{i}") for i in range(5)]

        scored_results = self.evaluator.evaluate_batch(rag_results)

        # Check that evaluate was called for each result
        self.assertEqual(self.evaluator.evaluate_called, 5)

        # Check that we got back the right number of results
        self.assertEqual(len(scored_results), 5)

        # Check that each result has the expected structure
        for scored_result in scored_results:
            self.assertIsInstance(scored_result, ScoredRAGResult)
            self.assertEqual(
                scored_result.scores.retrieval_score.scores["mock_score"], 0.5
            )
            self.assertEqual(
                scored_result.scores.generation_score.scores["mock_score"], 0.7
            )

    def test_evaluate_batch_empty_list(self):
        """Test that evaluate_batch handles empty input correctly"""
        scored_results = self.evaluator.evaluate_batch([])

        self.assertEqual(self.evaluator.evaluate_called, 0)
        self.assertEqual(len(scored_results), 0)


if __name__ == "__main__":
    unittest.main()
