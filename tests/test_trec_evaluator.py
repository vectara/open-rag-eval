import unittest

from open_rag_eval.evaluators.base_evaluator import Evaluator
from open_rag_eval.data_classes.rag_results import (
    RAGResult,
    RetrievalResult,
    AugmentedGenerationResult,
    GeneratedAnswerPart,
    MultiRAGResult,  # Add this import
)
from open_rag_eval.data_classes.eval_scores import (
    ScoredRAGResult,
    RAGScores,
    RetrievalScores,
    AugmentedGenerationScores,
    MultiScoredRAGResult,  # Add this import
)


class MockEvaluator(Evaluator):
    """Mock evaluator class for testing the abstract base class"""

    def __init__(self):
        self.evaluate_called = 0

    def evaluate(self,
                 multi_rag_result: MultiRAGResult) -> MultiScoredRAGResult:
        self.evaluate_called += 1
        # Create a simple mock ScoredRAGResult for each RAG result in the multi_rag_result
        scored_results = []

        for rag_result in multi_rag_result.rag_results:
            retrieval_scores = RetrievalScores(scores={"mock_score": 0.5})
            generation_scores = AugmentedGenerationScores(
                scores={"mock_score": 0.7})
            rag_scores = RAGScores(retrieval_score=retrieval_scores,
                                   generation_score=generation_scores)
            scored_result = ScoredRAGResult(rag_result=rag_result,
                                            scores=rag_scores)
            scored_results.append(scored_result)

        # Return MultiScoredRAGResult with all scored results
        return MultiScoredRAGResult(query=multi_rag_result.query,
                                    query_id=multi_rag_result.query_id,
                                    scored_rag_results=scored_results)

    # Make plot_metrics abstract by removing implementation
    @classmethod
    def plot_metrics(cls, csv_files, output_file="metrics_comparison.png", metrics_to_plot=None):
        raise NotImplementedError(
            "plot_metrics must be implemented by concrete evaluators")

    def to_csv(self, scored_results, output_file: str):
        raise NotImplementedError(
            "to_csv must be implemented by concrete evaluators")

    def get_consolidated_columns(self):
        raise NotImplementedError(
            "get_consolidated_columns must be implemented by concrete evaluators")

    def get_metrics_to_plot(self):
        raise NotImplementedError(
            "get_metrics_to_plot must be implemented by concrete evaluators")


def create_mock_multi_rag_result(query: str = "test query",
                                 query_id: str = "test_id") -> MultiRAGResult:
    """Helper function to create a mock MultiRAGResult for testing"""
    multi_result = MultiRAGResult(query=query, query_id=query_id)
    # Create first RAG result
    rag_result1 = create_mock_rag_result(query, "passage1", "answer1")
    multi_result.add_result(rag_result1)
    # Create second RAG result
    rag_result2 = create_mock_rag_result(query, "passage2", "answer2")
    multi_result.add_result(rag_result2)
    return multi_result


def create_mock_rag_result(query: str = "test query",
                           passage: str = "test passage",
                           answer: str = "test answer") -> RAGResult:
    """Helper function to create a mock RAG result for testing"""
    return RAGResult(
        retrieval_result=RetrievalResult(query=query,
                                         retrieved_passages={"doc1": passage}),
        generation_result=AugmentedGenerationResult(
            query=query,
            generated_answer=[
                GeneratedAnswerPart(text=answer, citations=["doc1"])
            ],
        ),
    )


class TestBaseEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = MockEvaluator()

    def test_evaluate_batch(self):
        """Test that evaluate_batch processes all results and uses ThreadPoolExecutor"""
        # Create a list of MultiRAGResult objects
        multi_rag_results = [
            create_mock_multi_rag_result(f"query_{i}", f"id_{i}")
            for i in range(5)
        ]

        scored_results = self.evaluator.evaluate_batch(multi_rag_results)

        # Check that evaluate was called for each result
        self.assertEqual(self.evaluator.evaluate_called, 5)

        # Check that we got back the right number of results
        self.assertEqual(len(scored_results), 5)

        # Check that each result has the expected structure
        for scored_result in scored_results:
            self.assertIsInstance(scored_result, MultiScoredRAGResult)
            self.assertTrue(len(scored_result.scored_rag_results) > 0)
            self.assertEqual(len(scored_result.scored_rag_results), 2)
            # Check the first scored result in the multi result
            first_scored = scored_result.scored_rag_results[0]
            self.assertIsInstance(first_scored, ScoredRAGResult)
            self.assertIsInstance(first_scored.scores, RAGScores)
            self.assertEqual(
                first_scored.scores.retrieval_score.scores["mock_score"], 0.5)
            self.assertEqual(
                first_scored.scores.generation_score.scores["mock_score"], 0.7)
            # Check the second scored result in the multi result
            second_scored = scored_result.scored_rag_results[1]
            self.assertIsInstance(second_scored, ScoredRAGResult)
            self.assertIsInstance(second_scored.scores, RAGScores)
            self.assertEqual(
                second_scored.scores.retrieval_score.scores["mock_score"], 0.5)
            self.assertEqual(
                second_scored.scores.generation_score.scores["mock_score"], 0.7)

    def test_evaluate_batch_empty_list(self):
        """Test that evaluate_batch handles empty input correctly"""
        scored_results = self.evaluator.evaluate_batch([])

        self.assertEqual(self.evaluator.evaluate_called, 0)
        self.assertEqual(len(scored_results), 0)


if __name__ == "__main__":
    unittest.main()
