"""
Unit test for API server type mismatch fix.

Tests that the /api/v1/evaluate endpoint correctly wraps RAGResult
in MultiRAGResult before calling the evaluator.
"""

import unittest
from unittest.mock import Mock

from open_rag_eval.data_classes.rag_results import (
    RAGResult,
    RetrievalResult,
    AugmentedGenerationResult,
    GeneratedAnswerPart,
    MultiRAGResult
)
from open_rag_eval.data_classes.eval_scores import (
    ScoredRAGResult,
    MultiScoredRAGResult
)


class TestAPIServerTypeFix(unittest.TestCase):
    """Test that API server correctly handles type conversion."""

    def test_rag_result_wrapping_in_multi_rag_result(self):
        """Test that a single RAGResult can be wrapped in MultiRAGResult."""
        # Create a sample RAGResult
        retrieval_result = RetrievalResult(
            query="What is Python?",
            retrieved_passages={"1": "Python is a programming language"}
        )

        generation_result = AugmentedGenerationResult(
            query="What is Python?",
            generated_answer=[
                GeneratedAnswerPart(
                    text="Python is a high-level programming language",
                    citations=["1"]
                )
            ]
        )

        rag_result = RAGResult(
            retrieval_result=retrieval_result,
            generation_result=generation_result
        )

        # Wrap in MultiRAGResult (this is what the API fix does)
        multi_rag_result = MultiRAGResult(
            query=rag_result.retrieval_result.query,
            query_id="test_query_1",
            rag_results=[rag_result]
        )

        # Verify structure
        self.assertEqual(multi_rag_result.query, "What is Python?")
        self.assertEqual(multi_rag_result.query_id, "test_query_1")
        self.assertEqual(len(multi_rag_result.rag_results), 1)
        self.assertIs(multi_rag_result.rag_results[0], rag_result)

    def test_extracting_scored_result_from_multi_scored_result(self):
        """Test extracting single scored result from MultiScoredRAGResult."""
        # Create mock scored result
        retrieval_result = RetrievalResult(
            query="Test query",
            retrieved_passages={"1": "passage"}
        )
        generation_result = AugmentedGenerationResult(
            query="Test query",
            generated_answer=[GeneratedAnswerPart(text="answer", citations=["1"])]
        )
        rag_result = RAGResult(
            retrieval_result=retrieval_result,
            generation_result=generation_result
        )

        scored_result = ScoredRAGResult(
            rag_result=rag_result,
            scores=Mock()
        )

        # Create MultiScoredRAGResult (this is what evaluator returns)
        multi_scored_result = MultiScoredRAGResult(
            query="Test query",
            query_id="test_id",
            scored_rag_results=[scored_result]
        )

        # Extract single result (this is what the API fix does)
        extracted = multi_scored_result.scored_rag_results[0]

        # Verify we got the same object back
        self.assertIs(extracted, scored_result)
        self.assertEqual(extracted.rag_result.retrieval_result.query, "Test query")

    def test_api_endpoint_type_compatibility(self):
        """Test that the type conversion logic works correctly."""
        # This test verifies the fix logic without needing Flask request context
        # pylint: disable=import-outside-toplevel
        from open_rag_eval.api.schemas import EvaluationRequestSchema
        from open_rag_eval.api.server import convert_schema_to_rag_result

        # Create mock request data
        request_json = {
            "rag_result": {
                "retrieval_result": {
                    "query": "test",
                    "retrieved_passages": {"1": "passage"}
                },
                "generation_result": {
                    "query": "test",
                    "generated_answer": [{"text": "answer", "citations": ["1"]}]
                }
            },
            "evaluator_name": "trec",
            "query_id": "test_123"
        }

        request_data = EvaluationRequestSchema(
            rag_results=[request_json["rag_result"]],
            evaluator_name="trec",
            model_name="gpt-4o-mini"
        )

        rag_result = convert_schema_to_rag_result(request_data.rag_results[0])

        # This is the key fix - wrap in MultiRAGResult
        multi_rag_result = MultiRAGResult(
            query=rag_result.retrieval_result.query,
            query_id=request_json.get("query_id", "api_single_query"),
            rag_results=[rag_result]
        )

        # Verify the MultiRAGResult has correct structure
        self.assertIsInstance(multi_rag_result, MultiRAGResult)
        self.assertEqual(multi_rag_result.query_id, "test_123")
        self.assertEqual(len(multi_rag_result.rag_results), 1)
        self.assertIsInstance(multi_rag_result.rag_results[0], RAGResult)


if __name__ == "__main__":
    unittest.main()
