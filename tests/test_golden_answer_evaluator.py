"""Unit tests for Golden Answer Evaluator and metrics.

These tests define the expected behavior for the golden answer evaluation feature:
- SemanticSimilarityMetric: embedding-based cosine similarity
- FactualCorrectnessMetric: claim decomposition + NLI for precision/recall/F1
- GoldenAnswerEvaluator: orchestrates all metrics, integrates with consistency
"""

import unittest

import numpy as np

from open_rag_eval.data_classes.rag_results import (
    RAGResult,
    RetrievalResult,
    AugmentedGenerationResult,
    GeneratedAnswerPart,
    MultiRAGResult,
)
from open_rag_eval.data_classes.eval_scores import (
    MultiScoredRAGResult,
)


# ============ Helper Functions ============

def create_mock_rag_result(query="test query", answer="test answer"):
    """Create a mock RAGResult for testing."""
    return RAGResult(
        retrieval_result=RetrievalResult(
            query=query,
            retrieved_passages={"doc1": "test passage"}
        ),
        generation_result=AugmentedGenerationResult(
            query=query,
            generated_answer=[GeneratedAnswerPart(text=answer, citations=["doc1"])]
        )
    )


def create_mock_multi_rag_result(
    query="test query",
    query_id="test_id",
    expected_answer="expected answer",
    num_results=1
):
    """Create a mock MultiRAGResult with expected_answer for testing."""
    multi_result = MultiRAGResult(query=query, query_id=query_id)
    multi_result.expected_answer = expected_answer
    for i in range(num_results):
        multi_result.add_result(create_mock_rag_result(query, f"answer version {i}"))
    return multi_result


# ============ Mock Models ============

class MockEmbeddingModel:
    """Mock embedding model for deterministic testing."""

    def __init__(self, dim=384):
        self.dim = dim

    def embed(self, texts):
        """Return deterministic embeddings based on text hash."""
        embeddings = []
        for text in texts:
            np.random.seed(hash(text) % (2**32))
            embeddings.append(np.random.randn(self.dim))
        return np.array(embeddings)

    def embed_single(self, text):
        """Embed a single text."""
        return self.embed([text])[0]


class MockLLMModel:
    """Mock LLM model for testing structured output parsing."""

    def __init__(self):
        self.parse_responses = {}

    def set_parse_response(self, response_format, response):
        """Set the response for a specific format."""
        self.parse_responses[response_format.__name__] = response

    def parse(self, prompt, response_format, model_kwargs=None):
        """Return preset response for the format."""
        format_name = response_format.__name__
        if format_name in self.parse_responses:
            return {"response": self.parse_responses[format_name], "metadata": {}}
        # Return empty default
        return {"response": response_format(), "metadata": {}}


# ============ Test Cases ============

class TestMultiRAGResultExpectedAnswer(unittest.TestCase):
    """Test that MultiRAGResult supports expected_answer field."""

    def test_multi_rag_result_has_expected_answer_field(self):
        """MultiRAGResult should have optional expected_answer field."""
        multi_result = MultiRAGResult(query="test", query_id="id1")
        # Default should be None
        self.assertIsNone(multi_result.expected_answer)

    def test_multi_rag_result_can_set_expected_answer(self):
        """Should be able to set expected_answer."""
        multi_result = MultiRAGResult(query="test", query_id="id1")
        multi_result.expected_answer = "The sky is blue"
        self.assertEqual(multi_result.expected_answer, "The sky is blue")


class TestSemanticSimilarityMetric(unittest.TestCase):
    """Test SemanticSimilarityMetric."""

    def setUp(self):
        # Import here to allow for the test to fail gracefully if not implemented
        from open_rag_eval.metrics.golden_answer_metrics import SemanticSimilarityMetric
        self.embedding_model = MockEmbeddingModel()
        self.metric = SemanticSimilarityMetric(self.embedding_model)

    def test_name(self):
        """Metric should have correct name."""
        self.assertEqual(self.metric.name, "semantic_similarity")

    def test_identical_texts_high_similarity(self):
        """Identical texts should have similarity of 1.0."""
        result = self.metric.compute(
            query="any query",
            generated_answer="The sky is blue",
            expected_answer="The sky is blue"
        )
        self.assertIn("semantic_similarity", result)
        self.assertAlmostEqual(result["semantic_similarity"], 1.0, places=5)

    def test_different_texts_have_similarity(self):
        """Different texts should have some similarity value."""
        result = self.metric.compute(
            query="any query",
            generated_answer="The sky is blue",
            expected_answer="Water is wet"
        )
        self.assertIn("semantic_similarity", result)
        # Should be some value between -1 and 1
        self.assertGreaterEqual(result["semantic_similarity"], -1.0)
        self.assertLessEqual(result["semantic_similarity"], 1.0)

    def test_returns_float_score(self):
        """Should return a float score."""
        result = self.metric.compute(
            query="query",
            generated_answer="answer",
            expected_answer="expected"
        )
        self.assertIsInstance(result["semantic_similarity"], float)


class TestFactualCorrectnessMetric(unittest.TestCase):
    """Test FactualCorrectnessMetric."""

    def setUp(self):
        from open_rag_eval.metrics.golden_answer_metrics import (
            FactualCorrectnessMetric,
            Claims,
            ClaimVerdicts,
            ClaimVerdict,
            NLIVerdict
        )
        self.Claims = Claims
        self.ClaimVerdicts = ClaimVerdicts
        self.ClaimVerdict = ClaimVerdict
        self.NLIVerdict = NLIVerdict
        self.llm_model = MockLLMModel()
        self.metric = FactualCorrectnessMetric(self.llm_model)

    def test_name(self):
        """Metric should have correct name."""
        self.assertEqual(self.metric.name, "factual_correctness")

    def test_compute_perfect_match(self):
        """Perfect match should have precision=recall=f1=1.0."""
        mock_claims = self.Claims(claims=["The sky is blue", "Water is wet"])
        mock_verdicts = self.ClaimVerdicts(verdicts=[
            self.ClaimVerdict(claim="The sky is blue", verdict=self.NLIVerdict.ENTAILMENT),
            self.ClaimVerdict(claim="Water is wet", verdict=self.NLIVerdict.ENTAILMENT)
        ])

        self.llm_model.set_parse_response(self.Claims, mock_claims)
        self.llm_model.set_parse_response(self.ClaimVerdicts, mock_verdicts)

        result = self.metric.compute(
            query="Tell me facts",
            generated_answer="The sky is blue. Water is wet.",
            expected_answer="The sky is blue. Water is wet."
        )

        self.assertIn("factual_correctness_precision", result)
        self.assertIn("factual_correctness_recall", result)
        self.assertIn("factual_correctness_f1", result)
        self.assertEqual(result["factual_correctness_precision"], 1.0)
        self.assertEqual(result["factual_correctness_recall"], 1.0)
        self.assertEqual(result["factual_correctness_f1"], 1.0)

    def test_compute_partial_match(self):
        """Partial match should have intermediate scores."""
        mock_claims = self.Claims(claims=["Claim A", "Claim B"])
        mock_verdicts_partial = self.ClaimVerdicts(verdicts=[
            self.ClaimVerdict(claim="Claim A", verdict=self.NLIVerdict.ENTAILMENT),
            self.ClaimVerdict(claim="Claim B", verdict=self.NLIVerdict.NEUTRAL)
        ])

        self.llm_model.set_parse_response(self.Claims, mock_claims)
        self.llm_model.set_parse_response(self.ClaimVerdicts, mock_verdicts_partial)

        result = self.metric.compute(
            query="query",
            generated_answer="answer",
            expected_answer="expected"
        )

        # With 1/2 entailed, precision and recall should be 0.5
        self.assertEqual(result["factual_correctness_precision"], 0.5)
        self.assertEqual(result["factual_correctness_recall"], 0.5)
        self.assertEqual(result["factual_correctness_f1"], 0.5)

    def test_returns_claims_in_result(self):
        """Result should include extracted claims."""
        mock_claims = self.Claims(claims=["Claim 1", "Claim 2"])
        mock_verdicts = self.ClaimVerdicts(verdicts=[
            self.ClaimVerdict(claim="Claim 1", verdict=self.NLIVerdict.ENTAILMENT),
            self.ClaimVerdict(claim="Claim 2", verdict=self.NLIVerdict.ENTAILMENT)
        ])

        self.llm_model.set_parse_response(self.Claims, mock_claims)
        self.llm_model.set_parse_response(self.ClaimVerdicts, mock_verdicts)

        result = self.metric.compute(
            query="q", generated_answer="a", expected_answer="e"
        )

        self.assertIn("generated_claims", result)
        self.assertIn("expected_claims", result)


class TestGoldenAnswerEvaluator(unittest.TestCase):
    """Test GoldenAnswerEvaluator."""

    def setUp(self):
        from open_rag_eval.evaluators.golden_answer_evaluator import GoldenAnswerEvaluator
        from open_rag_eval.metrics.golden_answer_metrics import (
            Claims,
            ClaimVerdicts,
            ClaimVerdict,
            NLIVerdict
        )

        self.llm_model = MockLLMModel()
        self.embedding_model = MockEmbeddingModel()

        # Setup default mock responses
        self.llm_model.set_parse_response(
            Claims,
            Claims(claims=["Claim 1", "Claim 2"])
        )
        self.llm_model.set_parse_response(
            ClaimVerdicts,
            ClaimVerdicts(verdicts=[
                ClaimVerdict(claim="Claim 1", verdict=NLIVerdict.ENTAILMENT),
                ClaimVerdict(claim="Claim 2", verdict=NLIVerdict.NEUTRAL)
            ])
        )

        self.evaluator = GoldenAnswerEvaluator(
            llm_model=self.llm_model,
            embedding_model=self.embedding_model,
            options={"run_consistency": False}
        )

    def test_evaluate_with_expected_answer(self):
        """Should evaluate when expected_answer is present."""
        multi_result = create_mock_multi_rag_result(
            expected_answer="The expected answer"
        )

        scored_result = self.evaluator.evaluate(multi_result)

        self.assertEqual(scored_result.query_id, "test_id")
        self.assertEqual(len(scored_result.scored_rag_results), 1)

        gen_scores = scored_result.scored_rag_results[0].scores.generation_score.scores
        self.assertIn("semantic_similarity", gen_scores)
        self.assertIn("factual_correctness_f1", gen_scores)

    def test_evaluate_without_expected_answer(self):
        """Should return empty results when expected_answer is None."""
        multi_result = create_mock_multi_rag_result()
        multi_result.expected_answer = None

        scored_result = self.evaluator.evaluate(multi_result)

        self.assertEqual(len(scored_result.scored_rag_results), 0)

    def test_evaluate_returns_multi_scored_rag_result(self):
        """Should return MultiScoredRAGResult."""
        multi_result = create_mock_multi_rag_result(expected_answer="expected")

        scored_result = self.evaluator.evaluate(multi_result)

        self.assertIsInstance(scored_result, MultiScoredRAGResult)

    def test_evaluate_multiple_runs_with_consistency(self):
        """With run_consistency=True, should evaluate all runs."""
        from open_rag_eval.evaluators.golden_answer_evaluator import GoldenAnswerEvaluator as GAE

        evaluator = GAE(
            llm_model=self.llm_model,
            embedding_model=self.embedding_model,
            options={"run_consistency": True}
        )

        multi_result = create_mock_multi_rag_result(
            expected_answer="expected",
            num_results=3
        )

        scored_result = evaluator.evaluate(multi_result)

        # Should have scored all 3 runs
        self.assertEqual(len(scored_result.scored_rag_results), 3)

    def test_evaluate_single_run_without_consistency(self):
        """With run_consistency=False, should only evaluate first run."""
        from open_rag_eval.evaluators.golden_answer_evaluator import GoldenAnswerEvaluator as GAE

        # Explicitly disable consistency with empty metrics list
        evaluator = GAE(
            llm_model=self.llm_model,
            embedding_model=self.embedding_model,
            options={"run_consistency": False, "metrics_to_run_consistency": []}
        )

        multi_result = create_mock_multi_rag_result(
            expected_answer="expected",
            num_results=3
        )

        scored_result = evaluator.evaluate(multi_result)

        # Should only have scored first run
        self.assertEqual(len(scored_result.scored_rag_results), 1)

    def test_collect_scores_for_consistency(self):
        """Should extract float scores for consistency evaluation."""
        from open_rag_eval.evaluators.golden_answer_evaluator import GoldenAnswerEvaluator as GAE

        evaluator = GAE(
            llm_model=self.llm_model,
            embedding_model=self.embedding_model,
            options={
                "run_consistency": True,
                "metrics_to_run_consistency": ["semantic_similarity", "factual_correctness_f1"]
            }
        )

        multi_result = create_mock_multi_rag_result(
            expected_answer="expected",
            num_results=3
        )
        scored_result = evaluator.evaluate(multi_result)

        scores_dict = {}
        result = evaluator.collect_scores_for_consistency(
            [scored_result], scores_dict
        )

        self.assertIn("test_id", result)
        self.assertIn("semantic_similarity", result["test_id"])
        self.assertIn("factual_correctness_f1", result["test_id"])

    def test_get_metrics_to_plot(self):
        """Should return list of metrics to plot."""
        metrics = self.evaluator.get_metrics_to_plot()
        self.assertIsInstance(metrics, list)
        self.assertTrue(len(metrics) > 0)

    def test_get_consolidated_columns(self):
        """Should return list of columns for consolidated CSV."""
        columns = self.evaluator.get_consolidated_columns()
        self.assertIsInstance(columns, list)
        self.assertIn("query_id", columns)
        self.assertIn("query", columns)

    def test_run_consistency_false_without_empty_metrics_list(self):
        """run_consistency=False should work without specifying metrics_to_run_consistency."""
        from open_rag_eval.evaluators.golden_answer_evaluator import GoldenAnswerEvaluator as GAE

        # Verifies that setting run_consistency=False disables consistency,
        # even if metrics_to_run_consistency is not specified (should default to empty list).
        evaluator = GAE(
            llm_model=self.llm_model,
            embedding_model=self.embedding_model,
            options={"run_consistency": False}  # Should work without metrics_to_run_consistency: []
        )
        self.assertFalse(evaluator.run_consistency)
        self.assertEqual(evaluator.metrics_to_run_consistency, [])


class TestGoldenAnswerMetricBase(unittest.TestCase):
    """Test GoldenAnswerMetric base class."""

    def test_base_class_is_abstract(self):
        """GoldenAnswerMetric should be abstract."""
        from open_rag_eval.metrics.golden_answer_metrics import GoldenAnswerMetric

        # Should not be able to instantiate directly
        with self.assertRaises(TypeError):
            GoldenAnswerMetric()


if __name__ == "__main__":
    unittest.main()
