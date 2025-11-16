"""
Unit tests for factuality backends (HHEM and Vectara API).
"""

import os
import unittest
from unittest.mock import MagicMock, Mock, patch

import requests

from open_rag_eval.data_classes.rag_results import (
    AugmentedGenerationResult,
    GeneratedAnswerPart,
    RAGResult,
    RetrievalResult
)
from open_rag_eval.metrics.factuality_backends import (
    HHEMBackend,
    VectaraAPIBackend
)
from open_rag_eval.metrics.hallucination_metric import HallucinationMetric


class TestHHEMBackend(unittest.TestCase):
    """Test cases for HHEM backend."""

    @patch('open_rag_eval.metrics.factuality_backends.AutoModelForSequenceClassification')
    def test_hhem_backend_initialization(self, mock_model_class):
        """Test HHEM backend initializes correctly."""
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        backend = HHEMBackend(
            model_name='test-model',
            detection_threshold=0.7,
            max_chars=4096
        )

        # Verify model was loaded with correct parameters
        mock_model_class.from_pretrained.assert_called_once_with(
            'test-model',
            trust_remote_code=True,
            token=os.environ.get('HF_TOKEN')
        )
        self.assertEqual(backend.detection_threshold, 0.7)
        self.assertEqual(backend.max_chars, 4096)

    @patch('open_rag_eval.metrics.factuality_backends.AutoModelForSequenceClassification')
    def test_hhem_backend_evaluate(self, mock_model_class):
        """Test HHEM backend evaluation."""
        mock_model = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.item.return_value = 0.85
        mock_model.predict.return_value = mock_prediction
        mock_model_class.from_pretrained.return_value = mock_model

        backend = HHEMBackend()

        sources = "Source text about AI."
        summary = "AI is a field of computer science."
        score = backend.evaluate(sources, summary)

        # Verify prediction was called correctly
        mock_model.predict.assert_called_once_with([(sources, summary)])
        self.assertEqual(score, 0.85)

    @patch('open_rag_eval.metrics.factuality_backends.AutoModelForSequenceClassification')
    def test_hhem_backend_truncates_long_sources(self, mock_model_class):
        """Test HHEM backend truncates sources exceeding max_chars."""
        mock_model = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.item.return_value = 0.75
        mock_model.predict.return_value = mock_prediction
        mock_model_class.from_pretrained.return_value = mock_model

        backend = HHEMBackend(max_chars=100)

        sources = "A" * 200  # 200 characters
        summary = "Summary"
        score = backend.evaluate(sources, summary)

        # Verify sources were truncated to 100 characters
        call_args = mock_model.predict.call_args[0][0]
        self.assertEqual(len(call_args[0][0]), 100)
        self.assertEqual(score, 0.75)


class TestVectaraAPIBackend(unittest.TestCase):
    """Test cases for Vectara API backend."""

    def test_vectara_backend_initialization_without_api_key(self):
        """Test Vectara backend raises error without API key."""
        with self.assertRaises(ValueError) as context:
            VectaraAPIBackend(api_key="")

        self.assertIn("api_key is required", str(context.exception))

    def test_vectara_backend_initialization_with_api_key(self):
        """Test Vectara backend initializes correctly with API key."""
        backend = VectaraAPIBackend(api_key="test-key-123")

        self.assertEqual(backend._api_key, "test-key-123")
        self.assertEqual(backend._base_url, "https://api.vectara.io")
        self.assertEqual(
            backend._endpoint,
            "https://api.vectara.io/v2/evaluate_factual_consistency"
        )

    def test_vectara_backend_custom_base_url(self):
        """Test Vectara backend accepts custom base URL."""
        backend = VectaraAPIBackend(
            api_key="test-key",
            base_url="https://custom.api.com/"
        )

        self.assertEqual(backend._base_url, "https://custom.api.com")
        self.assertEqual(
            backend._endpoint,
            "https://custom.api.com/v2/evaluate_factual_consistency"
        )

    @patch('open_rag_eval.metrics.factuality_backends.requests.post')
    def test_vectara_backend_successful_evaluation(self, mock_post):
        """Test Vectara backend makes correct API call and returns score."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"score": 0.92}
        mock_response.raise_for_status = Mock()  # Mock raise_for_status to not raise
        mock_post.return_value = mock_response

        backend = VectaraAPIBackend(api_key="test-key")

        sources = "Source text about machine learning."
        summary = "Machine learning is a subset of AI."
        score = backend.evaluate(sources, summary)

        # Verify API was called correctly
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]

        # Check headers
        self.assertEqual(call_kwargs['headers']['x-api-key'], "test-key")
        self.assertEqual(call_kwargs['headers']['Content-Type'], "application/json")

        # Check payload matches Vectara API specification
        payload = call_kwargs['json']
        self.assertEqual(payload['generated_text'], summary)
        self.assertEqual(payload['source_texts'], [sources])  # Should be a list

        # Check score
        self.assertEqual(score, 0.92)

    @patch('open_rag_eval.metrics.factuality_backends.requests.post')
    def test_vectara_backend_alternative_response_keys(self, mock_post):
        """Test Vectara backend handles different response key formats."""
        test_cases = [
            {"factual_consistency_score": 0.88},
            {"consistency_score": 0.91},
            {"result": 0.79},
        ]

        backend = VectaraAPIBackend(api_key="test-key")

        for response_data in test_cases:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = response_data
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            score = backend.evaluate("sources", "summary")
            expected_score = list(response_data.values())[0]
            self.assertEqual(score, expected_score)

    @patch('open_rag_eval.metrics.factuality_backends.requests.post')
    def test_vectara_backend_403_error(self, mock_post):
        """Test Vectara backend handles 403 Forbidden error."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.text = "Access denied"
        # Simulate raise_for_status behavior
        http_error = requests.exceptions.HTTPError("403 Client Error")
        http_error.response = mock_response
        mock_response.raise_for_status = Mock(side_effect=http_error)
        mock_post.return_value = mock_response

        backend = VectaraAPIBackend(api_key="invalid-key")

        with self.assertRaises(RuntimeError) as context:
            backend.evaluate("sources", "summary")

        self.assertIn("Access forbidden", str(context.exception))
        self.assertIn("403", str(context.exception))

    @patch('open_rag_eval.metrics.factuality_backends.requests.post')
    def test_vectara_backend_422_error(self, mock_post):
        """Test Vectara backend handles 422 Unsupported Language error."""
        mock_response = Mock()
        mock_response.status_code = 422
        mock_response.text = "Unsupported language"
        # Simulate raise_for_status behavior
        http_error = requests.exceptions.HTTPError("422 Client Error")
        http_error.response = mock_response
        mock_response.raise_for_status = Mock(side_effect=http_error)
        mock_post.return_value = mock_response

        backend = VectaraAPIBackend(api_key="test-key")

        with self.assertRaises(RuntimeError) as context:
            backend.evaluate("sources", "summary")

        self.assertIn("Unsupported language", str(context.exception))
        self.assertIn("422", str(context.exception))

    @patch('open_rag_eval.metrics.factuality_backends.requests.post')
    def test_vectara_backend_generic_error(self, mock_post):
        """Test Vectara backend handles generic API errors."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        # Simulate raise_for_status behavior
        http_error = requests.exceptions.HTTPError("500 Server Error")
        http_error.response = mock_response
        mock_response.raise_for_status = Mock(side_effect=http_error)
        mock_post.return_value = mock_response

        backend = VectaraAPIBackend(api_key="test-key")

        # Retry decorator will retry 3 times, then raise RuntimeError
        with self.assertRaises(RuntimeError) as context:
            backend.evaluate("sources", "summary")

        # Verify the error message contains information about the failure
        self.assertIn("Vectara API request failed", str(context.exception))

    @patch('open_rag_eval.metrics.factuality_backends.requests.post')
    def test_vectara_backend_timeout(self, mock_post):
        """Test Vectara backend handles timeout."""
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        backend = VectaraAPIBackend(api_key="test-key")

        # Retry decorator will retry 3 times, then RuntimeError will be raised
        with self.assertRaises(RuntimeError) as context:
            backend.evaluate("sources", "summary")

        # Verify it's related to timeout
        self.assertIn("timed out", str(context.exception).lower())

    @patch('open_rag_eval.metrics.factuality_backends.requests.post')
    def test_vectara_backend_unexpected_response_format(self, mock_post):
        """Test Vectara backend handles unexpected response format."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"unexpected_key": "value"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        backend = VectaraAPIBackend(api_key="test-key")

        with self.assertRaises(RuntimeError) as context:
            backend.evaluate("sources", "summary")

        self.assertIn("Could not extract score", str(context.exception))


class TestHallucinationMetric(unittest.TestCase):
    """Test cases for HallucinationMetric with backend selection."""

    def _create_test_rag_result(self):
        """Helper to create a test RAG result."""
        retrieval_result = RetrievalResult(
            query="What is AI?",
            retrieved_passages={
                "1": "First passage about AI.",
                "2": "Second passage about machine learning."
            }
        )

        generation_result = AugmentedGenerationResult(
            query="What is AI?",
            generated_answer=[
                GeneratedAnswerPart(
                    text="AI is a field of computer science.",
                    citations=["1"]
                ),
                GeneratedAnswerPart(
                    text="Machine learning is a subset of AI.",
                    citations=["1", "2"]
                )
            ]
        )

        return RAGResult(
            retrieval_result=retrieval_result,
            generation_result=generation_result
        )

    @patch('open_rag_eval.metrics.factuality_backends.AutoModelForSequenceClassification')
    def test_hallucination_metric_default_backend(self, mock_model_class):
        """Test HallucinationMetric uses HHEM backend by default."""
        mock_model = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.item.return_value = 0.88
        mock_model.predict.return_value = mock_prediction
        mock_model_class.from_pretrained.return_value = mock_model

        # Create metric without specifying backend (should default to HHEM)
        metric = HallucinationMetric()

        self.assertIsInstance(metric.backend, HHEMBackend)

        # Test compute
        rag_result = self._create_test_rag_result()
        result = metric.compute(rag_result)

        self.assertIn("hhem_score", result)
        self.assertEqual(result["hhem_score"], 0.88)

    @patch('open_rag_eval.metrics.factuality_backends.AutoModelForSequenceClassification')
    def test_hallucination_metric_explicit_hhem_backend(self, mock_model_class):
        """Test HallucinationMetric with explicit HHEM backend config."""
        mock_model = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.item.return_value = 0.82
        mock_model.predict.return_value = mock_prediction
        mock_model_class.from_pretrained.return_value = mock_model

        metric = HallucinationMetric(
            backend_type="hhem",
            model_name="custom-model",
            max_chars=4096
        )

        self.assertIsInstance(metric.backend, HHEMBackend)
        self.assertEqual(metric.backend.max_chars, 4096)

        rag_result = self._create_test_rag_result()
        result = metric.compute(rag_result)

        self.assertEqual(result["hhem_score"], 0.82)

    @patch('open_rag_eval.metrics.factuality_backends.requests.post')
    def test_hallucination_metric_vectara_backend(self, mock_post):
        """Test HallucinationMetric with Vectara API backend."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"score": 0.95}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        metric = HallucinationMetric(
            backend_type="vectara_api",
            api_key="test-api-key"
        )

        self.assertIsInstance(metric.backend, VectaraAPIBackend)

        rag_result = self._create_test_rag_result()
        result = metric.compute(rag_result)

        self.assertIn("hhem_score", result)
        self.assertEqual(result["hhem_score"], 0.95)

        # Verify API was called with correct sources and summary using Vectara API format
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs['json']

        expected_sources = "First passage about AI. Second passage about machine learning."
        expected_summary = "AI is a field of computer science. Machine learning is a subset of AI."

        # Vectara API expects generated_text and source_texts (as a list)
        self.assertEqual(payload['generated_text'], expected_summary)
        self.assertEqual(payload['source_texts'], [expected_sources])

    def test_hallucination_metric_invalid_backend(self):
        """Test HallucinationMetric raises error for invalid backend type."""
        with self.assertRaises(ValueError) as context:
            HallucinationMetric(backend_type="invalid_backend")

        self.assertIn("Unsupported backend_type", str(context.exception))
        self.assertIn("invalid_backend", str(context.exception))

    def test_hallucination_metric_vectara_without_api_key(self):
        """Test HallucinationMetric raises error for Vectara backend without API key."""
        # When api_key is not provided at all, it will raise TypeError
        # When api_key is empty string, it will raise ValueError
        with self.assertRaises((TypeError, ValueError)) as context:
            HallucinationMetric(backend_type="vectara_api")

        # Check for either missing argument or required key error
        self.assertTrue(
            "api_key" in str(context.exception).lower() or
            "missing 1 required positional argument" in str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
