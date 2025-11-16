"""
Unit tests for AnthropicModel parameter filtering.

Tests the _remove_invalid_kwargs method to ensure it properly handles
parameter conflicts and removes unsupported parameters.
"""

import unittest
from unittest.mock import patch

from open_rag_eval.models.llm_judges import AnthropicModel


# pylint: disable=protected-access
# Accessing _remove_invalid_kwargs is intentional for unit testing


class TestAnthropicModelKwargsFiltering(unittest.TestCase):
    """Test AnthropicModel parameter filtering without making API calls."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the Anthropic client to avoid API calls
        with patch('open_rag_eval.models.llm_judges.anthropic.Anthropic'):
            self.model = AnthropicModel({
                "name": "claude-sonnet-4-5",
                "api_key": "test-key"
            })

    def test_remove_openai_specific_parameters(self):
        """Test that OpenAI-specific parameters are removed."""
        model_kwargs = {
            "temperature": 0.0,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.0,
            "seed": 42,
        }

        filtered = self.model._remove_invalid_kwargs(model_kwargs)

        # Should keep temperature
        self.assertIn("temperature", filtered)
        self.assertEqual(filtered["temperature"], 0.0)

        # Should remove OpenAI-specific params
        self.assertNotIn("presence_penalty", filtered)
        self.assertNotIn("frequency_penalty", filtered)
        self.assertNotIn("seed", filtered)

    def test_temperature_and_top_p_conflict(self):
        """Test that top_p is removed when both temperature and top_p are present."""
        model_kwargs = {
            "temperature": 0.0,
            "top_p": 1.0,
        }

        filtered = self.model._remove_invalid_kwargs(model_kwargs)

        # Should keep temperature
        self.assertIn("temperature", filtered)
        self.assertEqual(filtered["temperature"], 0.0)

        # Should remove top_p to avoid conflict
        self.assertNotIn("top_p", filtered)

    def test_only_temperature_present(self):
        """Test that temperature is kept when only temperature is present."""
        model_kwargs = {
            "temperature": 0.5,
        }

        filtered = self.model._remove_invalid_kwargs(model_kwargs)

        # Should keep temperature
        self.assertIn("temperature", filtered)
        self.assertEqual(filtered["temperature"], 0.5)

    def test_only_top_p_present(self):
        """Test that top_p is kept when only top_p is present."""
        model_kwargs = {
            "top_p": 0.9,
        }

        filtered = self.model._remove_invalid_kwargs(model_kwargs)

        # Should keep top_p when it's the only sampling param
        self.assertIn("top_p", filtered)
        self.assertEqual(filtered["top_p"], 0.9)

    def test_umbrela_metric_kwargs(self):
        """Test filtering of actual UMBRELA metric kwargs."""
        # These are the actual kwargs from umbrela_metric.py
        model_kwargs = {
            "temperature": 0.0,
            "top_p": 1.0,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.0,
            "seed": 42,
        }

        filtered = self.model._remove_invalid_kwargs(model_kwargs)

        # Should only keep temperature
        self.assertEqual(filtered, {"temperature": 0.0})

    def test_original_kwargs_not_modified(self):
        """Test that the original kwargs dict is not modified."""
        model_kwargs = {
            "temperature": 0.0,
            "top_p": 1.0,
            "presence_penalty": 0.5,
        }

        original_keys = set(model_kwargs.keys())
        self.model._remove_invalid_kwargs(model_kwargs)

        # Original dict should remain unchanged
        self.assertEqual(set(model_kwargs.keys()), original_keys)

    def test_empty_kwargs(self):
        """Test that empty kwargs returns empty dict."""
        model_kwargs = {}
        filtered = self.model._remove_invalid_kwargs(model_kwargs)
        self.assertEqual(filtered, {})

    def test_other_valid_parameters_preserved(self):
        """Test that other valid parameters are preserved."""
        model_kwargs = {
            "temperature": 0.0,
            "max_tokens": 1024,
            "top_p": 1.0,
            "presence_penalty": 0.5,
        }

        filtered = self.model._remove_invalid_kwargs(model_kwargs)

        # Should keep temperature and max_tokens
        self.assertIn("temperature", filtered)
        self.assertIn("max_tokens", filtered)
        self.assertEqual(filtered["max_tokens"], 1024)

        # Should remove top_p and presence_penalty
        self.assertNotIn("top_p", filtered)
        self.assertNotIn("presence_penalty", filtered)


if __name__ == "__main__":
    unittest.main()
