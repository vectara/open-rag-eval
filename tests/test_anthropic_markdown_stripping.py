"""
Unit tests for Anthropic markdown code fence stripping.

Tests that the AnthropicModel correctly handles JSON responses wrapped in markdown.
"""

import json
import unittest
from unittest.mock import Mock, patch

from pydantic import BaseModel


class TestResponse(BaseModel):
    """Test Pydantic model for testing."""
    score: str


class TestAnthropicMarkdownStripping(unittest.TestCase):
    """Test Anthropic model's ability to strip markdown code fences from JSON responses."""

    def test_markdown_fence_stripping(self):
        """Test that markdown code fences are correctly stripped from JSON."""
        # This is what Claude actually returns
        markdown_wrapped_json = '''```json
{
  "score": "2"
}
```'''

        # Simulate the stripping logic
        cleaned_text = markdown_wrapped_json.strip()
        if cleaned_text.startswith('```'):
            lines = cleaned_text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            cleaned_text = '\n'.join(lines).strip()

        # Should successfully parse after stripping
        response_json = json.loads(cleaned_text)
        self.assertEqual(response_json, {"score": "2"})

    def test_plain_json_not_modified(self):
        """Test that plain JSON without markdown is not modified."""
        plain_json = '{"score": "3"}'

        # Simulate the stripping logic
        cleaned_text = plain_json.strip()
        if cleaned_text.startswith('```'):
            lines = cleaned_text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            cleaned_text = '\n'.join(lines).strip()

        # Should be unchanged
        response_json = json.loads(cleaned_text)
        self.assertEqual(response_json, {"score": "3"})

    def test_multiline_json_with_markdown(self):
        """Test multiline JSON wrapped in markdown."""
        markdown_json = '''```json
{
  "score": "1",
  "reason": "test"
}
```'''

        cleaned_text = markdown_json.strip()
        if cleaned_text.startswith('```'):
            lines = cleaned_text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            cleaned_text = '\n'.join(lines).strip()

        response_json = json.loads(cleaned_text)
        self.assertEqual(response_json, {"score": "1", "reason": "test"})

    @patch('open_rag_eval.models.llm_judges.anthropic.Anthropic')
    def test_anthropic_model_with_markdown_response(self, mock_anthropic_class):
        """Integration test with mocked Anthropic API returning markdown-wrapped JSON."""
        # pylint: disable=import-outside-toplevel
        from open_rag_eval.models.llm_judges import AnthropicModel

        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock response with markdown-wrapped JSON
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = '''```json
{
  "score": "2"
}
```'''
        mock_response.content = [mock_content]
        # Mock usage for token tracking
        mock_response.usage = Mock(input_tokens=100, output_tokens=50)
        mock_client.messages.create.return_value = mock_response

        # Create model and call parse
        model = AnthropicModel({"name": "claude-sonnet-4-5", "api_key": "test-key"})
        result = model.parse("test prompt", TestResponse, {"temperature": 0.0})

        # Verify the result - parse returns {"response": ..., "metadata": ...}
        self.assertIsInstance(result["response"], TestResponse)
        self.assertEqual(result["response"].score, "2")
        self.assertEqual(result["metadata"]["input_tokens"], 100)
        self.assertEqual(result["metadata"]["output_tokens"], 50)
        self.assertEqual(result["metadata"]["total_tokens"], 150)

    @patch('open_rag_eval.models.llm_judges.anthropic.Anthropic')
    def test_anthropic_model_with_plain_json_response(self, mock_anthropic_class):
        """Integration test with mocked Anthropic API returning plain JSON."""
        # pylint: disable=import-outside-toplevel
        from open_rag_eval.models.llm_judges import AnthropicModel

        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock response with plain JSON (no markdown)
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = '{"score": "3"}'
        mock_response.content = [mock_content]
        # Mock usage for token tracking
        mock_response.usage = Mock(input_tokens=80, output_tokens=20)
        mock_client.messages.create.return_value = mock_response

        # Create model and call parse
        model = AnthropicModel({"name": "claude-sonnet-4-5", "api_key": "test-key"})
        result = model.parse("test prompt", TestResponse, {"temperature": 0.0})

        # Verify the result - parse returns {"response": ..., "metadata": ...}
        self.assertIsInstance(result["response"], TestResponse)
        self.assertEqual(result["response"].score, "3")
        self.assertEqual(result["metadata"]["total_tokens"], 100)


if __name__ == "__main__":
    unittest.main()
