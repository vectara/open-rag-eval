"""Tests for LLM-based query generator."""

import unittest
from unittest.mock import Mock, patch

from open_rag_eval.query_generation.llm_generator import LLMQueryGenerator


class TestLLMQueryGenerator(unittest.TestCase):
    """Test cases for LLMQueryGenerator."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = Mock()
        self.generator = LLMQueryGenerator(
            model=self.mock_model,
            questions_per_doc=10
        )

    def test_init_requires_model(self):
        """Test that initialization requires a model."""
        with self.assertRaises(ValueError):
            LLMQueryGenerator(model=None)

    def test_init_requires_positive_questions_per_doc(self):
        """Test that questions_per_doc must be positive."""
        with self.assertRaises(ValueError):
            LLMQueryGenerator(model=self.mock_model, questions_per_doc=0)

    def test_init_requires_non_empty_language(self):
        """Test that language cannot be empty."""
        with self.assertRaises(ValueError):
            LLMQueryGenerator(model=self.mock_model, language="")

    def test_language_parameter_in_prompt(self):
        """Test that the language parameter is used in the prompt."""
        # Create generator with Spanish language
        generator = LLMQueryGenerator(
            model=self.mock_model,
            questions_per_doc=10,
            language="Spanish"
        )

        # Mock the model response
        self.mock_model.call.return_value = "¿Qué es esto?"

        # Generate questions
        generator.generate(documents=["Test document"], n_questions=1)

        # Verify the prompt contains Spanish
        call_args = self.mock_model.call.call_args[0][0]
        self.assertIn("Spanish", call_args)

    def test_generate_requires_documents(self):
        """Test that generate requires non-empty documents list."""
        with self.assertRaises(ValueError):
            self.generator.generate(documents=[])

    def test_generate_requires_positive_n_questions(self):
        """Test that n_questions must be positive."""
        with self.assertRaises(ValueError):
            self.generator.generate(documents=["test"], n_questions=0)

    def test_generate_validates_word_counts(self):
        """Test that max_words must be >= min_words."""
        with self.assertRaises(ValueError):
            self.generator.generate(
                documents=["test"],
                min_words=20,
                max_words=10
            )

    def test_generate_with_valid_response(self):
        """Test successful query generation."""
        # Mock LLM response
        self.mock_model.call.return_value = """What is machine learning?
How does neural network work?
Why is deep learning important?"""

        documents = ["Machine learning is a subset of AI."]
        queries = self.generator.generate(
            documents=documents,
            n_questions=3,
            min_words=2,
            max_words=10
        )

        # Verify we got questions
        self.assertGreater(len(queries), 0)
        self.assertLessEqual(len(queries), 3)

        # Verify all questions end with ?
        for query in queries:
            self.assertTrue(query.endswith('?'))

    def test_generate_filters_by_word_count(self):
        """Test that queries are filtered by word count."""
        # Mock LLM response with mixed word counts
        self.mock_model.call.return_value = """What?
What is this?
What is this long question about machine learning?"""

        documents = ["Test document"]
        queries = self.generator.generate(
            documents=documents,
            n_questions=10,
            min_words=3,
            max_words=5
        )

        # Only "What is this?" should pass (3 words)
        for query in queries:
            word_count = len(query.split())
            self.assertGreaterEqual(word_count, 3)
            self.assertLessEqual(word_count, 5)

    def test_generate_deduplicates_questions(self):
        """Test that duplicate questions are removed."""
        # Mock LLM response with duplicates
        self.mock_model.call.return_value = """What is AI?
What is AI?
How does AI work?"""

        documents = ["AI is artificial intelligence."]
        queries = self.generator.generate(
            documents=documents,
            n_questions=10,
            min_words=2,
            max_words=10
        )

        # Verify no duplicates
        self.assertEqual(len(queries), len(set(queries)))

    def test_generate_handles_model_errors(self):
        """Test that model errors are handled gracefully."""
        # Mock model to raise exception
        self.mock_model.call.side_effect = Exception("API Error")

        documents = ["Test document"]
        queries = self.generator.generate(
            documents=documents,
            n_questions=10,
            min_words=2,
            max_words=10
        )

        # Should return empty list or handle error gracefully
        self.assertEqual(len(queries), 0)

    def test_generate_cleans_question_formatting(self):
        """Test that questions are cleaned of bullets, numbers, etc."""
        # Mock LLM response with formatting
        self.mock_model.call.return_value = """- What is AI?
* How does ML work?
1. Why use deep learning?"""

        documents = ["Test document"]
        queries = self.generator.generate(
            documents=documents,
            n_questions=10,
            min_words=2,
            max_words=10
        )

        # Verify questions don't start with bullets or numbers
        for query in queries:
            self.assertFalse(query.startswith('-'))
            self.assertFalse(query.startswith('*'))
            self.assertFalse(query[0].isdigit())


if __name__ == '__main__':
    unittest.main()
