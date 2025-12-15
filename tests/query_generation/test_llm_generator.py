"""Tests for LLM-based query generator."""

import unittest
from unittest.mock import Mock

from open_rag_eval.query_generation.llm_generator import (
    LLMQueryGenerator,
    QueryWithAnswer
)


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
        self.mock_model.call.return_value = {
            "response": "¿Qué es esto?",
            "metadata": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        }

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
        self.mock_model.call.return_value = {
            "response": """What is machine learning?
How does neural network work?
Why is deep learning important?""",
            "metadata": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
        }

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
        self.mock_model.call.return_value = {
            "response": """What?
What is this?
What is this long question about machine learning?""",
            "metadata": {"input_tokens": 10, "output_tokens": 15, "total_tokens": 25}
        }

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
        self.mock_model.call.return_value = {
            "response": """What is AI?
What is AI?
How does AI work?""",
            "metadata": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}
        }

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
        self.mock_model.call.return_value = {
            "response": """- What is AI?
* How does ML work?
1. Why use deep learning?""",
            "metadata": {"input_tokens": 10, "output_tokens": 15, "total_tokens": 25}
        }

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

    def test_default_question_type_weights(self):
        """Test that default weights are equal for all types."""
        generator = LLMQueryGenerator(
            model=self.mock_model,
            questions_per_doc=10
        )

        # Verify default percentages are equal (25% each)
        percentages = generator.question_type_percentages
        self.assertEqual(percentages['directly_answerable'], 25.0)
        self.assertEqual(percentages['reasoning_required'], 25.0)
        self.assertEqual(percentages['unanswerable'], 25.0)
        self.assertEqual(percentages['partially_answerable'], 25.0)

    def test_custom_question_type_weights(self):
        """Test custom question type weights are normalized correctly."""
        weights = {
            'directly_answerable': 50,
            'reasoning_required': 30,
            'unanswerable': 0,
            'partially_answerable': 20
        }

        generator = LLMQueryGenerator(
            model=self.mock_model,
            questions_per_doc=10,
            question_type_weights=weights
        )

        # Verify weights are normalized to percentages
        percentages = generator.question_type_percentages
        self.assertEqual(percentages['directly_answerable'], 50.0)
        self.assertEqual(percentages['reasoning_required'], 30.0)
        self.assertEqual(percentages['unanswerable'], 0.0)
        self.assertEqual(percentages['partially_answerable'], 20.0)

    def test_auto_normalize_weights(self):
        """Test that weights are auto-normalized to sum to 100."""
        # Use weights that don't sum to 100
        weights = {
            'directly_answerable': 5,
            'reasoning_required': 3,
            'unanswerable': 0,
            'partially_answerable': 2
        }

        generator = LLMQueryGenerator(
            model=self.mock_model,
            questions_per_doc=10,
            question_type_weights=weights
        )

        # Verify normalized percentages
        percentages = generator.question_type_percentages
        self.assertAlmostEqual(percentages['directly_answerable'], 50.0)
        self.assertAlmostEqual(percentages['reasoning_required'], 30.0)
        self.assertAlmostEqual(percentages['unanswerable'], 0.0)
        self.assertAlmostEqual(percentages['partially_answerable'], 20.0)

        # Verify they sum to 100
        self.assertAlmostEqual(sum(percentages.values()), 100.0)

    def test_disable_question_type_with_zero_weight(self):
        """Test that setting weight to 0 disables that question type."""
        weights = {
            'directly_answerable': 1,
            'reasoning_required': 1,
            'unanswerable': 0,
            'partially_answerable': 1
        }

        generator = LLMQueryGenerator(
            model=self.mock_model,
            questions_per_doc=10,
            question_type_weights=weights
        )

        # Mock response
        self.mock_model.call.return_value = {
            "response": "What is this?",
            "metadata": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        }

        # Generate questions
        generator.generate(documents=["Test document"], n_questions=1)

        # Verify prompt doesn't mention unanswerable questions
        call_args = self.mock_model.call.call_args[0][0]
        self.assertNotIn("cannot be answered", call_args)
        self.assertNotIn("not be answerable", call_args)

    def test_all_zero_weights_raises_error(self):
        """Test that all weights being zero raises an error."""
        weights = {
            'directly_answerable': 0,
            'reasoning_required': 0,
            'unanswerable': 0,
            'partially_answerable': 0
        }

        with self.assertRaises(ValueError) as context:
            LLMQueryGenerator(
                model=self.mock_model,
                questions_per_doc=10,
                question_type_weights=weights
            )

        self.assertIn("at least one", str(context.exception).lower())

    def test_negative_weight_raises_error(self):
        """Test that negative weights raise an error."""
        weights = {
            'directly_answerable': 50,
            'reasoning_required': -10,
            'unanswerable': 25,
            'partially_answerable': 25
        }

        with self.assertRaises(ValueError) as context:
            LLMQueryGenerator(
                model=self.mock_model,
                questions_per_doc=10,
                question_type_weights=weights
            )

        self.assertIn("non-negative", str(context.exception))

    def test_invalid_question_type_key_raises_error(self):
        """Test that invalid question type keys raise an error."""
        weights = {
            'directly_answerable': 25,
            'invalid_type': 25,
            'unanswerable': 25,
            'partially_answerable': 25
        }

        with self.assertRaises(ValueError) as context:
            LLMQueryGenerator(
                model=self.mock_model,
                questions_per_doc=10,
                question_type_weights=weights
            )

        self.assertIn("Invalid question type keys", str(context.exception))

    def test_prompt_includes_enabled_question_types(self):
        """Test that prompt includes only enabled question types."""
        weights = {
            'directly_answerable': 60,
            'reasoning_required': 40,
            'unanswerable': 0,
            'partially_answerable': 0
        }

        generator = LLMQueryGenerator(
            model=self.mock_model,
            questions_per_doc=10,
            question_type_weights=weights
        )

        # Mock response
        self.mock_model.call.return_value = {
            "response": "What is this?",
            "metadata": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        }

        # Generate questions
        generator.generate(documents=["Test document"], n_questions=1)

        # Verify prompt content
        call_args = self.mock_model.call.call_args[0][0]

        # Should include enabled types with percentages
        self.assertIn("60%", call_args)
        self.assertIn("40%", call_args)
        self.assertIn("factual", call_args)
        self.assertIn("reasoning", call_args)

        # Should not include disabled types
        self.assertNotIn("not directly answerable", call_args)
        self.assertNotIn("partially", call_args)

    def test_partial_weights_specification(self):
        """Test that partial weight specification works (missing keys use 0)."""
        weights = {
            'directly_answerable': 1,
            'reasoning_required': 1
        }

        generator = LLMQueryGenerator(
            model=self.mock_model,
            questions_per_doc=10,
            question_type_weights=weights
        )

        # Verify only specified types have non-zero weights
        percentages = generator.question_type_percentages
        self.assertEqual(percentages['directly_answerable'], 50.0)
        self.assertEqual(percentages['reasoning_required'], 50.0)
        self.assertEqual(percentages.get('unanswerable', 0), 0)
        self.assertEqual(percentages.get('partially_answerable', 0), 0)


class TestQueryWithAnswer(unittest.TestCase):
    """Test cases for QueryWithAnswer dataclass."""

    def test_query_with_answer_creation(self):
        """Test that QueryWithAnswer can be created."""
        qa = QueryWithAnswer(
            query="What is machine learning?",
            expected_answer="Machine learning is a subset of AI."
        )
        self.assertEqual(qa.query, "What is machine learning?")
        self.assertEqual(qa.expected_answer, "Machine learning is a subset of AI.")


class TestGenerateWithAnswers(unittest.TestCase):
    """Test cases for generate_with_answers method."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = Mock()
        self.generator = LLMQueryGenerator(
            model=self.mock_model,
            questions_per_doc=10
        )

    def test_generate_with_answers_requires_documents(self):
        """Test that generate_with_answers requires non-empty documents list."""
        with self.assertRaises(ValueError):
            self.generator.generate_with_answers(documents=[])

    def test_generate_with_answers_requires_positive_n_questions(self):
        """Test that n_questions must be positive."""
        with self.assertRaises(ValueError):
            self.generator.generate_with_answers(documents=["test"], n_questions=0)

    def test_generate_with_answers_validates_word_counts(self):
        """Test that max_words must be >= min_words."""
        with self.assertRaises(ValueError):
            self.generator.generate_with_answers(
                documents=["test"],
                min_words=20,
                max_words=10
            )

    def test_generate_with_answers_valid_response(self):
        """Test successful QA pair generation."""
        # Mock LLM response with XML format
        self.mock_model.call.return_value = {
            "response": """<qa>
<question>What is machine learning?</question>
<answer>Machine learning is a subset of artificial intelligence.</answer>
</qa>

<qa>
<question>How does neural network work?</question>
<answer>Neural networks use layers of nodes to process information.</answer>
</qa>

<qa>
<question>Why is deep learning important?</question>
<answer>Deep learning enables complex pattern recognition in data.</answer>
</qa>""",
            "metadata": {"input_tokens": 50, "output_tokens": 100, "total_tokens": 150}
        }

        documents = ["Machine learning is a subset of AI."]
        qa_pairs = self.generator.generate_with_answers(
            documents=documents,
            n_questions=3,
            min_words=2,
            max_words=10
        )

        # Verify we got QA pairs
        self.assertGreater(len(qa_pairs), 0)

        # Verify structure
        for qa in qa_pairs:
            self.assertIsInstance(qa, QueryWithAnswer)
            self.assertTrue(qa.query.endswith('?'))
            self.assertGreater(len(qa.expected_answer), 0)

    def test_generate_with_answers_filters_by_word_count(self):
        """Test that QA pairs are filtered by question word count."""
        self.mock_model.call.return_value = {
            "response": """<qa>
<question>What?</question>
<answer>Something short.</answer>
</qa>

<qa>
<question>What is this thing?</question>
<answer>This is a test answer.</answer>
</qa>

<qa>
<question>What is this extremely long question about machine learning and artificial intelligence?</question>
<answer>Long answer here.</answer>
</qa>""",
            "metadata": {"input_tokens": 50, "output_tokens": 80, "total_tokens": 130}
        }

        documents = ["Test document"]
        qa_pairs = self.generator.generate_with_answers(
            documents=documents,
            n_questions=10,
            min_words=3,
            max_words=6
        )

        # Verify all questions are within word count range
        for qa in qa_pairs:
            word_count = len(qa.query.split())
            self.assertGreaterEqual(word_count, 3)
            self.assertLessEqual(word_count, 6)

    def test_generate_with_answers_deduplicates(self):
        """Test that duplicate questions are removed."""
        self.mock_model.call.return_value = {
            "response": """<qa>
<question>What is AI?</question>
<answer>AI is artificial intelligence.</answer>
</qa>

<qa>
<question>What is AI?</question>
<answer>AI refers to artificial intelligence systems.</answer>
</qa>

<qa>
<question>How does AI work?</question>
<answer>AI works by processing data.</answer>
</qa>""",
            "metadata": {"input_tokens": 50, "output_tokens": 80, "total_tokens": 130}
        }

        documents = ["AI is artificial intelligence."]
        qa_pairs = self.generator.generate_with_answers(
            documents=documents,
            n_questions=10,
            min_words=2,
            max_words=10
        )

        # Verify no duplicate queries
        queries = [qa.query for qa in qa_pairs]
        self.assertEqual(len(queries), len(set(queries)))

    def test_generate_with_answers_handles_model_errors(self):
        """Test that model errors are handled gracefully."""
        self.mock_model.call.side_effect = Exception("API Error")

        documents = ["Test document"]
        qa_pairs = self.generator.generate_with_answers(
            documents=documents,
            n_questions=10,
            min_words=2,
            max_words=10
        )

        # Should return empty list
        self.assertEqual(len(qa_pairs), 0)

    def test_generate_with_answers_filters_non_questions(self):
        """Test that responses without ? are filtered out."""
        self.mock_model.call.return_value = {
            "response": """<qa>
<question>What is AI?</question>
<answer>AI is artificial intelligence.</answer>
</qa>

<qa>
<question>Tell me about ML</question>
<answer>ML is machine learning.</answer>
</qa>

<qa>
<question>How does it work?</question>
<answer>It works by learning patterns.</answer>
</qa>""",
            "metadata": {"input_tokens": 50, "output_tokens": 80, "total_tokens": 130}
        }

        documents = ["Test document"]
        qa_pairs = self.generator.generate_with_answers(
            documents=documents,
            n_questions=10,
            min_words=2,
            max_words=10
        )

        # Verify all questions end with ?
        for qa in qa_pairs:
            self.assertTrue(qa.query.endswith('?'))

    def test_generate_with_answers_parses_xml_format(self):
        """Test parsing of XML-formatted QA pairs."""
        self.mock_model.call.return_value = {
            "response": """<qa>
<question>What is machine learning?</question>
<answer>Machine learning is a subset of artificial intelligence.</answer>
</qa>

<qa>
<question>How does neural network work?</question>
<answer>Neural networks use layers of nodes to process information.</answer>
</qa>""",
            "metadata": {"input_tokens": 50, "output_tokens": 60, "total_tokens": 110}
        }

        documents = ["Test document about ML"]
        qa_pairs = self.generator.generate_with_answers(
            documents=documents,
            n_questions=2,
            min_words=2,
            max_words=10
        )

        self.assertEqual(len(qa_pairs), 2)
        self.assertEqual(qa_pairs[0].query, "What is machine learning?")
        self.assertEqual(
            qa_pairs[0].expected_answer,
            "Machine learning is a subset of artificial intelligence."
        )

    def test_generate_with_answers_parses_multiline_answers(self):
        """Test parsing of multi-line answers within XML tags."""
        self.mock_model.call.return_value = {
            "response": """<qa>
<question>What is machine learning?</question>
<answer>Machine learning is a subset of artificial intelligence.
It enables systems to learn from data automatically.
This includes supervised, unsupervised, and reinforcement learning.</answer>
</qa>

<qa>
<question>How does it work?</question>
<answer>It works by:
1. Collecting training data
2. Training a model
3. Making predictions</answer>
</qa>""",
            "metadata": {"input_tokens": 50, "output_tokens": 100, "total_tokens": 150}
        }

        documents = ["Test document about ML"]
        qa_pairs = self.generator.generate_with_answers(
            documents=documents,
            n_questions=2,
            min_words=2,
            max_words=10
        )

        self.assertEqual(len(qa_pairs), 2)
        # First answer should contain all three lines
        self.assertIn("Machine learning is a subset", qa_pairs[0].expected_answer)
        self.assertIn("enables systems to learn", qa_pairs[0].expected_answer)
        self.assertIn("reinforcement learning", qa_pairs[0].expected_answer)

        # Second answer should contain the numbered list
        self.assertIn("Collecting training data", qa_pairs[1].expected_answer)
        self.assertIn("Making predictions", qa_pairs[1].expected_answer)

    def test_generate_with_answers_handles_whitespace_variations(self):
        """Test XML parsing handles various whitespace patterns."""
        self.mock_model.call.return_value = {
            "response": """<qa><question>What is AI?</question><answer>Artificial intelligence.</answer></qa>
<qa>
  <question>How does it work?</question>
  <answer>By processing data.</answer>
</qa>""",
            "metadata": {"input_tokens": 50, "output_tokens": 50, "total_tokens": 100}
        }

        documents = ["Test document"]
        qa_pairs = self.generator.generate_with_answers(
            documents=documents,
            n_questions=2,
            min_words=2,
            max_words=10
        )

        self.assertEqual(len(qa_pairs), 2)
        self.assertEqual(qa_pairs[0].query, "What is AI?")
        self.assertEqual(qa_pairs[0].expected_answer, "Artificial intelligence.")


if __name__ == '__main__':
    unittest.main()
