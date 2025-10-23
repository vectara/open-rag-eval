"""LLM-based query generator implementation."""

import logging
import random
import re
from typing import List

from .base_generator import QueryGenerator
from ..models.llm_judges import LLMJudgeModel

logger = logging.getLogger(__name__)


class LLMQueryGenerator(QueryGenerator):
    """
    Generate queries from documents using an LLM.

    This generator uses a language model to create diverse questions
    based on document content, including factual, reasoning, and
    unanswerable questions.
    """

    def __init__(
        self,
        model: LLMJudgeModel,
        questions_per_doc: int = 10,
        language: str = "English"
    ):
        """
        Initialize LLMQueryGenerator.

        Args:
            model: LLM model to use for query generation
            questions_per_doc: Number of questions to generate per document
            language: Language for generated questions (e.g., "English", "Spanish", "French")

        Raises:
            ValueError: If parameters are invalid
        """
        if not model:
            raise ValueError("Model is required")
        if questions_per_doc < 1:
            raise ValueError("questions_per_doc must be at least 1")
        if not language:
            raise ValueError("Language cannot be empty")

        self.model = model
        self.questions_per_doc = questions_per_doc
        self.language = language

    def generate(
        self,
        documents: List[str],
        n_questions: int = 50,
        min_words: int = 5,
        max_words: int = 20,
        **kwargs
    ) -> List[str]:
        """
        Generate queries from documents using an LLM.

        Args:
            documents: List of document texts
            n_questions: Total number of questions to generate
            min_words: Minimum number of words per question
            max_words: Maximum number of words per question
            **kwargs: Additional parameters (unused)

        Returns:
            List of generated query strings

        Raises:
            ValueError: If parameters are invalid or no documents provided
        """
        if not documents:
            raise ValueError("No documents provided for query generation")
        if n_questions < 1:
            raise ValueError("n_questions must be at least 1")
        if min_words < 1:
            raise ValueError("min_words must be at least 1")
        if max_words < min_words:
            raise ValueError("max_words must be >= min_words")

        logger.info(
            "Generating %d questions from %d documents (min_words=%d, max_words=%d)",
            n_questions,
            len(documents),
            min_words,
            max_words
        )

        all_questions = []
        questions_per_doc = min(n_questions // len(documents) + 5, self.questions_per_doc)

        for idx, doc in enumerate(documents):
            logger.debug("Processing document %d/%d", idx + 1, len(documents))

            try:
                questions = self._generate_questions_for_doc(
                    doc,
                    questions_per_doc,
                    min_words,
                    max_words
                )
                all_questions.extend(questions)
            except Exception as e:
                logger.warning(
                    "Failed to generate questions for document %d: %s",
                    idx + 1,
                    str(e)
                )
                continue

        # Post-processing: deduplication and filtering
        all_questions = self._post_process_questions(
            all_questions,
            n_questions,
            min_words,
            max_words
        )

        logger.info("Successfully generated %d questions", len(all_questions))
        return all_questions

    def _generate_questions_for_doc(
        self,
        doc: str,
        num_questions: int,
        min_words: int,
        max_words: int
    ) -> List[str]:
        """
        Generate questions for a single document.

        Args:
            doc: Document text
            num_questions: Number of questions to generate
            min_words: Minimum words per question
            max_words: Maximum words per question

        Returns:
            List of generated questions

        Raises:
            Exception: If LLM call fails
        """
        prompt = f"""Given the following document text, generate {num_questions} diverse questions.
Each question should have at least {min_words} words, and no more than {max_words} words.
Generate questions at varying lengths within this range (some shorter, some longer).
Vary the question types to include:
- Questions that can be answered directly from the text.
- Questions that require reasoning, thinking or inference.
- Questions that cannot be answered from the text.
- Questions that can be partially answered from the text.
A question should not mention or refer to the text it is based on.
Each question should end with a question mark.
Your response must be a list of questions, one per line.
Do not use bullets, numbers, blank lines, code fences, or any additional text.
Your response should always be in {self.language}.
The text is:
<document>
{doc}
</document>
Your response:
"""

        response = self.model.call(prompt)
        questions = response.strip().split('\n')

        # Clean up questions: remove bullets, numbers, and formatting
        cleaned_questions = []
        for q in questions:
            if len(q.strip()) == 0:
                continue

            # Remove leading bullets, dashes, asterisks
            q = q.strip()
            q = q.lstrip('-').lstrip('*').strip()

            # Remove numbered list prefixes (e.g., "1.", "10.", "1)")
            q = re.sub(r'^\d+[\.\)]\s*', '', q)

            # Final cleanup
            q = q.strip()

            if q:
                cleaned_questions.append(q)

        # Filter out non-questions (must end with ?)
        questions = [
            q for q in cleaned_questions
            if q and q.endswith('?')
        ]

        return questions

    def _post_process_questions(
        self,
        questions: List[str],
        n_questions: int,
        min_words: int,
        max_words: int
    ) -> List[str]:
        """
        Post-process generated questions: deduplicate, filter, and sample.

        Args:
            questions: List of raw generated questions
            n_questions: Target number of questions
            min_words: Minimum words per question
            max_words: Maximum words per question

        Returns:
            Filtered and sampled list of questions
        """
        # Deduplicate
        unique_questions = list(set(questions))
        logger.info(
            "Deduplicated: %d -> %d questions",
            len(questions),
            len(unique_questions)
        )

        # Filter by word count
        filtered_questions = [
            q for q in unique_questions
            if min_words <= len(q.split()) <= max_words
        ]
        logger.info(
            "Filtered by word count: %d -> %d questions",
            len(unique_questions),
            len(filtered_questions)
        )

        # Sample if we have more than needed
        if len(filtered_questions) > n_questions:
            filtered_questions = random.sample(filtered_questions, n_questions)
            logger.info("Sampled %d questions", n_questions)

        return filtered_questions
