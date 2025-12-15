"""LLM-based query generator implementation."""

import logging
import math
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from tqdm import tqdm

from .base_generator import QueryGenerator
from ..models.llm_judges import LLMJudgeModel

logger = logging.getLogger(__name__)


@dataclass
class QueryWithAnswer:
    """Container for a query with its expected answer."""
    query: str
    expected_answer: str


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
        language: str = "English",
        question_type_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize LLMQueryGenerator.

        Args:
            model: LLM model to use for query generation
            questions_per_doc: Number of questions to generate per document
            language: Language for generated questions (e.g., "English", "Spanish", "French")
            question_type_weights: Dictionary of question type weights for distribution control.
                Keys: 'directly_answerable', 'reasoning_required', 'unanswerable',
                      'partially_answerable'
                Values: Non-negative numbers (will be auto-normalized). Set to 0 to disable.
                Default: Equal weights for all types (25 each)

        Raises:
            ValueError: If parameters are invalid
        """
        if not model:
            raise ValueError("Model is required")
        if questions_per_doc < 1:
            raise ValueError("questions_per_doc must be at least 1")
        if not language:
            raise ValueError("Language cannot be empty")

        # Set default weights if not provided
        if question_type_weights is None:
            question_type_weights = {
                'directly_answerable': 25,
                'reasoning_required': 25,
                'unanswerable': 25,
                'partially_answerable': 25
            }

        # Validate weights
        self._validate_weights(question_type_weights)

        # Normalize weights to percentages
        self.question_type_percentages = self._normalize_weights(question_type_weights)

        self.model = model
        self.questions_per_doc = questions_per_doc
        self.language = language

    def _validate_weights(self, weights: Dict[str, float]) -> None:
        """
        Validate question type weights.

        Args:
            weights: Dictionary of question type weights

        Raises:
            ValueError: If weights are invalid
        """
        valid_keys = {'directly_answerable', 'reasoning_required',
                      'unanswerable', 'partially_answerable'}

        # Check for invalid keys
        invalid_keys = set(weights.keys()) - valid_keys
        if invalid_keys:
            raise ValueError(
                f"Invalid question type keys: {invalid_keys}. "
                f"Valid keys are: {valid_keys}"
            )

        # Check that all weights are non-negative
        for key, value in weights.items():
            if value < 0:
                raise ValueError(
                    f"Question type weight '{key}' must be non-negative, got {value}"
                )

        # Check that at least one weight is positive
        if all(v == 0 for v in weights.values()):
            raise ValueError(
                "At least one question type weight must be greater than 0"
            )

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to percentages that sum to 100.

        Args:
            weights: Dictionary of question type weights

        Returns:
            Dictionary of normalized percentages
        """
        valid_keys = ['directly_answerable', 'reasoning_required', 'unanswerable', 'partially_answerable']
        # Fill in missing keys with 0
        full_weights = {key: weights.get(key, 0) for key in valid_keys}
        total = sum(full_weights.values())
        if total == 0:
            raise ValueError("Cannot normalize weights: all weights are zero")
        return {key: (value / total) * 100 for key, value in full_weights.items()}

    def _build_question_type_instructions(self) -> str:
        """
        Build question type instructions based on configured percentages.

        Returns:
            Formatted string with question type instructions
        """
        instructions = []

        percentages = self.question_type_percentages

        if percentages.get('directly_answerable', 0) > 0:
            pct = percentages['directly_answerable']
            instructions.append(
                f"- Approximately {pct:.0f}% of questions should be factual "
                "questions with clear, direct answers."
            )

        if percentages.get('reasoning_required', 0) > 0:
            pct = percentages['reasoning_required']
            instructions.append(
                f"- Approximately {pct:.0f}% of questions should require "
                "reasoning, thinking or inference to answer."
            )

        if percentages.get('unanswerable', 0) > 0:
            pct = percentages['unanswerable']
            instructions.append(
                f"- Approximately {pct:.0f}% of questions should be related "
                "but not directly answerable with the given information."
            )

        if percentages.get('partially_answerable', 0) > 0:
            pct = percentages['partially_answerable']
            instructions.append(
                f"- Approximately {pct:.0f}% of questions should be "
                "only partially answerable with the given information."
            )

        if not instructions:
            # Fallback - should never happen due to validation
            instructions.append("- Generate diverse questions covering different aspects.")

        return "Vary the question types to include:\n" + "\n".join(instructions)

    def generate(
        self,
        documents: List[str],
        n_questions: int = 50,
        min_words: int = 5,
        max_words: int = 20,
        seed: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate queries from documents using an LLM.

        Args:
            documents: List of document texts
            n_questions: Total number of questions to generate
            min_words: Minimum number of words per question
            max_words: Maximum number of words per question
            seed: Random seed for reproducible sampling (None for random)
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
        # Calculate adaptive questions per doc with 1.5x buffer for deduplication/filtering
        questions_per_doc = math.ceil((n_questions / len(documents)) * 1.5)

        # Use tqdm for progress tracking
        for doc in tqdm(documents, desc="Generating queries from documents", unit="doc"):
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
                    "Failed to generate questions for document: %s",
                    str(e)
                )
                continue

        # Post-processing: deduplication and filtering
        all_questions = self._post_process_questions(
            all_questions,
            n_questions,
            min_words,
            max_words,
            seed
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
        # Build question type instructions based on configured percentages
        question_type_instructions = self._build_question_type_instructions()

        prompt = f"""Given the following document text, generate {num_questions} diverse questions.
Each question should have at least {min_words} words, and no more than {max_words} words.
Generate questions at varying lengths within this range (some shorter, some longer).
{question_type_instructions}
IMPORTANT: Questions must be standalone and self-contained. Do NOT use phrases like "mentioned in the document", "according to the text", "in the passage", "discussed above", or any similar references. Each question should read as if it were asked independently without any source material.
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

        result = self.model.call(prompt)
        response = result["response"]
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
        max_words: int,
        seed: Optional[int] = None
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
            if seed is not None:
                random.seed(seed)
            filtered_questions = random.sample(filtered_questions, n_questions)
            logger.info("Sampled %d questions", n_questions)

        return filtered_questions

    def generate_with_answers(
        self,
        documents: List[str],
        n_questions: int = 50,
        min_words: int = 5,
        max_words: int = 20,
        seed: Optional[int] = None,
    ) -> List[QueryWithAnswer]:
        """
        Generate queries with expected answers from documents using an LLM.

        This method generates question-answer pairs where the answer is derived
        from the same document used to generate the question. This is useful
        for creating golden answer evaluation datasets.

        Args:
            documents: List of document texts
            n_questions: Total number of question-answer pairs to generate
            min_words: Minimum number of words per question
            max_words: Maximum number of words per question
            seed: Random seed for reproducible sampling (None for random)
            **kwargs: Additional parameters (unused)

        Returns:
            List of QueryWithAnswer objects containing query and expected_answer

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
            "Generating %d question-answer pairs from %d documents",
            n_questions,
            len(documents)
        )

        all_qa_pairs = []
        # Calculate adaptive questions per doc with 1.5x buffer
        questions_per_doc = math.ceil((n_questions / len(documents)) * 1.5)

        for doc in tqdm(documents, desc="Generating QA pairs from documents", unit="doc"):
            try:
                qa_pairs = self._generate_qa_pairs_for_doc(
                    doc,
                    questions_per_doc,
                    min_words,
                    max_words
                )
                all_qa_pairs.extend(qa_pairs)
            except Exception as e:
                logger.warning(
                    "Failed to generate QA pairs for document: %s",
                    str(e)
                )
                continue

        # Post-processing
        all_qa_pairs = self._post_process_qa_pairs(
            all_qa_pairs,
            n_questions,
            min_words,
            max_words,
            seed
        )

        logger.info("Successfully generated %d QA pairs", len(all_qa_pairs))
        return all_qa_pairs

    def _parse_qa_xml(self, response: str) -> List[QueryWithAnswer]:
        """Parse QA pairs from XML-formatted response.

        Args:
            response: LLM response containing XML-formatted QA pairs

        Returns:
            List of QueryWithAnswer objects
        """
        qa_pairs = []

        # Find all <qa>...</qa> blocks with flexible whitespace handling
        qa_pattern = re.compile(
            r'<qa>\s*<question>(.*?)</question>\s*<answer>(.*?)</answer>\s*</qa>',
            re.DOTALL | re.IGNORECASE
        )

        for match in qa_pattern.finditer(response):
            question = match.group(1).strip()
            answer = match.group(2).strip()

            if question and answer:
                qa_pairs.append(QueryWithAnswer(
                    query=question,
                    expected_answer=answer
                ))

        return qa_pairs

    def _generate_qa_pairs_for_doc(
        self,
        doc: str,
        num_questions: int,
        min_words: int,
        max_words: int
    ) -> List[QueryWithAnswer]:
        """
        Generate question-answer pairs for a single document.

        Args:
            doc: Document text
            num_questions: Number of QA pairs to generate
            min_words: Minimum words per question
            max_words: Maximum words per question

        Returns:
            List of QueryWithAnswer objects

        Raises:
            Exception: If LLM call fails
        """
        question_type_instructions = self._build_question_type_instructions()

        prompt = f"""Given the following document text, generate {num_questions} question-answer pairs.

For each pair:
- The QUESTION should have at least {min_words} words and no more than {max_words} words
- The ANSWER should be a complete, accurate response based on all relevant content from the document.
- Generate questions at varying lengths within the word range

{question_type_instructions}

IMPORTANT:
- Questions must be standalone and self-contained
- Do NOT use phrases like "mentioned in the document", "according to the text", etc.
- Each question should end with a question mark
- Answers should be comprehensive but concise (1-3 sentences typically)
- Answers may span multiple lines if needed

Your response must use XML tags in this exact format:
<qa>
<question>[question text]</question>
<answer>[answer text]</answer>
</qa>

<qa>
<question>[question text]</question>
<answer>[answer text]</answer>
</qa>

Do not use bullets, numbers, code fences, or any text outside the XML tags.
Your response should always be in {self.language}.

The text is:
<document>
{doc}
</document>

Your response:
"""

        result = self.model.call(prompt)
        response = result["response"]

        # Parse XML-formatted QA pairs
        qa_pairs = self._parse_qa_xml(response)

        # Filter: questions must end with ?
        qa_pairs = [
            qa for qa in qa_pairs
            if qa.query and qa.query.endswith('?')
        ]

        return qa_pairs

    def _post_process_qa_pairs(
        self,
        qa_pairs: List[QueryWithAnswer],
        n_questions: int,
        min_words: int,
        max_words: int,
        seed: Optional[int] = None
    ) -> List[QueryWithAnswer]:
        """
        Post-process generated QA pairs: deduplicate, filter, and sample.

        Args:
            qa_pairs: List of raw generated QA pairs
            n_questions: Target number of QA pairs
            min_words: Minimum words per question
            max_words: Maximum words per question
            seed: Random seed for sampling

        Returns:
            Filtered and sampled list of QA pairs
        """
        # Deduplicate by query text
        seen_queries = set()
        unique_pairs = []
        for qa in qa_pairs:
            if qa.query not in seen_queries:
                seen_queries.add(qa.query)
                unique_pairs.append(qa)

        logger.info(
            "Deduplicated: %d -> %d QA pairs",
            len(qa_pairs),
            len(unique_pairs)
        )

        # Filter by word count
        filtered_pairs = [
            qa for qa in unique_pairs
            if min_words <= len(qa.query.split()) <= max_words
        ]
        logger.info(
            "Filtered by word count: %d -> %d QA pairs",
            len(unique_pairs),
            len(filtered_pairs)
        )

        # Sample if we have more than needed
        if len(filtered_pairs) > n_questions:
            if seed is not None:
                random.seed(seed)
            filtered_pairs = random.sample(filtered_pairs, n_questions)
            logger.info("Sampled %d QA pairs", n_questions)

        return filtered_pairs
