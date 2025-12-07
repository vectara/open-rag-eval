"""Golden answer evaluation metrics for comparing generated answers against reference answers.

Implements Golden answer metrics:
- Answer Relevance: Question generation + semantic similarity
- Semantic Similarity: Direct embedding similarity
- Factual Correctness: Claim-based NLI with precision/recall/F1
"""

from typing import Dict, List
from enum import Enum
import logging

import numpy as np
from pydantic import BaseModel

from open_rag_eval.metrics.base_metrics import GoldenAnswerMetric
from open_rag_eval.models.embedding_models import EmbeddingModel
from open_rag_eval.models.llm_judges import LLMJudgeModel


logger = logging.getLogger(__name__)


# ============ Pydantic Models for Structured Output ============

class GeneratedQuestions(BaseModel):
    """Container for questions generated from an answer."""
    questions: List[str]


class Claims(BaseModel):
    """Container for claims extracted from text."""
    claims: List[str]


class NLIVerdict(str, Enum):
    """NLI verdict types."""
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


class ClaimVerdict(BaseModel):
    """NLI verdict for a single claim."""
    claim: str
    verdict: NLIVerdict


class ClaimVerdicts(BaseModel):
    """Container for NLI verdicts on multiple claims."""
    verdicts: List[ClaimVerdict]


# ============ Metric Implementations ============

class SemanticSimilarityMetric(GoldenAnswerMetric):
    """Direct semantic similarity between generated and golden answers.

    Uses embedding cosine similarity to measure how semantically similar
    the generated answer is to the expected golden answer.
    """

    def __init__(self, embedding_model: EmbeddingModel):
        """Initialize Semantic Similarity metric.

        Args:
            embedding_model: Model for computing embeddings
        """
        self.embedding_model = embedding_model

    @property
    def name(self) -> str:
        return "semantic_similarity"

    def compute(
        self,
        query: str,
        generated_answer: str,
        expected_answer: str
    ) -> Dict[str, float]:
        """Compute semantic similarity between generated and expected answers.

        Returns:
            Dict with 'semantic_similarity' score (0-1)
        """
        try:
            # Embed both answers
            embeddings = self.embedding_model.embed([generated_answer, expected_answer])
            gen_embedding = embeddings[0]
            exp_embedding = embeddings[1]

            # Compute cosine similarity
            similarity = self._cosine_similarity(gen_embedding, exp_embedding)

            return {"semantic_similarity": similarity}

        except Exception as e:
            logger.error("Failed to compute semantic similarity: %s", e)
            return {"semantic_similarity": 0.0}

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


class AnswerRelevanceMetric(GoldenAnswerMetric):
    """Answer Relevance metric.

    Generates questions from the answer and measures how similar those
    questions are to the original query using embedding cosine similarity.

    Higher scores indicate the answer is more relevant to the query.
    """

    _QUESTION_GENERATION_PROMPT = """Given the following answer to a question, generate {num_questions} questions that this answer could be responding to.
The generated questions should be diverse and cover different aspects of the answer.

Answer: {answer}

Generate exactly {num_questions} questions."""

    def __init__(
        self,
        llm_model: LLMJudgeModel,
        embedding_model: EmbeddingModel,
        num_questions: int = 3
    ):
        """Initialize Answer Relevance metric.

        Args:
            llm_model: LLM for question generation
            embedding_model: Model for computing embeddings
            num_questions: Number of questions to generate from answer
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.num_questions = num_questions
        self.model_kwargs = {"temperature": 0.7}

    @property
    def name(self) -> str:
        return "answer_relevance"

    def compute(
        self,
        query: str,
        generated_answer: str,
        expected_answer: str  # Not used for this metric
    ) -> Dict[str, float]:
        """Compute answer relevance score.

        Returns:
            Dict with 'answer_relevance' score (0-1) and additional info
        """
        try:
            # Generate questions from the answer
            questions = self._generate_questions(generated_answer)

            if not questions:
                return {"answer_relevance": 0.0}

            # Embed original query and generated questions
            query_embedding = self.embedding_model.embed_single(query)
            question_embeddings = self.embedding_model.embed(questions)

            # Compute cosine similarities
            similarities = []
            for q_emb in question_embeddings:
                sim = self._cosine_similarity(query_embedding, q_emb)
                similarities.append(sim)

            # Average similarity is the relevance score
            relevance_score = float(np.mean(similarities))

            return {
                "answer_relevance": relevance_score,
                "generated_questions": questions,
                "question_similarities": similarities
            }

        except Exception as e:
            logger.error("Failed to compute answer relevance: %s", e)
            return {"answer_relevance": 0.0}

    def _generate_questions(self, answer: str) -> List[str]:
        """Generate questions that the answer could be responding to."""
        prompt = self._QUESTION_GENERATION_PROMPT.format(
            answer=answer,
            num_questions=self.num_questions
        )

        result = self.llm_model.parse(
            prompt,
            response_format=GeneratedQuestions,
            model_kwargs=self.model_kwargs
        )

        return result["response"].questions

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


class FactualCorrectnessMetric(GoldenAnswerMetric):
    """Factual correctness via claim-based NLI.

    1. Decompose both answers into atomic claims
    2. Use NLI to check:
       - Precision: How many generated claims are entailed by golden answer
       - Recall: How many golden claims are entailed by generated answer
    3. Compute F1 score
    """

    _CLAIM_EXTRACTION_PROMPT = """Extract all atomic factual claims from the following text.
Each claim should be a single, verifiable statement.
Break down complex sentences into simple atomic claims.

Text: {text}

Extract all claims."""

    _NLI_PROMPT = """For each claim below, determine if it is ENTAILED by, CONTRADICTED by, or NEUTRAL with respect to the reference text.

Reference text: {reference}

Claims to verify:
{claims}

For each claim, provide a verdict: "entailment" (claim is supported), "contradiction" (claim is contradicted), or "neutral" (claim cannot be verified from reference)."""

    def __init__(self, llm_model: LLMJudgeModel):
        """Initialize Factual Correctness metric.

        Args:
            llm_model: LLM for claim extraction and NLI
        """
        self.llm_model = llm_model
        self.model_kwargs = {"temperature": 0.0}

    @property
    def name(self) -> str:
        return "factual_correctness"

    def compute(
        self,
        query: str,
        generated_answer: str,
        expected_answer: str
    ) -> Dict[str, float]:
        """Compute factual correctness metrics.

        Returns:
            Dict with precision, recall, f1 scores
        """
        try:
            # Extract claims from both answers
            generated_claims = self._extract_claims(generated_answer)
            expected_claims = self._extract_claims(expected_answer)

            if not generated_claims or not expected_claims:
                return {
                    "factual_correctness_precision": 0.0,
                    "factual_correctness_recall": 0.0,
                    "factual_correctness_f1": 0.0,
                    "generated_claims": generated_claims,
                    "expected_claims": expected_claims
                }

            # Precision: generated claims verified against expected answer
            precision_verdicts = self._verify_claims(generated_claims, expected_answer)
            precision = self._compute_entailment_ratio(precision_verdicts)

            # Recall: expected claims verified against generated answer
            recall_verdicts = self._verify_claims(expected_claims, generated_answer)
            recall = self._compute_entailment_ratio(recall_verdicts)

            # F1 score
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            return {
                "factual_correctness_precision": precision,
                "factual_correctness_recall": recall,
                "factual_correctness_f1": f1,
                "generated_claims": generated_claims,
                "expected_claims": expected_claims,
                "precision_verdicts": [v.model_dump() for v in precision_verdicts],
                "recall_verdicts": [v.model_dump() for v in recall_verdicts]
            }

        except Exception as e:
            logger.error("Failed to compute factual correctness: %s", e)
            return {
                "factual_correctness_precision": 0.0,
                "factual_correctness_recall": 0.0,
                "factual_correctness_f1": 0.0
            }

    def _extract_claims(self, text: str) -> List[str]:
        """Extract atomic claims from text."""
        prompt = self._CLAIM_EXTRACTION_PROMPT.format(text=text)

        result = self.llm_model.parse(
            prompt,
            response_format=Claims,
            model_kwargs=self.model_kwargs
        )

        return result["response"].claims

    def _verify_claims(self, claims: List[str], reference: str) -> List[ClaimVerdict]:
        """Verify claims against reference text using NLI."""
        if not claims:
            return []

        claims_formatted = "\n".join([f"{i+1}. {c}" for i, c in enumerate(claims)])
        prompt = self._NLI_PROMPT.format(
            reference=reference,
            claims=claims_formatted
        )

        result = self.llm_model.parse(
            prompt,
            response_format=ClaimVerdicts,
            model_kwargs=self.model_kwargs
        )

        return result["response"].verdicts

    @staticmethod
    def _compute_entailment_ratio(verdicts: List[ClaimVerdict]) -> float:
        """Compute ratio of entailed claims."""
        if not verdicts:
            return 0.0

        entailed = sum(1 for v in verdicts if v.verdict == NLIVerdict.ENTAILMENT)
        return entailed / len(verdicts)
