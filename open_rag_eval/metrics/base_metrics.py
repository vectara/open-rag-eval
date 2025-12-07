from abc import ABC, abstractmethod
from typing import Dict


class RetrievalMetric(ABC):
    """This class is the base class for all retrieval metrics."""
    pass


class AugmentedGenerationMetric(ABC):
    """This class is the base class for all augmented generation metrics."""
    pass


class PairwiseAnswerSimilarityMetric(ABC):
    """Base class for all pairwise answer similarity metrics."""
    pass


class GoldenAnswerMetric(ABC):
    """Base class for metrics that compare generated answers to golden/reference answers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the metric name."""
        pass

    @abstractmethod
    def compute(
        self,
        query: str,
        generated_answer: str,
        expected_answer: str
    ) -> Dict[str, float]:
        """Compute metric score comparing generated to expected answer.

        Args:
            query: The original query
            generated_answer: The RAG-generated answer
            expected_answer: The golden/reference answer

        Returns:
            Dictionary with metric scores
        """
        pass
