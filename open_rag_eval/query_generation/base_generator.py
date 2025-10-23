"""Abstract base class for query generators."""

from abc import ABC, abstractmethod
from typing import List


class QueryGenerator(ABC):
    """
    Abstract base class for query generators.

    All query generators must implement the generate method to produce
    queries from a list of documents.
    """

    @abstractmethod
    def generate(
        self,
        documents: List[str],
        n_questions: int = 50,
        min_words: int = 5,
        max_words: int = 20,
        **kwargs
    ) -> List[str]:
        """
        Generate queries from a list of documents.

        Args:
            documents: List of document texts to generate queries from
            n_questions: Total number of questions to generate
            min_words: Minimum number of words per question
            max_words: Maximum number of words per question
            **kwargs: Additional generator-specific parameters

        Returns:
            List of generated query strings

        Raises:
            ValueError: If parameters are invalid
        """
        pass
