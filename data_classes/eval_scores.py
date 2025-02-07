from typing import Any
from dataclasses import dataclass

@dataclass
class RetrievalScore():
    """Holds the scores obtained from a retrieval evaluation metric."""
    # Maps the metric name to the actual scores defined by the metric.
    scores: dict[str, Any]

@dataclass
class AugmentedGenerationScore():
    """Holds the scores obtained from a generation evaluation metric."""
    # Maps the metric name to the actual scores defined by the metric.
    scores: dict[str, Any]


@dataclass
class RAGScore():
    """Holds the scores obtained from a RAG evaluation metric."""
    retrieval_score: RetrievalScore
    generation_score: AugmentedGenerationScore
