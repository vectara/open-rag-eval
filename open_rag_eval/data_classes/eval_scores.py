from typing import Any, List
from dataclasses import dataclass, field

from .rag_results import RAGResult

@dataclass
class RetrievalScores():
    """Holds the scores obtained from a retrieval evaluation metric."""
    # Maps the metric name to the actual scores defined by the metric.
    scores: dict[str, Any]

@dataclass
class AugmentedGenerationScores():
    """Holds the scores obtained from a generation evaluation metric."""
    # Maps the metric name to the actual scores defined by the metric.
    scores: dict[str, Any]


@dataclass
class RAGScores():
    """Holds the scores obtained from a RAG evaluation metric."""
    retrieval_score: RetrievalScores
    generation_score: AugmentedGenerationScores

@dataclass
class ScoredRAGResult():
    """Holds the RAG output and scores for it obtained from a RAG evaluation metric."""
    rag_result: RAGResult
    scores: RAGScores

@dataclass
class MultiScoredRAGResult:
    """Container class that holds multiple ScoredRAGResult objects for a single query."""
    query: str
    query_id: str
    scored_rag_results: List[ScoredRAGResult] = field(default_factory=list)

    def add_scored_result(self, scored_result: ScoredRAGResult):
        """Add a ScoredRAGResult to the list."""
        self.scored_rag_results.append(scored_result)