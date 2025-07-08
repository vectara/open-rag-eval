from typing import Any, List, Dict
from dataclasses import dataclass, field

import numpy as np

from .rag_results import RAGResult, MultiRAGResult


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
class ConsistencyScore:
    """Holds statistical measures for consistency evaluation."""
    values: List[float]
    stats: Dict[str, float] = None

    def __post_init__(self):
        if not self.values:
            raise ValueError("values list cannot be empty.")

        min_val = float(np.min(self.values))
        max_val = float(np.max(self.values))
        q1 = float(np.percentile(self.values, 25))
        q3 = float(np.percentile(self.values, 75))
        mean_val = float(np.mean(self.values))
        std_val = float(np.std(self.values))

        self.stats = {
            "mean": mean_val,
            "median": float(np.median(self.values)),
            "std": std_val,
            "max_min": max_val - min_val,  # range
            "iqr": q3 - q1,
            "consistency_adjusted_index": mean_val / (1 + std_val)
        }


@dataclass
class ConsistencyResult:
    """Holds consistency scores and multi-RAG results across multiple runs."""
    query: str
    query_id: str
    multi_rag_result: "MultiRAGResult"
    consistency_scores: Dict[str,
                             "ConsistencyScore"] = field(default_factory=dict)

    def add_score_from_values(self, name: str, values: List[float]):
        """Compute and add a consistency score from raw values."""
        self.consistency_scores[name] = ConsistencyScore(values=values)


@dataclass
class MultiScoredRAGResult:
    """Container for multiple scored results"""
    query: str
    query_id: str
    scored_rag_results: List["ScoredRAGResult"] = field(default_factory=list)
