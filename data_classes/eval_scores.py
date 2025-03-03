from typing import Any, List, Union

import pandas as pd

from dataclasses import dataclass
from data_classes.rag_results import RetrievalResult, AugmentedGenerationResult, RAGResult

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

    
def to_csv(scored_results: List[ScoredRAGResult], file_path: str) -> None:
    """Saves the scored results to a CSV file."""
    results_dict = []
    for result in scored_results:        
        result_dict = {
            "query": result.rag_result.retrieval_result.query,
            "retrieved_passages": result.rag_result.retrieval_result.retrieved_passages,
            "generated_answer": result.rag_result.generation_result.generated_answer,
        }
        # Unpack retrieval scores
        for key, value in result.scores.retrieval_score.scores.items():
            result_dict[f"retrieval_score_{key}"] = value
        
        # Unpack generation scores
        for key, value in result.scores.generation_score.scores.items():
            result_dict[f"generation_score_{key}"] = value
            
        results_dict.append(result_dict)

    df = pd.DataFrame(results_dict)
    df.to_csv(file_path, index=False)

    