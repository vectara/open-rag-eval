from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from tqdm import tqdm

from open_rag_eval.data_classes.rag_results import RAGResult
from open_rag_eval.data_classes.eval_scores import ScoredRAGResult


class Evaluator(ABC):

    @classmethod
    @abstractmethod
    def plot_metrics(cls, csv_files, output_file="metrics_comparison.png"):
        """Plot metrics from multiple CSV files."""
        pass

    @abstractmethod
    def evaluate(self, rag_results: RAGResult) -> ScoredRAGResult:
        """Evaluate a single RAG result."""
        pass

    def evaluate_batch(
        self, rag_results: List[RAGResult], max_workers: Optional[int] = 5
    ) -> List[ScoredRAGResult]:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            eval_scores = list(
                tqdm(
                    executor.map(self.evaluate, rag_results),
                    total=len(rag_results),
                    desc="Evaluating using TRECRAG evaluator.",
                )
            )
        return eval_scores
