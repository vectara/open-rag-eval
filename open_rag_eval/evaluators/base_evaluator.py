from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from tqdm import tqdm

from open_rag_eval.data_classes.rag_results import MultiRAGResult
from open_rag_eval.data_classes.eval_scores import MultiScoredRAGResult


class Evaluator(ABC):

    @classmethod
    @abstractmethod
    def plot_metrics(cls, csv_files, output_file="metrics_comparison.png"):
        """Plot metrics from multiple CSV files."""
        pass

    @abstractmethod
    def evaluate(self, multi_rag_result: MultiRAGResult):
        """Evaluate results for a single query (which may include multiple runs)."""
        pass

    def evaluate_batch(
            self, multi_rag_results: List[MultiRAGResult], max_workers: Optional[int] = 5
    ) -> List:
        """Evaluate a batch of results.

        This method can be overridden by child classes to return different result types.
        The default implementation returns a list of MultiScoredRAGResult objects.
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            eval_scores = list(
                tqdm(
                    executor.map(self.evaluate, multi_rag_results),
                    total=len(multi_rag_results),
                    desc=f"Evaluating using {self.__class__.__name__} evaluator.",
                )
            )
        return eval_scores
