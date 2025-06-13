from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, TypeVar, Generic, Dict
from tqdm import tqdm

from open_rag_eval.data_classes.rag_results import MultiRAGResult

T = TypeVar("T")


class Evaluator(ABC, Generic[T]):

    @classmethod
    @abstractmethod
    def plot_metrics(cls,
                     csv_files,
                     output_file="metrics_comparison.png",
                     metrics_to_plot=None):
        """Plot metrics from multiple CSV files."""
        pass

    @abstractmethod
    def to_csv(self, scored_results, output_file: str):
        """Write evaluation results to a CSV file."""
        pass

    @abstractmethod
    def get_consolidated_columns(self):
        """Return a list of columns to add to the consolidated CSV file."""
        pass

    @abstractmethod
    def get_metrics_to_plot(self):
        """Return a list of metrics to plot."""
        pass

    @abstractmethod
    def evaluate(self, multi_rag_result: MultiRAGResult) -> T:
        """Evaluate results for a single query (which may include multiple runs)."""
        pass

    def evaluate_batch(
        self,
        multi_rag_results: List[MultiRAGResult],
        max_workers: Optional[int] = 5,
        precomputed_metric_scores_by_query: Optional[Dict[str, Dict[
            str, List[float]]]] = None
    ) -> List[T]:
        """Evaluate a batch of results.
        This method can be overridden by child classes to return different result types.
        """
        _ = precomputed_metric_scores_by_query
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            eval_scores = list(
                tqdm(
                    executor.map(self.evaluate, multi_rag_results),
                    total=len(multi_rag_results),
                    desc=
                    f"Evaluating using {self.__class__.__name__} evaluator.",
                ))
        return eval_scores

    def collect_scores_for_consistency(
            self,
            scored_results: T,
            scores_for_consistency: dict[str, dict[str, list[float]]],
            max_workers: Optional[int] = 5
    ) -> dict[str, dict[str, list[float]]]:
        raise NotImplementedError
