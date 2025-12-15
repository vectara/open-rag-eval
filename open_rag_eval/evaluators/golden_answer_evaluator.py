"""Golden Answer Evaluator - Evaluates RAG answers against reference/golden answers.

Implements evaluation metrics when we have golden answers:
- Semantic Similarity
- Factual Correctness (Precision/Recall/F1)
"""

import json
import logging
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from open_rag_eval.data_classes.eval_scores import (
    AugmentedGenerationScores,
    MultiScoredRAGResult,
    RAGScores,
    RetrievalScores,
    ScoredRAGResult,
)
from open_rag_eval.data_classes.rag_results import MultiRAGResult
from open_rag_eval.metrics.golden_answer_metrics import (
    FactualCorrectnessMetric,
    SemanticSimilarityMetric,
)
from open_rag_eval.models.embedding_models import EmbeddingModel
from open_rag_eval.models.llm_judges import LLMJudgeModel

from .base_evaluator import Evaluator

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


class GoldenAnswerEvaluator(Evaluator):
    """Evaluator that compares RAG-generated answers against golden/reference answers.

    Requires MultiRAGResult objects to have the `expected_answer` field populated.
    """

    # Metrics that produce single float scores suitable for consistency
    CONSISTENCY_METRICS = [
        "semantic_similarity",
        "factual_correctness_f1"
    ]

    def __init__(
        self,
        llm_model: LLMJudgeModel,
        embedding_model: EmbeddingModel,
        options: Optional[dict] = None
    ):
        """Initialize Golden Answer Evaluator.

        Args:
            llm_model: LLM for NLI-based factual correctness
            embedding_model: Model for computing embeddings
            options: Optional configuration:
                - run_consistency: bool - Whether to support consistency evaluation
                - metrics_to_run_consistency: List[str] - Which metrics to include
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.options = options or {}

        # Initialize metrics
        self.semantic_similarity_metric = SemanticSimilarityMetric(embedding_model)
        self.factual_correctness_metric = FactualCorrectnessMetric(llm_model)

        # Consistency settings
        self.run_consistency = self.options.get("run_consistency", False)
        if self.run_consistency:
            self.metrics_to_run_consistency = self.options.get(
                "metrics_to_run_consistency", self.CONSISTENCY_METRICS
            )
        else:
            self.metrics_to_run_consistency = []

    def evaluate(self, multi_rag_result: MultiRAGResult) -> MultiScoredRAGResult:
        """Evaluate RAG results against the golden answer.

        Args:
            multi_rag_result: Container with RAG results and expected_answer

        Returns:
            MultiScoredRAGResult with golden answer comparison scores
        """
        all_scored_results = []
        expected_answer = multi_rag_result.expected_answer

        # Skip if no golden answer provided
        if not expected_answer:
            logger.warning(
                "No expected_answer for query %s. Skipping golden answer evaluation.",
                multi_rag_result.query_id
            )
            return MultiScoredRAGResult(
                query_id=multi_rag_result.query_id,
                query=multi_rag_result.query,
                scored_rag_results=[]
            )

        for i, rag_result in enumerate(multi_rag_result.rag_results):
            # Break after first if not running consistency
            if i > 0 and not self.run_consistency:
                break

            try:
                # Extract generated answer text
                generated_answer = " ".join([
                    part.text for part in rag_result.generation_result.generated_answer
                ])
                query = rag_result.generation_result.query

                # Compute all metrics
                similarity_scores = self.semantic_similarity_metric.compute(
                    query, generated_answer, expected_answer
                )
                correctness_scores = self.factual_correctness_metric.compute(
                    query, generated_answer, expected_answer
                )

                # Extract and aggregate token usage
                factual_tokens = correctness_scores.pop("token_usage", {})
                total_input_tokens = factual_tokens.get("input_tokens", 0)
                total_output_tokens = factual_tokens.get("output_tokens", 0)

                aggregated_token_usage = {
                    "factual_correctness_tokens": factual_tokens,
                    "total_input_tokens": total_input_tokens,
                    "total_output_tokens": total_output_tokens,
                    "total_tokens": total_input_tokens + total_output_tokens
                }

                # Combine into scores structure
                generation_scores = {
                    **similarity_scores,
                    **correctness_scores,
                    "expected_answer": expected_answer,
                    "token_usage": aggregated_token_usage
                }

                rag_scores = RAGScores(
                    RetrievalScores(scores={}),  # Not evaluating retrieval
                    AugmentedGenerationScores(scores=generation_scores)
                )

                scored_result = ScoredRAGResult(
                    rag_result=rag_result,
                    scores=rag_scores
                )
                all_scored_results.append(scored_result)

            except Exception as e:
                logger.exception("Error evaluating golden answer result: %s", e)
                rag_scores = RAGScores(
                    RetrievalScores(scores={}),
                    AugmentedGenerationScores(scores={})
                )
                scored_result = ScoredRAGResult(
                    rag_result=rag_result,
                    scores=rag_scores
                )
                all_scored_results.append(scored_result)

        return MultiScoredRAGResult(
            query_id=multi_rag_result.query_id,
            query=multi_rag_result.query,
            scored_rag_results=all_scored_results
        )

    def collect_scores_for_consistency(
        self,
        scored_results: List[MultiScoredRAGResult],
        scores_for_consistency: Dict[str, Dict[str, List[float]]],
        max_workers: Optional[int] = 5
    ) -> Dict[str, Dict[str, List[float]]]:
        """Gather scores for consistency evaluation.

        Args:
            scored_results: List of evaluated results
            scores_for_consistency: Dict to update with scores
            max_workers: Thread pool size

        Returns:
            Updated scores_for_consistency dict
        """
        lock = threading.Lock()

        def _extract_scores(score_dict, target: Dict[str, List[float]], metrics: List[str]):
            for metric in metrics:
                value = score_dict.get(metric)
                if isinstance(value, (int, float)):
                    target.setdefault(metric, []).append(float(value))

        def process_single_result(multi_scored_result: MultiScoredRAGResult):
            local_scores = {}
            query_id = multi_scored_result.query_id

            if not multi_scored_result.scored_rag_results:
                return

            for scored_result in multi_scored_result.scored_rag_results:
                if scored_result.scores and scored_result.scores.generation_score:
                    _extract_scores(
                        scored_result.scores.generation_score.scores,
                        local_scores,
                        self.metrics_to_run_consistency
                    )

            with lock:
                if query_id not in scores_for_consistency:
                    scores_for_consistency[query_id] = {}
                for metric, vals in local_scores.items():
                    scores_for_consistency[query_id].setdefault(metric, []).extend(vals)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_single_result, msr)
                for msr in scored_results
            ]
            for future in as_completed(futures):
                future.result()

        return scores_for_consistency

    @classmethod
    def plot_metrics(
        cls,
        csv_files: List[str],
        output_file: str = "golden_answer_metrics.png",
        metrics_to_plot: Optional[List[str]] = None
    ):
        """Plot golden answer evaluation metrics."""
        if metrics_to_plot is None:
            metrics_to_plot = [
                "generation_score_semantic_similarity",
                "generation_score_factual_correctness_precision",
                "generation_score_factual_correctness_recall",
                "generation_score_factual_correctness_f1"
            ]

        num_metrics = len(metrics_to_plot)
        ncols = min(3, num_metrics)
        nrows = math.ceil(num_metrics / ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
        axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

        display_names = {
            "generation_score_semantic_similarity": "Semantic Similarity",
            "generation_score_factual_correctness_precision": "Factual Precision",
            "generation_score_factual_correctness_recall": "Factual Recall",
            "generation_score_factual_correctness_f1": "Factual F1"
        }

        if len(csv_files) == 1:
            df = pd.read_csv(csv_files[0])
            for i, metric in enumerate(metrics_to_plot):
                ax = axs[i]
                col_name = metric if metric in df.columns else f"run_1_{metric}"

                if col_name in df.columns:
                    values = df[col_name].dropna().values
                    ax.boxplot(
                        [values],
                        patch_artist=True,
                        boxprops={"facecolor": "skyblue"}
                    )
                    mean_val = np.mean(values) if len(values) > 0 else 0
                    ax.axhline(
                        mean_val, color="red", linestyle="--",
                        label=f"Mean: {mean_val:.3f}"
                    )
                    ax.set_title(display_names.get(metric, metric))
                    ax.set_ylim(0, 1)
                    ax.legend()
                else:
                    ax.text(
                        0.5, 0.5, f"No data for {metric}",
                        ha="center", va="center", transform=ax.transAxes
                    )
        else:
            # Multiple CSV comparison
            for i, metric in enumerate(metrics_to_plot):
                ax = axs[i]
                data_list = []
                labels = []

                for csv_file in csv_files:
                    df = pd.read_csv(csv_file)
                    labels.append(os.path.basename(csv_file))
                    col_name = metric if metric in df.columns else f"run_1_{metric}"

                    if col_name in df.columns:
                        data_list.append(df[col_name].dropna().values)
                    else:
                        data_list.append(np.array([]))

                ax.boxplot(data_list, patch_artist=True, labels=labels)
                ax.set_title(display_names.get(metric, metric))
                ax.set_ylim(0, 1)
                ax.tick_params(axis='x', rotation=45)

        for ax in axs[num_metrics:]:
            ax.set_visible(False)

        fig.suptitle("Golden Answer Evaluation Metrics", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Golden answer metrics plot saved to {output_file}")

    def to_csv(
        self,
        scored_results: List[MultiScoredRAGResult],
        output_file: str
    ) -> None:
        """Save golden answer evaluation results to CSV."""
        results_dict = []

        # Calculate max runs across all results for column generation
        max_runs = 1
        for multi_scored_result in scored_results:
            if multi_scored_result.scored_rag_results:
                max_runs = max(max_runs, len(multi_scored_result.scored_rag_results))

        for multi_scored_result in scored_results:
            if not multi_scored_result.scored_rag_results:
                continue

            result_dict = {
                "query_id": multi_scored_result.query_id,
                "query": multi_scored_result.query
            }

            for i, result in enumerate(multi_scored_result.scored_rag_results):
                run_id = f"run_{i + 1}_"

                if result.rag_result and result.rag_result.generation_result:
                    generated_answer = [
                        {"text": part.text, "citations": part.citations}
                        for part in result.rag_result.generation_result.generated_answer
                    ]
                    result_dict[f"{run_id}generated_answer"] = json.dumps(generated_answer)

                if result.scores and result.scores.generation_score:
                    for key, value in result.scores.generation_score.scores.items():
                        if isinstance(value, (int, float, str)):
                            result_dict[f"{run_id}generation_score_{key}"] = value
                        else:
                            result_dict[f"{run_id}generation_score_{key}"] = json.dumps(value)

                    # Add flattened token usage columns for easier analysis
                    token_usage = result.scores.generation_score.scores.get(
                        "token_usage", {}
                    )
                    if token_usage:
                        result_dict[f"{run_id}total_input_tokens"] = token_usage.get(
                            "total_input_tokens", 0
                        )
                        result_dict[f"{run_id}total_output_tokens"] = token_usage.get(
                            "total_output_tokens", 0
                        )
                        result_dict[f"{run_id}total_tokens"] = token_usage.get(
                            "total_tokens", 0
                        )

            results_dict.append(result_dict)

        df = pd.DataFrame(results_dict)
        df.to_csv(output_file, index=False)
        if not df.empty:
            print(f"Golden answer scores saved to {output_file}")

    def get_consolidated_columns(self, num_runs: int = 1) -> List[str]:
        """Return columns for consolidated CSV.

        Args:
            num_runs: Number of runs to include columns for (default 1)

        Returns:
            List of column names for consolidated output
        """
        columns = ["query_id", "query"]
        metrics = [
            "semantic_similarity",
            "factual_correctness_precision",
            "factual_correctness_recall",
            "factual_correctness_f1",
            "expected_answer"
        ]

        for i in range(num_runs):
            run_id = f"run_{i + 1}_"
            columns.append(f"{run_id}generated_answer")
            for metric in metrics:
                columns.append(f"{run_id}generation_score_{metric}")

        return columns

    def get_metrics_to_plot(self) -> List[str]:
        """Return metrics to plot."""
        return [
            "generation_score_semantic_similarity",
            "generation_score_factual_correctness_f1"
        ]
