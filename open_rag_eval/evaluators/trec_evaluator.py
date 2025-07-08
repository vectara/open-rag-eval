import json
import logging
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from open_rag_eval.data_classes.eval_scores import (AugmentedGenerationScores,
                                                    MultiScoredRAGResult,
                                                    RAGScores, RetrievalScores,
                                                    ScoredRAGResult)
from open_rag_eval.data_classes.rag_results import MultiRAGResult
from open_rag_eval.metrics import (AutoNuggetMetric, CitationMetric,
                                   HallucinationMetric, NoAnswerMetric,
                                   UMBRELAMetric)
from open_rag_eval.models.llm_judges import LLMJudgeModel

from .base_evaluator import Evaluator

matplotlib.use("Agg")

class TRECEvaluator(Evaluator):

    def __init__(self, model: LLMJudgeModel, options: Optional[dict] = None):
        self.model = model
        self.retrieval_metric = UMBRELAMetric(model)
        self.generation_metric = AutoNuggetMetric(model)
        self.citation_metric = CitationMetric(model)
        self.hallucination_metric = HallucinationMetric()
        self.no_answer_metric = NoAnswerMetric(model)

        if not options:
            self.k_values = [1, 3, 5]
            self.run_consistency = False
        else:
            self.k_values = options["k_values"]
            self.run_consistency = options.get("run_consistency", False)
            self.metrics_to_run_consistency = options.get(
                "metrics_to_run_consistency", [])
            # if there is at least one metric to run consistency, set run_consistency to True
            if self.metrics_to_run_consistency:
                self.run_consistency = True
        self.to_repeat = 1

    def evaluate(self,
                 multi_rag_result: MultiRAGResult) -> MultiScoredRAGResult:
        """ Evaluate the RAG results for a single query.
        This method processes each RAG result, computes various metrics,
        and returns a MultiScoredRAGResult containing all scored results.
        Args:
            multi_rag_result (MultiRAGResult): The RAG results for a single query.
        Returns:
            MultiScoredRAGResult: A container with scored RAG results for the query.
        """

        # Create a list to store all scored results for this query
        all_scored_results = []

        # Process each RAG result
        for i, rag_result in enumerate(multi_rag_result.rag_results):
            # Break after the first result if consistency is not needed
            if i > 0 and not self.run_consistency:
                break

            try:
                retrieval_scores = self.retrieval_metric.compute(
                    rag_result.retrieval_result, self.k_values)
                umbrela_scores = retrieval_scores["umbrela_scores"]
                autonugget_scores = self.generation_metric.compute(
                    rag_result, umbrela_scores)
                hallucination_scores = self.hallucination_metric.compute(
                    rag_result)
                citation_scores = self.citation_metric.compute(rag_result)
                no_answer_score = self.no_answer_metric.compute(
                    rag_result.generation_result)

                # Create aggregate scores
                mean_umbrela_score = sum(
                    umbrela_scores.values()) / len(umbrela_scores)
                assignment_scores = autonugget_scores["assignment_scores"]
                mean_assignment_score = sum(assignment_scores) / len(
                    assignment_scores)
                hallucination_score = hallucination_scores["hhem_score"]

                rag_scores = RAGScores(
                    RetrievalScores(
                        scores={
                            "umbrela_scores":
                                umbrela_scores,
                            "precision_metrics":
                                retrieval_scores["retrieval_scores"],
                            "mean_umbrela_score":
                                mean_umbrela_score,
                        }),
                    AugmentedGenerationScores(
                        scores={
                            "autonugget_scores":
                                autonugget_scores,
                            "mean_nugget_assignment_score":
                                mean_assignment_score,
                            "vital_nuggetizer_score":
                                autonugget_scores["nuggetizer_scores"]["Vital"],
                            "hallucination_score":
                                hallucination_score,
                            "citation_scores":
                                citation_scores,
                            "citation_f1_score":
                                citation_scores["f1"],
                            "no_answer_score":
                                no_answer_score,
                        }),
                )

                scored_result = ScoredRAGResult(rag_result=rag_result,
                                                scores=rag_scores)
                all_scored_results.append(scored_result)

            except Exception as e:
                logging.exception("Error evaluating result: %s", str(e))
                rag_scores = RAGScores(RetrievalScores(scores={}),
                                       AugmentedGenerationScores(scores={}))

                scored_result = ScoredRAGResult(rag_result=rag_result,
                                                scores=rag_scores)
                all_scored_results.append(scored_result)
        return MultiScoredRAGResult(query_id=multi_rag_result.query_id,
                                    query=multi_rag_result.query,
                                    scored_rag_results=all_scored_results)

    @classmethod
    def plot_metrics(cls,
                     csv_files: list,
                     output_file: str = "metrics_comparison.png",
                     metrics_to_plot=None):
        """
        Plot metrics from CSV files as subplots in a single figure.

        - For a single CSV file, each subplot displays a boxplot for the metric,
        with a horizontal mean line spanning only the width of the box and
        a legend indicating both mean and median values.
        - For multiple CSV files, each subplot displays grouped boxplots (one per CSV)
        with horizontal mean lines drawn for each box (only the first box is labeled
        for clarity), mimicking the single CSV style.
        - Additionally displays a bar graph showing the percentage of questions answered
        for each CSV file.

        Parameters:
            csv_files (list): List of CSV file paths.
            output_file (str): File name to save the resulting figure.
            metrics_to_plot (list, optional): List of metric column names to plot.
        """

        if metrics_to_plot is None:
            metrics_to_plot = [
                "retrieval_score_mean_umbrela_score",
                "generation_score_vital_nuggetizer_score",
                "generation_score_hallucination_score",
                "generation_score_citation_f1_score",
                "questions_answered",
            ]

        def get_answered_percentage(df):
            answered_count = 0
            total_valid_rows = 0
            for idx, row in df.iterrows():
                value = row.get(
                    "generation_score_no_answer_score", None) or row.get(
                        "run_1_generation_score_no_answer_score", None)
                if pd.isna(value):
                    continue
                try:
                    answer_data = json.loads(value)
                    total_valid_rows += 1
                    if answer_data.get("query_answered", "") == "yes":
                        answered_count += 1
                except Exception:
                    logging.error("Invalid JSON at row index %s", idx)
            return (answered_count /
                    total_valid_rows) * 100 if total_valid_rows > 0 else 0

        def plot_boxplot(ax,
                         data_list,
                         xtick_labels,
                         metric_title,
                         single=True):
            positions = [1] if single else list(range(1, len(data_list) + 1))
            bp = ax.boxplot(
                data_list,
                vert=True,
                patch_artist=True,
                positions=positions,
                boxprops={
                    "facecolor": "skyblue",
                    "color": "black"
                },
                medianprops={
                    "color": "darkorange",
                    "linewidth": 2
                },
            )
            ax.set_title(metric_title, fontsize=16 if single else 12)
            ax.set_ylabel("Value", fontsize=14 if single else 10)
            if not single:
                ax.set_xticks(positions)
                ax.set_xticklabels(xtick_labels, rotation=45, fontsize=10)
                ax.grid(axis="y", linestyle="--", alpha=0.7)
            else:
                ax.set_xticks([])

            for i, d in enumerate(data_list):
                if len(d) > 0:
                    mean_val = np.mean(d)
                    box_path = bp["boxes"][i].get_path()
                    box_x_data = box_path.vertices[:, 0]
                    left, right = np.min(box_x_data), np.max(box_x_data)
                    ax.hlines(mean_val,
                              left,
                              right,
                              color="blue",
                              linestyle="--",
                              linewidth=2,
                              label="Mean" if i == 0 else None)
                    x_pos = positions[i] if not single else 1
                    ax.scatter(x_pos, mean_val, color="blue", s=50, zorder=5)
                    if i == 0:
                        bp["medians"][i].set_label("Median")
            ax.legend(fontsize=12 if single else 9)

        num_metrics = len(metrics_to_plot)
        ncols = min(3, num_metrics)
        nrows = math.ceil(num_metrics / ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
        axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

        if len(csv_files) == 1:
            df = pd.read_csv(csv_files[0])
            for i, metric in enumerate(metrics_to_plot):
                ax = axs[i]
                if metric == "questions_answered":
                    percentage = get_answered_percentage(df)
                    ax.bar([os.path.basename(csv_files[0])], [percentage],
                           color="skyblue")
                    ax.set_title("Questions Answered", fontsize=16)
                    ax.set_ylabel("Percentage (%)", fontsize=14)
                    ax.set_ylim(0, 100)
                    ax.grid(axis="y", linestyle="--", alpha=0.7)
                    ax.text(0,
                            percentage + 1,
                            f"{percentage:.1f}%",
                            ha="center",
                            fontsize=12)
                    continue

                if metric in df.columns or f"run_1_{metric}" in df.columns:
                    values = df[metric].dropna(
                    ).values if metric in df.columns else df[
                        f"run_1_{metric}"].dropna().values
                    plot_boxplot(ax, [values], [os.path.basename(csv_files[0])],
                                 metric.replace("_", " ").title(),
                                 single=True)
                    ax.set_ylim(
                        0, 3 if metric == "retrieval_score_mean_umbrela_score"
                        else 1)
                else:
                    ax.text(0.5,
                            0.5,
                            f"No data for {metric}",
                            transform=ax.transAxes,
                            ha="center",
                            va="center",
                            fontsize=10)

        else:
            for i, metric in enumerate(metrics_to_plot):
                ax = axs[i]
                data_list = []
                labels = []
                if metric == "questions_answered":
                    percentages = []
                    for csv_file in csv_files:
                        df = pd.read_csv(csv_file)
                        percentages.append(get_answered_percentage(df))
                        labels.append(os.path.basename(csv_file))
                    ax.bar(range(len(percentages)),
                           percentages,
                           color="skyblue")
                    ax.set_title("Questions Answered", fontsize=12)
                    ax.set_ylabel("Percentage (%)", fontsize=10)
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels, rotation=45, fontsize=10)
                    ax.set_ylim(0, 100)
                    ax.grid(axis="y", linestyle="--", alpha=0.7)
                    for j, v in enumerate(percentages):
                        ax.text(j, v + 1, f"{v:.1f}%", ha="center", fontsize=10)
                    continue

                for csv_file in csv_files:
                    df = pd.read_csv(csv_file)
                    labels.append(os.path.basename(csv_file))
                    if metric in df.columns or f"run_1_{metric}" in df.columns:
                        data_list.append(df[metric].dropna().values if metric in
                                         df.columns else df[f"run_1_{metric}"].
                                         dropna().values)
                    else:
                        data_list.append(np.array([]))

                plot_boxplot(ax,
                             data_list,
                             labels,
                             metric.replace("_", " ").title(),
                             single=False)
                ax.set_ylim(
                    0,
                    3 if metric == "retrieval_score_mean_umbrela_score" else 1)

        for ax in axs[num_metrics:]:
            ax.set_visible(False)

        fig.suptitle("Open RAG Eval Metrics", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"TREC metrics plot saved to {output_file}")

    def to_csv(self, scored_results: List[MultiScoredRAGResult],
               output_file: str) -> None:
        """
        Saves the scored results to a CSV file.

        Note: For simplicity, only the first scored RAG result from each query is saved
        to the CSV file, even when multiple runs were evaluated. The complete results
        for all runs are preserved in the MultiScoredRAGResult objects and can be used
        by the ConsistencyEvaluator (CE) for further analysis.
        """
        results_dict = []
        for multi_scored_result in scored_results:
            # Skip empty results
            if not multi_scored_result.scored_rag_results:
                continue

            self.to_repeat = max(self.to_repeat,
                                 len(multi_scored_result.scored_rag_results))
            result_dict = {"query_id": multi_scored_result.query_id, "query": multi_scored_result.query}
            for i, result in enumerate(multi_scored_result.scored_rag_results):
                run_id = f"run_{i + 1}_"
                # Get fields if they exist
                if result.rag_result and result.rag_result.retrieval_result:
                    result_dict[f"{run_id}retrieved_passages"] = json.dumps(
                        result.rag_result.retrieval_result.retrieved_passages)

                if result.rag_result and result.rag_result.generation_result:
                    generated_answer_dict = [
                        {
                            "text": part.text,
                            "citations": part.citations
                        } for part in
                        result.rag_result.generation_result.generated_answer
                    ]
                    result_dict[f"{run_id}generated_answer"] = json.dumps(
                        generated_answer_dict)

                # Add scores if they exist
                if result.scores and result.scores.retrieval_score:
                    for key, value in result.scores.retrieval_score.scores.items(
                    ):
                        result_dict[
                            f"{run_id}retrieval_score_{key}"] = json.dumps(
                                value)

                if result.scores and result.scores.generation_score:
                    for key, value in result.scores.generation_score.scores.items(
                    ):
                        result_dict[
                            f"{run_id}generation_score_{key}"] = json.dumps(
                                value)

            results_dict.append(result_dict)

        df = pd.DataFrame(results_dict)
        df.to_csv(output_file, index=False)
        if not df.empty:
            print(f"TREC scores saved to {output_file}")

    def get_consolidated_columns(self) -> List[str]:
        """
        Returns a list of columns to add to the consolidated CSV file.
        This includes all retrieval and generation score columns.
        """
        columns = ["query_id", "query"]
        for i in range(self.to_repeat):

            run_id = f"run_{i + 1}_"
            columns.extend([
                f"{run_id}retrieved_passages", f"{run_id}generated_answer",
                f"{run_id}retrieval_score_umbrela_scores",
                f"{run_id}retrieval_score_precision_metrics",
                f"{run_id}retrieval_score_mean_umbrela_score",
                f"{run_id}generation_score_autonugget_scores",
                f"{run_id}generation_score_mean_nugget_assignment_score",
                f"{run_id}generation_score_vital_nuggetizer_score",
                f"{run_id}generation_score_hallucination_score",
                f"{run_id}generation_score_citation_scores",
                f"{run_id}generation_score_citation_f1_score",
                f"{run_id}generation_score_no_answer_score"
            ])
        return columns

    def get_metrics_to_plot(self) -> List[str]:
        """ Returns a list of metrics to plot."""
        return [
            "retrieval_score_mean_umbrela_score",
            "generation_score_vital_nuggetizer_score",
            "generation_score_hallucination_score",
            "generation_score_citation_f1_score", "questions_answered"
        ]

    def collect_scores_for_consistency(
            self,
            scored_results: List[MultiScoredRAGResult],
            scores_for_consistency: dict[str, dict[str, list[float]]],
            max_workers: Optional[int] = 5
    ) -> dict[str, dict[str, list[float]]]:
        """ Gather scores for consistency evaluation from multiple scored results.
        This method processes the scored results and extracts scores for specified metrics
        to be used in consistency evaluation.
        Args:
            scored_results (List[MultiScoredRAGResult]): List of MultiScoredRAGResult objects.
            scores_for_consistency (dict[str, dict[str, list[float]]]): Dictionary to store scores for consistency.
            Maps query IDs to a dictionary of metric names and their corresponding lists of scores.
            max_workers (Optional[int]): Number of threads to use for processing. Default is 5.
        Returns:
            dict[str, dict[str, list[float]]]: Updated dictionary with scores for consistency.
        """
        lock = threading.Lock()  # to safely update shared dict

        def _extract_scores(score_dict, target: dict[str, list[float]],
                            metrics: list[str]):
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
                if scored_result.scores:
                    if scored_result.scores.retrieval_score:
                        _extract_scores(
                            scored_result.scores.retrieval_score.scores,
                            local_scores, self.metrics_to_run_consistency)

                    if scored_result.scores.generation_score:
                        _extract_scores(
                            scored_result.scores.generation_score.scores,
                            local_scores, self.metrics_to_run_consistency)

            with lock:
                if query_id not in scores_for_consistency:
                    scores_for_consistency[query_id] = {}
                for metric, vals in local_scores.items():
                    scores_for_consistency[query_id].setdefault(metric,
                                                                []).extend(vals)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_single_result, msr)
                for msr in scored_results
            ]
            for future in as_completed(futures):
                future.result()  # Wait for all futures to complete

        return scores_for_consistency
