import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from open_rag_eval.data_classes.eval_scores import ConsistencyResult
from open_rag_eval.data_classes.rag_results import MultiRAGResult
from open_rag_eval.metrics import HallucinationMetric
from open_rag_eval.metrics.bert_score_similarity_metric import \
    BERTScoreSimilarityMetric
from open_rag_eval.metrics.rouge_score_similarity_metric import \
    ROUGEScoreSimilarityMetric
from open_rag_eval.utils.constants import (BERT_SCORE, CONSISTENCY,
                                           HALLUCINATION_SCORE, ROUGE_SCORE)

from .base_evaluator import Evaluator

matplotlib.use("Agg")  # Use a non-GUI backend

class ConsistencyEvaluator(Evaluator):
    """
    Evaluator that analyzes the consistency of RAG results across multiple runs.
    This evaluator expects MultiRAGResult objects that contain multiple runs of
    the same query to properly assess consistency.
    """

    def __init__(self, options: Optional[dict] = None):
        """
        Initialize the ConsistencyEvaluator with configured metrics.

        Args:
            options: List of metric configurations from YAML config
        """
        self.options = options or {}
        self.metric_calculators = []
        self.metric_names = {f"{CONSISTENCY}_{metric}" for metric in [BERT_SCORE, ROUGE_SCORE, HALLUCINATION_SCORE]}
        # Default metric names and their default constructors
        default_metrics = {
            ROUGE_SCORE: ROUGEScoreSimilarityMetric(),
            BERT_SCORE: BERTScoreSimilarityMetric(model_type="xlm-roberta-large"
                                                 ),
        }

        # Keep track of which metrics were already configured
        configured_metrics = {}

        # Parse metrics from config
        if self.options and "metrics" in self.options:
            metrics_list = self.options["metrics"]

            for metric_config in metrics_list:
                if isinstance(metric_config, dict):
                    for metric_name, metric_params in metric_config.items():
                        configured_metrics[
                            metric_name] = True  # mark as configured

                        if metric_name == BERT_SCORE:
                            model_type = metric_params.get(
                                "model_type", "xlm-roberta-large")
                            lang = metric_params.get("lang", "en")
                            rescale_with_baseline = metric_params.get(
                                "rescale_with_baseline", True)
                            self.metric_calculators.append(
                                BERTScoreSimilarityMetric(
                                    model_type=model_type,
                                    lang=lang,
                                    rescale_with_baseline=rescale_with_baseline)
                            )
                        elif metric_name == ROUGE_SCORE:
                            self.metric_calculators.append(
                                ROUGEScoreSimilarityMetric())
                        else:
                            logging.warning("Unknown metric: %s. Skipping.",
                                            metric_name)
                else:
                    logging.warning("Invalid metric config: %s. Skipping.",
                                    metric_config)
        else:
            print(
                "No metric config found for ConsistencyEvaluator—will apply defaults."
            )

        # add any missing default metrics
        for metric_name, constructor in default_metrics.items():
            if metric_name not in configured_metrics:
                print(f"Adding missing default metric: {metric_name}")
                self.metric_calculators.append(constructor)

        self.hallucination_metric = HallucinationMetric()

    def evaluate(
        self,
        multi_rag_result: MultiRAGResult,
        precomputed_metric_scores: Optional[Dict[str, List[float]]] = None
    ) -> ConsistencyResult:
        """
        Evaluate consistency across multiple RAG results for the same query.

        Args:
            multi_rag_result: Container with multiple RAG results for a single query
            precomputed_metric_scores: Optional precomputed scores for the query, if available

        Returns:
            ConsistencyResult with consistency metrics
        """
        # We need at least 3 results to measure consistency
        if len(multi_rag_result.rag_results) <= 2:
            logging.warning(
                "Query %s has fewer than 3 RAG results. "
                "Cannot compute consistency metrics.",
                multi_rag_result.query_id)
            return ConsistencyResult(query_id=multi_rag_result.query_id,
                                     query=multi_rag_result.query,
                                     multi_rag_result=multi_rag_result)

        # Calculate consistency results
        consistency_result = ConsistencyResult(
            query_id=multi_rag_result.query_id,
            query=multi_rag_result.query,
            multi_rag_result=multi_rag_result)

        try:
            # If precomputed scores are provided, use them
            if precomputed_metric_scores:
                for metric_name, scores in precomputed_metric_scores.items():
                    if len(scores) >= 3:
                        consistency_result.add_score_from_values(
                            metric_name, scores)

            if not precomputed_metric_scores or HALLUCINATION_SCORE not in precomputed_metric_scores:
                # Compute hallucination scores if not precomputed
                hallucination_scores = []
                for rag_result in multi_rag_result.rag_results:
                    hallucination_scores.append(
                        self.hallucination_metric.compute(rag_result)
                        ["hhem_score"])
                if len(hallucination_scores) >= 3:
                    consistency_result.add_score_from_values(
                        "hallucination_score", hallucination_scores)

            # Apply each metric calculator
            for calculator in self.metric_calculators:
                scores = calculator.compute(multi_rag_result)
                if scores:
                    consistency_result.add_score_from_values(
                        calculator.name, scores)

        except Exception as e:
            logging.exception("Error computing consistency metrics: %s", str(e))
            return ConsistencyResult(query_id=multi_rag_result.query_id,
                                     query=multi_rag_result.query,
                                     multi_rag_result=multi_rag_result)

        # Return results with consistency scores
        return consistency_result

    def evaluate_batch(
        self,
        multi_rag_results: List[MultiRAGResult],
        max_workers: Optional[int] = 5,
        precomputed_metric_scores_by_query: Optional[Dict[str, Dict[
            str, List[float]]]] = None
    ) -> List[ConsistencyResult]:
        if precomputed_metric_scores_by_query:
            for metric in precomputed_metric_scores_by_query[list(
                    precomputed_metric_scores_by_query.keys())[0]].keys():
                self.metric_names.add(f"{CONSISTENCY}_{metric}")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            eval_scores = list(
                tqdm(
                    executor.map(
                        lambda r: self.evaluate(
                            r,
                            precomputed_metric_scores=
                            precomputed_metric_scores_by_query.get(r.query_id)
                            if precomputed_metric_scores_by_query else None),
                        multi_rag_results),
                    total=len(multi_rag_results),
                    desc=f"Evaluating using {self.__class__.__name__}",
                ))
        return eval_scores

    @classmethod
    def plot_single_csv(cls, csv_files, metrics_to_plot, axs, max_possible_scores, score_name_to_display_name):
        """
        Plot Consistency-Adjusted Index (CAI) from a single CSV file using per-query statistics.
        Args:
            csv_files (list): List of CSV file paths (should contain only one file).
            metrics_to_plot (list): List of metric column names to include in the plot.
            axs (list): List of matplotlib Axes objects to plot on.
            max_possible_scores (dict): Dictionary mapping metric names to their maximum possible scores.
            score_name_to_display_name (dict): Dictionary mapping metric names to their display names.
        """
        df = pd.read_csv(csv_files[0])

        for i, metric in enumerate(metrics_to_plot):
            if metric not in df.columns:
                axs[i].text(0.5,
                            0.5,
                            f"No {metric.upper()} data available",
                            ha="center",
                            va="center",
                            fontsize=12)
                continue
            per_query_consistency_adjusted_indices = []

            for value in df[metric].dropna().values:
                try:
                    metric_data = json.loads(value)
                    consistency_adjusted_index = float(
                        metric_data["stats"]["consistency_adjusted_index"])
                    consistency_adjusted_index = consistency_adjusted_index / max_possible_scores[
                            metric] if metric in max_possible_scores else consistency_adjusted_index
                    per_query_consistency_adjusted_indices.append(consistency_adjusted_index)
                except (KeyError, ValueError, TypeError,
                        json.JSONDecodeError) as e:
                    logging.warning(
                        f"Skipping invalid value in {csv_files[0]} for metric {metric}: {e}"
                    )
                    continue
            if not per_query_consistency_adjusted_indices:
                axs[i].text(0.5,
                            0.5,
                            f"No valid {metric.upper()} scores found",
                            ha="center",
                            va="center",
                            fontsize=12)
                continue

            axs[i].boxplot(
                per_query_consistency_adjusted_indices,
                patch_artist=True,
                boxprops={"facecolor": "skyblue", "edgecolor": "black", "linewidth": 1.2},
                medianprops={"color": "orange", "linewidth": 2},
                whiskerprops={"linewidth": 1.2},
                capprops={"linewidth": 1.2},
                flierprops={"marker": "o", "markersize": 4, "alpha": 0.5}
            )

            mean_val = np.mean(per_query_consistency_adjusted_indices)
            median_val = np.median(per_query_consistency_adjusted_indices)
            axs[i].axhline(mean_val,
                           color="red",
                           linestyle="--",
                           label=f"Mean: {mean_val:.3f}")
            axs[i].axhline(median_val,
                           color="orange",
                           linestyle="-",
                           label=f"Median: {median_val:.3f}")

            if metric in score_name_to_display_name:
                display_name = score_name_to_display_name.get(metric)
            else:
                display_name = " ".join(
                    w.capitalize() for w in metric.split("_"))

            axs[i].set_title(display_name)

            axs[i].set_ylabel("Consistency-Adjusted Index")
            ymin = min(0, np.min(per_query_consistency_adjusted_indices)
                       )  # ensures y-axis starts at 0 or lower
            ymax = np.max(
                per_query_consistency_adjusted_indices) * 1.1  # adds 10% headroom
            axs[i].set_ylim(ymin, ymax)

            axs[i].legend()
            axs[i].grid(True, linestyle="--", alpha=0.6)

    @classmethod
    def plot_multiple_csv(cls, csv_files, metrics_to_plot, axs, max_possible_scores, score_name_to_display_name):
        """ Plot Consistency-Adjusted Index (CAI) from multiple CSV files using per-query statistics.
        Args:
            csv_files (list): List of CSV file paths (should contain only one file).
            metrics_to_plot (list): List of metric column names to include in the plot.
            axs (list): List of matplotlib Axes objects to plot on.
            max_possible_scores (dict): Dictionary mapping metric names to their maximum possible scores.
            score_name_to_display_name (dict): Dictionary mapping metric names to their display names.
        """
        for i, metric in enumerate(metrics_to_plot):
            consistency_adjusted_indices_across_csv = []
            labels = []

            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                label = os.path.basename(csv_file)
                labels.append(label)

                consistency_adjusted_indices = []

                if metric in df.columns:
                    for value in df[metric].dropna().values:
                        try:
                            consistency_adjusted_index = float(json.loads(value)["stats"]["consistency_adjusted_index"])
                            consistency_adjusted_index = consistency_adjusted_index / max_possible_scores[metric] \
                                if metric in max_possible_scores else consistency_adjusted_index
                            consistency_adjusted_indices.append(consistency_adjusted_index)
                        except (KeyError, ValueError, TypeError,
                                json.JSONDecodeError) as e:
                            logging.warning(
                                f"Skipping invalid value in {csv_file} for metric {metric}: {e}"
                            )
                            continue

                consistency_adjusted_indices_across_csv.append(consistency_adjusted_indices)

            positions = list(range(1, len(csv_files) + 1))

            consistency_adjusted_indices_across_csv_flat = [
                val for sublist in consistency_adjusted_indices_across_csv
                for val in sublist
            ]
            if not consistency_adjusted_indices_across_csv_flat:
                axs[i].text(0.5,
                            0.5,
                            f"No valid {metric.upper()} scores found",
                            ha="center",
                            va="center",
                            fontsize=12)
                continue

            # Determine display name
            if metric in score_name_to_display_name:
                display_name = score_name_to_display_name.get(metric)
            else:
                display_name = " ".join(
                    w.capitalize() for w in metric.split("_"))

            # Boxplot
            axs[i].boxplot(
                consistency_adjusted_indices_across_csv,
                patch_artist=True,
                boxprops={"facecolor": "skyblue", "edgecolor": "black", "linewidth": 1.2},
                medianprops={"color": "orange", "linewidth": 2},
                whiskerprops={"linewidth": 1.2},
                capprops={"linewidth": 1.2},
                flierprops={"marker": "o", "markersize": 4, "alpha": 0.5}
            )

            # Axis labels
            axs[i].set_xticks(positions)
            axs[i].set_xticklabels(labels, rotation=45)
            axs[i].set_title(display_name)
            axs[i].set_ylabel("Consistency-Adjusted Index")

            # Limits
            axs[i].set_ylim(
                min(0, np.min(consistency_adjusted_indices_across_csv_flat)),
                np.max(consistency_adjusted_indices_across_csv_flat) * 1.1)

            # Grid
            axs[i].grid(True, linestyle="--", alpha=0.6)

            box_width = 0.4  # width of each box on the x-axis
            for pos, (label, values_group) in enumerate(zip(
                    labels, consistency_adjusted_indices_across_csv),
                    start=1):
                group_mean = np.mean(values_group)
                group_median = np.median(values_group)

                axs[i].hlines(y=group_mean,
                              xmin=pos - box_width / 4,
                              xmax=pos + box_width / 4,
                              color="red",
                              linestyle="--",
                              linewidth=2,
                              label=f"{label} Mean: {group_mean:.3f}")

                axs[i].hlines(y=group_median,
                              xmin=pos - box_width / 4,
                              xmax=pos + box_width / 4,
                              color="orange",
                              linestyle="-",
                              linewidth=2,
                              label=f"{label} Median: {group_median:.3f}")

            axs[i].legend()

    @classmethod
    def plot_metrics(cls,
                     csv_files,
                     output_file="consistency_metrics.png",
                     metrics_to_plot=None):
        """
        Plot Consistency-Adjusted Index (CAI) from CSV files using per-query statistics.

        For each query, a CAI is computed as:
            mean / (1 + std)
        where mean and std are computed over the multiple generations for that query.

        - For a single CSV file, each subplot displays a boxplot of these per-query CAIs.
          This highlights how consistent the model is across different queries for a given metric.
          Horizontal lines indicate the overall mean (dashed red) and median (solid orange) of CAI.

        - For multiple CSV files, each subplot displays grouped boxplots (one per file),
          allowing comparison of per-query consistency distributions across systems.
          Mean and median lines are drawn per group to show central tendency.

        Args:
            csv_files (list): List of CSV file paths.
            output_file (str): Path to save the output plot.
            metrics_to_plot (list, optional): List of metric column names to include in the plot.
        """

        score_name_to_display_name = {
            "consistency_vital_nuggetizer_score":
                "Groundedness (CAI)",
            "consistency_citation_f1_score":
                "Citation (CAI)",
            "consistency_hallucination_score":
                "Factuality (CAI)",
            "consistency_bert_score":
                "BERT (CAI)",
            "consistency_rouge_score":
                "ROUGE-L (CAI)",
            "consistency_mean_umbrela_score":
                "Retrieval (CAI)",
        }
        max_possible_scores = {
            "consistency_mean_umbrela_score": 3.0,
        }

        if metrics_to_plot is None:
            metrics_to_plot = [
                f"{CONSISTENCY}_{metric}"
                for metric in [BERT_SCORE, ROUGE_SCORE, HALLUCINATION_SCORE]
            ]
        num_metrics = len(metrics_to_plot)
        ncols = min(2, num_metrics)
        nrows = math.ceil(num_metrics / ncols)

        plt.rcParams.update({
            "font.size": 12,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 10,
        })

        fig, axs = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
        fig.suptitle("Per-Metric Consistency-Adjusted Index (CAI) Distribution", fontsize=16)
        # Ensure axs is always a flat list
        axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

        # hide any extra subplot if total metrics < grid slots
        for ax in axs[num_metrics:]:
            ax.set_visible(False)

        if len(csv_files) == 1:
            ConsistencyEvaluator.plot_single_csv(csv_files, metrics_to_plot, axs, max_possible_scores, score_name_to_display_name)
        else:
            ConsistencyEvaluator.plot_multiple_csv(csv_files, metrics_to_plot, axs, max_possible_scores, score_name_to_display_name)

        # One shared caption at the bottom center
        fig.text(
            0.5,
            -0.07,
            "Boxplots show the distribution of per-query Consistency-Adjusted Index (CAI) for each metric.\n"
            "This index is computed as: mean / (1 + std), where mean and std are calculated across multiple generations of a query.\n"
            "Higher index indicate both high-quality and stable outputs. Red dashed lines show the mean; orange solid lines show the median.\n"
            "When comparing multiple systems, grouped boxplots illustrate the relative consistency across queries for each system.",
            ha="center",
            fontsize=9,
            bbox={
                "facecolor": "white",
                "edgecolor": "gray",
                "boxstyle": "round,pad=0.4",
                "alpha": 0.9
            })

        plt.tight_layout()
        fig.savefig(output_file, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"Consistency metrics plot saved to {output_file}")

    def to_csv(self, scored_results: List[ConsistencyResult],
               output_file: str) -> None:
        """
        Saves the consistency stats to a CSV file with one column per metric
        containing JSON-serialized values.

        Args:
            scored_results: List of ConsistencyResult objects with consistency stats
            output_file: Path to save the CSV file
        """
        results_dict = []

        for consistency_result in scored_results:
            # Skip if no consistency result
            if not consistency_result.consistency_scores:
                continue

            row = {
                "query_id": consistency_result.query_id,
                "query": consistency_result.query,
            }

            # Add each consistency metric as a JSON column
            for metric_name, metric_score in consistency_result.consistency_scores.items(
            ):
                # Create a JSON representation of the metric
                metric_data = {
                    "values": metric_score.values,
                    "stats": metric_score.stats
                }
                row[f"{CONSISTENCY}_{metric_name}"] = json.dumps(metric_data)

            # Add each individual RAG result (generated answers and retrieved passages)
            if consistency_result.multi_rag_result:
                # Access the rag_results directly from the original multi_rag_result
                for i, rag_result in enumerate(
                        consistency_result.multi_rag_result.rag_results):
                    # Add generated answer
                    if rag_result.generation_result and rag_result.generation_result.generated_answer:
                        generated_answer_dict = [
                            {
                                "text": part.text,
                                "citations": part.citations
                            } for part in
                            rag_result.generation_result.generated_answer
                        ]
                        row[f"run_{i + 1}_generated_answer"] = json.dumps(
                            generated_answer_dict)

                    # Add retrieved passages
                    if rag_result.retrieval_result and rag_result.retrieval_result.retrieved_passages:
                        row[f"run_{i + 1}_retrieved_passages"] = json.dumps(
                            rag_result.retrieval_result.retrieved_passages)

            results_dict.append(row)

        # Save to CSV
        df = pd.DataFrame(results_dict)
        df.to_csv(output_file, index=False)
        if not df.empty:
            print(f"Consistency stats saved to {output_file}")

    def get_consolidated_columns(self) -> List[str]:
        """
        Returns the list of columns that this evaluator will add to a consolidated CSV file.
        """
        columns = ["query_id", "query"]
        columns.extend(list(self.metric_names))
        return columns

    def get_metrics_to_plot(self):
        """
        Returns the list of metrics that this evaluator will plot.
        """
        return list(self.metric_names)
