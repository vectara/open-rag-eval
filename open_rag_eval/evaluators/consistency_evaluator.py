import math
from typing import Optional, List
import logging
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from open_rag_eval.data_classes.rag_results import MultiRAGResult
from open_rag_eval.data_classes.eval_scores import (
    ConsistencyResult,)
from .base_evaluator import Evaluator
from open_rag_eval.metrics.bert_score_similarity_metric import BERTScoreSimilarityMetric
from open_rag_eval.metrics.rouge_score_similarity_metric import ROUGEScoreSimilarityMetric
from open_rag_eval.utils.constants import BERT_SCORE, ROUGE_SCORE


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

        # Default metric names and their default constructors
        default_metrics = {
            ROUGE_SCORE:
                lambda: ROUGEScoreSimilarityMetric(),
            BERT_SCORE:
                lambda: BERTScoreSimilarityMetric(model_type="xlm-roberta-large"
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
                        configured_metrics[metric_name] = True  # mark as configured

                        if metric_name == BERT_SCORE:
                            model_type = metric_params.get("model_type", "xlm-roberta-large")
                            lang = metric_params.get("lang", "en")
                            rescale_with_baseline = metric_params.get("rescale_with_baseline", True)
                            self.metric_calculators.append(
                                BERTScoreSimilarityMetric(
                                    model_type=model_type,
                                    lang=lang,
                                    rescale_with_baseline=rescale_with_baseline
                                )
                            )
                        elif metric_name == ROUGE_SCORE:
                            self.metric_calculators.append(ROUGEScoreSimilarityMetric())
                        else:
                            logging.warning(f"Unknown metric: {metric_name}. Skipping.")
                else:
                    logging.warning(f"Invalid metric config: {metric_config}. Skipping.")
        else:
            print("No metric config found for ConsistencyEvaluator—will apply defaults.")

        # add any missing default metrics
        for metric_name, constructor in default_metrics.items():
            if metric_name not in configured_metrics:
                print(f"Adding missing default metric: {metric_name}")
                self.metric_calculators.append(constructor())

    def evaluate(self, multi_rag_result: MultiRAGResult) -> ConsistencyResult:
        """
        Evaluate consistency across multiple RAG results for the same query.

        Args:
            multi_rag_result: Container with multiple RAG results for a single query

        Returns:
            ConsistencyResult with consistency metrics
        """
        # We need at least 3 results to measure consistency
        if len(multi_rag_result.rag_results) <= 2:
            logging.warning(
                f"Query {multi_rag_result.query_id} has fewer than 2 RAG results. "
                "Cannot compute consistency metrics.")
            return ConsistencyResult(query_id=multi_rag_result.query_id,
                                     query=multi_rag_result.query,
                                     multi_rag_result=multi_rag_result)

        # Calculate consistency results
        consistency_result = ConsistencyResult(
            query_id=multi_rag_result.query_id,
            query=multi_rag_result.query,
            multi_rag_result=multi_rag_result)

        try:
            # Apply each metric calculator
            for calculator in self.metric_calculators:
                scores = calculator.compute(multi_rag_result)
                if scores:
                    consistency_result.add_score_from_values(
                        calculator.name, scores)

        except Exception as e:
            logging.exception(f"Error computing consistency metrics: {str(e)}")
            return ConsistencyResult(query_id=multi_rag_result.query_id,
                                     query=multi_rag_result.query,
                                     multi_rag_result=multi_rag_result)

        # Return results with consistency scores
        return consistency_result

    @classmethod
    def plot_metrics(cls,
                     csv_files,
                     output_file="consistency_metrics.png",
                     metrics_to_plot=None):
        """
        Plot consistency metrics from CSV files as subplots in a single figure.

        - For a single CSV file, each subplot displays a boxplot for the metric,
          based on all pairwise similarity scores (across all queries).
          It overlays query-level mean ± IQR, mean ± STD, and median.
        - For multiple CSV files, grouped boxplots show distributions,
          and dots indicate per-query aggregated statistics.
        - A gray caption explains the plot content on each subplot.

        Args:
            csv_files (list): List of CSV file paths.
            output_file (str): Path to save the output plot.
            metrics_to_plot (list, optional): List of metric names to plot.
        """

        if metrics_to_plot is None:
            metrics_to_plot = [BERT_SCORE, ROUGE_SCORE]
        num_metrics = len(metrics_to_plot)
        ncols = min(3, num_metrics)
        nrows = math.ceil(num_metrics / ncols)

        fig, axs = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))

        # Ensure axs is always a flat list
        axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

        # Optional: hide any extra subplot if total metrics < grid slots
        for ax in axs[num_metrics:]:
            ax.set_visible(False)

        if len(csv_files) == 1:
            df = pd.read_csv(csv_files[0])

            for i, metric in enumerate(metrics_to_plot):
                if metric not in df.columns:
                    axs[i].text(0.5,
                                0.5,
                                f"No {metric.upper()} data available",
                                ha='center',
                                va='center',
                                fontsize=12)
                    continue

                all_values = []
                means, stds, iqrs, medians = [], [], [], []

                for json_str in df[metric].dropna():
                    try:
                        metric_data = json.loads(json_str)
                        values = metric_data["values"]
                        stats = metric_data["stats"]

                        all_values.extend(values)
                        means.append(stats["mean"])
                        stds.append(stats["std"])
                        iqrs.append(stats["iqr"])
                        medians.append(np.median(values))
                    except Exception as e:
                        logging.warning(f"Failed to parse {metric} entry: {e}")
                if not all_values:
                    axs[i].text(0.5, 0.5, f"No valid {metric.upper()} scores found",
                                ha='center', va='center', fontsize=12)
                    continue
                axs[i].boxplot(all_values,
                               patch_artist=True,
                               boxprops=dict(facecolor='skyblue'))

                mean_val = np.mean(all_values)
                median_val = np.median(all_values)
                axs[i].axhline(mean_val,
                               color='red',
                               linestyle='--',
                               label=f"Mean (all): {mean_val:.3f}")
                axs[i].axhline(median_val,
                               color='green',
                               linestyle='-',
                               label=f"Median (all): {median_val:.3f}")

                axs[i].errorbar([1.1], [np.mean(means)],
                                yerr=[[0], [np.mean(iqrs)]],
                                fmt='o',
                                color='purple',
                                label='Avg. IQR (25–75%)')

                axs[i].errorbar([1.2], [np.mean(means)],
                                yerr=[[0], [np.mean(stds)]],
                                fmt='o',
                                color='darkgreen',
                                label='Avg. STD (±1σ)')

                axs[i].scatter([1.3], [np.mean(medians)],
                               marker='x',
                               color='black',
                               s=70,
                               label='Avg. Median')

                axs[i].set_title(f"{metric.upper()} Consistency")
                axs[i].set_ylabel("Value")
                axs[i].set_ylim(0, 1)
                axs[i].legend()
                axs[i].grid(True, linestyle='--', alpha=0.6)

        else:
            for i, metric in enumerate(metrics_to_plot):
                data = []
                means, stds, iqrs, medians = [], [], [], []
                labels = []

                for csv_file in csv_files:
                    df = pd.read_csv(csv_file)
                    label = os.path.basename(csv_file)
                    labels.append(label)

                    all_vals = []
                    local_means, local_stds, local_iqrs, local_medians = [], [], [], []

                    if metric in df.columns:
                        for json_str in df[metric].dropna():
                            try:
                                metric_data = json.loads(json_str)
                                values = metric_data["values"]
                                stats = metric_data["stats"]

                                all_vals.extend(values)
                                local_means.append(stats["mean"])
                                local_stds.append(stats["std"])
                                local_iqrs.append(stats["iqr"])
                                local_medians.append(np.median(values))
                            except Exception as e:
                                logging.warning(
                                    f"Error parsing {metric} data in {csv_file}: {e}"
                                )
                    data.append(all_vals)
                    means.append(np.mean(local_means))
                    stds.append(np.mean(local_stds))
                    iqrs.append(np.mean(local_iqrs))
                    medians.append(np.mean(local_medians))

                positions = list(range(1, len(csv_files) + 1))

                # Mean and median across all pairwise values across all CSVs
                all_values_flat = [val for sublist in data for val in sublist]
                if not all_values_flat:
                    axs[i].text(0.5, 0.5, f"No valid {metric.upper()} scores found",
                                ha='center', va='center', fontsize=12)
                    continue
                axs[i].boxplot(data,
                               patch_artist=True,
                               boxprops=dict(facecolor='skyblue'),
                               positions=positions)
                axs[i].set_xticks(positions)
                axs[i].set_xticklabels(labels, rotation=45)
                axs[i].set_title(f"{metric.upper()}")
                axs[i].set_ylabel("Pairwise Similarity Score")
                axs[i].set_ylim(0, 1)
                axs[i].grid(True, linestyle='--', alpha=0.6)

                if all_values_flat:
                    mean_val = np.mean(all_values_flat)
                    median_val = np.median(all_values_flat)
                    axs[i].axhline(mean_val,
                                   color='red',
                                   linestyle='--',
                                   label=f"Mean (all): {mean_val:.3f}")
                    axs[i].axhline(median_val,
                                   color='green',
                                   linestyle='-',
                                   label=f"Median (all): {median_val:.3f}")

                axs[i].errorbar(positions,
                                means,
                                yerr=iqrs,
                                fmt='o',
                                color='purple',
                                label='Avg. IQR (25–75%)')

                axs[i].errorbar([p + 0.1 for p in positions],
                                means,
                                yerr=stds,
                                fmt='o',
                                color='darkgreen',
                                label='Avg. STD (±1σ)')

                axs[i].scatter([p + 0.2 for p in positions],
                               medians,
                               marker='x',
                               color='black',
                               s=70,
                               label='Avg. Median')

                axs[i].legend()

        # One shared caption at the bottom center
        fig.text(
            0.5,
            -0.03,
            "Box: all pairwise scores • Dots: per-query stats (mean, median, IQR, STD)",
            ha='center',
            fontsize=11,
            color='black',
            fontweight='semibold',
            bbox=dict(facecolor='white',
                      edgecolor='gray',
                      boxstyle='round,pad=0.3',
                      alpha=0.85))

        plt.tight_layout()
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Consistency metrics plot saved to {output_file}")

    def to_csv(self, consistency_results: List[ConsistencyResult],
               file_path: str) -> None:
        """
        Saves the consistency scores to a CSV file with one column per metric
        containing JSON-serialized values.

        Args:
            consistency_results: List of ConsistencyResult objects with consistency scores
            file_path: Path to save the CSV file
        """
        results_dict = []

        for consistency_result in consistency_results:
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
                row[metric_name] = json.dumps(metric_data)

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
        df.to_csv(file_path, index=False)
        print(f"Consistency scores saved to {file_path}")

    def get_consolidated_columns(self) -> List[str]:
        """
        Returns the list of columns that this evaluator will add to a consolidated CSV file.
        """
        columns = ["query_id", "query"]
        for metric in self.metric_calculators:
            columns.append(metric.name)
        return columns

    def get_metrics_to_plot(self):
        """
        Returns the list of metrics that this evaluator will plot.
        """
        return [metric.name for metric in self.metric_calculators]
