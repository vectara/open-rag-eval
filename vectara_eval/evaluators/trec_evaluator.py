from typing import List
import logging
import numpy as np
from tqdm import tqdm

from vectara_eval.data_classes.rag_results import RAGResult
from vectara_eval.data_classes.eval_scores import AugmentedGenerationScores, RetrievalScores, RAGScores, ScoredRAGResult
from vectara_eval.models.llm_judges import LLMJudgeModel
from vectara_eval.metrics import AutoNuggetMetric, CitationMetric, HallucinationMetric, UMBRELAMetric
from .base_evaluator import Evaluator

class TRECEvaluator(Evaluator):
    def __init__(self, model: LLMJudgeModel):
        self.model = model
        self.retrieval_metric = UMBRELAMetric(model)
        self.generation_metric = AutoNuggetMetric(model)
        self.citation_metric = CitationMetric(model)
        self.hallucination_metric = HallucinationMetric()

    def evaluate(self, rag_results: RAGResult) -> ScoredRAGResult:
        try:
            umbrela_scores = self.retrieval_metric.compute(rag_results.retrieval_result)
            autonugget_scores = self.generation_metric.compute(rag_results, umbrela_scores)
            hallucination_scores = self.hallucination_metric.compute(rag_results)
            citation_scores = self.citation_metric.compute(rag_results)

            # Create aggregate example scores where needed from the finegrained scores.
            mean_umbrela_score = sum(umbrela_scores.values()) / len(umbrela_scores)

            assignment_scores = autonugget_scores['assignment_scores']
            mean_assignment_score = sum(assignment_scores) / len(assignment_scores)

            hallucination_scores = hallucination_scores['hhem_score']

            rag_scores = RAGScores(
                RetrievalScores(scores={"umbrela_scores": umbrela_scores,
                                        "mean_umbrela_score": mean_umbrela_score}),
                AugmentedGenerationScores(
                    scores={
                        "autonugget_scores": autonugget_scores,
                        "mean_nugget_assignment_score": mean_assignment_score,
                        "vital_nuggetizer_score": autonugget_scores['nuggetizer_scores']['Vital'],
                        "hallucination_scores": hallucination_scores,
                        "citation_scores": citation_scores,
                        "citation_f1_score": citation_scores["f1"],
                    }
                )
            )

            return ScoredRAGResult(rag_result=rag_results, scores=rag_scores)

        except Exception as e:
            logging.exception("Error in TRECEvaluator.evaluate: %s", str(e))
            rag_scores = RAGScores(RetrievalScores(scores={}), AugmentedGenerationScores(scores={}))
            return ScoredRAGResult(rag_result=rag_results, scores=rag_scores)

    def evaluate_batch(self, rag_results: List[RAGResult]) -> List[ScoredRAGResult]:
        eval_scores = []
        for result in tqdm(rag_results, desc="Evaluating using TRECRAG evaluator."):
            eval_scores.append(self.evaluate(result))

        return eval_scores

    @classmethod
    def plot_metrics(cls, csv_files: list, output_file: str = 'metrics_comparison.png'):
        """
        Plot metrics from CSV files as subplots in a single figure.

        - For a single CSV file, each subplot displays a boxplot for the metric,
        with a horizontal mean line spanning only the width of the box and
        a legend indicating both mean and median values.
        - For multiple CSV files, each subplot displays grouped boxplots (one per CSV)
        with horizontal mean lines drawn for each box (only the first box is labeled
        for clarity), mimicking the single CSV style.

        Parameters:
            csv_files (list): List of CSV file paths.
            output_file (str): File name to save the resulting figure.
        """

        # Helper function to draw a boxplot on the given axis.
        def plot_boxplot(ax, data_list, xtick_labels, metric_title, single=True):
            # Positions: single CSV gets position [1], multiple CSVs get positions 1...n.
            positions = [1] if single else list(range(1, len(data_list) + 1))
            bp = ax.boxplot(
                data_list,
                vert=True,
                patch_artist=True,
                positions=positions,
                boxprops=dict(facecolor='skyblue', color='black'),
                medianprops=dict(color='darkorange', linewidth=2)
            )
            ax.set_title(metric_title, fontsize=16 if single else 12)
            ax.set_ylabel('Value', fontsize=14 if single else 10)
            if not single:
                ax.set_xticks(positions)
                ax.set_xticklabels(xtick_labels, rotation=45, fontsize=10)
                ax.grid(axis='y', linestyle='--', alpha=0.7)
            else:
                ax.set_xticks([])

            # For each box, add a horizontal mean line spanning only the box width.
            # Use the box's path vertices to determine the box boundaries.
            for i, d in enumerate(data_list):
                if len(d) > 0:
                    mean_val = np.mean(d)
                    median_val = np.median(d)
                    # Retrieve the x-coordinates of the box using its path vertices.
                    box_path = bp['boxes'][i].get_path()
                    box_x_data = box_path.vertices[:, 0]
                    left, right = np.min(box_x_data), np.max(box_x_data)
                    if i == 0:
                        ax.hlines(
                            mean_val, left, right, color='blue', linestyle='--', linewidth=2,
                            label=f'Mean: {mean_val:.4f}'
                        )
                    else:
                        ax.hlines(mean_val, left, right, color='blue', linestyle='--', linewidth=2)
                    # Scatter the mean point at the box's center.
                    x_pos = positions[i] if not single else 1
                    ax.scatter(x_pos, mean_val, color='blue', s=50, zorder=5)
                    # Only label the first median for the legend.
                    if i == 0:
                        bp['medians'][i].set_label(f'Median: {median_val:.4f}')
            ax.legend(fontsize=12 if single else 9)
