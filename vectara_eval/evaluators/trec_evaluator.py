from typing import List
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
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
    def plot_metrics(self, csv_files, output_file='metrics_comparison.png'):
        """Plot bar graphs for specified metrics across multiple CSV files and save to a file. """
        # Metrics to plot
        metrics = [
            'retrieval_score_mean_umbrela_score',
            'generation_score_vital_nuggetizer_score',
            'generation_score_hallucination_scores',
            'generation_score_citation_f1_score'
        ]

        # Set up the plot
        plt.figure(figsize=(16, 10))

        # Color palette for visual appeal
        colors = plt.get_cmap('Spectral')(np.linspace(0, 1, len(csv_files)))

        # Width of each bar
        bar_width = 0.8 / len(csv_files)

        # Iterate through metrics
        for metric_idx, metric in enumerate(metrics):
            # Subplot for each metric
            plt.subplot(2, 2, metric_idx + 1)

            # Collect values for this metric
            metric_values = []

            # Iterate through CSV files
            for file_idx, csv_file in enumerate(csv_files):
                # Read CSV
                df = pd.read_csv(csv_file)

                # Calculate mean of the metric
                mean_value = df[metric].mean()
                metric_values.append(mean_value)

                # Plot bar with label and color
                plt.bar(
                    file_idx,
                    mean_value,
                    width=bar_width,
                    color=colors[file_idx],
                    label=f'{os.path.basename(csv_file)} ({mean_value:.4f})'
                )

            # Customize subplot
            plt.title(f'Mean {metric.replace("_", " ").title()}', fontsize=12)
            plt.ylabel('Mean Value', fontsize=10)
            plt.xticks([])
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.grid(axis='y', which='minor', linestyle=':', alpha=0.4)

            # Add value labels on bars
            for i, v in enumerate(metric_values):
                plt.text(
                    i,
                    v,
                    f'{v:.4f}',
                    ha='center',
                    va='bottom',
                    fontweight='bold'
                )

            # Add legend
            plt.legend(title='CSV Files', loc='best', bbox_to_anchor=(1, 1), fontsize=8)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free up memory

        print(f"Graph saved to {output_file}")
