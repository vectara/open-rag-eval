import logging
import itertools
from typing import List

from rouge import Rouge

from open_rag_eval.data_classes.rag_results import MultiRAGResult
from open_rag_eval.metrics.base_metrics import PairwiseAnswerSimilarityMetric
from open_rag_eval.utils.constants import ROUGE_SCORE


class ROUGEScoreSimilarityMetric(PairwiseAnswerSimilarityMetric):
    """Computes pairwise lexical similarity between generated answers using ROUGE metrics.

    Higher scores indicate greater lexical similarity between answers,
    which can be interpreted as higher consistency in the RAG system output.
    ROUGE-L F1 score is used as the similarity metric to capture longest
    common subsequence similarity between answer pairs.
    """

    @property
    def name(self) -> str:
        return ROUGE_SCORE

    def compute(self, multi_rag_result: MultiRAGResult) -> List[float]:
        """Compute ROUGE similarities between all pairs of generated answers."""
        # Extract generated text from each RAG result
        answers = [
            " ".join([
                part.text
                for part in result.generation_result.generated_answer
            ])
            for result in multi_rag_result.rag_results
            if result.generation_result and
            result.generation_result.generated_answer
        ]
        answers = [a.strip() for a in answers if a.strip()]

        # Skip if we don't have enough answers
        if len(answers) <= 2:
            return []

        # Initialize ROUGE
        rouge = Rouge()

        # Calculate ROUGE scores for all unique pairs
        rouge_scores = []
        for i, j in itertools.combinations(range(len(answers)), 2):
            a, b = answers[i], answers[j]
            if not a or not b:
                continue

            try:
                scores_ab = rouge.get_scores(a, b)[0]
                scores_ba = rouge.get_scores(b, a)[0]
                rouge_l_f1 = (scores_ab["rouge-l"]["f"] +
                              scores_ba["rouge-l"]["f"]) / 2
            except Exception as e:
                logging.warning(
                    f"ROUGE score computation failed for pair ({i}, {j}): {e}")
                continue

            rouge_scores.append(float(rouge_l_f1))

        return rouge_scores
