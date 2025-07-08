from typing import List
import logging
from itertools import combinations

from bert_score import score as bert_score
from open_rag_eval.data_classes.rag_results import MultiRAGResult
from open_rag_eval.metrics.base_metrics import PairwiseAnswerSimilarityMetric
from open_rag_eval.utils.constants import BERT_SCORE


class BERTScoreSimilarityMetric(PairwiseAnswerSimilarityMetric):
    """Compute BERTScore similarity between pairs of answers."""

    def __init__(self,
                 model_type: str = "xlm-roberta-large",
                 lang: str = "en",
                 rescale_with_baseline: bool = True):
        self.model_type = model_type
        self.lang = lang
        self.rescale_with_baseline = rescale_with_baseline

    @property
    def name(self) -> str:
        return BERT_SCORE

    def compute(self, multi_rag_result: MultiRAGResult) -> List[float]:
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
        if len(answers) <= 2:
            return []

        answer_pairs = list(combinations(answers, 2))
        f1_scores = []

        for a, b in answer_pairs:
            f1_ab = self._get_bert_score(
                a, b, use_baseline=self.rescale_with_baseline)
            f1_ba = self._get_bert_score(
                b, a, use_baseline=self.rescale_with_baseline)

            if f1_ab is None or f1_ba is None:
                # Try fallback without baseline
                f1_ab = self._get_bert_score(a, b, use_baseline=False)
                f1_ba = self._get_bert_score(b, a, use_baseline=False)

            if f1_ab is not None and f1_ba is not None:
                f1 = (f1_ab + f1_ba) / 2
            else:
                continue

            f1_scores.append(f1)

        return f1_scores

    def _get_bert_score(self, a: str, b: str, use_baseline: bool) -> float:
        try:
            _, _, f1 = bert_score([a], [b],
                                  model_type=self.model_type,
                                  lang=self.lang,
                                  rescale_with_baseline=use_baseline,
                                  verbose=False)
            return float(f1[0])
        except Exception as e:
            logging.warning(
                f"BERTScore {'with' if use_baseline else 'without'} baseline failed: {e}"
            )
            return None
