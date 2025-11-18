from typing import List
import logging
from itertools import combinations
import torch
from torchmetrics.text.bert import BERTScore
from transformers import AutoTokenizer

from open_rag_eval.data_classes.rag_results import MultiRAGResult
from open_rag_eval.metrics.base_metrics import PairwiseAnswerSimilarityMetric
from open_rag_eval.utils.constants import BERT_SCORE


class BERTScoreSimilarityMetric(PairwiseAnswerSimilarityMetric):
    """Compute BERTScore similarity between pairs of answers using torchmetrics."""

    def __init__(self,
                 model_type: str = "xlm-roberta-large",
                 lang: str = "en",
                 rescale_with_baseline: bool = True,
                 device: str = None,
                 max_length: int = 512):
        self.model_type = model_type
        self.lang = lang
        self.rescale_with_baseline = rescale_with_baseline
        self.max_length = max_length
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        # Initialize BERTScore metric once during initialization
        self.bertscore_metric = BERTScore(
            model_name_or_path=self.model_type,
            device=self.device,
            rescale_with_baseline=self.rescale_with_baseline
        )
        # Initialize tokenizer for truncation
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)

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
            f1_ab = self._get_bert_score(a, b)
            f1_ba = self._get_bert_score(b, a)  # pylint: disable=arguments-out-of-order

            if f1_ab is not None and f1_ba is not None:
                f1 = (f1_ab + f1_ba) / 2
                f1_scores.append(f1)

        return f1_scores

    def _truncate_text(self, text: str) -> str:
        """
        Truncate text to max_length tokens using the model's tokenizer.

        Args:
            text: Input text to truncate

        Returns:
            Truncated text that fits within max_length tokens
        """
        tokens = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True
        )
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _get_bert_score(self, a: str, b: str) -> float:
        try:
            # Truncate inputs to avoid tensor size mismatches
            a_truncated = self._truncate_text(a)
            b_truncated = self._truncate_text(b)

            result = self.bertscore_metric([a_truncated], [b_truncated])
            f1_tensor = result['f1']
            # Handle both scalar and vector tensors
            if f1_tensor.dim() == 0:
                f1_score = f1_tensor.item()
            else:
                f1_score = f1_tensor[0].item()
            return float(f1_score)
        except Exception as e:
            logging.warning(
                f"BERTScore computation failed: {e}"
            )
            return None
