import logging
import itertools
from typing import List

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from open_rag_eval.data_classes.rag_results import MultiRAGResult
from open_rag_eval.metrics.base_metrics import PairwiseAnswerSimilarityMetric


class BERTAnswerSimilarityMetric(PairwiseAnswerSimilarityMetric):
    """
    Computes pairwise semantic similarity between generated answers using BERT embeddings.
    Designed for multi-response consistency evaluation.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name)
        except ImportError:
            logging.warning("sentence-transformers not available.")
            self.model = None

    @property
    def name(self) -> str:
        return "bert"

    def compute(self, multi_rag_result: MultiRAGResult) -> List[float]:

        if self.model is None:
            return []

        answers = [
            " ".join([
                part.text
                for part in result.generation_result.generated_answer
            ])
            for result in multi_rag_result.rag_results
            if result.generation_result and result.generation_result.generated_answer
        ]
        answers = [a.strip() for a in answers if a.strip()]
        if len(answers) < 2:
            return []

        embeddings = self.model.encode(answers)
        return [
            float(cosine_similarity([embeddings[i]], [embeddings[j]])[0][0])
            for i, j in itertools.combinations(range(len(embeddings)), 2)
        ]


