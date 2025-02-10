from typing import List

from data_classes.rag_results import RAGResult
from data_classes.eval_scores import RetrievalScore 
from evaluators.base_evaluator import Evaluator
import metrics
import models


class TRECEvaluator(Evaluator):
    def __init__(self, model: models.LLMJudgeModel):
        self.model = model
        self.retrieval_metric = metrics.UMBRELAMetric(model)
        # self.generation_metric = metrics.AutoNuggetMetric(model)

    def evaluate(self, rag_results: RAGResult) -> dict[str, int]:
        retrieval_scores = self.retrieval_metric.compute(rag_results.retrieval_result)
        # augmented_generation_scores = self.augmented_generation_metric.compute(rag_results)

        evaluation_scores = RetrievalScore()
        evaluation_scores.retrieval_score = {"umbrela_scores": retrieval_scores}

        return evaluation_scores
    
    def evaluate_batch(self, rag_results: List[RAGResult]) -> dict[str, int]:
        eval_scores = []
        for result in rag_results:
            eval_scores.append(self.evaluate(result))

        return eval_scores
            

