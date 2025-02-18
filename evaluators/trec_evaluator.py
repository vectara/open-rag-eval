from typing import List

import logging 

from data_classes.rag_results import RAGResult
from data_classes.eval_scores import RetrievalScores 
from evaluators.base_evaluator import Evaluator
from models.llm_judges import LLMJudgeModel
from metrics import UMBRELAMetric


class TRECEvaluator(Evaluator):
    def __init__(self, model: LLMJudgeModel):
        self.model = model
        self.retrieval_metric = UMBRELAMetric(model)
        # self.generation_metric = metrics.AutoNuggetMetric(model)

    def evaluate(self, rag_results: RAGResult) -> dict[str, int]:
        try:
            retrieval_scores = self.retrieval_metric.compute(rag_results.retrieval_result)
            # augmented_generation_scores = self.augmented_generation_metric.compute(rag_results)
            evaluation_scores = RetrievalScores(scores={"umbrela_scores": retrieval_scores})
            return evaluation_scores
        except Exception as e:
            # Log the error
            logging.error(f"Error in TRECEvaluator.evaluate: {str(e)}")
            # Return empty scores on error
            return RetrievalScores(scores={"umbrela_scores": {}})
        
    def evaluate_batch(self, rag_results: List[RAGResult]) -> dict[str, int]:
        eval_scores = []
        for result in rag_results:
            eval_scores.append(self.evaluate(result))

        return eval_scores        