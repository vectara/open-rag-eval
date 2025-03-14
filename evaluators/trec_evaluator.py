from typing import List

import logging 

from data_classes.rag_results import RAGResult
from data_classes.eval_scores import AugmentedGenerationScores, RetrievalScores, RAGScores, ScoredRAGResult
from evaluators.base_evaluator import Evaluator
from models.llm_judges import LLMJudgeModel
from metrics import AutoNuggetMetric, CitationMetric, HallucinationMetric, UMBRELAMetric

from tqdm import tqdm


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

            rag_scores= RAGScores(
                RetrievalScores(scores={"umbrela_scores": umbrela_scores,
                                        "mean_umbrela_score": mean_umbrela_score}),
                AugmentedGenerationScores(scores={"autonugget_scores": autonugget_scores,
                                                  "mean_nugget_assignment_score": mean_assignment_score,
                                                  "vital_nuggetizer_score": autonugget_scores['nuggetizer_scores']['Vital'],
                                                  "hallucination_scores": hallucination_scores,
                                                  "citation_scores": citation_scores,
                                                  "citation_f1_score": citation_scores["f1"],
                                                  }))
            
            return ScoredRAGResult(rag_result=rag_results, scores=rag_scores)

        except Exception as e:
            logging.exception(f"Error in TRECEvaluator.evaluate: {str(e)}")
            rag_scores = RAGScores(RetrievalScores(scores={}), AugmentedGenerationScores(scores={}))
            return ScoredRAGResult(rag_result=rag_results, scores=rag_scores)
            
        
    def evaluate_batch(self, rag_results: List[RAGResult]) -> List[ScoredRAGResult]:
        eval_scores = []
        for result in tqdm(rag_results):
            eval_scores.append(self.evaluate(result))
            
        return eval_scores