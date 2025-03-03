from typing import List, Dict

from transformers import AutoModelForSequenceClassification
from metrics.base_metrics import AugmentedGenerationMetric
from data_classes.rag_results import RAGResult

class HallucinationMetric(AugmentedGenerationMetric):
    """ This metric uses the Vectara Hallucination Evaluation Model to detect hallucinations in RAG output. """

    def __init__(self, model_name: str = 'vectara/hallucination_evaluation_model', detection_threshold: float = 0.5):
        """Initialize the Hallucination metric.
        
        Args:
            model_name (str): The name of the model to use for hallucination detection.
            detection_threshold (float): The threshold fordetecting hallucinations.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
        self.detection_threshold = detection_threshold

    def compute(self, rag_result: RAGResult) -> Dict[str, int]:
        # Create source and summary pair.
        passage_text_collection = []
        retrieval_results = rag_result.retrieval_result
        for id, passage in retrieval_results.retrieved_passages.items():
            passage_text_collection.append(passage)

        summary_text_collection = []
        for id, sentence in rag_result.generation_result.generated_answer.items():
            summary_text_collection.append(sentence)

        sources = " ".join(passage_text_collection)
        summary = " ".join(summary_text_collection)

        # Call the hallucination detection model.
        score = self.model.predict([(sources, summary)]).item()
                
        return {"hhem_score": score}
    
