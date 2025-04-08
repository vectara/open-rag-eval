from typing import Dict
import torch

from transformers import AutoModelForSequenceClassification
from open_rag_eval.metrics.base_metrics import AugmentedGenerationMetric
from open_rag_eval.data_classes.rag_results import RAGResult

# Set number of cores to 2 to avoid heavy CPU usage
torch.set_num_threads(2)

class HallucinationMetric(AugmentedGenerationMetric):
    """ This metric uses the Vectara Hallucination Evaluation Model to detect hallucinations in RAG output. """

    def __init__(self, model_name: str = 'vectara/hallucination_evaluation_model', detection_threshold: float = 0.5, max_chars: int = 8192):
        """Initialize the Hallucination metric.

        Args:
            model_name (str): The name of the model to use for hallucination detection.
            detection_threshold (float): The threshold fordetecting hallucinations.
            max_chars (int): The maximum number of characters to process. Inputs longer than this will be truncated.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
        self.detection_threshold = detection_threshold
        self.max_chars = max_chars

    def compute(self, rag_result: RAGResult) -> Dict[str, int]:
        # Create source and summary pair.
        passage_text_collection = []
        retrieval_results = rag_result.retrieval_result
        for _, passage in retrieval_results.retrieved_passages.items():
            passage_text_collection.append(passage)

        summary_text_collection = [generated_answer_part.text for generated_answer_part in rag_result.generation_result.generated_answer]

        sources = " ".join(passage_text_collection)
        summary = " ".join(summary_text_collection)

        if len(sources) > self.max_chars:
            sources = sources[:self.max_chars]

        # Call the hallucination detection model.
        score = self.model.predict([(sources, summary)]).item()

        return {"hhem_score": score}
