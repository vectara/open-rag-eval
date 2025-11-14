from typing import Dict

from open_rag_eval.metrics.base_metrics import AugmentedGenerationMetric
from open_rag_eval.data_classes.rag_results import RAGResult
from open_rag_eval.metrics.factuality_backends import (
    FactualityBackend,
    HHEMBackend,
    VectaraAPIBackend
)


class HallucinationMetric(AugmentedGenerationMetric):
    """
    Metric for detecting hallucinations in RAG output using configurable backends.

    Supports multiple backends:
    - HHEM (default): Open-source Vectara Hallucination Evaluation Model
    - Vectara API: Commercial Vectara Factual Consistency API
    """

    def __init__(self, backend_type: str = "hhem", **backend_config):
        """
        Initialize the Hallucination metric with a specific backend.

        Args:
            backend_type: Type of backend to use ("hhem" or "vectara_api")
            **backend_config: Configuration passed to the backend
                For HHEM backend:
                    - model_name (str): Hugging Face model name (default: 'vectara/hallucination_evaluation_model')
                    - detection_threshold (float): Threshold for hallucination detection (default: 0.5)
                    - max_chars (int): Max characters to process (default: 8192)
                For Vectara API backend:
                    - api_key (str): Vectara API key (required)
                    - base_url (str): API base URL (default: 'https://api.vectara.io')

        Raises:
            ValueError: If backend_type is not supported or required config is missing
        """
        self.backend = self._create_backend(backend_type, backend_config)

    def _create_backend(self, backend_type: str, config: Dict) -> FactualityBackend:
        """
        Factory method to create the appropriate backend.

        Args:
            backend_type: Type of backend ("hhem" or "vectara_api")
            config: Configuration dictionary for the backend

        Returns:
            Initialized FactualityBackend instance

        Raises:
            ValueError: If backend_type is not supported
        """
        if backend_type == "hhem":
            return HHEMBackend(**config)
        if backend_type == "vectara_api":
            return VectaraAPIBackend(**config)

        raise ValueError(
            f"Unsupported backend_type: {backend_type}. "
            f"Supported types: 'hhem', 'vectara_api'"
        )

    def compute(self, rag_result: RAGResult) -> Dict[str, float]:
        """
        Compute hallucination score for a RAG result.

        Args:
            rag_result: RAG result containing retrieval and generation results

        Returns:
            Dictionary with 'hhem_score' key containing the factual consistency score
        """
        # Extract source passages
        passage_text_collection = []
        retrieval_results = rag_result.retrieval_result
        for _, passage in retrieval_results.retrieved_passages.items():
            passage_text_collection.append(passage)

        # Extract generated answer
        summary_text_collection = [
            generated_answer_part.text
            for generated_answer_part in rag_result.generation_result.generated_answer
        ]

        sources = " ".join(passage_text_collection)
        summary = " ".join(summary_text_collection)

        # Delegate to backend for evaluation
        score = self.backend.evaluate(sources, summary)

        # Keep 'hhem_score' key for backward compatibility
        return {"hhem_score": score}
