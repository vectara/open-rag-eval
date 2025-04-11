from typing import Dict
from enum import Enum
from pydantic import BaseModel

from open_rag_eval.models.llm_judges import LLMJudgeModel
from open_rag_eval.metrics.base_metrics import AugmentedGenerationMetric
from open_rag_eval.data_classes.rag_results import RAGResult

class CitationSupportValues(str, Enum):
    FULL = "full_support"
    PARTIAL = "partial_support"
    NONE = "no_support"

class CitationSupport(BaseModel):
    support: CitationSupportValues

class CitationMetric(AugmentedGenerationMetric):
    """
    This metric uses LLM as a judge to determine if the generated answer is supported by the retrieved passages
    that it cites.
    """

    _CITATION_PROMPT = """
        In this task, you will evaluate whether each statement is
        supported by its corresponding citations. Note that the system
        responses may appear very fluent and well-formed, but contain
        slight inaccuracies that are not easy to discern at first glance.
        Pay close attention to the text.

        You will be provided with a statement and its corresponding
        citation. It may be helpful to ask yourself whether it is
        accurate to say "according to the citation" with a
        statement following this phrase. Be sure to check all of the
        information in the statement. You will be given three options:

        - Full Support: All of the information in the statement is
        supported in the citation.

        - Partial Support: Some parts of the information are supported in
        the citation, but other parts are missing from the citation.

        - No Support: This citation does not support any part of the
        statement.

        Please provide your response based on the information in the
        citation. If you are unsure, use your best judgment. Respond as
        either ``full_support'', ``partial_support'', or ``no_support''
        with no additional information.

        Statement: {statement}

        Citation: {citation}
    """

    def __init__(self, model: LLMJudgeModel):
        """Initialize the Citation metric.

        Args:
            model (LLMJudgeModel): The model to use for the metric assessment.
        """
        self.model = model
        self.score_map = {
            "full_support": 1,
            "partial_support": 0.5,
            "no_support": 0
        }

    def compute(self, rag_result: RAGResult) -> Dict[str, str]:
        scores = {}
        retrieval_result = rag_result.retrieval_result
        generation_result = rag_result.generation_result

        for generated_answer_part in generation_result.generated_answer:
            answer_sentence, citations = generated_answer_part.text, generated_answer_part.citations
            if len(citations) == 0:
                continue
            key = citations[0]
            try:
                passage = retrieval_result.retrieved_passages.get(key, "")
                if not passage:
                    raise ValueError(f"No corresponding passage found for key: {key}")

                prompt = self._CITATION_PROMPT.format(
                    statement=answer_sentence,
                    citation=passage
                )
                response = self.model.parse(prompt, response_format=CitationSupport)
                if not response.support:
                    raise ValueError(f"Failed to parse response: {response.refusal}")

                label = response.support.value
                scores[f"citation_score_{key}"] = self.score_map[label]
            except Exception as e:
                raise Exception(f"Error computing Citation score: {str(e)}") from e

        # Weighted precision i.e.the weighted proportion of "citations" that support the answer sentence.
        p = sum(scores.values()) / len(scores) if scores else 0.0
        # Weighted recall i.e. proportion of answer sentences that are correctly cited.
        r = sum(scores.values()) / len(generation_result.generated_answer) if generation_result.generated_answer else 0.0

        scores["weighted_precision"] = p
        scores["weighted_recall"] = r

        if scores["weighted_precision"] + scores["weighted_recall"] == 0:
            scores["f1"] = 0.0
        else:
            scores["f1"] = 2 * (scores["weighted_precision"] * scores["weighted_recall"]) / (scores["weighted_precision"] + scores["weighted_recall"])

        return scores
