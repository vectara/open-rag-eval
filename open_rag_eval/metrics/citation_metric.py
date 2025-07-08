from typing import Dict
from collections import defaultdict

from enum import Enum
import logging
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

    def compute(self, rag_result: RAGResult) -> Dict[str, float]:
        """Compute the citation metric scores for the given RAG result.
        Args:
            rag_result (RAGResult): The RAG result containing retrieval and generation results.
        Returns:
            Dict[str, float]: A dictionary containing:
                - weighted_precision: Sum of citation average scores / total citations
                - weighted_recall: Sum of part average scores / total parts
                - f1: Harmonic mean of precision and recall
                - Individual citation and part scores
        """

        retrieval_result = rag_result.retrieval_result
        generation_result = rag_result.generation_result

        citation_to_scores = defaultdict(list)
        part_to_scores = defaultdict(list)
        for part_idx, generated_answer_part in enumerate(
                generation_result.generated_answer, start=1):
            answer_sentence, citations = (
                generated_answer_part.text,
                generated_answer_part.citations,
            )
            if len(citations) == 0:
                part_to_scores[f"part_score_{part_idx}"] = []
                continue
            for citation_key in citations:
                try:
                    passage = retrieval_result.retrieved_passages.get(
                        citation_key, "")
                    if not passage:
                        logging.error(
                            "While calculating citation metrics: Passage not found for key: %s Skipping this citation.",
                            citation_key)
                        continue

                    prompt = self._CITATION_PROMPT.format(
                        statement=answer_sentence, citation=passage
                    )
                    response = self.model.parse(
                        prompt,
                        response_format=CitationSupport,
                        model_kwargs={
                            "temperature": 0.0,
                            "seed": 42
                        }
                    )
                    if not response.support:
                        logging.error(
                            "While calculating citation metrics: failed to parse response – %s",
                            getattr(response, "refusal", "No error details available"),
                        )
                        continue

                    label = response.support.value
                    score = self.score_map[label]
                    citation_to_scores[f"citation_score_{citation_key}"].append(
                        score)
                    part_to_scores[f"part_score_{part_idx}"].append(score)
                except Exception as e:
                    raise Exception(
                        f"Error computing Citation score: {str(e)}") from e

        citation_averages = {
            citation_id: (sum(scores) / len(scores)) if scores else 0.0
            for citation_id, scores in citation_to_scores.items()
        }
        part_averages = {
            part_id: (sum(scores) / len(scores)) if scores else 0.0
            for part_id, scores in part_to_scores.items()
        }

        # Weighted precision i.e.the weighted proportion of "citations" that support the answer sentence.
        precision = (sum(citation_averages.values()) /
                     len(citation_averages) if citation_averages else 0.0)

        # Weighted recall i.e. proportion of answer parts that are correctly cited.
        recall = (sum(part_averages.values()) /
                  len(part_averages) if part_averages else 0.0)
        scores = citation_averages
        scores.update(part_averages)
        scores["weighted_precision"] = precision
        scores["weighted_recall"] = recall

        if scores["weighted_precision"] + scores["weighted_recall"] == 0:
            scores["f1"] = 0.0
        else:
            scores["f1"] = (
                2 * (scores["weighted_precision"] * scores["weighted_recall"]) /
                (scores["weighted_precision"] + scores["weighted_recall"]))

        return scores
