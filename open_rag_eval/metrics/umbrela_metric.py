from enum import Enum

from pydantic import BaseModel

from open_rag_eval.models.llm_judges import LLMJudgeModel
from open_rag_eval.metrics.base_metrics import RetrievalMetric
from open_rag_eval.data_classes.rag_results import RetrievalResult


class UMBRELAScoreValues(str, Enum):
    NO_RELEVANCE = "0"
    RELATED = "1"
    PARTIAL_ANSWER = "2"
    EXACT_ANSWER = "3"


class UMBRELAScore(BaseModel):
    score: UMBRELAScoreValues


class UMBRELAMetric(RetrievalMetric):
    """
    This metric is based on the UMBRELA: UMbrela is the (Open-Source Reproduction of the)
    Bing RELevance Assessor paper: https://arxiv.org/pdf/2406.06519
    """

    _UMBRELA_PROMPT = """Given a query and a passage, you must provide a score on an
                        integer scale of 0 to 3 with the following meanings:
                        0 = represent that the passage has nothing to do with the query,
                        1 = represents that the passage seems related to the query but
                        does not answer it,
                        2 = represents that the passage has some answer for the query,
                        but the answer may be a bit unclear, or hidden amongst extraneous
                        information and
                        3 = represents that the passage is dedicated to the query and
                        contains the exact answer.
                        Important Instruction: Assign category 1 if the passage is
                        somewhat related to the topic but not completely, category 2 if
                        passage presents something very important related to the entire
                        topic but also has some extra information and category 3 if the
                        passage only and entirely refers to the topic. If none of the
                        above satisfies give it category 0.
                        Query: {query}
                        Passage: {passage}
                        Split this problem into steps:
                        Consider the underlying intent of the search.
                        Measure how well the content matches a likely intent of the query
                        (M).
                        Measure how trustworthy the passage is (T).
                        Consider the aspects above and the relative importance of each,
                        and decide on a final score (O). Final score must be an integer
                        value only.
                        Do not provide any code in result. Provide each score in the
                        format of: a single integer without any reasoning."""

    def __init__(self, model: LLMJudgeModel):
        """Initialize the UMBRELA metric.

        Args:
            model (str): The model to use for the metric assesment.
            prompt_override (str): An optional prompt to override the default UMBRELA prompt.
                Must hvae placeholders for {query} and {passage}.
        """
        self.model = model
        # kwargs to match the UMBRELA paper.
        self.model_kwargs = {
            "temperature": 0.0,
            "top_p": 1.0,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.0,
        }
        self.prompt = self._UMBRELA_PROMPT
        # Any UMBRELA score above this threshold is considered relevant for
        # calculation of traditional retrieval metrics like MAP, Precison@k, etc.
        self._umbrela_relevant_threshold = 2

    def compute(self, retrieval_result: RetrievalResult) -> dict[str, int]:
        scores = {}
        score_map = {"0": 0, "1": 1, "2": 2, "3": 3}

        for key, passage in retrieval_result.retrieved_passages.items():
            try:
                query = retrieval_result.query
                prompt = self.prompt.format(query=query, passage=passage)
                response = self.model.parse(prompt, UMBRELAScore, self.model_kwargs)

                if not response.score:
                    raise ValueError(f"Failed to parse response: {response.refusal}")

                scores[key] = score_map[response.score.value]

            except Exception as e:
                raise Exception(f"Error computing UMBRELA score: {str(e)}") from e

        # Calculate traditional retrieval metrics.
        self.add_retrieval_metrics(scores)

        return scores

    def add_retrieval_metrics(self, scores: dict[str, int]) -> None:
        """Add traditional retrieval metrics to the scores dictionary.

        Args:
            scores (dict): The scores dictionary to update.
        """

        # Calculate precision@K where K is the number of retrieved passages.
        k = len(scores)
        if k == 0:
            return
        
        scores[f"precision_at_{k}"] = sum(
            1 for score in scores.values() if score >= self._umbrela_relevant_threshold
        ) / len(scores)

        # Calculate Average Precision (AP@k )
        binary_relevance = [
            1 if score >= self._umbrela_relevant_threshold else 0
            for score in scores.values()
        ]
        total_relevant = sum(binary_relevance)

        if total_relevant == 0:
            scores[f"ap_at_{k}"] = 0.0
        else:
            precision_at_k = []
            relevant_so_far = 0

            for i, is_relevant in enumerate(binary_relevance, start=1):
                if is_relevant == 1:
                    relevant_so_far += 1
                    precision = relevant_so_far / i
                    precision_at_k.append(precision)

            # Calculate AP as the average of precision values at relevant positions
            ap = sum(precision_at_k) / len(precision_at_k)
            scores[f"ap_at_{k}"] = ap

        # Mean Reciprocal Rank (MRR).
        scores["MRR"] = 0.0
        for i, (_, score) in enumerate(scores.items(), start=1):
            if score >= self._umbrela_relevant_threshold:
                scores["MRR"] = 1.0 / i
                # Only consider the first relevant item for MRR.
                break
