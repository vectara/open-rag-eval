
from typing import List, Dict, Tuple

import logging

from enum import Enum
from pydantic import BaseModel

from open_rag_eval.data_classes.rag_results import RAGResult
from open_rag_eval.metrics.base_metrics import AugmentedGenerationMetric
from open_rag_eval.models.llm_judges import LLMJudgeModel


# These classes are used to ensure structured output from the Judge LLM.
class Nuggets(BaseModel):
    nuggets: list[str]

class NuggetImportanceValues(str, Enum):
    VITAL = "vital"
    OKAY = "okay"

class NuggetImportance(BaseModel):
    importance: list[NuggetImportanceValues]


class NuggetAssignmentValues(str, Enum):
    SUPPORT = "support"
    PARTIAL_SUPPORT = "partial_support"
    NOT_SUPPORT = "not_support"

class NuggetAssignment(BaseModel):
    assignment: list[NuggetAssignmentValues]


class AutoNuggetMetric(AugmentedGenerationMetric):
    """Implements the AutoNuggetizer evaluation metric from the TREC 2024 RAG Track.
       For more details, please refer to the following paper:
       https://arxiv.org/pdf/2411.09607
    """

    _NUGGET_CREATION_PROMPT = """
        You are NuggetizeLLM, an intelligent assistant that can update a list of atomic nuggets to best provide all the information required for the query.
        Update the list of atomic nuggets of information (1-12 words), if needed, so they best provide the information required for the query. Leverage only
        the initial list of nuggets (if exists) and the provided context (this is an iterative process). Return only the final list of all nuggets in a Pythonic list format (even if no updates).
        Make sure there is no redundant information. Ensure the updated nugget list has at most {max_nuggets} nuggets (can be less), keeping only the most vital ones. Order them in decreasing order of importance.
        Prefer nuggets that provide more interesting information.

        Search Query: {query}

        Context:
        {context}

        Initial Nugget List: {initial_nuggets}
        Initial Nugget List Length: {initial_nuggets_length}

        Only update the list of atomic nuggets (if needed, else return as is). Do not explain.
        Always answer in short nuggets (not questions). List in the form ["a", "b", ...] and a and b are strings with no mention of ".

        Updated Nugget List:
        """

    _NUGGET_IMPORTANCE_PROMPT = """
        You are NuggetizeScoreLLM, an intelligent assistant that can label a list of atomic nuggets
        based on their importance for a given search query

        Based on the query, label each of the {len_nuggets} nuggets either as vital or okay based on the
        following criteria. Vital nuggets represent concepts that must be present in a “good” answer; on the other
        hand, okay nuggets contribute worthwhile information about the target but are not essential. Return the
        list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input
        nuggets. Make sure to provide a label for each nugget.
        Search Query: {query}
        Nugget List: {nuggets}
        Only return the list of labels (List[str]). Do not explain your answer.
        Labels:
    """

    _NUGGET_ASSIGNMENT_PROMPT = """
        You are NuggetizeAssignerLLM, an intelligent assistant that can label a list of atomic nuggets
        based on if they are captured by a given passage.

        Based on the query and passage, label each of the {len_nuggets} nuggets either as support, partial_support, or not_support using the following criteria.
        A nugget that is fully captured in the passage should be labeled as support. A nugget that is partially captured in the passage should be labeled as
        partial_support. If the nugget is not captured at all, label it as not_support. Return the list of labels in a
        Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure
        to provide a label for each nugget.
        Search Query: {query}

        Passage: {generated_passage}

        Nugget List: {nuggets}

        Only return the list of labels (List[str]). Do not explain your answer.

        Labels:
    """

    def __init__(self, model: LLMJudgeModel, nugget_creation_iters: int = 5):
        self.model = model
        self.nugget_creation_iters = nugget_creation_iters
        self.max_nuggets = 30
        self.assignment_score_map = {"support": 1.0, "partial_support": 0.5, "not_support": 0.0}

    def compute(self, rag_result: RAGResult, umbrela_scores: Dict[str, int]) -> Dict[str, int]:
        retrieval_result = rag_result.retrieval_result
        try:
            nuggets = self._create_nuggets(retrieval_result.query, retrieval_result.retrieved_passages, umbrela_scores)
            sorted_nuggets, sorted_labels = self._score_and_sort_nuggets(retrieval_result.query, nuggets)
            nugget_assignments = self._assign_nuggets(rag_result.generation_result.query,
                                                      rag_result.generation_result.generated_answer,
                                                      sorted_nuggets)

            scores = {}
            scores["nuggetizer_scores"] = self._evaluate_answer(sorted_nuggets, sorted_labels, nugget_assignments)
            scores["nuggets"] = sorted_nuggets
            scores["labels"] = sorted_labels
            scores["assignments"] = nugget_assignments
            scores["assignment_scores"] = [self.assignment_score_map[assignment] for assignment in nugget_assignments]

            return scores
        except Exception as e:
            logging.error("Failed to compute nugget metric: %s", str(e))
            return {}

    def _create_nuggets(self, query: str, retrieved_passages: Dict[str, str], umbrela_scores: Dict[str, int]) -> List[str]:
        """
        Creates nuggets (concise information units) from retrieved passages based on a query.

        This method filters passages based on umbrella scores and iteratively generates nuggets
        using a language model until the maximum number of nuggets is reached or iterations complete.

        """
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        filtered_passages = {k: v for k, v in retrieved_passages.items() if umbrela_scores.get(k, 0) >= 1}
        context = "\n".join(f"[{i+1}] {seg}" for i, (_, seg) in enumerate(filtered_passages.items()))
        nuggets = []
        for _ in range(self.nugget_creation_iters):
            prompt = self._NUGGET_CREATION_PROMPT.format(
                query=query,
                context=context,
                initial_nuggets=nuggets,
                initial_nuggets_length=len(nuggets),
                max_nuggets=self.max_nuggets
            )
            try:
                response = self.model.parse(prompt, response_format=Nuggets)
            except Exception as e:
                logging.error(f"Failed to create nuggets: {e}")
                raise e

            if response.nuggets:
                nuggets = response.nuggets
            else:
                logging.error(f"Failed to parse nuggets from response: {response.refusal} for query {query}.")
                raise ValueError(f"Failed to parse nuggets from response: {response.refusal} for query {query}.")

            if len(nuggets) >= self.max_nuggets:
                break

        return nuggets

    def _score_and_sort_nuggets(self, query: str, nuggets: List[str]) -> Tuple[List[str], List[str]]:
        """
        Evaluates and ranks a list of text nuggets based on their relevance to a query.
        Processes nuggets in batches of 10, scores them using an LLM, and returns the top
        20 most relevant nuggets along with their importance labels.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if not nuggets:
            return [], []
        labels = []
        for i in range(0, len(nuggets), 10):
            prompt = self._NUGGET_IMPORTANCE_PROMPT.format(
                query=query,
                len_nuggets=len(nuggets[i:i+10]),
                nuggets=nuggets[i:i+10]
            )
            try:
                response = self.model.parse(prompt, response_format=NuggetImportance)
            except Exception as e:
                logging.error(f"Failed to create nuggets: {e}")
                raise e

            if response.importance:
                labels.extend(response.importance)
            else:
                logging.error(f"Failed to score nuggets from response: {response.refusal} for query {query}.")
                raise ValueError(f"Failed to score nuggets from response: {response.refusal} for query {query}.")

        if len(labels) != len(nuggets):
            raise ValueError("Number of labels does not match number of nuggets.")
        sorted_pairs = sorted(zip(nuggets, labels), key=lambda x: x[1] == "okay")
        sorted_nuggets, sorted_labels = zip(*sorted_pairs)
        return list(sorted_nuggets[:20]), list(sorted_labels[:20])

    def _assign_nuggets(self, query: str, generated_answer: Dict[str, str], nuggets: List[str]) -> List[str]:
        """Evaluates how well each nugget is covered in the generated passage by assigning
        support/partial_support/not_support labels"""

        # Generated passage maps passage ID to passage text, we need to convert this to a single string.
        generated_passage = " ".join([generated_answer_part.text for generated_answer_part in generated_answer])

        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if not generated_passage.strip():
            raise ValueError("Generated passage cannot be empty.")
        if not nuggets:
            return []
        assignments = []
        for i in range(0, len(nuggets), 10):
            prompt = self._NUGGET_ASSIGNMENT_PROMPT.format(
                query=query,
                len_nuggets=len(nuggets[i:i+10]),
                nuggets=nuggets[i:i+10],
                generated_passage=generated_passage
            )
            try:
                response = self.model.parse(prompt, response_format=NuggetAssignment)
            except Exception as e:
                logging.error(f"Failed to create nuggets: {e}")
                raise e
            if response.assignment:
                assignments.extend(response.assignment)
            else:
                logging.error(f"Failed to assign nuggets from response: {response.refusal} for query {query}.")
                raise ValueError(f"Failed to assign nuggets from response: {response.refusal} for query {query}.")

        if len(assignments) != len(nuggets):
            raise ValueError("Number of assignments does not match number of nuggets.")
        return assignments

    def _evaluate_answer(self, nuggets: List[str], labels: List[str], nugget_assignments: List[str]) -> Dict[str, float]:
        """
        Calculates various nugget evaluation scores by comparing nugget assignments with their labels.
        Computes both strict and lenient scores, with weighted versions accounting for vital and okay labels.
        """
        if len(nugget_assignments) != len(nuggets):
            raise ValueError(f"Nugget assignments length ({len(nugget_assignments)}) must match nuggets length ({len(nuggets)}).")
        vital_scores, okay_scores = [], []
        strict_vital_scores, strict_okay_scores = [], []
        all_scores, all_strict_scores = [], []

        for label, assignment in zip(labels, nugget_assignments):
            score = self.assignment_score_map.get(assignment, 0.0)
            strict_score = 1.0 if assignment == "support" else 0.0
            all_scores.append(score)
            all_strict_scores.append(strict_score)
            if label == "vital":
                vital_scores.append(score)
                strict_vital_scores.append(strict_score)
            elif label == "okay":
                okay_scores.append(score)
                strict_okay_scores.append(strict_score)

        num_nuggets = max(len(nuggets), 1)
        num_vital = max(len(vital_scores), 1)
        num_okay = max(len(okay_scores), 1)

        all_score = sum(all_scores) / num_nuggets
        all_strict_score = sum(all_strict_scores) / num_nuggets
        vital_score = sum(vital_scores) / num_vital
        vital_strict_score = sum(strict_vital_scores) / num_vital
        weighted_score = (sum(vital_scores) + 0.5 * sum(okay_scores)) / (num_vital + 0.5 * num_okay)
        weighted_strict_score = (sum(strict_vital_scores) + 0.5 * sum(strict_okay_scores)) / (num_vital + 0.5 * num_okay)

        return {
            "All": all_score,
            "All Strict": all_strict_score,
            "Vital": vital_score,
            "Vital Strict": vital_strict_score,
            "Weighted": weighted_score,
            "Weighted Strict": weighted_strict_score
        }
