import ast

from metrics.base_metrics import AugmentedGenerationMetric
from models import LLMJudgeModel
from data_classes.rag_results import RAGResult
from typing import List, Dict, Optional
from enum import Enum
from dataclasses import dataclass


class NuggetImportance(Enum):
    VITAL = "vital"
    OKAY = "okay"

class NuggetSupport(Enum):
    SUPPORT = "support"
    PARTIAL_SUPPORT = "partial_support"
    NOT_SUPPORT = "not_support"

@dataclass
class Nugget:
    text: str
    importance: NuggetImportance


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
    
    def __init__(self, model: LLMJudgeModel, nugget_creation_iters: int = 5):
        self.model = model        
        self.nugget_creation_iters = nugget_creation_iters
        self.max_nuggets = 30

    def compute(self, rag_result: RAGResult, umbrela_scores: dict[str, int]) -> dict[str, int]:
        retrieval_result = rag_result.retrieval_result
        nuggets = self._create_nuggets(retrieval_result.query, retrieval_result.retrieved_passages, umbrela_scores)


    def _create_nuggets(self, query: str, retrieved_passages: dict[str, str], umbrela_scores: dict[str, int]) -> List[Nugget]:
        #TODO: Filter the retrieved passages based on the umbrela scores.
        context = "\n".join(f"[{i+1}] {seg}" for i, (_, seg) in enumerate(retrieved_passages.items()))
        nuggets = []
        for _ in range(self.nugget_creation_iters):
            prompt = self._NUGGET_CREATION_PROMPT.format(query=query, context=context, initial_nuggets=nuggets, initial_nuggets_length=len(nuggets), max_nuggets=self.max_nuggets)
            response = self.model.call(prompt)
            nuggets = ast.literal_eval(response)
            if len(nuggets) >= self.max_nuggets:
                break

        return nuggets

class AutoNuggetizer:

    def _create_importance_prompt(self, query: str, nuggets: List[str]) -> str:
        """Create the importance labeling prompt as shown in the paper."""
        return f"""You are NuggetizeScoreLLM, an intelligent assistant that can label a list of atomic nuggets based on their importance for a given search query.

Based on the query, label each of the {len(nuggets)} nuggets either a vital or okay based on the following criteria. Vital nuggets represent concepts that must be present in a "good" answer; on the other hand, okay nuggets contribute worthwhile information about the target but are not essential. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget.

Search Query: {query}
Nugget List: {nuggets}

Only return the list of labels (List[str]). Do not explain."""

    def _create_assignment_prompt(self, query: str, answer: str, nuggets: List[str]) -> str:
        """Create the nugget assignment prompt as shown in the paper."""
        return f"""You are NuggetizeAssignerLLM, an intelligent assistant that can label a list of atomic nuggets based on if they are captured by a given passage.

Based on the query and passage, label each of the {len(nuggets)} nuggets either as support, partial_support, or not_support using the following criteria. A nugget that is fully captured in the passage should be labeled as support. A nugget that is partially captured in the passage should be labeled as partial_support. If the nugget is not captured at all, label it as not_support. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget.

Search Query: {query}
Passage: {answer}
Nugget List: {nuggets}

Only return the list of labels (List[str]). Do not explain."""

    def _process_llm_chunks(self, items: List, chunk_size: int, process_func) -> List:
        """Process items in chunks to avoid token limits."""
        results = []
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            results.extend(process_func(chunk))
        return results

    def generate_nuggets(self, query: str, context_segments: List[str]) -> List[Nugget]:
        """
        Generate nuggets for a query using the provided context.
        
        Args:
            query: The search query
            context_segments: List of relevant document segments
            
        Returns:
            List of Nugget objects
        """
        # First generate raw nuggets
        prompt = self._create_nugget_prompt(query, context_segments)
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        try:
            nugget_texts = ast.literal_eval(response.choices[0].message.content)
        except:
            nugget_texts = []
            
        # Then determine importance for each nugget
        def process_importance_chunk(nugget_chunk):
            prompt = self._create_importance_prompt(query, nugget_chunk)
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return ast.literal_eval(response.choices[0].message.content)
            
        importance_labels = self._process_llm_chunks(nugget_texts, 10, process_importance_chunk)
        
        # Create Nugget objects
        nuggets = []
        for text, importance in zip(nugget_texts, importance_labels):
            nuggets.append(Nugget(
                text=text,
                importance=NuggetImportance(importance)
            ))
            
        # Sort by importance (vital first) and limit to 20
        nuggets.sort(key=lambda x: x.importance == NuggetImportance.OKAY)
        return nuggets[:20]

    def assign_nuggets(self, query: str, answer: str, nuggets: List[Nugget]) -> List[NuggetSupport]:
        """
        Assign support levels to nuggets for a given answer.
        
        Args:
            query: The search query
            answer: The system's answer text
            nuggets: List of Nugget objects
            
        Returns:
            List of NuggetSupport values
        """
        nugget_texts = [n.text for n in nuggets]
        
        def process_assignment_chunk(nugget_chunk):
            prompt = self._create_assignment_prompt(query, answer, nugget_chunk)
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return ast.literal_eval(response.choices[0].message.content)
            
        assignments = self._process_llm_chunks(nugget_texts, 10, process_assignment_chunk)
        return [NuggetSupport(a) for a in assignments]

class NuggetEvaluator:
    def __init__(self, nuggets: List[Nugget]):
        """Initialize the evaluator with a list of nuggets."""
        self.nuggets = nuggets
        
    def evaluate_answer(self, nugget_assignments: List[NuggetSupport]) -> Dict:
        """Evaluate an answer based on nugget assignments."""
        if len(nugget_assignments) != len(self.nuggets):
            raise ValueError("Number of assignments must match number of nuggets")
        
        # Count nuggets by importance and support level
        vital_supported = 0
        vital_partial = 0
        vital_total = 0
        okay_supported = 0
        okay_partial = 0
        okay_total = 0
        
        for nugget, support in zip(self.nuggets, nugget_assignments):
            if nugget.importance == NuggetImportance.VITAL:
                vital_total += 1
                if support == NuggetSupport.SUPPORT:
                    vital_supported += 1
                elif support == NuggetSupport.PARTIAL_SUPPORT:
                    vital_partial += 1
            else:  # OKAY
                okay_total += 1
                if support == NuggetSupport.SUPPORT:
                    okay_supported += 1
                elif support == NuggetSupport.PARTIAL_SUPPORT:
                    okay_partial += 1
        
        # Calculate scores
        vital_score = (vital_supported + 0.5 * vital_partial) / vital_total if vital_total > 0 else 0
        okay_score = (okay_supported + 0.5 * okay_partial) / okay_total if okay_total > 0 else 0
        
        # Calculate weighted combined score (vital nuggets weighted more heavily)
        combined_score = (0.7 * vital_score + 0.3 * okay_score) if (vital_total + okay_total) > 0 else 0
        
        return {
            "vital_score": vital_score,
            "okay_score": okay_score,
            "combined_score": combined_score,
            "vital_stats": {
                "supported": vital_supported,
                "partial": vital_partial,
                "total": vital_total
            },
            "okay_stats": {
                "supported": okay_supported,
                "partial": okay_partial,
                "total": okay_total
            }
        }    