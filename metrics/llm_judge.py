# A metric class which uses an LLM as a judge to evaluate the quality of a given answer  given a prompt.

from typing import List, Optional
from base_metrics import RetrievalMetric

class UMBERLAMetric(RetrievalMetric):
    """ This metric is based on the UMBRELA: UMbrela is the (Open-Source Reproduction of the) 
        Bing RELevance Assessor paper: https://arxiv.org/pdf/2406.06519"""        
    
    _UMBRELA_DEFAULT_PROMPT = """Given a query and a passage, you must provide a score on an
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
                                format of: ##final score: score without providing any reasoning."""

    def __init__(self, model: Optional[str] = None, prompt_override: Optional[str] = None):
        """Initialize the UMBRELA metric.
        
        Args:
            model (str): The model to use for the metric assesment.
            prompt_override (str): An optional prompt to override the default UMBRELA prompt.
        """
        self.model = model
        self.prompt = prompt_override if prompt_override else self._UMBRELA_DEFAULT_PROMPT
        

    def compute(self, predictions: List[str], references: List[str]) -> float:
        pass