from typing import List

from models.llm_judges import LLMJudgeModel
from metrics.base_metrics import RetrievalMetric
from data_classes.rag_results import RetrievalResult

class UMBRELAMetric(RetrievalMetric):
    """ This metric is based on the UMBRELA: UMbrela is the (Open-Source Reproduction of the) 
        Bing RELevance Assessor paper: https://arxiv.org/pdf/2406.06519"""        
    
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
                        format of: ##final score: score without providing any reasoning."""

    def __init__(self, model: LLMJudgeModel):
        """Initialize the UMBRELA metric.
        
        Args:
            model (str): The model to use for the metric assesment.
            prompt_override (str): An optional prompt to override the default UMBRELA prompt. 
                Must hvae placeholders for {query} and {passage}.
        """
        self.model = model
        #kwargs to match the UMBRELA paper.
        self.model_kwargs = {
            "temperature": 0.0,
            "top_p": 1.0,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.0,                
        }         
        self.prompt = self._UMBRELA_PROMPT   

    def compute(self, retrieval_result: RetrievalResult) -> dict[str, int]:
        scores = {}
        for key, passage in retrieval_result.retrieved_passages.items():
            try:
                query = retrieval_result.query
                prompt = self.prompt.format(query=query, passage=passage)
                raw_score = self.model.call(prompt, self.model_kwargs)
                
                # Extract numeric score from the response
                try:
                    # Look for "##final score: X" pattern and extract X
                    score_str = raw_score.split("##final score:")[-1].strip()
                    score = int(score_str)
                    
                    # Validate score range
                    if score not in [0, 1, 2, 3]:
                        raise ValueError(f"Invalid score {score}. Must be between 0 and 3.")
                    
                    scores[key] = score
                except (ValueError, IndexError) as e:
                    raise ValueError(f"Failed to parse valid score from model response: {raw_score}. Error: {str(e)}")
                    
            except Exception as e:
                raise Exception(f"Error computing UMBRELA score: {str(e)}")
                
        return scores
            
            
