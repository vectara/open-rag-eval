# This method defines classes to hold results from a RAG system which need to be evaluated.

from typing import List, Optional

@dataclass
class RetrievalResult:
    """This class holds the output from a retrieval system."""
    # The query that was used to retrieve the passages
    query: str      
    # The passages that were retrieved, each entry in the dict is a passage 
    # with the key being the passage ID.    
    retrieved_passages: dict[str, str]


@dataclass
class GenerationResult:
    """This class holds the output from a generation system."""
    # The query that was used to generate the answer
    query: str
    # The generated answer, the values in the dicts are parts of the answer and the keys 
    # are the passage ID that the information was derived from.
    generated_answer: dict[str, str] 

@dataclass
class RAGResult:
    """This class holds the output from a RAG system."""
    retrieval_result: RetrievalResult
    generation_result: GenerationResult
    
    