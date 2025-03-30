from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class RetrievalResultSchema(BaseModel):
    query: str
    retrieved_passages: Dict[str, str]


class GeneratedAnswerPartSchema(BaseModel):
    text: str
    citations: List[str]


class AugmentedGenerationResultSchema(BaseModel):
    query: str
    generated_answer: List[GeneratedAnswerPartSchema]


class RAGResultSchema(BaseModel):
    retrieval_result: RetrievalResultSchema
    generation_result: AugmentedGenerationResultSchema


class EvaluationRequestSchema(BaseModel):
    rag_results: List[RAGResultSchema]
    evaluator_name: str = Field(default="trec")
    model_name: str = Field(default="gpt-4o-mini")


class CSVEvaluationRequestSchema(BaseModel):
    csv_path: str
    evaluator_name: str = Field(default="trec")
    model_name: str = Field(default="gpt-4o-mini")


class ErrorResponseSchema(BaseModel):
    error: str
    details: Optional[str] = None
