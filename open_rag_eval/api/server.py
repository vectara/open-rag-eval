import os
import logging
from typing import Dict, List
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from open_rag_eval.data_classes.rag_results import RAGResult, RetrievalResult, AugmentedGenerationResult
from open_rag_eval.data_classes.eval_scores import ScoredRAGResult
from open_rag_eval.evaluators.trec_evaluator import TRECEvaluator
from open_rag_eval.models.llm_judges import OpenAIModel

from open_rag_eval.api.schemas import (
    EvaluationRequestSchema,
    RAGResultSchema,
    ErrorResponseSchema
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Dictionary mapping evaluator names to their class constructors
EVALUATOR_MAP = {
    "trec": TRECEvaluator
}

def get_evaluator(evaluator_name: str, model_name: str):
    """Get the appropriate evaluator instance based on name."""
    if evaluator_name not in EVALUATOR_MAP:
        raise ValueError(f"Unsupported evaluator: {evaluator_name}")

    # Create model
    model = OpenAIModel(
        model_name=model_name,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create and return evaluator
    return EVALUATOR_MAP[evaluator_name](model=model)

def convert_schema_to_rag_result(schema: RAGResultSchema) -> RAGResult:
    """Convert RAGResultSchema to RAGResult."""
    retrieval_result = RetrievalResult(
        query=schema.retrieval_result.query,
        retrieved_passages=schema.retrieval_result.retrieved_passages
    )

    generation_result = AugmentedGenerationResult(
        query=schema.generation_result.query,
        generated_answer=schema.generation_result.generated_answer
    )

    return RAGResult(
        retrieval_result=retrieval_result,
        generation_result=generation_result
    )

def convert_scored_results_to_dict(scored_results: List[ScoredRAGResult]) -> List[Dict]:
    """Convert ScoredRAGResult objects to JSON-serializable dictionaries."""
    results = []
    for result in scored_results:
        # Convert generated answer parts to dict for JSON serialization
        generated_answer = result.rag_result.generation_result.generated_answer
        generated_answer_list = [{"text": part.text, "citations": part.citations}
                                 for part in generated_answer]

        if result.scores:
            # Convert the rag_result and scores to dict for JSON serialization
            result_dict = {
                "rag_result": {
                    "retrieval_result": {
                        "query": result.rag_result.retrieval_result.query,
                        "retrieved_passages": result.rag_result.retrieval_result.retrieved_passages
                    },
                    "generation_result": {
                        "query": result.rag_result.generation_result.query,
                        "generated_answer": generated_answer_list
                    }
                },
                "scores": {
                    "retrieval_scores": result.scores.retrieval_score.scores,
                    "generation_scores": result.scores.generation_score.scores
                }
            }
            results.append(result_dict)
        else:
            # Handle case where evaluation failed
            results.append({
                "rag_result": {
                    "retrieval_result": {
                        "query": result.rag_result.retrieval_result.query,
                        "retrieved_passages": result.rag_result.retrieval_result.retrieved_passages
                    },
                    "generation_result": {
                        "query": result.rag_result.generation_result.query,
                        "generated_answer": generated_answer_list
                    }
                },
                "scores": None
            })
    return results

@app.route("/api/v1/evaluate", methods=["POST"])
def evaluate():
    """Evaluate a single RAG output."""
    try:
        # Validate request body
        data = request.json
        request_data = EvaluationRequestSchema(
            rag_results=[data["rag_result"]],
            evaluator_name=data.get("evaluator_name", "trec"),
            model_name=data.get("model_name", "gpt-4o-mini")
        )

        # Get evaluator
        evaluator = get_evaluator(
            request_data.evaluator_name,
            request_data.model_name
        )

        # Convert schema to RAGResult
        rag_result = convert_schema_to_rag_result(request_data.rag_results[0])

        # Evaluate
        scored_result = evaluator.evaluate(rag_result)

        # Convert result to JSON-serializable format
        result_dict = convert_scored_results_to_dict([scored_result])[0]

        return jsonify(result_dict)

    except Exception as e:
        logger.exception("Error in evaluate endpoint")
        error_response = ErrorResponseSchema(
            error="Evaluation failed",
            details=str(e)
        )
        return jsonify(error_response.dict()), 400

@app.route("/api/v1/evaluate_batch", methods=["POST"])
def evaluate_batch():
    """Evaluate a batch of RAG outputs."""
    try:
        # Validate request body
        data = request.json
        request_data = EvaluationRequestSchema(**data)

        # Get evaluator
        evaluator = get_evaluator(
            request_data.evaluator_name,
            request_data.model_name
        )

        # Convert schema to RAGResults
        rag_results = [
            convert_schema_to_rag_result(result)
            for result in request_data.rag_results
        ]

        # Evaluate
        scored_results = evaluator.evaluate_batch(rag_results)

        # Convert results to JSON-serializable format
        results_dict = convert_scored_results_to_dict(scored_results)

        return jsonify(results_dict)

    except Exception as e:
        logger.exception("Error in evaluate_batch endpoint")
        error_response = ErrorResponseSchema(
            error="Batch evaluation failed",
            details=str(e)
        )
        return jsonify(error_response.dict()), 400

@app.route("/api/v1/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})

def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the Flask server."""
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server(debug=True)
