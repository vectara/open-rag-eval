from dotenv import load_dotenv
import os

from data_classes import eval_scores

from data_classes.rag_results import AugmentedGenerationResult, RAGResult, RetrievalResult
from evaluators.trec_evaluator import TRECEvaluator
from models.llm_judges import OpenAIModel

def create_dummy_data():
    query = "What is the capital of France?"
    retrieved_passages = {
        "doc1": "France is a country in Western Europe.",
        "doc2": "Paris is the capital and most populous city of France. Situated on the Seine River, in the north of the country, it is in the centre of the Île-de-France region",
        "doc3": "France is known for its wine and cheese.",
        "doc4": "Paris is known for its Eiffel Tower.",
        "doc5": "Spain is a country in Western Europe.",
    }

    # The original output may be "Paris is the capital of France. [2]"
    # but gets converted to this kev: value format.
    generated_answer = {"doc2": "Paris is the capital of France."}

    retrieval_result = RetrievalResult(
        query=query,
        retrieved_passages=retrieved_passages
    )

    generation_result = AugmentedGenerationResult(
        query=query,
        generated_answer=generated_answer
    )

    return RAGResult(
        retrieval_result=retrieval_result,
        generation_result=generation_result
    )

def run_eval():

    load_dotenv()
    # Get some data to evaluate. We need to support import from a database, CSV and JSON formats.
    rag_result = create_dummy_data()

    # Create an evaluator with the model you need as the judge.
    evaluator = TRECEvaluator(model=OpenAIModel(model_name="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")))

    # Run the evaluation, this can be done in a batch or on a single example. Single threaded or multi-threaded.
    scored_results = evaluator.evaluate_batch([rag_result])

    # Save the results, we need to provide options to save as CSV/JSON or other formats.
    eval_scores.to_csv(scored_results, "results.csv")


if __name__ == "__main__":
    run_eval()    