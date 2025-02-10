from data_classes.rag_results import RAGResult, RetrievalResult, AugmentedGenerationResult
from evaluators.trec_evaluator import TRECEvaluator

def create_dummy_data():
    query = "What is "
    retrieved_passages = {
        "doc1": "France is a country in Western Europe.",
        "doc2": "Paris is the capital and most populous city of France. Situated on the Seine River, in the north of the country, it is in the centre of the Île-de-France region",
        "doc3": "France is known for its wine and cheese.",
        "doc4": "Paris is known for its Eiffel Tower.",
        "doc5": "Spain is a country in Western Europe.",
    }
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

    # Get some data to evaluate. We need to support import from a database, CSV and JSON formats.
    rag_result = create_dummy_data()

    # Create an evaluator with the model you need as the judge.
    evaluator = TRECEvaluator()

    # Run the evaluation, this can be done in a batch or on a single example. Single threaded or multi-threaded.
    metrics = evaluator.evaluate(rag_result)

    # Save the results, we need to provide options to save as CSV/JSON or other formats.
    print(metrics)


if __name__ == "__main__":
    run_eval()

    