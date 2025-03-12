
import dotenv
import importlib
import sys
import os

from data_classes import eval_scores
from data_classes.rag_results import AugmentedGenerationResult, RAGResult, RetrievalResult
from evaluators.trec_evaluator import Evaluator
from models.llm_judges import OpenAIModel
from omegaconf import OmegaConf
from pathlib import Path


from typing import Any, Dict


def get_evaluator(config: Dict[str, Any]) -> Evaluator:
    """
    Dynamically import and instantiate an evaluator class based on configuration.
    
    Args:
        config: Configuration dictionary containing evaluator settings
        
    Returns:
        An instance of the specified evaluator
    """
    evaluator_type = config.evaluator.type.lower()
    evaluator_class_name = f"{evaluator_type.upper()}Evaluator"
    
    # Add evaluators directory to path
    evaluators_dir = os.path.join(os.path.dirname(__file__), 'evaluators')
    sys.path.insert(0, os.path.abspath(evaluators_dir))
    
    try:
        # Import the evaluator module
        module_name = f"{evaluator_type}_evaluator"
        module = importlib.import_module(module_name)
        
        # Get the evaluator class
        evaluator_class = getattr(module, evaluator_class_name)
        
        # Verify it's a subclass of Evaluator
        if not issubclass(evaluator_class, Evaluator):
            raise TypeError(f"{evaluator_class_name} is not a subclass of Evaluator")
        
        # Create the model instance based on config
        model_config = config.evaluator.model
        if model_config.type.lower() == "openai":
            model = OpenAIModel(
                model_name=model_config.name,
                api_key=model_config.api_key
            )
        else:
            raise ValueError(f"Unknown model type: {model_config.type}")
            
        # Instantiate the evaluator with the model
        return evaluator_class(model=model)
        
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not load evaluator {evaluator_class_name}: {str(e)}")
    finally:
        # Remove evaluators directory from path
        sys.path.remove(os.path.abspath(evaluators_dir))


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

def run_eval(config_path: str):
    dotenv.load_dotenv()
    
    # Load configuration
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    
    # Get some data to evaluate
    rag_result = create_dummy_data()

    # Create an evaluator based on config
    evaluator = get_evaluator(config)

    # Run the evaluation
    scored_results = evaluator.evaluate_batch([rag_result])

    # Save the results
    eval_scores.to_csv(scored_results, "results.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run RAG evaluation')
    parser.add_argument('--config', type=str, default='eval_config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    run_eval(args.config)