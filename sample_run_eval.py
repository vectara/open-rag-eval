import os

import connectors
import evaluators

from data_classes import eval_scores
from data_classes.rag_results import AugmentedGenerationResult, RAGResult, RetrievalResult
from connectors.connector import Connector
from evaluators.base_evaluator import Evaluator
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
    evaluator_type = config.evaluator.type    
    try:
        # Import the evaluator module
        evaluator_class = getattr(evaluators, evaluator_type)
        
        # Verify it's a subclass of Evaluator
        if not issubclass(evaluator_class, Evaluator):
            raise TypeError(f"{evaluator_type} is not a subclass of Evaluator")
        
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
        raise ImportError(f"Could not load evaluator {evaluator_type}: {str(e)}")

def get_connector(config: Dict[str, Any]) -> Connector:
    """
    Dynamically import and instantiate a connector class based on configuration.
    
    Args:
        config: Configuration dictionary containing connector settings
        
    Returns:
        An instance of the specified connector
    """
    if 'connector' not in config:
        return None
    connector_type = config.connector.type
    try:
        # Import the connector module
        connector_class = getattr(connectors, connector_type)

        # Create connector instance with options from config
        connector_options =config.connector.options
        return connector_class(connector_options.customer_id,
                               connector_options.api_key, 
                               connector_options.corpus_key)
        
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not load connector {connector_type}: {str(e)}")


def run_eval(config_path: str):    
    # Load configuration
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    
    # Create an evaluator based on config
    evaluator = get_evaluator(config)

    # Get a connector based on config.
    connector = get_connector(config)

    if connector:
        config["input_results"] = config.options.output_csv
        # Run queries and save results using connector if one is present.
        connector.fetch_data(
            config.connector.options.query_config,
            config.connector.options.input_csv,
            config.connector.options.output_csv)
        
    rag_results = connectors.CSVConnector(config.input_results).fetch_data()

    # Run the evaluation
    scored_results = evaluator.evaluate_batch(rag_results)

    # Save the results to the configured output folder
    eval_scores.to_csv(scored_results, config.output_folder)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run RAG evaluation')
    parser.add_argument('--config', type=str, default='eval_config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    run_eval(args.config)