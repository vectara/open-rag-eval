"""
This script evaluates the performance of a retrieval-augmented generation (RAG) system.
"""

from typing import Any, Dict
import argparse

from pathlib import Path
from omegaconf import OmegaConf

from open_rag_eval import connectors, data_classes, models
from open_rag_eval import evaluators

def get_evaluator(config: Dict[str, Any]) -> evaluators.Evaluator:
    """
    Dynamically import and instantiate an evaluator class based on configuration.

    Args:
        config: Configuration dictionary containing evaluator settings

    Returns:
        An instance of the specified evaluator
    """
    evaluator_type = config.evaluator.type
    try:
        evaluator_class = getattr(evaluators, f'{evaluator_type}')

        # Verify it's a subclass of Evaluator
        if not issubclass(evaluator_class, evaluators.Evaluator):
            raise TypeError(f"{evaluator_type} is not a subclass of Evaluator")

        # Create the model instance based on config
        model_config = config.evaluator.model
        model_class = getattr(models, model_config.type)

        # Verify it's a subclass of LLMJudgeModel
        if not issubclass(model_class, models.LLMJudgeModel):
            raise TypeError(f"{model_config.type} is not a subclass of LLMJudgeModel")

        # Instantiate the model with config parameters
        model = model_class(
            model_name=model_config.name,
            api_key=model_config.api_key
        )

        # Instantiate the evaluator with the model
        return evaluator_class(model=model)

    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not load evaluator {evaluator_type}: {str(e)}") from e

def get_connector(config: Dict[str, Any]) -> connectors.Connector:
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
        connector_options = config.connector.options
        return connector_class(
            connector_options.api_key,
            connector_options.corpus_key)

    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not load connector {connector_type}: {str(e)}") from e


def run_eval(config_path: str):
    """
    Main function to run the evaluation process.
    Args:
        config_path: Path to the configuration file
    """
    # Load configuration
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = OmegaConf.load(config_path)

    # Create an evaluator based on config
    evaluator = get_evaluator(config)

    # Get a connector based on config.
    connector = get_connector(config)

    if connector:
        config["input_results"] = config.connector.options.generated_answers
        # Run queries and save results using connector if one is present.
        connector.fetch_data(
            config.connector.options.query_config,
            config.connector.options.input_queries,
            config.connector.options.generated_answers
        )

    rag_results = connectors.CSVConnector(config.input_results).fetch_data()

    # Run the evaluation
    scored_results = evaluator.evaluate_batch(rag_results)

    # Save the results to the configured output folder
    data_classes.eval_scores.to_csv(scored_results, config.evaluation_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run RAG evaluation')
    parser.add_argument(
        '--config', type=str, default='eval_config.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()

    run_eval(args.config)
