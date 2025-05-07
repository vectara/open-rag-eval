"""
This script evaluates the performance of a retrieval-augmented generation (RAG) system.
"""

from typing import Any, Dict
import argparse
import os
import shutil

from pathlib import Path
from omegaconf import OmegaConf

from open_rag_eval import connectors, data_classes, models
from open_rag_eval import evaluators
from open_rag_eval.rag_results_loader import RAGResultsLoader


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
        evaluator_class = getattr(evaluators, f"{evaluator_type}")

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
        model = model_class(model_name=model_config.name, api_key=model_config.api_key)

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
    if "connector" not in config:
        return None
    connector_type = config.connector.type
    try:
        connector_class = getattr(connectors, connector_type)
        return connector_class(config, **config.connector.options)

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

    # Create output folder.
    results_folder = config.results_folder
    if os.path.exists(results_folder):
        print(f"WARNING: Output folder {results_folder} already exists...")
    os.makedirs(results_folder, exist_ok=True)

    # Copy the config file from config_path to the output folder.
    config_file_name = os.path.basename(config_path)
    shutil.copy2(config_path, os.path.join(results_folder, config_file_name))

    # If connector configured - run it to generate results (or read results)
    connector = get_connector(config)
    if connector:
        connector.fetch_data()

    # Run evaluation
    evaluator = get_evaluator(config)
    answer_path = os.path.join(config.results_folder, config.generated_answers)
    rag_results = RAGResultsLoader(answer_path).load()
    scored_results = evaluator.evaluate_batch(rag_results)

    eval_results_path = os.path.join(results_folder, config.eval_results_file)
    data_classes.eval_scores.to_csv(scored_results, eval_results_path)

    # Plot the metrics.
    evaluator.plot_metrics(
        csv_files=[eval_results_path],
        output_file=os.path.join(results_folder, config.metrics_file),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="eval_config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    run_eval(args.config)
