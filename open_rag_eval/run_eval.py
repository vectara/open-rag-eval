"""
This script evaluates the performance of a retrieval-augmented generation (RAG) system.
"""

from typing import Any, Dict
import argparse
import os
import shutil

from pathlib import Path
from omegaconf import OmegaConf, ListConfig

from open_rag_eval import connectors, data_classes, models
from open_rag_eval import evaluators
from open_rag_eval.rag_results_loader import RAGResultsLoader


def get_evaluator(evaluator_config: Dict[str, Any]) -> evaluators.Evaluator:
    """
    Dynamically import and instantiate an evaluator class based on configuration.

    Args:
        evaluator_config: Configuration dictionary containing evaluator settings

    Returns:
        An instance of the specified evaluator
    """
    evaluator_type = evaluator_config.type
    try:
        evaluator_class = getattr(evaluators, f"{evaluator_type}")

        # Verify it's a subclass of Evaluator
        if not issubclass(evaluator_class, evaluators.Evaluator):
            raise TypeError(f"{evaluator_type} is not a subclass of Evaluator")

        # Create the model instance based on config
        model_config = evaluator_config.model
        model_class = getattr(models, model_config.type)

        # Verify it's a subclass of LLMJudgeModel
        if not issubclass(model_class, models.LLMJudgeModel):
            raise TypeError(f"{model_config.type} is not a subclass of LLMJudgeModel")

        # Instantiate the model with config parameters
        model = model_class(model_options = model_config)

        # Instantiate the evaluator with the model
        options = evaluator_config.options if hasattr(evaluator_config, "options") else None
        return evaluator_class(model=model, options=options)

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

    answer_path = os.path.join(results_folder, config.generated_answers)
    rag_results = RAGResultsLoader(answer_path).load()

    # Run evaluation
    evaluator_configs = config.evaluator if isinstance(config.evaluator, ListConfig) else [config.evaluator]
    for eval_config in evaluator_configs:
        evaluator_type = eval_config.type
        evaluator = get_evaluator(eval_config)
        scored_results = evaluator.evaluate_batch(rag_results)
        eval_results_file = f"{evaluator_type}-{config.eval_results_file}"
        eval_results_path = os.path.join(results_folder, eval_results_file)
        evaluator.to_csv(scored_results, eval_results_path)

        # Plot individual metrics for this evaluator
        metrics_file = f'{evaluator_type}-{config.metrics_file}'
        evaluator.plot_metrics(
            csv_files=[eval_results_path],
            output_file=os.path.join(results_folder, metrics_file),
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
