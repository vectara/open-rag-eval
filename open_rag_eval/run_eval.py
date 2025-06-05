"""
This script evaluates the performance of a retrieval-augmented generation (RAG) system.
"""
import os
import logging
import shutil
import pandas as pd
from typing import Any, Dict
import argparse

from pathlib import Path
from omegaconf import OmegaConf, ListConfig

from open_rag_eval import connectors, models
from open_rag_eval import evaluators
from open_rag_eval.rag_results_loader import RAGResultsLoader


def get_evaluator(evaluator_config: Dict[str, Any]) -> evaluators.Evaluator:
    """
    Dynamically import and instantiate an evaluator class based on configuration.
    Supports evaluators that require a model and those that don't.

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

        # Get options if they exist
        options = evaluator_config.options if hasattr(evaluator_config, "options") else None

        # Check if model config exists in the evaluator_config
        has_model_config = hasattr(evaluator_config, "model")

        if has_model_config:
            # Create the model instance based on config
            model_config = evaluator_config.model
            model_class = getattr(models, model_config.type)

            # Verify it's a subclass of LLMJudgeModel
            if not issubclass(model_class, models.LLMJudgeModel):
                raise TypeError(f"{model_config.type} is not a subclass of LLMJudgeModel")

            # Instantiate the model with config parameters
            model = model_class(model_options=model_config)

            # Instantiate the evaluator with the model
            return evaluator_class(model=model, options=options)
        else:
            # Instantiate without the model parameter
            return evaluator_class(options=options)

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



def merge_eval_results(results_folder, config):
    """
    Merge evaluation results from multiple evaluators into a single CSV.
    Takes all columns from TREC evaluator and only specific metrics from other evaluators.

    Args:
        results_folder: Path to folder containing evaluation results
        config: Configuration object containing file names
    """
    # Find TREC evaluator results file (assumed to always exist)
    trec_file = os.path.join(results_folder, f"TRECEvaluator-{config.eval_results_file}")
    if not os.path.exists(trec_file):
        logging.error(f"TREC evaluation file not found: {trec_file}")
        return

    # Start with TREC evaluator data as base
    merged_df = pd.read_csv(trec_file)
    logging.info(f"Using TREC evaluator data as base: {trec_file}")

    # Dictionary mapping evaluator types to columns we want to extract
    columns_to_extract = {
        "ConsistencyEvaluator": ["bert", "rouge"]
        # Add more mappings here for other evaluator types
    }

    # Find all other evaluator result files
    for evaluator_type, columns in columns_to_extract.items():
        evaluator_file = os.path.join(results_folder, f"{evaluator_type}-{config.eval_results_file}")
        if not os.path.exists(evaluator_file):
            logging.info(f"No {evaluator_type} file found: {evaluator_file}")
            continue

        try:
            # Load the evaluator's data
            evaluator_df = pd.read_csv(evaluator_file)

            # Determine merge key (query_id preferred, fall back to query)
            if "query_id" in merged_df.columns and "query_id" in evaluator_df.columns:
                merge_key = "query_id"
            elif "query" in merged_df.columns and "query" in evaluator_df.columns:
                merge_key = "query"
            else:
                logging.warning(f"No common key found to merge with {evaluator_type}")
                continue

            # Select only the columns we want from this evaluator
            columns_to_merge = [col for col in columns if col in evaluator_df.columns]
            columns_to_merge.append(merge_key)  # Include merge key

            # Merge with the base dataframe
            if columns_to_merge:
                evaluator_subset = evaluator_df[columns_to_merge]
                merged_df = pd.merge(merged_df, evaluator_subset, on=merge_key, how='left')
                logging.info(f"Merged {len(columns_to_merge) - 1} columns from {evaluator_type}")
        except Exception as e:
            logging.error(f"Error merging data from {evaluator_type}: {str(e)}")

    # Save merged dataframe
    output_file = os.path.join(results_folder, config.eval_results_file)
    merged_df.to_csv(output_file, index=False)
    logging.info(f"Merged evaluation results saved to {output_file}")

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

    # Merge results from all evaluators into a single CSV file
    # Assuming TRECEvaluator is always present as the base
    merge_eval_results(results_folder, config)
    print(f"Created consolidated results file: {config.eval_results_file}")

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
