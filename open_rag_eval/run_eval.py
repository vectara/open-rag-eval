"""
This script evaluates the performance of a retrieval-augmented generation (RAG) system.
"""
import shutil
from pandas.errors import EmptyDataError
from typing import Any, Dict
import argparse

import os
import logging
import pandas as pd

from pathlib import Path
from omegaconf import OmegaConf, ListConfig, DictConfig

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
        options = evaluator_config.options if hasattr(evaluator_config,
                                                      "options") else None
        if isinstance(options, DictConfig):
            options = OmegaConf.to_container(options, resolve=True)

        # Check if model config exists in the evaluator_config
        has_model_config = hasattr(evaluator_config, "model")

        if has_model_config:
            # Create the model instance based on config
            model_config = evaluator_config.model
            model_class = getattr(models, model_config.type)

            # Verify it's a subclass of LLMJudgeModel
            if not issubclass(model_class, models.LLMJudgeModel):
                raise TypeError(
                    f"{model_config.type} is not a subclass of LLMJudgeModel")

            # Instantiate the model with config parameters
            model = model_class(model_options=model_config)

            # Instantiate the evaluator with the model
            return evaluator_class(model=model, options=options)
        else:
            # Instantiate without the model parameter
            return evaluator_class(options=options)

    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Could not load evaluator {evaluator_type}: {str(e)}") from e


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
        raise ImportError(
            f"Could not load connector {connector_type}: {str(e)}") from e



def merge_eval_results(results_folder, config, per_evaluator_columns=None):
    """
    Merge evaluation results from multiple evaluators into a single CSV file using 'query_id' as the join key.
    Requires the user to include the merge key in per_evaluator_columns for each evaluator.

    Args:
        results_folder: Path to folder containing evaluation results.
        config: Configuration object containing output file names (expects 'eval_results_file' attribute).
        per_evaluator_columns: Dict mapping evaluator types to lists of column names to include (must include query_id).

    Returns:
        None: Merged CSV is saved to disk as specified in config.eval_results_file.
    """
    if not per_evaluator_columns:
        logging.warning("No evaluator columns provided for merging.")
        return

    merged_df = None
    merge_key = None
    print("Merging evaluation results...")

    for evaluator_type, columns in per_evaluator_columns.items():
        evaluator_file = os.path.join(results_folder, f"{evaluator_type}-{config.eval_results_file}")
        if not os.path.exists(evaluator_file):
            logging.warning(f"{evaluator_type} file not found: {evaluator_file}. Skipping.")
            continue

        try:
            evaluator_df = pd.read_csv(evaluator_file)

            # Determine merge key once
            if merge_key is None:
                if "query_id" in evaluator_df.columns:
                    merge_key = "query_id"
                else:
                    logging.warning(f"'query_id' not found in {evaluator_type}. Skipping.")
                    continue

            # Enforce that user-provided columns include the merge key
            desired_columns = set(columns)
            if merge_key not in desired_columns:
                logging.warning(f"Merge key '{merge_key}' not included in columns for '{evaluator_type}'. Skipping.")
                continue

            if merge_key not in evaluator_df.columns:
                logging.warning(f"Merge key '{merge_key}' not found in {evaluator_type} file. Skipping.")
                continue

            available_columns = set(evaluator_df.columns)
            existing_columns = set(merged_df.columns) if merged_df is not None else set()

            # Keep only desired & available columns, skip those already in merged_df (except merge_key)
            columns_to_merge = list((desired_columns & available_columns) - (existing_columns - {merge_key}))

            if not columns_to_merge:
                logging.info(f"No new columns to merge from {evaluator_type}. Skipping.")
                continue

            evaluator_subset = evaluator_df[columns_to_merge]

            if merged_df is None:
                merged_df = evaluator_subset
            else:
                merged_df = pd.merge(merged_df, evaluator_subset, on=merge_key, how='left')

        except Exception as e:
            logging.error(f"Error processing {evaluator_type}: {str(e)}")

    if merged_df is not None:
        # Reorder 'query' and 'query_id' to appear first if available
        priority = [col for col in ['query', 'query_id'] if col in merged_df.columns]
        remaining = [col for col in merged_df.columns if col not in priority]
        merged_df = merged_df[priority + remaining]

        output_file = os.path.join(results_folder, config.eval_results_file)
        merged_df.to_csv(output_file, index=False)
        print(f"Merged evaluation results saved to {output_file}")
    else:
        logging.warning("No data was merged. Final CSV not created.")


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
    per_evaluator_columns = {}
    evaluator_configs = config.evaluator if isinstance(
        config.evaluator, ListConfig) else [config.evaluator]
    for eval_config in evaluator_configs:
        evaluator_type = eval_config.type
        evaluator = get_evaluator(eval_config)
        scored_results = evaluator.evaluate_batch(rag_results)
        eval_results_file = f"{evaluator_type}-{config.eval_results_file}"
        eval_results_path = os.path.join(results_folder, eval_results_file)
        evaluator.to_csv(scored_results, eval_results_path)

        try:
            df = pd.read_csv(eval_results_path)
            if not df.empty:
                per_evaluator_columns[evaluator_type] = evaluator.get_consolidated_columns()
                metrics_to_plot = evaluator.get_metrics_to_plot()
                metrics_file = f'{evaluator_type}-{config.metrics_file}'
                evaluator.plot_metrics(
                    csv_files=[eval_results_path],
                    output_file=os.path.join(results_folder, metrics_file),
                    metrics_to_plot=metrics_to_plot
                )
            else:
                logging.warning(f"Skipping plot: {eval_results_path} is empty.")
        except FileNotFoundError:
            logging.warning(f"CSV file not found: {eval_results_path}")
        except EmptyDataError:
            logging.warning(f"Skipping plot: {eval_results_path} is completely empty (no header, no data).")
        except Exception as e:
            logging.exception(f"Failed to read or plot metrics from {eval_results_path}: {str(e)}")

    merge_eval_results(results_folder,
                       config,
                       per_evaluator_columns=per_evaluator_columns)


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
