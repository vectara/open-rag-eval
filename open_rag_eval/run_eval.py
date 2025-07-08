"""
This script evaluates the performance of a retrieval-augmented generation (RAG) system.
"""

import argparse
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from pandas.errors import EmptyDataError
from omegaconf import OmegaConf, ListConfig, DictConfig

from open_rag_eval import connectors, models
from open_rag_eval import evaluators
from open_rag_eval.rag_results_loader import RAGResultsLoader
from open_rag_eval.utils.constants import CONSISTENCYEVALUATOR, CONSISTENCY
from open_rag_eval._version import __version__


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
        options = (
            evaluator_config.options if hasattr(evaluator_config, "options") else None
        )
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
                    f"{model_config.type} is not a subclass of LLMJudgeModel"
                )

            # Instantiate the model with config parameters
            model = model_class(model_options=model_config)

            # Instantiate the evaluator with the model
            return evaluator_class(model=model, options=options)

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
        evaluator_file = os.path.join(
            results_folder, f"{evaluator_type}-{config.eval_results_file}"
        )
        if not os.path.exists(evaluator_file):
            logging.warning(
                f"{evaluator_type} file not found: {evaluator_file}. Skipping."
            )
            continue

        try:
            evaluator_df = pd.read_csv(evaluator_file)

            # Determine merge key once
            if merge_key is None:
                if "query_id" in evaluator_df.columns:
                    merge_key = "query_id"
                else:
                    logging.warning(
                        f"'query_id' not found in {evaluator_type}. Skipping."
                    )
                    continue

            # Enforce that user-provided columns include the merge key
            desired_columns = set(columns)
            if merge_key not in desired_columns:
                logging.warning(
                    f"Merge key '{merge_key}' not included in columns for '{evaluator_type}'. Skipping."
                )
                continue

            if merge_key not in evaluator_df.columns:
                logging.warning(
                    f"Merge key '{merge_key}' not found in {evaluator_type} file. Skipping."
                )
                continue

            available_columns = set(evaluator_df.columns)
            existing_columns = (
                set(merged_df.columns) if merged_df is not None else set()
            )

            # Keep only desired & available columns, skip those already in merged_df (except merge_key)
            columns_to_merge = list(
                (desired_columns & available_columns) - (existing_columns - {merge_key})
            )

            if not columns_to_merge:
                logging.info(
                    f"No new columns to merge from {evaluator_type}. Skipping."
                )
                continue

            evaluator_subset = evaluator_df[columns_to_merge]

            if merged_df is None:
                merged_df = evaluator_subset
            else:
                merged_df = pd.merge(
                    merged_df, evaluator_subset, on=merge_key, how="left"
                )

        except Exception as e:
            logging.error(f"Error processing {evaluator_type}: {str(e)}")

    if merged_df is not None:
        # Reorder 'query' and 'query_id' to appear first if available
        priority = [col for col in ["query", "query_id"] if col in merged_df.columns]
        remaining = [col for col in merged_df.columns if col not in priority]
        merged_df = merged_df[priority + remaining]

        output_file = os.path.join(results_folder, config.eval_results_file)
        merged_df.to_csv(output_file, index=False)
        print(f"Merged evaluation results saved to {output_file}")
    else:
        logging.warning("No data was merged. Final CSV not created.")


def create_openeval_report(results_folder, eval_results_file):
    csv_file = os.path.join(results_folder, eval_results_file)
    json_report_name = f'{".".join(eval_results_file.split(".")[:-1])}.json'
    json_path = os.path.join(results_folder, json_report_name)

    df = pd.read_csv(csv_file)

    # Identify run-based prefixes
    run_prefixes = {"_".join(col.split("_")[:2]) for col in df.columns if col.startswith("run_")}

    # Identify consistency metric columns
    consistency_cols = [col for col in df.columns if col.startswith(CONSISTENCY)]

    # Build structured JSON output
    structured_output = []

    for _, row in df.iterrows():
        entry = {
            "query_id": row["query_id"],
            "query": row["query"],
            "runs": [],
            "consistency": {},
        }

        # Extract each run
        for prefix in sorted(run_prefixes):
            run_data = {}
            for col in df.columns:
                if col.startswith(prefix):
                    field = col[len(prefix) + 1 :] if col != prefix else col
                    try:
                        run_data[field] = json.loads(row[col])
                    except (json.JSONDecodeError, TypeError):
                        run_data[field] = row[col]

            if run_data:
                entry["runs"].append(run_data)

        # Extract consistency fields
        for col in consistency_cols:
            metric_name = "_".join(col.split("_")[1:])
            entry["consistency"][metric_name] = json.loads(row[col])

        structured_output.append(entry)

    # Wrap in outer object for versioning
    json_output = {
        "version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "evaluation": structured_output,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2)
    print(f"Open Evaluation json report saved to {json_path}")


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
    precomputed_metric_scores_by_query = {}

    # Normalize to list
    evaluator_configs = (
        config.evaluator
        if isinstance(config.evaluator, ListConfig)
        else [config.evaluator]
    )

    # Separate consistency evaluator if present
    evaluator_configs_filtered = []
    consistency_eval_config = None
    for eval_config in evaluator_configs:
        if eval_config.type == CONSISTENCYEVALUATOR:
            consistency_eval_config = eval_config
        else:
            evaluator_configs_filtered.append(eval_config)

    # Append consistency evaluator last, if it exists
    if consistency_eval_config:
        evaluator_configs_filtered.append(consistency_eval_config)

    # Run all evaluators
    for eval_config in evaluator_configs_filtered:
        evaluator_type = eval_config.type
        evaluator = get_evaluator(eval_config)

        # Evaluate (pass precomputed scores only for consistency evaluator)
        if evaluator_type == CONSISTENCYEVALUATOR:
            results = evaluator.evaluate_batch(
                rag_results,
                precomputed_metric_scores_by_query=precomputed_metric_scores_by_query,
            )
        else:
            results = evaluator.evaluate_batch(rag_results)
            if getattr(eval_config, "options", {}).get("run_consistency", False):
                precomputed_metric_scores_by_query = (
                    evaluator.collect_scores_for_consistency(
                        results, precomputed_metric_scores_by_query
                    )
                )

        # Save results
        eval_results_path = os.path.join(
            results_folder, f"{evaluator_type}-{config.eval_results_file}"
        )
        evaluator.to_csv(results, eval_results_path)

        # Plot results
        try:
            df = pd.read_csv(eval_results_path)
            if df.empty:
                logging.warning(f"Skipping plot: {eval_results_path} is empty.")
                continue

            per_evaluator_columns[evaluator_type] = evaluator.get_consolidated_columns()
            evaluator.plot_metrics(
                csv_files=[eval_results_path],
                output_file=os.path.join(
                    results_folder, f"{evaluator_type}-{config.metrics_file}"
                ),
                metrics_to_plot=evaluator.get_metrics_to_plot(),
            )
            print(
                f"Graph saved to {os.path.join(results_folder, f'{evaluator_type}-{config.metrics_file}')}"
            )
        except (FileNotFoundError, EmptyDataError):
            logging.warning(f"Skipping plot: {eval_results_path} not found or empty.")
        except Exception as e:
            logging.exception(
                f"Failed to read or plot metrics from {eval_results_path}: {str(e)}"
            )

    # Merge results from all evaluators into a single CSV file
    merge_eval_results(
        results_folder, config, per_evaluator_columns=per_evaluator_columns
    )

    create_openeval_report(results_folder, config.eval_results_file)


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
