"""
This script evaluates the performance of a retrieval-augmented generation (RAG) system.
"""

import inspect
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf
from pandas.errors import EmptyDataError

from open_rag_eval import connectors, evaluators, models
from open_rag_eval._version import __version__
from open_rag_eval.chunking import ChunkingStrategy, parse_chunking_strategies
from open_rag_eval import chunking_comparison
from open_rag_eval.rag_results_loader import RAGResultsLoader
from open_rag_eval.utils.constants import CONSISTENCY, CONSISTENCYEVALUATOR


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
        has_embedding_config = hasattr(evaluator_config, "embedding_model")

        # Special handling for GoldenAnswerEvaluator (requires both LLM and embedding model)
        if evaluator_type == "GoldenAnswerEvaluator":
            if not has_model_config:
                raise ValueError("GoldenAnswerEvaluator requires 'model' configuration")
            if not has_embedding_config:
                raise ValueError("GoldenAnswerEvaluator requires 'embedding_model' configuration")

            # Create LLM model
            model_config = evaluator_config.model
            model_class = getattr(models, model_config.type)
            if not issubclass(model_class, models.LLMJudgeModel):
                raise TypeError(f"{model_config.type} is not a subclass of LLMJudgeModel")
            llm_model = model_class(model_options=model_config)

            # Create embedding model
            emb_config = evaluator_config.embedding_model
            emb_class = getattr(models, emb_config.type)
            if not issubclass(emb_class, models.EmbeddingModel):
                raise TypeError(f"{emb_config.type} is not a subclass of EmbeddingModel")
            embedding_model = emb_class(model_options=emb_config)

            return evaluator_class(
                llm_model=llm_model,
                embedding_model=embedding_model,
                options=options
            )

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


def get_connector(
    config: Dict[str, Any],
    chunking_strategy: Optional[ChunkingStrategy] = None,
    output_filename: Optional[str] = None,
) -> connectors.Connector:
    """
    Dynamically import and instantiate a connector class based on configuration.

    Args:
        config: Configuration dictionary containing connector settings
        chunking_strategy: Optional chunking strategy to apply. Only forwarded to
            connectors whose constructor accepts a ``chunking_strategy`` argument.
        output_filename: Optional override for the generated-answers filename.
            Only forwarded to connectors that accept an ``output_filename`` argument.

    Returns:
        An instance of the specified connector
    """
    if "connector" not in config:
        return None
    connector_type = config.connector.type
    try:
        connector_class = getattr(connectors, connector_type)

        # Only forward chunking-related kwargs to connectors that support them,
        # keeping connectors like VectaraConnector untouched.
        params = inspect.signature(connector_class.__init__).parameters
        extra = {}
        if chunking_strategy is not None and "chunking_strategy" in params:
            extra["chunking_strategy"] = chunking_strategy
        if output_filename is not None and "output_filename" in params:
            extra["output_filename"] = output_filename

        return connector_class(config, **config.connector.options, **extra)

    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not load connector {connector_type}: {str(e)}") from e


def _connector_supports_chunking(config: Dict[str, Any]) -> bool:
    """Return True if the configured connector accepts a chunking_strategy."""
    if "connector" not in config:
        return False
    try:
        connector_class = getattr(connectors, config.connector.type)
    except AttributeError:
        return False
    return "chunking_strategy" in inspect.signature(connector_class.__init__).parameters


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
    run_prefixes = {
        "_".join(col.split("_")[:2]) for col in df.columns if col.startswith("run_")
    }

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
            try:
                entry["consistency"][metric_name] = json.loads(row[col])
            except (json.JSONDecodeError, TypeError, ValueError):
                # Handle NaN or invalid JSON (e.g., when metric failed)
                entry["consistency"][metric_name] = None

        structured_output.append(entry)

    # Wrap in outer object for versioning
    json_output = {
        "version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "evaluation": structured_output,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_omit_empty_consistency(json_output), f, indent=2)
    print(f"Open Evaluation json report saved to {json_path}")


def _omit_empty_consistency(report: dict) -> dict:
    """Return a copy of `report` with the 'consistency' key removed if its value is falsy.

    The returned dictionary excludes 'consistency' when present but empty or None.
    All other keys and values are preserved.
    """
    return {k: v for k, v in report.items() if k != "consistency" or v}


def _print_token_usage_summary(results, evaluator_type: str):
    """
    Print a summary of token usage across all evaluation results.

    Args:
        results: List of MultiScoredRAGResult objects
        evaluator_type: Type of evaluator (e.g., "TRECEvaluator", "GoldenAnswerEvaluator")
    """
    total_input = 0
    total_output = 0

    # Define metric token keys based on evaluator type
    if evaluator_type == "GoldenAnswerEvaluator":
        metric_keys = {
            "factual_correctness": "factual_correctness_tokens",
        }
    else:
        # Default to TREC evaluator metrics
        metric_keys = {
            "umbrela": "umbrela_tokens",
            "autonugget": "autonugget_tokens",
            "citation": "citation_tokens",
            "no_answer": "no_answer_tokens",
        }

    metric_tokens = {name: {"input": 0, "output": 0} for name in metric_keys}

    # Aggregate tokens from all results
    for multi_scored_result in results:
        if not hasattr(multi_scored_result, 'scored_rag_results'):
            continue
        for scored_result in multi_scored_result.scored_rag_results:
            if not scored_result.scores or not scored_result.scores.generation_score:
                continue

            token_usage = scored_result.scores.generation_score.scores.get("token_usage", {})
            if not token_usage:
                continue

            # Add to total
            total_input += token_usage.get("total_input_tokens", 0)
            total_output += token_usage.get("total_output_tokens", 0)

            # Add to metric-specific counts
            for metric_name, token_key in metric_keys.items():
                metric_data = token_usage.get(token_key, {})
                metric_tokens[metric_name]["input"] += metric_data.get("input_tokens", 0)
                metric_tokens[metric_name]["output"] += metric_data.get("output_tokens", 0)

    total_tokens = total_input + total_output

    # Print summary if there are tokens to report
    if total_tokens > 0:
        print(f"\n=== Token Usage Summary ({evaluator_type}) ===")
        print(f"Total Input Tokens:  {total_input:,}")
        print(f"Total Output Tokens: {total_output:,}")
        print(f"Total Tokens:        {total_tokens:,}")
        print("\nBreakdown by Metric:")

        for metric_name, tokens in metric_tokens.items():
            metric_total = tokens["input"] + tokens["output"]
            if metric_total > 0:
                percentage = metric_total / total_tokens * 100
                print(f"  {metric_name.upper():20} {metric_total:8,} tokens ({percentage:5.1f}%)")

        print("=" * 50 + "\n")


def _run_evaluators(config, results_folder, rag_results, file_suffix="", plot=True):
    """Run all configured evaluators over `rag_results` and write per-evaluator CSVs.

    Args:
        config: Loaded evaluation config.
        results_folder: Folder to write evaluator CSVs and plots into.
        rag_results: List of MultiRAGResult to evaluate.
        file_suffix: String inserted before the eval_results_file name, used by
            the chunking-comparison layer to tag per-strategy output files
            (e.g. "small-" -> "TRECEvaluator-small-results.csv"). Empty by default
            for the standard single-pass behavior.
        plot: When True, write per-evaluator metric plots.

    Returns:
        A tuple (per_evaluator_columns, results_paths) where per_evaluator_columns
        maps evaluator type -> consolidated column list (for merging) and
        results_paths maps evaluator type -> path of its written results CSV.
    """
    per_evaluator_columns = {}
    results_paths = {}
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
            results_folder, f"{evaluator_type}-{file_suffix}{config.eval_results_file}"
        )
        evaluator.to_csv(results, eval_results_path)
        results_paths[evaluator_type] = eval_results_path

        # Print token usage summary
        _print_token_usage_summary(results, evaluator_type)

        # Plot results
        try:
            df = pd.read_csv(eval_results_path)
            if df.empty:
                logging.warning(f"Skipping plot: {eval_results_path} is empty.")
                continue

            per_evaluator_columns[evaluator_type] = evaluator.get_consolidated_columns()
            if plot:
                metrics_plot_path = os.path.join(
                    results_folder, f"{evaluator_type}-{file_suffix}{config.metrics_file}"
                )
                evaluator.plot_metrics(
                    csv_files=[eval_results_path],
                    output_file=metrics_plot_path,
                    metrics_to_plot=evaluator.get_metrics_to_plot(),
                )
                print(f"Graph saved to {metrics_plot_path}")
        except (FileNotFoundError, EmptyDataError):
            logging.warning(f"Skipping plot: {eval_results_path} not found or empty.")
        except Exception as e:
            logging.exception(
                f"Failed to read or plot metrics from {eval_results_path}: {str(e)}"
            )

    return per_evaluator_columns, results_paths


def _load_queries_df(config):
    """Load the queries CSV (with optional golden answers), or None if absent."""
    if hasattr(config, 'input_queries') and config.input_queries:
        queries_path = config.input_queries
        if os.path.exists(queries_path):
            queries_df = pd.read_csv(queries_path)
            if 'expected_answer' in queries_df.columns:
                num_golden = queries_df['expected_answer'].notna().sum()
                print(f"Loaded {num_golden} golden answers from {queries_path}")
            return queries_df
    return None


def _run_chunking_comparison(config, results_folder, queries_df):
    """Run the pipeline once per chunking strategy and report the best one.

    For each strategy, the connector re-indexes the documents with that strategy
    and writes a strategy-tagged answers CSV; evaluators then produce
    strategy-tagged result CSVs. Strategies are ranked by mean UMBRELA retrieval
    score, and a comparison plot + JSON report + console summary are emitted.
    """
    strategies = parse_chunking_strategies(config.chunking)
    print(f"Comparing {len(strategies)} chunking strategies: "
          f"{', '.join(s.name for s in strategies)}")

    strategy_results = []
    trec_csvs = {}
    for strategy in strategies:
        print(f"\n--- Chunking strategy: {strategy.name} "
              f"(chunk_size={strategy.chunk_size}, chunk_overlap={strategy.chunk_overlap}) ---")
        answers_filename = f"answers_{strategy.name}.csv"

        # Re-index and generate answers for this strategy.
        connector = get_connector(
            config, chunking_strategy=strategy, output_filename=answers_filename
        )
        connector.fetch_data()

        answer_path = os.path.join(results_folder, answers_filename)
        rag_results = RAGResultsLoader(answer_path, queries_df=queries_df).load()

        # Evaluate this strategy; skip per-strategy plots to reduce clutter.
        _, results_paths = _run_evaluators(
            config, results_folder, rag_results,
            file_suffix=f"{strategy.name}-", plot=False,
        )

        trec_path = results_paths.get("TRECEvaluator")
        strategy_results.append({
            "name": strategy.name,
            "chunk_size": strategy.chunk_size,
            "chunk_overlap": strategy.chunk_overlap,
            "results_file": trec_path,
        })
        if trec_path:
            trec_csvs[strategy.name] = trec_path

    # Rank strategies and emit comparison artifacts.
    ranked = chunking_comparison.rank_strategies(strategy_results)
    chunking_comparison.print_summary(ranked)
    json_path = chunking_comparison.write_comparison(results_folder, ranked, __version__)
    print(f"Chunking comparison report saved to {json_path}")

    # Comparison plot: reuse the TREC grouped-boxplot path with one CSV per strategy.
    if len(trec_csvs) >= 1:
        try:
            plot_path = os.path.join(
                results_folder, chunking_comparison.COMPARISON_PLOT_FILENAME
            )
            evaluators.TRECEvaluator.plot_metrics(
                csv_files=[trec_csvs[s["name"]] for s in ranked if s["name"] in trec_csvs],
                output_file=plot_path,
                metrics_to_plot=[
                    chunking_comparison.RANKING_METRIC,
                    "generation_score_vital_nuggetizer_score",
                ],
            )
            print(f"Chunking comparison plot saved to {plot_path}")
        except Exception as e:
            logging.exception(f"Failed to plot chunking comparison: {str(e)}")


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

    # Load queries with expected_answer (golden answers) if available
    queries_df = _load_queries_df(config)

    # Chunking-comparison mode: run the pipeline once per chunking strategy and
    # report the best one. Only supported for connectors that control document
    # chunking (LangChain, LlamaIndex).
    if "chunking" in config and config.chunking:
        if _connector_supports_chunking(config):
            _run_chunking_comparison(config, results_folder, queries_df)
            return
        logging.warning(
            "A 'chunking' block was configured but the connector does not support "
            "chunking strategies. Running a single standard evaluation instead."
        )

    # Standard single-pass evaluation.
    # If connector configured - run it to generate results (or read results)
    connector = get_connector(config)
    if connector:
        connector.fetch_data()

    answer_path = os.path.join(results_folder, config.generated_answers)
    rag_results = RAGResultsLoader(answer_path, queries_df=queries_df).load()

    # Run evaluation
    per_evaluator_columns, _ = _run_evaluators(
        config, results_folder, rag_results
    )

    # Merge results from all evaluators into a single CSV file
    merge_eval_results(
        results_folder, config, per_evaluator_columns=per_evaluator_columns
    )

    create_openeval_report(results_folder, config.eval_results_file)


def main():
    """CLI entry point for standalone execution.

    This function maintains backwards compatibility by redirecting to the main CLI.
    It prepends 'eval' to sys.argv to invoke the correct subcommand.
    """
    import sys  # pylint: disable=import-outside-toplevel,reimported
    # Redirect to the main CLI with the eval subcommand
    sys.argv.insert(1, 'eval')
    from open_rag_eval.cli import main as cli_main  # pylint: disable=import-outside-toplevel,cyclic-import
    cli_main()


if __name__ == "__main__":
    main()
