"""
Command-line interface for Open RAG Eval.
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

from open_rag_eval.run_eval import run_eval
from open_rag_eval.plot_results import plot_metrics
from open_rag_eval.run_query_generation import run_query_generation


def main():
    """Main CLI entry point that dispatches to subcommands."""
    # Load .env file if it exists in the current directory
    dotenv_path = os.path.join(os.getcwd(), '.env')
    load_dotenv(dotenv_path=dotenv_path, override=False)

    parser = argparse.ArgumentParser(
        description="Open RAG Eval - A toolkit for evaluating RAG systems",
        prog="open-rag-eval",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Run RAG evaluation")
    eval_parser.add_argument(
        "--config",
        type=str,
        default="eval_config.yaml",
        help="Path to configuration file (default: eval_config.yaml)",
    )

    # Plot subcommand
    plot_parser = subparsers.add_parser(
        "plot", help="Plot metrics from evaluation results"
    )
    plot_parser.add_argument(
        "csv_files",
        nargs="+",
        help="Path(s) to CSV file(s) containing evaluation metrics",
    )
    plot_parser.add_argument(
        "--evaluator",
        type=str,
        required=True,
        choices=["trec", "consistency", "golden_answer"],
        help="Type of evaluator used to generate the CSV files",
    )
    plot_parser.add_argument(
        "--output-file",
        type=str,
        default="metrics_comparison.png",
        help="Output file name for the plot (default: metrics_comparison.png)",
    )
    plot_parser.add_argument(
        "--metrics-to-plot",
        nargs="+",
        help="Specific metric column names to plot. If not specified, all metrics will be plotted.",
    )

    # Generate-queries subcommand
    gen_parser = subparsers.add_parser(
        "generate-queries", help="Generate synthetic queries from documents"
    )
    gen_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to query generation configuration file",
    )
    gen_parser.add_argument(
        "--output",
        type=str,
        help="Output file path (overrides config)",
    )
    gen_parser.add_argument(
        "--num-queries",
        type=int,
        help="Number of queries to generate (overrides config)",
    )
    gen_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration and document count without generating",
    )
    gen_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.command == "eval":
        run_eval(args.config)
    elif args.command == "plot":
        plot_metrics(
            evaluator_type=args.evaluator,
            csv_files=args.csv_files,
            output_file=args.output_file,
            metrics_to_plot=args.metrics_to_plot,
        )
    elif args.command == "generate-queries":
        # Configure logging based on verbose flag
        if args.verbose:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Suppress noisy third-party loggers to prevent progress bar interruption
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('openai').setLevel(logging.WARNING)
        logging.getLogger('anthropic').setLevel(logging.WARNING)

        run_query_generation(
            config_path=args.config,
            output_file=args.output,
            num_queries=args.num_queries,
            dry_run=args.dry_run,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
