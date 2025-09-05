"""
Command-line interface for Open RAG Eval.
"""

import argparse
import sys

from open_rag_eval.run_eval import run_eval
from open_rag_eval.plot_results import plot_metrics


def main():
    """Main CLI entry point that dispatches to subcommands."""
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
        choices=["trec", "consistency"],
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
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
