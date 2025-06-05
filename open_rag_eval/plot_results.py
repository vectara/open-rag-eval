"""This script is useful for visualizing the results of RAG evaluators.

It can be used to plot results from multiple runs of different evaluators on the same plot
allowing for a comparative analysis of the results. This script reads CSV files
containing evaluation metrics and generates plots for the specified metrics.

Usage:
    python plot_results.py --evaluator trec metrics_1.csv metrics_2.csv
    python plot_results.py --evaluator consistency metrics_1.csv metrics_2.csv

The plot is saved by default to a 'metrics_comparison.png' file.
"""

import argparse
from typing import List
from open_rag_eval.evaluators.trec_evaluator import TRECEvaluator
from open_rag_eval.evaluators.consistency_evaluator import ConsistencyEvaluator

def plot_metrics(evaluator_type: str, csv_files: List[str], output_file: str):
    """
    Plot metrics based on the evaluator type.

    Args:
        evaluator_type: Type of evaluator ('trec', 'consistency', etc.)
        csv_files: List of CSV files containing metrics
        output_file: Output file path for the generated plot
    """
    evaluator_type = evaluator_type.lower()

    if evaluator_type == 'trec':
        TRECEvaluator.plot_metrics(csv_files=csv_files, output_file=output_file)
    elif evaluator_type == 'consistency':
        ConsistencyEvaluator.plot_metrics(csv_files=csv_files, output_file=output_file)
    else:
        raise ValueError(f"Unsupported evaluator type: {evaluator_type}")

def main():
    """
    Main function to parse command line arguments and plot metrics.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plot metrics from CSV files')
    parser.add_argument(
        'csv_files',
        nargs='+',
        help='List of CSV files to process'
    )
    parser.add_argument(
        '-e', '--evaluator',
        required=True,
        choices=['trec', 'consistency'],
        help='Type of evaluator to use for plotting'
    )
    parser.add_argument(
        '-o', '--output',
        default='metrics_comparison.png',
        help='Output filename for the graph (default: metrics_comparison.png)'
    )

    # Parse arguments
    args = parser.parse_args()

    # Plot the metrics using the appropriate evaluator
    plot_metrics(args.evaluator, args.csv_files, args.output)

if __name__ == '__main__':
    main()