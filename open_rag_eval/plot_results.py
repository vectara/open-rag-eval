"""This script is useful for visualizing the results of the TREC RAG evaluator.

It can be used to plot results from multiple runs of the evaluator on the same plot
allowing for a comparative analysis of the results. This script reads CSV files
containing evaluation metrics and generates bar plots for the specified metrics.

Usage:
    python plot_results.py metrics_1.csv metrics_2.csv

The plot is saved by default to a 'metrics_comparison.png' file.
"""

import argparse
from open_rag_eval.evaluators.trec_evaluator import TRECEvaluator

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
        '-o', '--output',
        default='metrics_comparison.png',
        help='Output filename for the graph (default: metrics_comparison.png)'
    )

    # Parse arguments
    args = parser.parse_args()

    # Plot the metrics
    TRECEvaluator.plot_metrics(csv_files=args.csv_files, output_file=args.output)


if __name__ == '__main__':
    main()
