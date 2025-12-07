"""
This script is useful for visualizing the results of RAG evaluators.

It can be used to plot results from multiple runs of different evaluators on the same plot
allowing for a comparative analysis of the results. This script reads CSV files
containing evaluation metrics and generates plots for the specified metrics.

Usage:
    # Recommended (via unified CLI):
    open-rag-eval plot metrics_1.csv metrics_2.csv --evaluator trec --metrics-to-plot retrieval_score_mean_umbrela_score
    open-rag-eval plot metrics_1.csv metrics_2.csv --evaluator consistency --metrics-to-plot rouge_score bert_score

    # Legacy (still supported for backwards compatibility):
    python plot_results.py -e trec metrics_1.csv metrics_2.csv --metrics-to-plot retrieval_score_mean_umbrela_score

The plot is saved by default to a 'metrics_comparison.png' file.
"""

from typing import List, Optional
from open_rag_eval.evaluators.trec_evaluator import TRECEvaluator
from open_rag_eval.evaluators.consistency_evaluator import ConsistencyEvaluator
from open_rag_eval.evaluators.golden_answer_evaluator import GoldenAnswerEvaluator

def plot_metrics(
    evaluator_type: str,
    csv_files: List[str],
    output_file: str,
    metrics_to_plot: Optional[List[str]] = None
):
    """
    Plot metrics based on the evaluator type.

    Args:
        evaluator_type: Type of evaluator ('trec', 'consistency', 'golden_answer', etc.)
        csv_files: List of CSV files containing metrics
        output_file: Output file path for the generated plot
        metrics_to_plot: List of metric column names to plot
    """
    evaluator_type = evaluator_type.lower()

    if evaluator_type == 'trec':
        TRECEvaluator.plot_metrics(csv_files=csv_files, output_file=output_file, metrics_to_plot=metrics_to_plot)
    elif evaluator_type == 'consistency':
        ConsistencyEvaluator.plot_metrics(csv_files=csv_files, output_file=output_file, metrics_to_plot=metrics_to_plot)
    elif evaluator_type == 'golden_answer':
        GoldenAnswerEvaluator.plot_metrics(csv_files=csv_files, output_file=output_file, metrics_to_plot=metrics_to_plot)
    else:
        raise ValueError(f"Unsupported evaluator type: {evaluator_type}")

def main():
    """CLI entry point for standalone execution.

    This function maintains backwards compatibility by redirecting to the main CLI.
    It prepends 'plot' to sys.argv and normalizes flag names to invoke the correct subcommand.
    """
    import sys  # pylint: disable=import-outside-toplevel,reimported

    # Normalize old-style flags to match cli.py conventions
    # Replace -e with --evaluator, -o with --output-file
    normalized_argv = []
    i = 0
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '-e':
            normalized_argv.append('--evaluator')
            if i + 1 < len(sys.argv):
                normalized_argv.append(sys.argv[i + 1])
                i += 2
                continue
        elif arg == '-o':
            normalized_argv.append('--output-file')
            if i + 1 < len(sys.argv):
                normalized_argv.append(sys.argv[i + 1])
                i += 2
                continue
        else:
            normalized_argv.append(arg)
        i += 1

    # Redirect to the main CLI with the plot subcommand
    sys.argv = normalized_argv
    sys.argv.insert(1, 'plot')
    from open_rag_eval.cli import main as cli_main  # pylint: disable=import-outside-toplevel,cyclic-import
    cli_main()


if __name__ == '__main__':
    main()
