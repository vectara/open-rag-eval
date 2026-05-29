"""Ranking and reporting for the chunking-strategy comparison layer.

After each chunking strategy has been evaluated and written its own
TRECEvaluator results CSV, these helpers rank the strategies by mean UMBRELA
retrieval score, write a ``chunking_comparison.json`` report, and print a
console summary highlighting the best-performing strategy.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# The metric used to decide the best-performing chunking strategy. Chunking
# primarily affects retrieval quality, so we rank by mean UMBRELA.
RANKING_METRIC = "retrieval_score_mean_umbrela_score"

COMPARISON_JSON_FILENAME = "chunking_comparison.json"
COMPARISON_PLOT_FILENAME = "chunking_comparison.png"


def rank_strategies(
    strategy_results: List[Dict[str, Any]],
    metric: str = RANKING_METRIC,
) -> List[Dict[str, Any]]:
    """Rank chunking strategies by the mean of a metric column.

    Args:
        strategy_results: One dict per strategy, each with keys ``name``,
            ``chunk_size``, ``chunk_overlap`` and ``results_file`` (path to that
            strategy's TRECEvaluator results CSV).
        metric: The CSV column to average per strategy. Defaults to
            :data:`RANKING_METRIC`.

    Returns:
        A new list of dicts (copies of the inputs) each augmented with a
        ``score`` key (mean of the metric column, or ``None`` if unavailable),
        sorted by ``score`` descending. Strategies whose score could not be
        computed sort last.
    """
    ranked: List[Dict[str, Any]] = []
    for entry in strategy_results:
        score = _mean_metric(entry.get("results_file"), metric)
        ranked.append({**entry, "score": score})

    # Sort by score descending; None scores sort last.
    ranked.sort(key=lambda e: (e["score"] is not None, e["score"] or 0.0), reverse=True)
    return ranked


def _mean_metric(results_file: Optional[str], metric: str) -> Optional[float]:
    """Return the mean of `metric` in `results_file`, or None if unavailable."""
    if not results_file or not os.path.exists(results_file):
        logger.warning("Results file not found for ranking: %s", results_file)
        return None
    try:
        df = pd.read_csv(results_file)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Failed to read %s for ranking: %s", results_file, e)
        return None
    # TRECEvaluator.to_csv writes per-run columns prefixed with "run_1_". Mirror
    # the resolution used by TRECEvaluator.plot_metrics: prefer the bare metric
    # name (present in the consolidated CSV), else fall back to the run_1_ form.
    if metric in df.columns:
        column = metric
    elif f"run_1_{metric}" in df.columns:
        column = f"run_1_{metric}"
    else:
        logger.warning("Metric '%s' not found in %s.", metric, results_file)
        return None
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean())


def write_comparison(
    results_folder: str,
    ranked: List[Dict[str, Any]],
    version: str,
    metric: str = RANKING_METRIC,
) -> str:
    """Write the chunking comparison JSON report to `results_folder`.

    Args:
        results_folder: Folder to write ``chunking_comparison.json`` into.
        ranked: Ranked strategy dicts as returned by :func:`rank_strategies`.
        version: Package version to stamp into the report.
        metric: The ranking metric name to record.

    Returns:
        The path to the written JSON file.
    """
    best = next((e["name"] for e in ranked if e.get("score") is not None), None)
    report = {
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ranking_metric": metric,
        "best_strategy": best,
        "strategies": [
            {
                "name": e["name"],
                "chunk_size": e.get("chunk_size"),
                "chunk_overlap": e.get("chunk_overlap"),
                "score": e.get("score"),
                "results_file": e.get("results_file"),
            }
            for e in ranked
        ],
    }
    json_path = os.path.join(results_folder, COMPARISON_JSON_FILENAME)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return json_path


def print_summary(ranked: List[Dict[str, Any]], metric: str = RANKING_METRIC) -> None:
    """Print a ranked table of chunking strategies and the winner to stdout."""
    print(f"\n=== Chunking Strategy Comparison (ranked by {metric}) ===")
    print(f"{'Rank':<5} {'Strategy':<16} {'size/overlap':<14} {'Score':<8}")
    for rank, e in enumerate(ranked, start=1):
        score = e.get("score")
        score_str = f"{score:.4f}" if score is not None else "N/A"
        size_overlap = f"{e.get('chunk_size')}/{e.get('chunk_overlap')}"
        print(f"{rank:<5} {e['name']:<16} {size_overlap:<14} {score_str:<8}")

    best = next((e for e in ranked if e.get("score") is not None), None)
    if best:
        print(f"\n🏆 Best chunking strategy: {best['name']} "
              f"(chunk_size={best.get('chunk_size')}, "
              f"chunk_overlap={best.get('chunk_overlap')}, "
              f"{metric}={best['score']:.4f})")
    else:
        print("\nNo chunking strategy could be scored.")
    print("=" * 60 + "\n")
