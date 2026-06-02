import json
import os
import tempfile
import unittest

import pandas as pd

from open_rag_eval import chunking_comparison
from open_rag_eval.chunking_comparison import (
    RANKING_METRIC,
    rank_strategies,
    write_comparison,
)


class TestChunkingComparison(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _write_trec_csv(self, name, scores):
        """Write a minimal TREC-style results CSV with the ranking metric column."""
        path = os.path.join(self.tmp, f"TRECEvaluator-{name}-results.csv")
        pd.DataFrame({
            "query_id": [f"q{i}" for i in range(len(scores))],
            RANKING_METRIC: scores,
        }).to_csv(path, index=False)
        return path

    def test_rank_strategies_orders_by_mean(self):
        strategy_results = [
            {"name": "small", "chunk_size": 256, "chunk_overlap": 32,
             "results_file": self._write_trec_csv("small", [0.2, 0.4])},   # mean 0.3
            {"name": "medium", "chunk_size": 512, "chunk_overlap": 64,
             "results_file": self._write_trec_csv("medium", [0.8, 0.6])},  # mean 0.7
            {"name": "large", "chunk_size": 1024, "chunk_overlap": 128,
             "results_file": self._write_trec_csv("large", [0.5, 0.5])},   # mean 0.5
        ]
        ranked = rank_strategies(strategy_results)
        self.assertEqual([r["name"] for r in ranked], ["medium", "large", "small"])
        self.assertAlmostEqual(ranked[0]["score"], 0.7)
        self.assertAlmostEqual(ranked[2]["score"], 0.3)

    def test_missing_file_scores_none_and_sorts_last(self):
        strategy_results = [
            {"name": "good", "chunk_size": 512, "chunk_overlap": 64,
             "results_file": self._write_trec_csv("good", [0.9])},
            {"name": "missing", "chunk_size": 256, "chunk_overlap": 32,
             "results_file": os.path.join(self.tmp, "does-not-exist.csv")},
        ]
        ranked = rank_strategies(strategy_results)
        self.assertEqual(ranked[0]["name"], "good")
        self.assertEqual(ranked[1]["name"], "missing")
        self.assertIsNone(ranked[1]["score"])

    def test_run_prefixed_metric_column_is_resolved(self):
        # TRECEvaluator.to_csv writes the metric prefixed with "run_1_"; ranking
        # must resolve it the same way TRECEvaluator.plot_metrics does.
        path = os.path.join(self.tmp, "run-prefixed.csv")
        pd.DataFrame({
            "query_id": ["q0", "q1"],
            f"run_1_{RANKING_METRIC}": [1.2, 1.4],
        }).to_csv(path, index=False)
        ranked = rank_strategies([
            {"name": "x", "chunk_size": 512, "chunk_overlap": 64, "results_file": path}
        ])
        self.assertAlmostEqual(ranked[0]["score"], 1.3)

    def test_missing_metric_column_scores_none(self):
        path = os.path.join(self.tmp, "no-metric.csv")
        pd.DataFrame({"query_id": ["q0"], "some_other_col": [1.0]}).to_csv(path, index=False)
        ranked = rank_strategies([
            {"name": "x", "chunk_size": 256, "chunk_overlap": 32, "results_file": path}
        ])
        self.assertIsNone(ranked[0]["score"])

    def test_write_comparison_outputs_valid_json(self):
        strategy_results = [
            {"name": "small", "chunk_size": 256, "chunk_overlap": 32,
             "results_file": self._write_trec_csv("small", [0.2])},
            {"name": "big", "chunk_size": 1024, "chunk_overlap": 128,
             "results_file": self._write_trec_csv("big", [0.9])},
        ]
        ranked = rank_strategies(strategy_results)
        json_path = write_comparison(self.tmp, ranked, version="9.9.9")

        self.assertTrue(os.path.exists(json_path))
        with open(json_path, encoding="utf-8") as f:
            report = json.load(f)

        self.assertEqual(report["version"], "9.9.9")
        self.assertEqual(report["ranking_metric"], RANKING_METRIC)
        self.assertEqual(report["best_strategy"], "big")
        self.assertEqual(len(report["strategies"]), 2)
        self.assertEqual(report["strategies"][0]["name"], "big")
        self.assertEqual(report["strategies"][0]["chunk_size"], 1024)

    def test_write_comparison_best_is_none_when_unscored(self):
        ranked = rank_strategies([
            {"name": "missing", "chunk_size": 256, "chunk_overlap": 32,
             "results_file": os.path.join(self.tmp, "nope.csv")}
        ])
        json_path = write_comparison(self.tmp, ranked, version="1.0.0")
        with open(json_path, encoding="utf-8") as f:
            report = json.load(f)
        self.assertIsNone(report["best_strategy"])


if __name__ == "__main__":
    unittest.main()
