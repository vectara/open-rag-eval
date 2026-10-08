# tests/test_results_json_consistency_field.py
import json
from pathlib import Path

import pandas as pd

from open_rag_eval.run_eval import _omit_empty_consistency, create_openeval_report


def _write_results_and_report(tmp_path: Path, row: dict) -> dict:
    pd.DataFrame([row]).to_csv(tmp_path / "results.csv", index=False)
    create_openeval_report(str(tmp_path), "results.csv")
    return json.loads((tmp_path / "results.json").read_text())["evaluation"][0]


def test_report_entry_omits_empty_consistency(tmp_path: Path):
    entry = _write_results_and_report(
        tmp_path, {"query_id": "q1", "query": "x", "run_1_generated_answer": "a"}
    )
    assert "consistency" not in entry


def test_report_entry_keeps_consistency_scores(tmp_path: Path):
    entry = _write_results_and_report(
        tmp_path,
        {"query_id": "q1", "query": "x", "run_1_generated_answer": "a",
         "consistency_bert_score": json.dumps({"mean": 0.9})},
    )
    assert entry["consistency"] == {"bert_score": {"mean": 0.9}}


def test_omit_empty_consistency_removes_key(tmp_path: Path):
    report = {"metadata": {"evaluator": "trec"}, "consistency": {}}
    cleaned = _omit_empty_consistency(report.copy())
    assert "consistency" not in cleaned

    # sanity: when written to disk, key shouldn't appear
    p = tmp_path / "results.json"
    p.write_text(json.dumps(_omit_empty_consistency(report.copy())))
    loaded = json.loads(p.read_text())
    assert "consistency" not in loaded


def test_keep_nonempty_consistency(tmp_path: Path):
    report = {
        "metadata": {"evaluator": "consistency"},
        "consistency": {"example_metric": 0.77},  # any non-empty dict is fine
    }
    cleaned = _omit_empty_consistency(report.copy())
    assert "consistency" in cleaned

    p = tmp_path / "results.json"
    p.write_text(json.dumps(_omit_empty_consistency(report.copy())))
    loaded = json.loads(p.read_text())
    assert "consistency" in loaded
