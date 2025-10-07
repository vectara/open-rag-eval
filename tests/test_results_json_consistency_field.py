# tests/test_results_json_consistency_field.py
import json
from pathlib import Path

from open_rag_eval.run_eval import _omit_empty_consistency


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
