import json
from pathlib import Path

import pandas as pd

from open_rag_eval.run_eval import create_openeval_report


def test_runs_are_ordered_numerically_and_kept_separate(tmp_path: Path):
    row = {"query_id": "q1", "query": "What is a blackhole?"}
    for run in range(1, 12):
        row[f"run_{run}_generated_answer"] = f"answer {run}"
    pd.DataFrame([row]).to_csv(tmp_path / "results.csv", index=False)

    create_openeval_report(str(tmp_path), "results.csv")

    runs = json.loads((tmp_path / "results.json").read_text())["evaluation"][0]["runs"]
    assert [run["generated_answer"] for run in runs] == [f"answer {n}" for n in range(1, 12)]
    assert all(list(run) == ["generated_answer"] for run in runs)
