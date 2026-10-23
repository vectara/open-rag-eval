"""Helpers for plotting evaluation result tables."""

import json
import logging

import pandas as pd


def get_answered_percentage(df: pd.DataFrame) -> float:
    """Return the answered-query percentage from consolidated or run-one scores.

    Consolidated result CSVs can contain a score column that is blank for some
    queries while the run-specific column still has a valid score. Treat NaN as
    missing explicitly so it does not suppress the fallback.
    """
    answered_count = 0
    total_valid_rows = 0
    for idx, row in df.iterrows():
        value = row.get("generation_score_no_answer_score")
        if pd.isna(value) or value == "":
            value = row.get("run_1_generation_score_no_answer_score")
        if pd.isna(value):
            continue
        try:
            answer_data = json.loads(value)
            total_valid_rows += 1
            if answer_data.get("query_answered", "") == "yes":
                answered_count += 1
        except Exception:
            logging.error("Invalid JSON at row index %s", idx)
    return (answered_count / total_valid_rows) * 100 if total_valid_rows > 0 else 0
