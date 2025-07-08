import json
from ast import literal_eval
import re
from collections import defaultdict

import pandas as pd
import streamlit as st

from open_rag_eval.utils.constants import CONSISTENCY


def load_data(file):
    df = pd.read_csv(file)
    return df


def parse_retrieved_passages(passages_str):
    try:
        # Parse the JSON string into a Python dictionary
        passages_dict = json.loads(passages_str)
        # Create formatted passages
        formatted_passages = {}
        for key, value in passages_dict.items():
            formatted_passages[key] = value
        return formatted_passages
    except Exception as e:
        st.error(f"Error parsing retrieved passages: {e}")
        return {}


def parse_generated_answer(answer_str):
    try:
        # Parse the string into a Python list of dictionaries
        answer_list = json.loads(answer_str)
        # Format the answer with citations
        formatted_answer = ""
        for item in answer_list:
            text = item.get("text", "")
            citations = item.get("citations", [])
            if citations:
                citation_str = " ".join(citations)
                formatted_answer += f"{text} {citation_str}\n\n"
            else:
                formatted_answer += f"{text}\n\n"
        return formatted_answer
    except Exception as e:
        st.error(f"Error parsing generated answer: {e}")
        return ""


def parse_json_column(json_str):
    """Parse a column that might contain JSON or just a numeric value"""
    # Check if it's a numeric value
    try:
        float_val = float(json_str)
        return float_val
    except (ValueError, TypeError):
        pass

    # Try to parse as JSON
    try:
        return json.loads(json_str)
    except Exception:
        # Try as literal Python structure
        try:
            return literal_eval(json_str)
        except Exception:
            return json_str


def style_umbrela_score(score):
    """Style UMBRELA score with emoji indicators"""
    if score == "N/A":
        return "❓ N/A"
    try:
        score = int(score)
        score_str = str(score)
        if score == 0:
            score_str = "🔴 0"
        elif score == 1:
            score_str = "🟠 1"
        elif score == 2:
            score_str = "🟡 2"
        elif score == 3:
            score_str = "🟢 3"
        return score_str
    except Exception:
        return str(score)


def format_assignment(assignment):
    """Format assignment with color indicators"""
    if assignment == "support":
        return "🟢 Support"
    if assignment == "partial_support":
        return "🟡 Partial"
    if assignment == "not_support":
        return "🔴 No Support"
    return assignment


def format_no_answer_score(score_data):
    """Format no answer score with color indicators"""
    if isinstance(score_data, dict) and "query_answered" in score_data:
        if score_data["query_answered"].lower() == "no":
            return "🔴 Answer not attempted"

        return "🟢 Answer Provided"
    return "❓ Unknown"


def create_nugget_dataframe(data):
    """Create a formatted DataFrame for nugget visualization"""
    if not isinstance(data, dict):
        return None

    try:
        df = pd.DataFrame({
            "Nugget": data.get("nuggets", []),
            "Label": data.get("labels", []),
            "Assignment": data.get("assignments", []),
        })

        # Format assignments with colors
        df["Assignment"] = df["Assignment"].apply(format_assignment)

        return df
    except Exception:
        return None


def format_aggregate_metrics(metrics_dict):
    """Format aggregate metrics with consistent styling"""
    formatted = {}
    for key, value in metrics_dict.items():
        if key in ["precision_at_5", "ap_at_5", "MRR"]:
            formatted[key] = f"{value:.3f}"
    return formatted


def extract_runs(row):
    """
    Detect columns named like 'run_1_generated_answer', 'run_2_retrieved_passages', …
    Returns dict  {run_id: {field_name: cell_value, …}, …}
    """
    run_col_re = re.compile(r"run_(\d+)_(.+)")
    runs = defaultdict(dict)
    for col, val in row.items():
        m = run_col_re.match(col)
        if m:
            run_id, field = m.groups()
            runs[run_id][field] = val
    return runs

def visualize_selected_run(selected_run, consistency_fields=None):
    # Retrieved Passages + UMBRELA
    passages = {}
    if "retrieved_passages" in selected_run:
        passages = parse_retrieved_passages(
            selected_run["retrieved_passages"])
    if passages:
        st.subheader("Retrieved Passages")

    umbrela_scores = {}
    if "retrieval_score_umbrela_scores" in selected_run:
        umbrela_scores = parse_json_column(
            selected_run["retrieval_score_umbrela_scores"])

    aggregate_metrics = {
        k: v
        for k, v in umbrela_scores.items()
        if k in ["precision_at_5", "ap_at_5", "MRR"]
    }
    if aggregate_metrics:
        with st.expander("Aggregate Retrieval Metrics"):
            formatted_metrics = format_aggregate_metrics(
                aggregate_metrics)
            for metric, value in formatted_metrics.items():
                st.text(f"{metric}: {value}")

    passage_scores = {
        k: v
        for k, v in umbrela_scores.items()
        if k not in aggregate_metrics
    }
    for passage_id, passage_text in passages.items():
        score = passage_scores.get(passage_id, "N/A")
        styled_score = style_umbrela_score(score)
        with st.expander(
                f"Passage {passage_id} (UMBRELA: {styled_score})"):
            st.text(passage_text)

    # Generated Answer
    if "generated_answer" in selected_run:
        st.subheader("Generated Answer")
        answer = parse_generated_answer(
            selected_run["generated_answer"])
        st.text(answer)

    # No Answer Score
    if "generation_score_no_answer_score" in selected_run:
        st.subheader("Query Answer Attempted")
        no_answer_data = parse_json_column(
            selected_run["generation_score_no_answer_score"])
        st.text(format_no_answer_score(no_answer_data))

    # Evaluation Metrics
    st.subheader("Per Run Evaluation Metrics")
    base_metrics = [
        "retrieval_score_mean_umbrela_score",
        "retrieval_score_precision_metrics",
        "generation_score_autonugget_scores",
        "generation_score_mean_nugget_assignment_score",
        "generation_score_hallucination_score",
        "generation_score_citation_scores",
        "generation_score_citation_f1_score",
    ]

    metrics_columns = base_metrics
    for column in metrics_columns:
        if column not in selected_run:
            continue

        parsed_data = parse_json_column(selected_run[column])
        with st.expander(f"{column}"):
            if column == "retrieval_score_precision_recall_metrics" and isinstance(
                    parsed_data, dict):
                st.subheader("Precision@k")
                if "precision@" in parsed_data:
                    prec_df = pd.DataFrame(
                        parsed_data["precision@"].items(),
                        columns=["k", "Precision"])
                    prec_df["Precision"] = prec_df["Precision"].apply(
                        lambda x: f"{x:.3f}")
                    st.dataframe(prec_df, hide_index=True)

                st.subheader("Average Precision@k")
                if "AP@" in parsed_data:
                    ap_df = pd.DataFrame(parsed_data["AP@"].items(),
                                         columns=["k", "AP"])
                    ap_df["AP"] = ap_df["AP"].apply(
                        lambda x: f"{x:.3f}")
                    st.dataframe(ap_df, hide_index=True)

                if "MRR" in parsed_data:
                    st.subheader("Mean Reciprocal Rank")
                    st.text(f"{parsed_data['MRR']:.3f}")
                continue

            if column == "generation_score_autonugget_scores" and isinstance(
                    parsed_data, dict):
                st.subheader("Overall Scores")
                if "nuggetizer_scores" in parsed_data:
                    scores_df = pd.DataFrame(
                        parsed_data["nuggetizer_scores"].items(),
                        columns=["Metric", "Scores"])
                    scores_df["Scores"] = scores_df["Scores"].apply(
                        lambda x: f"{x:.2%}")
                    st.dataframe(scores_df, hide_index=True)

                st.subheader("Nugget Analysis")
                nugget_df = create_nugget_dataframe(parsed_data)
                if nugget_df is not None:
                    st.dataframe(nugget_df, hide_index=True)
                continue

            if isinstance(parsed_data, dict):
                st.json(parsed_data)
            else:
                st.text(f"{parsed_data:.2f}" if isinstance(
                    parsed_data, float) else parsed_data)

    if consistency_fields:
        st.subheader("Consistency Metrics")
        with st.expander("Consistency Metrics"):
            selected_metric = st.selectbox(
                "Choose a consistency metric",
                options=list(consistency_fields.keys()),
                format_func=lambda x: x.replace(
                    CONSISTENCY, "").replace("_", " ").title())

            parsed = consistency_fields[selected_metric]
            st.json(parsed)

def visualize_row(df, row_index):
    selected_row = df.iloc[row_index]

    # Display query
    st.subheader("Query")
    st.text(selected_row["query"])

    run_map = extract_runs(selected_row)

    st.subheader("Choose a run to view details")
    run_ids = sorted(run_map.keys(), key=int)  # ['1','2',…]
    sel_run = st.selectbox("Choose a run",
                           run_ids,
                           format_func=lambda r: f"Run {r}")
    selected_run = run_map[sel_run]
    consistency_fields = {
        col: parse_json_column(selected_row[col])
        for col in selected_row.index
        if col.startswith(CONSISTENCY)
    }
    visualize_selected_run(selected_run, consistency_fields)


def main():
    st.title("Open RAG Evaluation Viewer")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        st.subheader("Select a row to view details")
        row_index = st.selectbox(
            "Select row",
            range(len(df)),
            format_func=lambda x:
            f"Row {x} - Query: {df.iloc[x]['query'][:50]}...",
        )

        if row_index is not None:
            visualize_row(df, row_index)


if __name__ == "__main__":
    main()
