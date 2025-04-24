import streamlit as st
import pandas as pd
import json
from ast import literal_eval

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
        if score == 0:
            return "🔴 0"
        elif score == 1:
            return "🟠 1"
        elif score == 2:
            return "🟡 2"
        elif score == 3:
            return "🟢 3"
        return str(score)
    except Exception:
        return str(score)

def format_assignment(assignment):
    """Format assignment with color indicators"""
    if assignment == "support":
        return "🟢 Support"
    elif assignment == "partial_support":
        return "🟡 Partial"
    elif assignment == "not_support":
        return "🔴 No Support"
    return assignment

def format_no_answer_score(score_data):
    """Format no answer score with color indicators"""
    if isinstance(score_data, dict) and 'query_answered' in score_data:
        if score_data['query_answered'].lower() == 'no':
            return "🔴 Answer not attempted"
        else:
            return "🟢 Answer Provided"
    return "❓ Unknown"

def create_nugget_dataframe(data):
    """Create a formatted DataFrame for nugget visualization"""
    if not isinstance(data, dict):
        return None

    try:
        df = pd.DataFrame({
            'Nugget': data.get('nuggets', []),
            'Label': data.get('labels', []),
            'Assignment': data.get('assignments', []),
        })

        # Format assignments with colors
        df['Assignment'] = df['Assignment'].apply(format_assignment)

        return df
    except Exception:
        return None

def main():
    st.title("Open RAG Evaluation Viewer")

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)

        # Display row selector
        st.subheader("Select a row to view details")
        row_index = st.selectbox("Select row", range(len(df)), format_func=lambda x: f"Row {x} - Query: {df.iloc[x]['query'][:50]}...")

        if row_index is not None:
            selected_row = df.iloc[row_index]

            # Display query
            st.subheader("Query")
            st.text(selected_row['query'])

            # Display retrieved passages
            st.subheader("Retrieved Passages")
            passages = parse_retrieved_passages(selected_row['retrieved_passages'])

            # Parse umbrella scores
            umbrela_scores = parse_json_column(selected_row['retrieval_score_umbrela_scores'])

            for passage_id, passage_text in passages.items():
                score = umbrela_scores.get(passage_id, "N/A")
                styled_score = style_umbrela_score(score)
                with st.expander(f"Passage {passage_id} (UMBRELA: {styled_score})"):
                    st.text(passage_text)

            # Display generated answer
            st.subheader("Generated Answer")
            answer = parse_generated_answer(selected_row['generated_answer'])
            st.text(answer)

            # Display no answer score
            if 'generation_score_no_answer_score' in selected_row:
                st.subheader("Query Answer Attempted")
                no_answer_data = parse_json_column(selected_row['generation_score_no_answer_score'])
                st.text(format_no_answer_score(no_answer_data))

            # Display evaluation metrics
            st.subheader("Evaluation Metrics")
            metrics_columns = [
                'retrieval_score_mean_umbrela_score',
                'generation_score_autonugget_scores',
                'generation_score_mean_nugget_assignment_score',
                'generation_score_hallucination_scores',
                'generation_score_citation_scores',
                'generation_score_citation_f1_score'
            ]

            for column in metrics_columns:
                if column in selected_row:
                    with st.expander(f"{column}"):
                        parsed_data = parse_json_column(selected_row[column])

                        # Special handling for autonugget scores
                        if column == 'generation_score_autonugget_scores':
                            if isinstance(parsed_data, dict):
                                # Display overall scores
                                st.subheader("Overall Scores")
                                if 'nuggetizer_scores' in parsed_data:
                                    scores_df = pd.DataFrame(
                                        parsed_data['nuggetizer_scores'].items(),
                                        columns=['Metric', 'Scores']
                                    )
                                    scores_df['Scores'] = scores_df['Scores'].apply(lambda x: f"{x:.2%}")
                                    st.dataframe(scores_df, hide_index=True)

                                # Display nugget details
                                st.subheader("Nugget Analysis")
                                nugget_df = create_nugget_dataframe(parsed_data)
                                if nugget_df is not None:
                                    st.dataframe(nugget_df, hide_index=True)
                        else:
                            if isinstance(parsed_data, dict):
                                st.json(parsed_data)
                            else:
                                st.text(f"{parsed_data:.2f}" if isinstance(parsed_data, float) else parsed_data)


if __name__ == "__main__":
    main()
