import os
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from open_rag_eval.connectors.llama_index_connector import (
    LlamaIndexConnector,
)
import omegaconf
import pandas as pd

# Dummy response JSON to simulate the Vectara API response.
DUMMY_RESPONSE = {
    "summary": "Test summary[1]",
    "search_results": [{"text": "Passage one"}, {"text": "Passage two"}],
}

class TestVectaraConnector(unittest.TestCase):
    def setUp(self):
        # Create a temporary CSV file with one test query.
        self.outputs_path = 'tests/outputs'
        self.data_path = 'data/pdfs/'
        os.makedirs(self.outputs_path, exist_ok=True)
        self.input_queries = os.path.join(self.outputs_path, "test_llamaindex_queries.csv")

        self.queries = ["What is the meaning of life?", "what is a transformer?", "waht is attention?"]
        queries_df = pd.DataFrame(self.queries, columns=["query"])
        queries_df["query_id"] = [f"query_{inx}" for inx in range(len(self.queries))]
        queries_df.to_csv(self.input_queries, index=False)

        # Output CSV file for testing.
        self.generated_answers = os.path.join(self.outputs_path,'results.csv')
        self.connector = LlamaIndexConnector(
            config=omegaconf.OmegaConf.create({
                'input_queries': self.input_queries,
                'results_folder': '.',
                'generated_answers': self.generated_answers
            }),
            folder = self.data_path,
        )

    def tearDown(self):
        # Cleanup the temporary test CSV and output CSV.
        if os.path.exists(self.input_queries):
            Path(self.input_queries).unlink()
        if os.path.exists(self.generated_answers):
            Path(self.generated_answers).unlink()

    @patch("requests.post")
    def test_fetch_data(self, mock_post):
        # Configure the mock to return a dummy response.
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = DUMMY_RESPONSE
        mock_post.return_value = mock_response

        # Call the fetch_data method.
        self.connector.fetch_data()

        # Now read the output CSV and validate its contents.
        results = pd.read_csv(self.generated_answers, header=0, encoding="utf-8")
        self.assertEqual(results.shape[0], len(self.queries) * 2)

        # Check the first row: it should have the generated summary and passage_id "[1]"
        count = results["query_id"].value_counts()[0]
        for idx, row in results.iterrows():
            query_idx = idx // count
            passage_idx = idx % count
            self.assertEqual(row["query_id"], f"query_{query_idx}")
            self.assertEqual(row["query"], self.queries[query_idx])
            self.assertEqual(row["passage_id"], f"[{passage_idx+1}]")


if __name__ == "__main__":
    unittest.main()
