import os
import unittest
from pathlib import Path
from open_rag_eval.connectors.langchain_connector import (
    LangChainConnector
)
import omegaconf
import pandas as pd

# Dummy response JSON to simulate the Vectara API response.
DUMMY_RESPONSE = {
    "summary": "Test summary[1]",
    "search_results": [{"text": "Passage one"}, {"text": "Passage two"}],
}

TOP_K = 10

class TestLangchainConnector(unittest.TestCase):
    def setUp(self):
        # Create a temporary CSV file with one test query.
        self.outputs_path = 'tests/outputs'
        self.data_path = 'data/pdfs/'
        os.makedirs(self.outputs_path, exist_ok=True)

        self.queries = ["What is the meaning of life?", "what is a transformer?", "what is attention?"]
        queries_df = pd.DataFrame(self.queries, columns=["query"])
        queries_df["query_id"] = [f"query_{inx}" for inx in range(len(self.queries))]
        self.input_queries = os.path.join(self.outputs_path, "test_langchain_queries.csv")
        queries_df.to_csv(self.input_queries, index=False)

        # Output CSV file for testing.
        self.generated_answers = os.path.join(self.outputs_path,'results.csv')
        self.connector = LangChainConnector(
            config=omegaconf.OmegaConf.create({
                'input_queries': self.input_queries,
                'results_folder': '.',
                'generated_answers': self.generated_answers
            }),
            folder = self.data_path,
            top_k=TOP_K
        )

    def tearDown(self):
        # Cleanup the temporary test CSV and output CSV.
        if os.path.exists(self.input_queries):
            Path(self.input_queries).unlink()
        if os.path.exists(self.generated_answers):
            Path(self.generated_answers).unlink()

    def test_fetch_data(self):
        # Call the fetch_data method.
        self.connector.fetch_data()

        # Now read the output CSV and validate its contents.
        results = pd.read_csv(self.generated_answers, header=0, encoding="utf-8")
        self.assertEqual(results.shape[0], len(self.queries) * TOP_K)

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
