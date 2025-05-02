import os
import csv
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from open_rag_eval.connectors.vectara_connector import (
    VectaraConnector,
)
import omegaconf

# Dummy response JSON to simulate the Vectara API response.
DUMMY_RESPONSE = {
    "summary": "Test summary[1]",
    "search_results": [{"text": "Passage one"}, {"text": "Passage two"}],
}

DEFAULT_VECTARA_CONFIG = {
    "search": {
        "lexical_interpolation": 0.005,
        "limit": 100,
        "context_configuration": {
            "sentences_before": 3,
            "sentences_after": 3,
            "start_tag": "<em>",
            "end_tag": "</em>"
        },
        "reranker": {
            "type": "chain",
            "rerankers": [
                {
                    "type": "customer_reranker",
                    "reranker_name": "Rerank_Multilingual_v1",
                    "limit": 50
                },
                {
                    "type": "mmr",
                    "diversity_bias": 0.01,
                    "limit": 10
                }
            ]
        }
    },
    "generation": {
        "generation_preset_name": "vectara-summary-table-md-query-ext-jan-2025-gpt-4o",
        "max_used_search_results": 5,
        "response_language": "eng",
        "citations": {"style": "numeric"},
        "enable_factual_consistency_score": False
    },
    "intelligent_query_rewriting": False,
    "save_history": True,
}

class TestVectaraConnector(unittest.TestCase):
    def setUp(self):
        # Create a temporary CSV file with one test query.
        self.data_path = 'tests/data'
        os.makedirs(self.data_path, exist_ok=True)
        self.input_queries = os.path.join(self.data_path, "test_vectara_connector.csv")

        with open(self.input_queries, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["query_id", "query"])
            writer.writeheader()
            writer.writerow(
                {"query_id": "query_1", "query": "What is the meaning of life?"}
            )

        # Output CSV file for testing.
        self.generated_answers = os.path.join(self.data_path,'results.csv')

        # Retrieve test credentials (or set dummy values for unit testing)
        api_key = os.getenv("VECTARA_API_KEY", "dummy_api_key")
        corpus_key = os.getenv("VECTARA_CORPUS_KEY", "dummy_corpus_key")
        self.connector = VectaraConnector(
            config=omegaconf.OmegaConf.create({
                'input_queries': self.input_queries,
                'results_folder': '.',
                'generated_answers': self.generated_answers
            }),
            api_key=api_key,
            corpus_key=corpus_key,
            query_config=omegaconf.OmegaConf.create(DEFAULT_VECTARA_CONFIG)
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
        with open(self.generated_answers, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

        # We expect two rows since our dummy response returns 1 search result with citations.
        self.assertEqual(len(rows), 2)

        # Check the first row: it should have the generated summary and passage_id "[1]"
        row1 = rows[0]
        self.assertEqual(row1["query_id"], "query_1")
        self.assertEqual(row1["query"], "What is the meaning of life?")
        self.assertEqual(row1["passage_id"], "[1]")
        self.assertEqual(row1["passage"], "Passage one")
        self.assertEqual(row1["generated_answer"], "Test summary[1]")

        row2 = rows[1]
        self.assertEqual(row2["query_id"], "query_1")
        self.assertEqual(row2["query"], "What is the meaning of life?")
        self.assertEqual(row2["passage_id"], "[2]")
        self.assertEqual(row2["passage"], "Passage two")
        self.assertEqual(row2["generated_answer"], "")


if __name__ == "__main__":
    unittest.main()
