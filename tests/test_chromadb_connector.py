import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import omegaconf

from open_rag_eval.connectors.chromadb_connector import ChromaDBConnector


TOP_K = 3


class TestChromaDBConnector(unittest.TestCase):

    @patch("open_rag_eval.connectors.chromadb_connector.chromadb.PersistentClient")
    @patch("open_rag_eval.connectors.chromadb_connector.OpenAIEmbeddingFunction")
    def setUp(self, mock_embedding_fn, mock_client):
        self.outputs_path = "tests/outputs"
        os.makedirs(self.outputs_path, exist_ok=True)

        # Create test queries CSV
        self.queries = ["What is RAG?", "What is ChromaDB?", "What is an embedding?"]
        queries_df = pd.DataFrame(self.queries, columns=["query"])
        queries_df["query_id"] = [f"query_{i}" for i in range(len(self.queries))]
        self.input_queries = os.path.join(
            self.outputs_path, "test_chromadb_queries.csv"
        )
        queries_df.to_csv(self.input_queries, index=False)
        self.generated_answers = os.path.join(self.outputs_path, "chromadb_results.csv")

        # Mock the ChromaDB collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 10
        mock_collection.query.return_value = {
            "documents": [["Passage one", "Passage two", "Passage three"]],
            "ids": [["id1", "id2", "id3"]],
        }
        mock_client.return_value.get_collection.return_value = mock_collection

        # Mock the LLM chain
        with patch("open_rag_eval.connectors.chromadb_connector.ChatOpenAI"):
            self.connector = ChromaDBConnector(
                config=omegaconf.OmegaConf.create(
                    {
                        "input_queries": self.input_queries,
                        "results_folder": self.outputs_path,
                        "generated_answers": self.generated_answers,
                    }
                ),
                collection_name="test_collection",
                chroma_db_path="./fake_chroma_db",
                top_k=TOP_K,
            )
            # Mock llm_chain directly
            self.connector.llm_chain = MagicMock()
            self.connector.llm_chain.invoke.return_value = "This is a test answer."

    def tearDown(self):
        if os.path.exists(self.input_queries):
            Path(self.input_queries).unlink()
        if os.path.exists(self.generated_answers):
            Path(self.generated_answers).unlink()

    def test_process_query_returns_correct_rows(self):
        query = {"queryId": "query_0", "query": "What is RAG?"}
        rows = self.connector.process_query(query, run_idx=1)

        self.assertEqual(len(rows), TOP_K)
        self.assertEqual(rows[0]["query_id"], "query_0")
        self.assertEqual(rows[0]["generated_answer"], "This is a test answer.")
        self.assertEqual(rows[1]["generated_answer"], "")  # only first row has answer
        self.assertEqual(rows[0]["passage"], "Passage one")

    def test_process_query_error_handling(self):
        self.connector.collection.query.side_effect = Exception("DB connection failed")
        query = {"queryId": "query_1", "query": "What is ChromaDB?"}
        rows = self.connector.process_query(query, run_idx=1)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["passage_id"], "ERROR")
        self.assertIn("DB connection failed", rows[0]["passage"])


if __name__ == "__main__":
    unittest.main()
