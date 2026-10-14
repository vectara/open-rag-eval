import os
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from open_rag_eval.connectors.connector import Connector
from open_rag_eval.connectors.langchain_connector import LangChainConnector

TOP_K = 10


class TestLangchainConnector(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.queries = [
            "What is the meaning of life?",
            "what is a transformer?",
            "what is attention?",
        ]
        queries_df = pd.DataFrame(self.queries, columns=["query"])
        queries_df["query_id"] = [f"query_{inx}" for inx in range(len(self.queries))]
        self.input_queries = os.path.join(self.temp_dir.name, "queries.csv")
        queries_df.to_csv(self.input_queries, index=False)

        self.generated_answers = os.path.join(self.temp_dir.name, "results.csv")
        self.connector = LangChainConnector.__new__(LangChainConnector)
        Connector.__init__(
            self.connector,
            queries_csv=self.input_queries,
            output_path=self.generated_answers,
            max_workers=1,
        )
        self.connector.top_k = TOP_K
        self.connector.rag_chain = Mock()
        self.connector.rag_chain.invoke.return_value = "Test answer"
        self.connector.retriever = Mock()
        self.connector.retriever.invoke.return_value = [
            SimpleNamespace(page_content=f"Passage {idx}")
            for idx in range(1, TOP_K + 1)
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_fetch_data(self):
        self.connector.fetch_data()

        results = pd.read_csv(self.generated_answers, header=0, encoding="utf-8")
        self.assertEqual(results.shape[0], len(self.queries) * TOP_K)

        for idx, row in results.iterrows():
            query_idx = idx // TOP_K
            passage_idx = idx % TOP_K
            self.assertEqual(row["query_id"], f"query_{query_idx}")
            self.assertEqual(row["query"], self.queries[query_idx])
            self.assertEqual(row["query_run"], 1)
            self.assertEqual(row["passage_id"], f"[{passage_idx + 1}]")
            self.assertEqual(row["passage"], f"Passage {passage_idx + 1}")
            if passage_idx == 0:
                self.assertEqual(row["generated_answer"], "Test answer")
            else:
                self.assertTrue(pd.isna(row["generated_answer"]))

        self.assertEqual(self.connector.rag_chain.invoke.call_count, len(self.queries))
        self.assertEqual(self.connector.retriever.invoke.call_count, len(self.queries))


if __name__ == "__main__":
    unittest.main()
