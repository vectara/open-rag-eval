import csv
import logging
import os

from tqdm import tqdm

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.base.base_query_engine import BaseQueryEngine

from open_rag_eval.connectors.connector import Connector

# Configure logging for tenacity
logger = logging.getLogger(__name__)

class LlamaIndexConnector(Connector):
    def __init__(
        self,
        config: dict,
        folder: str,
    ) -> BaseQueryEngine:
        documents = SimpleDirectoryReader(folder).load_data()
        index = VectorStoreIndex.from_documents(documents)
        self.query_engine = index.as_query_engine()
        self.queries_csv = config.input_queries
        self.outputs_csv = os.path.join(config.results_folder, config.generated_answers)

    def fetch_data(
        self,
    ) -> None:
        if self.query_engine is None:
            raise ValueError("Query engine is not initialized. Call read_docs() first.")

        queries = self.read_queries(self.queries_csv)

        # Open the output CSV file and write header.
        with open(self.outputs_csv, "w", newline='', encoding='utf-8') as csvfile:
            fieldnames = ["query_id", "query", "passage_id", "passage",
                          "generated_answer"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for query in tqdm(queries, desc="Running LlamaIndex queries"):
                try:
                    res = self.query_engine.query(query["query"])
                except Exception as ex:
                    print(f"Failed to process query {query['queryId']}: {ex}")
                    continue

                # Get the overall summary (generated answer).
                generated_answer = res.response
                if not generated_answer:
                    # Skip queries with no generated answer.
                    continue

                # Get the search results.
                for idx, node in enumerate(res.source_nodes, start=1):
                    row = {
                        "query_id": query["queryId"],
                        "query": query["query"],
                        "passage_id": f"[{idx}]",
                        "passage": node.text,
                        "generated_answer": generated_answer if idx == 1 else ""
                    }
                    writer.writerow(row)
