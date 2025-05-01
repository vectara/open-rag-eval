import csv
import logging
import uuid
from tqdm import tqdm

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, LLM, BaseQueryEngine

from open_rag_eval.connectors.connector import Connector

# Configure logging for tenacity
logger = logging.getLogger(__name__)

class LlamaIndexConnector(Connector):
    def __init__(
            self, 
            llm: LLM
        ) -> None:
        self.llm = llm
        self.query_engine = None

    def read_docs(
            self, 
            docs_folder: str
        ) -> BaseQueryEngine:
        documents = SimpleDirectoryReader(docs_folder).load_data()
        index = VectorStoreIndex.from_documents(documents)
        self.query_engine = index.as_query_engine(llm=self.llm)

    def fetch_data(
            self, 
            input_csv="queries.csv", 
            output_csv="results.csv"
        ) -> None:
        if self.query_engine is None:
            raise ValueError("Query engine is not initialized. Call read_docs() first.")

        # Read queries from CSV file.
        queries = []
        with open(input_csv, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                query_text = row.get("query")
                if not query_text:
                    print(f"Skipping row without query: {row}")
                    continue  # skip rows without a query
                # Use provided query_id or generate one if not present.
                query_id = row.get("query_id") or str(uuid.uuid4())
                queries.append({"query": query_text, "queryId": query_id})

        # Open the output CSV file and write header.
        with open(output_csv, "w", newline='', encoding='utf-8') as csvfile:
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
                        "passage": node.text_resource,
                        "generated_answer": generated_answer if idx == 1 else ""
                    }
                    writer.writerow(row)
