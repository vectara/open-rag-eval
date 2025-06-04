import csv
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.query_engine.citation_query_engine import CitationQueryEngine

from open_rag_eval.connectors.connector import Connector
from open_rag_eval.utils.constants import NO_ANSWER, API_ERROR

# Configure logging for tenacity
logger = logging.getLogger(__name__)


class LlamaIndexConnector(Connector):

    def __init__(
        self,
        config: dict,
        folder: str,
        top_k: int = 10,
        max_workers: int = -1,
    ) -> BaseQueryEngine:
        documents = SimpleDirectoryReader(folder).load_data()
        index = VectorStoreIndex.from_documents(documents)
        retriever = index.as_retriever(similarity_top_k=top_k)
        self.top_k = top_k
        self.query_engine = CitationQueryEngine.from_args(
            index,
            retriever=retriever,
            citation_chunk_size=65536,
            citation_chunk_overlap=0,  # make every node be a single chunk
        )
        self.queries_csv = config.input_queries
        self.outputs_csv = os.path.join(config.results_folder,
                                        config.generated_answers)
        self.parallel = max_workers > 0 or max_workers == -1
        if max_workers == -1:
            self.max_workers = min(32, os.cpu_count() * 4)
        else:
            self.max_workers = max_workers

    def fetch_data(self) -> None:
        if self.query_engine is None:
            raise ValueError(
                "Query engine is not initialized. Call read_docs() first.")

        queries = self.read_queries(self.queries_csv)
        fieldnames = [
            "query_id", "query", "passage_id", "passage", "generated_answer"
        ]

        if not self.parallel:
            with open(self.outputs_csv, "w", newline="",
                      encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for query in tqdm(queries,
                                  total=len(queries),
                                  desc="Running LlamaIndex queries"):
                    results = self.process_query(query)
                    if results:
                        for row in results:
                            writer.writerow(row)
        else:
            # Use ThreadPoolExecutor to process queries in parallel
            indexed_queries = list(enumerate(queries))  # (index, query)
            results_buffer = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_index = {
                    executor.submit(self.process_query, query): idx
                    for idx, query in indexed_queries
                }
                for future in tqdm(as_completed(future_to_index),
                                   total=len(queries),
                                   desc="Running LlamaIndex queries"):
                    idx = future_to_index[future]
                    results = future.result()
                    if results:
                        for row in results:
                            results_buffer.append((idx, row))
            with open(self.outputs_csv, "w", newline="",
                      encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                results_buffer.sort(key=lambda x: x[0])
                for _, row in results_buffer:
                    writer.writerow(row)

        logger.info(
            "LlamaIndex query processing is complete. Results saved to %s",
            self.outputs_csv)

    def process_query(self, query):
        """ Process a single query using the LlamaIndex query engine.
        Args:
            query: A dictionary containing the query text and its ID.
        Returns:
            A list of dictionaries with query results, including passage text and generated answer.
        """
        try:
            res = self.query_engine.query(query["query"])
            generated_answer = res.response
            if not generated_answer:
                logger.warning("No generated answer for query %s. Skipping...",
                               query["query"])
                return [{
                    "query_id": query["queryId"],
                    "query": query["query"],
                    "passage_id": "NA",
                    "passage": "NA",
                    "generated_answer": NO_ANSWER
                }]

            rows = []
            for idx, node in enumerate(res.source_nodes[:self.top_k], start=1):
                row = {
                    "query_id": query["queryId"],
                    "query": query["query"],
                    "passage_id": f"[{idx}]",
                    "passage": node.text,
                    "generated_answer": generated_answer if idx == 1 else ""
                }
                rows.append(row)
            return rows
        except Exception as ex:
            logger.error(
                "Failed to process query_id %s ('%s'): %s",
                query["queryId"],
                query["query"],
                str(ex),
                exc_info=True,
            )
            return [{
                "query_id": query["queryId"],
                "query": query["query"],
                "passage_id": "ERROR",
                "passage": f"Runtime error: {ex}",
                "generated_answer": API_ERROR,
            }]
