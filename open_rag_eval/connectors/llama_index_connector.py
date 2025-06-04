import logging
import os

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
            repeat_query: int = 1,  # Add repeat_query parameter
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

        queries_csv = config.get("input_queries", "")
        results_folder = config.get("results_folder",
                                    ".")  # Default to current directory
        generated_answers_filename = config.get(
            "generated_answers", "llamaindex_generated_answers.csv")
        outputs_csv = os.path.join(results_folder, generated_answers_filename)

        # Initialize base class
        super().__init__(queries_csv=queries_csv,
                         output_path=outputs_csv,
                         max_workers=max_workers,
                         repeat_query=repeat_query)

    def process_query(self, query, run_idx=1):
        """ Process a single query using the LlamaIndex query engine.
        Args:
            query: A dictionary containing the query text and its ID.
            run_idx: The index of the query run (1-based)
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
                    "query_run": run_idx,
                    "passage_id": "NA",
                    "passage": "NA",
                    "generated_answer": NO_ANSWER
                }]

            rows = []
            for idx, node in enumerate(res.source_nodes[:self.top_k], start=1):
                row = {
                    "query_id": query["queryId"],
                    "query": query["query"],
                    "query_run": run_idx,
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
                "query_run": run_idx,
                "passage_id": "ERROR",
                "passage": f"Runtime error: {ex}",
                "generated_answer": API_ERROR,
            }]
