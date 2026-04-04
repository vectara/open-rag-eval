import logging
import os

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from open_rag_eval.connectors.connector import Connector
from open_rag_eval.utils.constants import API_ERROR

logger = logging.getLogger(__name__)


class ChromaDBConnector(Connector):

    def __init__(
        self,
        config: dict,
        collection_name: str,
        chroma_db_path: str = "./chroma_db",
        top_k: int = 10,
        openai_embedding_model: str = "text-embedding-3-large",
        openai_llm_model: str = "gpt-4o",
        max_workers: int = -1,
        repeat_query: int = 1,
    ) -> None:
        self.top_k = top_k

        # Config paths
        queries_csv = config.get("input_queries", "")
        results_folder = config.get("results_folder", ".")
        generated_answers_filename = config.get(
            "generated_answers", "chromadb_generated_answers.csv"
        )
        outputs_csv = os.path.join(results_folder, generated_answers_filename)

        super().__init__(
            queries_csv=queries_csv,
            output_path=outputs_csv,
            max_workers=max_workers,
            repeat_query=repeat_query,
        )

        os.makedirs(results_folder, exist_ok=True)

        # Connect to existing ChromaDB collection
        logger.info(
            "Connecting to ChromaDB at path: %s, collection: %s",
            chroma_db_path,
            collection_name,
        )
        embedding_fn = OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY"), model_name=openai_embedding_model
        )
        client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = client.get_collection(
            name=collection_name, embedding_function=embedding_fn
        )
        logger.info(
            "Connected to collection '%s' with %d documents.",
            collection_name,
            self.collection.count(),
        )

        # LLM for answer generation
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise.\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{question}"),
            ]
        )
        llm = ChatOpenAI(model_name=openai_llm_model, temperature=0)
        self.llm_chain = prompt | llm | StrOutputParser()

    def process_query(self, query, run_idx=1):
        """Process a single query using the existing ChromaDB collection.

        Args:
            query (dict): A dictionary containing the query text and its ID.
            run_idx (int): The index of the query run (1-based).

        Returns:
            list: A list of dictionaries with query results.
        """
        query_id = query["queryId"]
        actual_query = query["query"]
        try:
            # Query the existing ChromaDB collection directly
            results = self.collection.query(
                query_texts=[actual_query], n_results=self.top_k
            )
            passages = results["documents"][0]
            ids = results["ids"][0]

            context = "\n\n".join(passages)
            generated_answer = self.llm_chain.invoke(
                {"context": context, "question": actual_query}
            )

            rows = []
            for idx, (passage, pid) in enumerate(zip(passages, ids), start=1):
                rows.append(
                    {
                        "query_id": query_id,
                        "query": actual_query,
                        "query_run": run_idx,
                        "passage_id": pid,
                        "passage": passage,
                        "generated_answer": generated_answer if idx == 1 else "",
                    }
                )
            return rows

        except Exception as e:
            logger.error(
                "Failed to process query_id %s ('%s'): %s",
                query_id,
                actual_query,
                str(e),
                exc_info=True,
            )
            return [
                {
                    "query_id": query_id,
                    "query": actual_query,
                    "query_run": run_idx,
                    "passage_id": "ERROR",
                    "passage": f"Runtime error: {e}",
                    "generated_answer": API_ERROR,
                }
            ]
