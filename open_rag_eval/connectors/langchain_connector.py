import csv
import logging
import os

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub

from open_rag_eval.connectors.connector import Connector
from open_rag_eval.utils.constants import API_ERROR

logger = logging.getLogger(__name__)


class LangChainConnector(Connector):

    def __init__(
        self,
        config: dict,
        folder: str,
        top_k: int = 10,
        max_workers: int = -1,
    ) -> None:
        super().__init__()  # Call to the base class constructor if needed

        self.top_k = top_k
        self.parallel = max_workers > 0 or max_workers == -1
        if max_workers == -1:
            self.max_workers = min(32, os.cpu_count() * 4)
        else:
            self.max_workers = max_workers

        # Configuration for paths
        self.queries_csv = config.get("input_queries")
        if not self.queries_csv:
            logger.error("Config dictionary must contain 'input_queries' path.")
            raise ValueError(
                "Config dictionary must contain 'input_queries' path.")

        results_folder = config.get("results_folder",
                                    ".")  # Default to current directory
        generated_answers_filename = config.get(
            "generated_answers", "langchain_generated_answers.csv")
        self.outputs_csv = os.path.join(results_folder,
                                        generated_answers_filename)

        # Ensure the results directory exists
        os.makedirs(results_folder, exist_ok=True)

        logger.info("Loading documents from folder: %s", folder)
        loader = DirectoryLoader(folder, glob="**/*.*")
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                       chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits,
                                            embedding=OpenAIEmbeddings())
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        logger.info("Loaded %d documents and split into %d chunks.", len(docs),
                    len(splits))
        prompt = hub.pull("rlm/rag-prompt")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        self.rag_chain = ({
            "context": self.retriever | format_docs,
            "question": RunnablePassthrough()
        } | prompt | llm | StrOutputParser())

    def fetch_data(self) -> None:
        queries = self.read_queries(
            self.queries_csv)  # Using method from base or this class
        logger.info("Starting to process %d queries using LangChain connector.",
                    len(queries))
        fieldnames = [
            "query_id",
            "query",
            "passage_id",
            "passage",
            "generated_answer",
        ]

        if not self.parallel:
            with open(self.outputs_csv, "w", newline="",
                      encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for query in tqdm(queries,
                                  total=len(queries),
                                  desc="Running LangChain queries"):
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
                                   desc="Running LangChain queries"):
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
            "LangChain query processing is complete. Results saved to %s",
            self.outputs_csv)

    def process_query(self, query_data):
        """ Process a single query using the LangChain RAG chain.
        Args:
            query_data (dict): A dictionary containing the query and its ID.
                Expected keys are "query" and "queryId".
        Returns:
            list: A list of dictionaries containing the query ID, query text,
                passage ID, passage content, and generated answer.
        """
        query_id = query_data["queryId"]
        actual_query = query_data["query"]
        try:
            generated_answer = self.rag_chain.invoke(actual_query)
            source_documents = self.retriever.invoke(actual_query)
            rows = []
            for idx, doc in enumerate(source_documents, start=1):
                rows.append({
                    "query_id": query_id,
                    "query": actual_query,
                    "passage_id": f"[{idx}]",
                    "passage": doc.page_content,
                    "generated_answer": generated_answer if idx == 1 else "",
                })
            return rows
        except Exception as e:
            logger.error(
                "Failed to process query_id %s ('%s'): %s",
                query_id,
                actual_query,
                str(e),
                exc_info=True,
            )
            return [{
                "query_id": query_id,
                "query": actual_query,
                "passage_id": "ERROR",
                "passage": f"Runtime error: {e}",
                "generated_answer": API_ERROR,
            }]
