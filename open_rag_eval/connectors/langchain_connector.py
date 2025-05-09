import csv
import logging
import os

from tqdm import tqdm

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub

from open_rag_eval.connectors.connector import Connector

logger = logging.getLogger(__name__)


class LangchainConnector(Connector):
    def __init__(
        self,
        config: dict,
        folder: str,
        top_k: int = 10,
    ) -> None:
        super().__init__()  # Call to the base class constructor if needed

        self.top_k = top_k

        # Configuration for paths
        self.queries_csv = config.get("input_queries")
        if not self.queries_csv:
            logger.error("Config dictionary must contain 'input_queries' path.")
            raise ValueError("Config dictionary must contain 'input_queries' path.")

        results_folder = config.get(
            "results_folder", "."
        )  # Default to current directory
        generated_answers_filename = config.get(
            "generated_answers", "langchain_generated_answers.csv"
        )
        self.outputs_csv = os.path.join(results_folder, generated_answers_filename)

        # Ensure the results directory exists
        os.makedirs(results_folder, exist_ok=True)

        logger.info(f"Loading documents from folder: {folder}")
        loader = DirectoryLoader(folder, glob="**/*.*")
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=splits, embedding=OpenAIEmbeddings()
        )
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        logger.info(
            f"Loaded {len(docs)} documents and split into {len(splits)} chunks."
        )
        prompt = hub.pull("rlm/rag-prompt")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def fetch_data(self) -> None:
        queries = self.read_queries(
            self.queries_csv
        )  # Using method from base or this class
        logger.info(
            f"Starting to process {len(queries)} queries using LangChain connector."
        )
        with open(self.outputs_csv, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "query_id",
                "query",
                "passage_id",
                "passage",
                "generated_answer",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for query_data in tqdm(queries, desc="Running LangChain queries"):
                query_id = query_data["queryId"]
                actual_query = query_data["query"]

                try:
                    generated_answer = self.rag_chain.invoke(actual_query)
                    source_documents = self.retriever.invoke(actual_query)
                    for idx, doc in enumerate(source_documents, start=1):
                        passage_text = doc.page_content
                        row_data = {
                            "query_id": query_id,
                            "query": actual_query,
                            "passage_id": f"[{idx}]",  # Match LlamaIndex output format
                            "passage": passage_text,
                            "generated_answer": generated_answer if idx == 1 else "",
                        }
                        writer.writerow(row_data)

                except Exception as e:
                    logger.error(
                        f"Failed to process query_id {query_id} ('{actual_query}'): {e}",
                        exc_info=True,
                    )
                    # Write a row with error information for this specific query
                    writer.writerow(
                        {
                            "query_id": query_id,
                            "query": actual_query,
                            "passage_id": "ERROR",
                            "passage": f"Runtime error: {e}",
                            "generated_answer": "ERROR",
                        }
                    )
                    continue  # Continue with the next query

        logger.info(
            f"LangChain processing complete. Results saved to {self.outputs_csv}"
        )
