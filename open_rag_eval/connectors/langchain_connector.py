import logging
import os
from typing import Optional

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from open_rag_eval.chunking import ChunkingStrategy
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
            repeat_query: int = 1,  # Add repeat_query parameter
            chunking_strategy: Optional[ChunkingStrategy] = None,
            output_filename: Optional[str] = None,
    ) -> None:
        self.top_k = top_k

        # Configuration for paths
        queries_csv = config.get("input_queries", "")
        results_folder = config.get("results_folder",
                                    ".")  # Default to current directory
        # When the chunking-comparison layer drives this connector, it routes
        # each strategy to its own answers file via output_filename.
        generated_answers_filename = output_filename or config.get(
            "generated_answers", "langchain_generated_answers.csv")
        outputs_csv = os.path.join(results_folder, generated_answers_filename)

        super().__init__(queries_csv=queries_csv,
                         output_path=outputs_csv,
                         max_workers=max_workers,
                         repeat_query=repeat_query)

        # Ensure the results directory exists
        os.makedirs(results_folder, exist_ok=True)

        logger.info("Loading documents from folder: %s", folder)
        loader = DirectoryLoader(folder, glob="**/*.*")
        docs = loader.load()

        # Build the text splitter from the chunking strategy. Defaults preserve
        # the previous hardcoded behavior (chunk_size=1000, chunk_overlap=200).
        chunk_size = chunking_strategy.chunk_size if chunking_strategy else 1000
        chunk_overlap = chunking_strategy.chunk_overlap if chunking_strategy else 200
        if chunking_strategy:
            logger.info(
                "Using chunking strategy '%s' (chunk_size=%d, chunk_overlap=%d).",
                chunking_strategy.name, chunk_size, chunk_overlap)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits,
                                            embedding=OpenAIEmbeddings())
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        logger.info("Loaded %d documents and split into %d chunks.", len(docs),
                    len(splits))

        # Define the RAG prompt template directly
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise.\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        self.rag_chain = ({
            "context": self.retriever | format_docs,
            "question": RunnablePassthrough()
        } | prompt | llm | StrOutputParser())

    def process_query(self, query, run_idx=1):
        """ Process a single query using the LangChain RAG chain.
        Args:
            query (dict): A dictionary containing the query and its ID.
            run_idx (int): The index of the query run (1-based).
        Returns:
            list: A list of dictionaries containing the query ID, query text,
                  passage ID, passage text, and generated answer.
        """
        query_id = query["queryId"]
        actual_query = query["query"]
        try:
            generated_answer = self.rag_chain.invoke(actual_query)
            source_documents = self.retriever.invoke(actual_query)
            rows = []
            for idx, doc in enumerate(source_documents, start=1):
                rows.append({
                    "query_id": query_id,
                    "query": actual_query,
                    "query_run": run_idx,  # Add run_idx to output
                    "passage_id": f"[{idx}]",
                    "passage": doc.page_content,
                    "generated_answer": generated_answer if idx == 1 else ""
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
                "query_run": run_idx,  # Add run_idx to error output
                "passage_id": "ERROR",
                "passage": f"Runtime error: {e}",
                "generated_answer": API_ERROR,
            }]
