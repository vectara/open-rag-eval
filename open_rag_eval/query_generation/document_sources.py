"""Document source adapters for query generation."""

import csv
import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DocumentSource(ABC):
    """Abstract base class for document sources."""

    @abstractmethod
    def fetch_random_documents(
        self,
        min_doc_size: int = 0,
        max_num_docs: Optional[int] = None,
        seed: Optional[int] = None
    ) -> List[str]:
        """
        Fetch random documents from the source.

        Args:
            min_doc_size: Minimum document size in characters
            max_num_docs: Maximum number of documents to load (None for all)
            seed: Random seed for reproducible sampling (None for random)

        Returns:
            List of document texts

        Raises:
            ValueError: If source configuration is invalid
            IOError: If documents cannot be loaded
        """
        pass


class VectaraCorpusSource(DocumentSource):
    """Load documents from a Vectara corpus via API."""

    def __init__(self, api_key: str, corpus_key: str):
        """
        Initialize VectaraCorpusSource.

        Args:
            api_key: Vectara API key
            corpus_key: Vectara corpus key

        Raises:
            ValueError: If api_key or corpus_key is missing
        """
        if not api_key:
            raise ValueError("Vectara API key is required")
        if not corpus_key:
            raise ValueError("Vectara corpus key is required")

        self.api_key = api_key
        self.corpus_key = corpus_key
        self.api_url = "https://api.vectara.io"
        self.session = requests.Session()

    def list_docs(self) -> List[str]:
        """
        List all document IDs in the corpus.

        Returns:
            List of document IDs

        Raises:
            IOError: If API request fails
        """
        page_key = None
        doc_ids = []

        while True:
            params = {"limit": 100}
            if page_key:
                params["page_key"] = page_key

            post_headers = {
                'x-api-key': self.api_key,
            }
            response = self.session.get(
                f"{self.api_url}/v2/corpora/{self.corpus_key}/documents",
                headers=post_headers,
                params=params,
                timeout=30
            )

            if response.status_code != 200:
                raise IOError(
                    f"Error listing documents: status={response.status_code}, "
                    f"response={response.text}"
                )

            res = response.json()

            for doc in res.get('documents', []):
                doc_ids.append(doc['id'])

            response_metadata = res.get('metadata', None)
            if not response_metadata or not response_metadata.get('page_key'):
                break
            page_key = response_metadata['page_key']

        return doc_ids

    def get_doc_text(self, doc_id: str) -> str:
        """
        Retrieve text content of a document.

        Args:
            doc_id: Document ID

        Returns:
            Document text content

        Raises:
            IOError: If API request fails
        """
        post_headers = {
            'x-api-key': self.api_key,
        }
        response = self.session.get(
            f"{self.api_url}/v2/corpora/{self.corpus_key}/documents/{doc_id}",
            headers=post_headers,
            timeout=30
        )

        if response.status_code != 200:
            raise IOError(
                f"Error retrieving document {doc_id}: status={response.status_code}, "
                f"response={response.text}"
            )

        res = response.json()
        return '\n'.join((p['text'] for p in res.get('parts', [])))

    def fetch_random_documents(
        self,
        min_doc_size: int = 0,
        max_num_docs: Optional[int] = None,
        seed: Optional[int] = None
    ) -> List[str]:
        """
        Fetch random documents from Vectara corpus.

        Args:
            min_doc_size: Minimum document size in characters
            max_num_docs: Maximum number of documents to load (None for all)
            seed: Random seed for reproducible sampling (None for random)

        Returns:
            List of document texts

        Raises:
            IOError: If API requests fail
        """
        logger.info("Loading document IDs from Vectara corpus: %s", self.corpus_key)
        doc_ids = self.list_docs()
        logger.info("Found %d documents in corpus", len(doc_ids))

        max_docs_default = 5000
        if not max_num_docs and len(doc_ids) > max_docs_default:
            logger.warning(
                f"Corpus contains {len(doc_ids)} documents. Limiting to {max_docs_default} documents by default to avoid long load times."
                " Consider setting max_num_docs to load a smaller subset."
            )
            max_num_docs = max_docs_default

        if max_num_docs and max_num_docs < len(doc_ids):
            if seed is not None:
                random.seed(seed)
            doc_ids = random.sample(doc_ids, max_num_docs)
            logger.info("Randomly sampled %d documents", max_num_docs)

        documents = []
        for doc_id in tqdm(doc_ids, desc="Loading documents from Vectara", unit="doc"):
            try:
                doc_text = self.get_doc_text(doc_id)
                if len(doc_text) >= min_doc_size:
                    documents.append(doc_text)
            except IOError as e:
                logger.warning("Failed to load document %s: %s", doc_id, str(e))
                continue

        logger.info(
            "Loaded %d documents (after filtering by min_doc_size=%d)",
            len(documents),
            min_doc_size
        )
        return documents


class LocalFileSource(DocumentSource):
    """Load documents from local text files."""

    def __init__(self, path: str, file_extensions: Optional[List[str]] = None):
        """
        Initialize LocalFileSource.

        Args:
            path: Path to directory containing text files
            file_extensions: List of file extensions to include (e.g., ['.txt', '.md'])
                           Defaults to ['.txt', '.md']

        Raises:
            ValueError: If path is invalid
        """
        if not path:
            raise ValueError("Path is required")

        self.path = Path(path)
        if not self.path.exists():
            raise ValueError(f"Path does not exist: {path}")
        if not self.path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        self.file_extensions = file_extensions or ['.txt', '.md']

    def fetch_random_documents(
        self,
        min_doc_size: int = 0,
        max_num_docs: Optional[int] = None,
        seed: Optional[int] = None
    ) -> List[str]:
        """
        Fetch random documents from local files.

        Args:
            min_doc_size: Minimum document size in characters
            max_num_docs: Maximum number of documents to load (None for all)
            seed: Random seed for reproducible sampling (None for random)

        Returns:
            List of document texts

        Raises:
            IOError: If files cannot be read
        """
        documents = []
        file_paths = []

        # Collect all matching files
        for ext in self.file_extensions:
            file_paths.extend(self.path.glob(f"**/*{ext}"))

        logger.info("Found %d files with extensions %s", len(file_paths), self.file_extensions)

        if max_num_docs and max_num_docs < len(file_paths):
            if seed is not None:
                random.seed(seed)
            file_paths = random.sample(file_paths, max_num_docs)
            logger.info("Randomly sampled %d files", max_num_docs)

        for file_path in tqdm(file_paths, desc="Loading documents from local files", unit="file"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content) >= min_doc_size:
                        documents.append(content)
            except IOError as e:
                logger.warning("Failed to read file %s: %s", file_path, str(e))
                continue

        logger.info(
            "Loaded %d documents (after filtering by min_doc_size=%d)",
            len(documents),
            min_doc_size
        )
        return documents


class CSVSource(DocumentSource):
    """Load documents from a CSV file."""

    def __init__(self, csv_path: str, text_column: str = "text"):
        """
        Initialize CSVSource.

        Args:
            csv_path: Path to CSV file
            text_column: Name of column containing document text

        Raises:
            ValueError: If csv_path is invalid
        """
        if not csv_path:
            raise ValueError("CSV path is required")

        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise ValueError(f"CSV file does not exist: {csv_path}")

        self.text_column = text_column

    def fetch_random_documents(
        self,
        min_doc_size: int = 0,
        max_num_docs: Optional[int] = None,
        seed: Optional[int] = None
    ) -> List[str]:
        """
        Fetch random documents from CSV file.

        Args:
            min_doc_size: Minimum document size in characters
            max_num_docs: Maximum number of documents to load (None for all)
            seed: Random seed for reproducible sampling (None for random)

        Returns:
            List of document texts

        Raises:
            ValueError: If text_column is not found in CSV
            IOError: If CSV cannot be read
        """
        documents = []

        try:
            with open(self.csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)

                if self.text_column not in reader.fieldnames:
                    raise ValueError(
                        f"Column '{self.text_column}' not found in CSV. "
                        f"Available columns: {reader.fieldnames}"
                    )

                for row in reader:
                    text = row.get(self.text_column, "").strip()
                    if text and len(text) >= min_doc_size:
                        documents.append(text)

        except IOError as e:
            raise IOError(f"Failed to read CSV file {self.csv_path}: {str(e)}") from e

        logger.info("Loaded %d documents from CSV", len(documents))

        if max_num_docs and max_num_docs < len(documents):
            if seed is not None:
                random.seed(seed)
            documents = random.sample(documents, max_num_docs)
            logger.info("Randomly sampled %d documents", max_num_docs)

        return documents
