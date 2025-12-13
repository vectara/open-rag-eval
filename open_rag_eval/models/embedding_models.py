"""Embedding model implementations for semantic similarity computations."""

from abc import ABC, abstractmethod
from typing import List
import logging

import numpy as np
import openai


logger = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        pass

    @abstractmethod
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: String to embed

        Returns:
            numpy array of shape (embedding_dim,)
        """
        pass


class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI embedding model implementation."""

    def __init__(self, model_options: dict):
        """Initialize OpenAI embedding model.

        Args:
            model_options: Dict containing:
                - name: Model name (e.g., "text-embedding-3-large")
                - api_key: OpenAI API key
        """
        self.model_name = model_options["name"]
        self.client = openai.OpenAI(api_key=model_options["api_key"])

    def embed(self, texts: List[str]) -> np.ndarray:
        """Batch embed multiple texts."""
        if not texts:
            return np.array([])

        # OpenAI API limits batch size, handle large batches
        batch_size = 2048
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch
            )
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)

        return np.array(all_embeddings)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.embed([text])[0]
