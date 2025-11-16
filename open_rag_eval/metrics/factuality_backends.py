"""
Backend implementations for factual consistency evaluation.

This module provides different backend implementations for evaluating factual consistency
of generated text against source documents. Backends can be swapped based on configuration.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict

import requests
import torch
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from transformers import AutoModelForSequenceClassification

# Set number of cores to 2 to avoid heavy CPU usage
torch.set_num_threads(2)

logger = logging.getLogger(__name__)


class FactualityBackend(ABC):
    """Abstract base class for factual consistency evaluation backends."""

    @abstractmethod
    def evaluate(self, sources: str, summary: str) -> float:
        """
        Evaluate factual consistency of summary against sources.

        Args:
            sources: Source text(s) that the summary should be consistent with
            summary: Generated summary text to evaluate

        Returns:
            Float score representing factual consistency (typically 0-1 range)
        """
        pass


class HHEMBackend(FactualityBackend):
    """Backend using the open-source Vectara Hallucination Evaluation Model (HHEM)."""

    def __init__(
        self,
        model_name: str = 'vectara/hallucination_evaluation_model',
        detection_threshold: float = 0.5,
        max_chars: int = 8192
    ):
        """
        Initialize the HHEM backend.

        Args:
            model_name: The Hugging Face model name to use for hallucination detection.
            detection_threshold: The threshold for detecting hallucinations.
            max_chars: Maximum number of characters to process. Inputs longer than this will be truncated.
        """
        hf_token = os.environ.get('HF_TOKEN')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token
        )
        self.detection_threshold = detection_threshold
        self.max_chars = max_chars
        logger.info(f"Initialized HHEM backend with model: {model_name}")

    def evaluate(self, sources: str, summary: str) -> float:
        """
        Evaluate factual consistency using the HHEM model.

        Args:
            sources: Source text(s) concatenated together
            summary: Generated summary text

        Returns:
            Float score from the HHEM model (0-1 range, higher is more consistent)
        """
        # Truncate sources if needed
        if len(sources) > self.max_chars:
            sources = sources[:self.max_chars]
            logger.debug(f"Truncated sources to {self.max_chars} characters")

        # Call the hallucination detection model
        score = self.model.predict([(sources, summary)]).item()
        return score


class VectaraAPIBackend(FactualityBackend):
    """Backend using the commercial Vectara Factual Consistency API."""

    def __init__(self, api_key: str, base_url: str = "https://api.vectara.io"):
        """
        Initialize the Vectara API backend.

        Args:
            api_key: Vectara API key for authentication
            base_url: Base URL for Vectara API (defaults to production)

        Raises:
            ValueError: If api_key is not provided
        """
        if not api_key:
            raise ValueError(
                "api_key is required for VectaraAPIBackend. "
                "Set VECTARA_API_KEY environment variable or provide it in config."
            )
        self._api_key = api_key
        self._base_url = base_url.rstrip('/')
        self._endpoint = f"{self._base_url}/v2/evaluate_factual_consistency"
        logger.info("Initialized Vectara API backend")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((requests.exceptions.RequestException,)),
        reraise=True
    )
    def _make_api_request(self, sources: str, summary: str) -> Dict:
        """
        Make API request to Vectara with retry logic.

        Args:
            sources: Source text(s) concatenated together
            summary: Generated summary text

        Returns:
            JSON response from API

        Raises:
            requests.exceptions.RequestException: If API request fails (will be retried)
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-api-key": self._api_key
        }

        # Payload format as per Vectara API specification
        # source_texts expects a list of strings
        payload = {
            "generated_text": summary,
            "source_texts": [sources]
        }

        response = requests.post(
            self._endpoint,
            headers=headers,
            json=payload,
            timeout=30
        )

        # Use raise_for_status - this raises HTTPError (a RequestException subclass)
        # which will be caught by the retry decorator
        response.raise_for_status()

        return response.json()

    def evaluate(self, sources: str, summary: str) -> float:
        """
        Evaluate factual consistency using the Vectara API.

        Args:
            sources: Source text(s) concatenated together
            summary: Generated summary text

        Returns:
            Float score from the Vectara API (0-1 range, higher is more consistent)

        Raises:
            RuntimeError: If API request fails or response format is unexpected
        """
        try:
            response = self._make_api_request(sources, summary)

        except requests.exceptions.HTTPError as ex:
            # Provide specific error messages for common HTTP status codes
            status_code = ex.response.status_code if ex.response else None

            if status_code == 403:
                error_msg = (
                    "Access forbidden (403). Please verify your Vectara API key "
                    "has permissions for factual consistency evaluation."
                )
            elif status_code == 422:
                error_msg = (
                    "Unsupported language (422). The Vectara API does not support "
                    "the language in your text."
                )
            else:
                error_msg = f"Vectara API request failed: {ex}"

            logger.error(error_msg)
            raise RuntimeError(error_msg) from ex

        except requests.exceptions.Timeout as ex:
            error_msg = "Vectara API request timed out after 30 seconds"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from ex

        except requests.exceptions.RequestException as ex:
            error_msg = f"Vectara API request failed: {ex}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from ex

        except ValueError as ex:  # json.JSONDecodeError is a subclass of ValueError
            error_msg = f"Failed to parse Vectara API response: {ex}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from ex

        # Parse response - try common field names
        # The actual field name should be validated with real API
        score = None
        for possible_key in ["score", "factual_consistency_score", "consistency_score", "result"]:
            if possible_key in response:
                score = response[possible_key]
                break

        if score is None:
            logger.warning(
                f"Unexpected API response format. Response keys: {list(response.keys())}. "
                f"Full response: {response}"
            )
            # Try to extract any numeric value
            for value in response.values():
                if isinstance(value, (int, float)):
                    score = float(value)
                    logger.info(f"Using numeric value from response: {score}")
                    break

        if score is None:
            error_msg = (
                f"Could not extract score from Vectara API response. "
                f"Response: {response}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        return float(score)
