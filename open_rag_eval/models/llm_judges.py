from abc import ABC
import json

import openai
from google import genai
from google.genai.errors import APIError
from pydantic import BaseModel, parse_obj_as

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

class LLMJudgeModel(ABC):
    """Abstract base class for LLM judge models."""

    pass


class OpenAIModel(LLMJudgeModel):
    """Supports any model that conforms to the OpenAI API spec."""

    def __init__(self, model_options: dict):
        self.model_name = model_options["name"]
        openai.api_key = model_options["api_key"]
        self.base_url = model_options.get("base_url", None)
        self.client = openai.OpenAI(base_url=self.base_url)

    @retry(
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.APIError,
                ValueError,  # catch our “none‐response” too
            )
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def call(self, prompt: str, model_kwargs=None) -> str:
        """
        Call the OpenAI API compatible model with the given prompt.

        Args:
            prompt (str): The input prompt for the model
            model_kwargs (dict, optional): Additional kwargs for the API call

        Returns:
            str: The model's response text

        Raises:
            ValueError: If the prompt is empty or model_kwargs is invalid
            openai.APIError: If there's an API-related error
            openai.RateLimitError: If rate limit is exceeded
            openai.APIConnectionError: If there's a network error
            Exception: For other unexpected errors
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        model_kwargs = model_kwargs or {}

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **model_kwargs,
            )
            return response.choices[0].message.content
        except openai.RateLimitError:
            raise
        except openai.APIConnectionError:
            raise
        except openai.APIError:
            raise
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}") from e

    def parse(self, prompt: str, response_format: BaseModel, model_kwargs=None):
        model_kwargs = model_kwargs or {}
        completion = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that follows user instructions precisely and provides accurate information.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format=response_format,
            **model_kwargs,
        )

        message = completion.choices[0].message
        return message.parsed


class GeminiModel(LLMJudgeModel):
    """LLMJudge that supports Google Gemini models."""

    def __init__(self, model_options: dict):
        self.model_name = model_options["name"]
        self.client = genai.Client(api_key=model_options["api_key"])

    @retry(
        retry=retry_if_exception_type((APIError, ValueError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def call(self, prompt: str, model_kwargs=None) -> str:
        """
        Call the Gemini API model with the given prompt.

        Args:
            prompt (str): The input prompt for the model
            model_kwargs (dict, optional): Additional kwargs for the API call

        Returns:
            str: The model's response text

        Raises:
            ValueError: If the prompt is empty or model_kwargs is invalid
            Exception: For API or other unexpected errors
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        model_kwargs = model_kwargs or {}

        try:
            response = self.client.models.generate_content(
                model=self.model_name, contents=prompt, **model_kwargs
            )
            return response.text
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}") from e

    def parse(self, prompt: str, response_format: BaseModel, model_kwargs=None):
        """
        Parse structured output from a Gemini model according to a Pydantic schema.

        Args:
            prompt (str): The input prompt
            response_format (BaseModel): Pydantic model defining the expected response structure
            model_kwargs (dict, optional): Additional kwargs for the API call

        Returns:
            The parsed response matching the provided schema
        """
        model_kwargs = model_kwargs or {}
        config = {
            "response_mime_type": "application/json",
            "response_schema": response_format,
            **model_kwargs,
        }

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )

        response_json = json.loads(response.text)
        parsed_response = parse_obj_as(response_format, response_json)

        return parsed_response
