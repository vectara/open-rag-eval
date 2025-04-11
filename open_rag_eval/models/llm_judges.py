import json
import openai

from abc import ABC
from google import genai
from pydantic import BaseModel
from pydantic.tools import parse_obj_as


class LLMJudgeModel(ABC):
    """Abstract base class for LLM judge models."""
    pass


class OpenAIModel(LLMJudgeModel):
    """Supports any model that conforms to the OpenAI API spec."""
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        openai.api_key = api_key
        self.client = openai.OpenAI()

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
                **model_kwargs
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

    def parse(self, prompt: str, response_format: BaseModel):
        completion = self.client.beta.chat.completions.parse(
            model=self.model_name,  # Use appropriate model
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that follows user instructions precisely and provides accurate information."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format=response_format,
        )

        message = completion.choices[0].message

        return message.parsed


class GeminiModel(LLMJudgeModel):
    """LLMJudge that supports Google Gemini models."""
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)


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
                model=self.model_name,
                contents=prompt,
                **model_kwargs
            )
            return response.text
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}") from e
        

    def parse(self, prompt: str, response_format: BaseModel):
        """
        Parse structured output from a Gemini model according to a Pydantic schema.

        Args:
            prompt (str): The input prompt
            response_format (BaseModel): Pydantic model defining the expected response structure

        Returns:
            The parsed response matching the provided schema
        """
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': response_format,
            },
        )

        response_json = json.loads(response.text)
        parsed_response = parse_obj_as(response_format, response_json)

        return parsed_response
