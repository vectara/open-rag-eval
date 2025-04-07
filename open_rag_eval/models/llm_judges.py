from abc import ABC
from pydantic import BaseModel

import openai

class LLMJudgeModel(ABC):
    """Abstract base class for LLM judge models."""
    pass


class OpenAIModel(LLMJudgeModel):
    """OpenAI model."""
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        openai.api_key = api_key
        self.client = openai.OpenAI()

    def call(self, prompt: str, model_kwargs=None) -> str:
        """
        Call the OpenAI model with the given prompt.

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
            model="gpt-4o-mini",  # Use appropriate model
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that categorizes items as either 'vital' or 'okay'."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format=response_format,
        )

        message = completion.choices[0].message

        return message
