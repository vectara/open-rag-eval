from abc import ABC
import json
import re

import openai
from google import genai
from google.genai.errors import APIError
import anthropic
import together
from pydantic import BaseModel, TypeAdapter

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

    def _remove_invalid_kwargs(self, model_kwargs) -> dict:
        if bool(re.match(r"^gemini-2\.5.*", self.model_name, re.IGNORECASE)):
            model_kwargs = model_kwargs.copy()
            invalid_kwargs = ["presence_penalty"]

            for kwarg in invalid_kwargs:
                if kwarg in model_kwargs:
                    del model_kwargs[kwarg]

        return model_kwargs

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
        model_kwargs = self._remove_invalid_kwargs(model_kwargs)

        try:
            response = self.client.models.generate_content(
                model=self.model_name, contents=prompt, config=model_kwargs
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
        model_kwargs = self._remove_invalid_kwargs(model_kwargs)
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
        parsed_response = TypeAdapter(response_format).validate_python(response_json)
        return parsed_response


class AnthropicModel(LLMJudgeModel):
    """LLMJudge that supports Anthropic models."""

    def __init__(self, model_options: dict):
        self.model_name = model_options["name"]
        self.client = anthropic.Anthropic(api_key=model_options["api_key"])

    def _remove_invalid_kwargs(self, model_kwargs) -> dict:
        model_kwargs = model_kwargs.copy()
        invalid_kwargs = ["presence_penalty", "frequency_penalty", "seed"]

        for kwarg in invalid_kwargs:
            if kwarg in model_kwargs:
                del model_kwargs[kwarg]

        return model_kwargs

    @retry(
        retry=retry_if_exception_type(
            (
                anthropic.InternalServerError,
                anthropic.APITimeoutError,
                anthropic.APIConnectionError,
                ValueError,
            )
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def call(self, prompt: str, model_kwargs=None) -> str:
        """
        Call the Anthropic API model with the given prompt.

        Args:
            prompt (str): The input prompt for the model
            model_kwargs (dict, optional): Additional kwargs for the API call

        Returns:
            str: The model's response text

        Raises:
            ValueError: If the prompt is empty or model_kwargs is invalid
            anthropic.InternalServerError: If there's a server-side issue
            anthropic.APITimeoutError: If the request timeout limit is exceeded
            anthropic.APIConnectionError: If there's a network error
            Exception: For API or other unexpected errors
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        model_kwargs = model_kwargs or {}
        model_kwargs = self._remove_invalid_kwargs(model_kwargs)

        try:
            max_tokens = None
            if "max_tokens" in model_kwargs:
                max_tokens = model_kwargs.pop("max_tokens")
            else:
                max_tokens = 16384

            response = self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                **model_kwargs,
            )
            return response.content[0].text
        except anthropic.InternalServerError:
            raise
        except anthropic.APITimeoutError:
            raise
        except anthropic.APIConnectionError:
            raise
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}") from e

    def parse(self, prompt: str, response_format: BaseModel, model_kwargs=None):
        """
        Parse structured output from an Anthropic model according to a Pydantic schema.

        Args:
            prompt (str): The input prompt
            response_format (BaseModel): Pydantic model defining the expected response structure
            model_kwargs (dict, optional): Additional kwargs for the API call

        Returns:
            str: The parsed response matching the provided schema
        """
        model_kwargs = model_kwargs or {}
        model_kwargs = self._remove_invalid_kwargs(model_kwargs)
        schema = response_format.model_json_schema()

        structured_prompt = f"""{prompt}

Please respond with a JSON object that matches this exact schema:
{json.dumps(schema, indent=2)}

Return only the JSON object, no other text."""

        max_tokens = None
        if "max_tokens" in model_kwargs:
            max_tokens = model_kwargs.pop("max_tokens")
        else:
            max_tokens = 16384

        response = self.client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": structured_prompt}],
            max_tokens=max_tokens,
            **model_kwargs,
        )

        response_json = json.loads(response.content[0].text)
        parsed_response = TypeAdapter(response_format).validate_python(response_json)
        return parsed_response


class TogetherModel(LLMJudgeModel):
    """LLMJudge that supports Together models."""

    def __init__(self, model_options: dict):
        self.model_name = model_options["name"]
        self.client = together.Together(api_key=model_options["api_key"])

    @retry(
        retry=retry_if_exception_type(
            (
                together.error.Timeout,
                together.error.APIConnectionError,
                together.error.RateLimitError,
                ValueError,
            )
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def call(self, prompt: str, model_kwargs=None) -> str:
        """
        Call the Together API model with the given prompt.

        Args:
            prompt (str): The input prompt for the model
            model_kwargs (dict, optional): Additional kwargs for the API call

        Returns:
            str: The model's response text

        Raises:
            ValueError: If the prompt is empty or model_kwargs is invalid
            together.error.Timeout: If the request timeout limit is exceeded
            together.error.APIConnectionError: If there's a network error
            together.error.RateLimitError: If rate limit is exceeded
            Exception: For API or other unexpected errors
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
        except together.error.Timeout:
            raise
        except together.error.APIConnectionError:
            raise
        except together.error.RateLimitError:
            raise
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}") from e

    def parse(self, prompt: str, response_format: BaseModel, model_kwargs=None):
        """
        Parse structured output from a Together model according to a Pydantic schema.

        Args:
            prompt (str): The input prompt
            response_format (BaseModel): Pydantic model defining the expected response structure
            model_kwargs (dict, optional): Additional kwargs for the API call

        Returns:
            The parsed response matching the provided schema
        """
        model_kwargs = model_kwargs or {}

        # Get the raw schema and flatten it to remove $ref constructs (caused by AutoNuggetizer)
        schema = response_format.model_json_schema()
        flattened_schema = self._flatten_schema(schema)

        config = {
            "type": "json_schema",
            "schema": flattened_schema,
        }

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format=config,
                **model_kwargs,
            )

            response_json = json.loads(response.choices[0].message.content)
            parsed_response = TypeAdapter(response_format).validate_python(
                response_json
            )
            return parsed_response
        except Exception as e:
            # If grammar validation fails, fall back to prompt-based approach like AnthropicModel
            if "grammar" in str(e).lower():
                return self._fallback_parse(prompt, response_format, model_kwargs)
            raise e

    def _fallback_parse(
        self, prompt: str, response_format: BaseModel, model_kwargs=None
    ):
        """
        Fallback parsing method that uses prompt-based JSON generation instead of grammar validation.
        """
        model_kwargs = model_kwargs or {}
        schema = response_format.model_json_schema()

        structured_prompt = f"""{prompt}

Please respond with a JSON object that matches this exact schema:
{json.dumps(schema, indent=2)}

Return only the JSON object, no other text."""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": structured_prompt}],
            **model_kwargs,
        )

        response_text = response.choices[0].message.content.strip()

        # Extract JSON from response text (in case there's extra text)
        try:
            # Try to find JSON object in the response
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                response_json = json.loads(json_str)
            else:
                # Fallback: try to parse the entire response as JSON
                response_json = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON from Together response: {response_text}"
            ) from e

        try:
            parsed_response = TypeAdapter(response_format).validate_python(
                response_json
            )
            return parsed_response
        except Exception as e:
            raise ValueError(
                f"Failed to validate response against schema: {response_json}"
            ) from e

    def _flatten_schema(self, schema):
        """
        Flatten JSON schema by resolving $ref constructs that Together's grammar validator doesn't support.
        """
        if not isinstance(schema, dict):
            return schema

        # Copy the schema to avoid mutating the original
        flattened = schema.copy()

        # If there are $defs, resolve them inline
        if "$defs" in schema:
            defs = schema["$defs"]
            flattened = self._resolve_refs(flattened, defs)
            # Remove $defs from the final schema since all refs are resolved
            flattened.pop("$defs", None)

        return flattened

    def _resolve_refs(self, obj, defs):
        """
        Recursively resolve $ref constructs by replacing them with their definitions.
        """
        if isinstance(obj, dict):
            if "$ref" in obj:
                # Extract the reference path (e.g., "#/$defs/NuggetImportanceValues")
                ref_path = obj["$ref"]
                if ref_path.startswith("#/$defs/"):
                    def_name = ref_path.split("/")[-1]
                    if def_name in defs:
                        return self._resolve_refs(defs[def_name], defs)
                return obj
            return {k: self._resolve_refs(v, defs) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._resolve_refs(item, defs) for item in obj]
        return obj
