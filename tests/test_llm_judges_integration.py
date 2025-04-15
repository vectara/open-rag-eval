import os
import unittest

from dotenv import load_dotenv
from enum import Enum
from pydantic import BaseModel

from open_rag_eval.models.llm_judges import OpenAIModel, GeminiModel


class CitationSupportValues(str, Enum):
    FULL = "full_support"
    PARTIAL = "partial_support"
    NONE = "no_support"


class CitationSupport(BaseModel):
    support: CitationSupportValues


_STRUCTURED_OUTPUT_TEST_PROMPT = """
    In this task, you will evaluate whether each statement is
    supported by its corresponding citations. Note that the system
    responses may appear very fluent and well-formed, but contain
    slight inaccuracies that are not easy to discern at first glance.
    Pay close attention to the text.

    You will be provided with a statement and its corresponding
    citation. It may be helpful to ask yourself whether it is
    accurate to say "according to the citation" with a
    statement following this phrase. Be sure to check all of the
    information in the statement. You will be given three options:

    - Full Support: All of the information in the statement is
    supported in the citation.

    - Partial Support: Some parts of the information are supported in
    the citation, but other parts are missing from the citation.

    - No Support: This citation does not support any part of the
    statement.

    Please provide your response based on the information in the
    citation. If you are unsure, use your best judgment. Respond as
    either ``full_support'', ``partial_support'', or ``no_support''
    with no additional information.

    Statement: {statement}

    Citation: {citation}
"""


class TestLLMJudgesIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv()
        # Check for required environment variables
        required_vars = {"openai": ["OPENAI_API_KEY"], "gemini": ["GOOGLE_API_KEY"]}

        cls.available_models = []

        # Check OpenAI credentials
        if all(os.getenv(var) for var in required_vars["openai"]):
            cls.openai_key = os.getenv("OPENAI_API_KEY")
            cls.available_models.append("openai")

        # Check Gemini credentials
        if all(os.getenv(var) for var in required_vars["gemini"]):
            cls.gemini_key = os.getenv("GOOGLE_API_KEY")
            cls.available_models.append("gemini")

        if not cls.available_models:
            raise unittest.SkipTest(
                "Skipping LLMJudge integration tests - no API keys configured"
            )
        cls.model_kwargs = {
            "temperature": 0.0,
            "top_p": 1.0,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.0,
        }

    def setUp(self):
        if "openai" in self.available_models:
            self.openai_model = OpenAIModel(
                model_name="gpt-4o-mini", api_key=self.openai_key
            )
        if "gemini" in self.available_models:
            self.gemini_model = GeminiModel(
                model_name="gemini-2.0-flash", api_key=self.gemini_key
            )

    def test_openai_integration(self):
        """Test OpenAI model with actual API calls"""
        if "openai" not in self.available_models:
            self.skipTest("OpenAI API key not configured")

        prompt = "What is 2+2? Answer with just the number."
        response = self.openai_model.call(prompt)

        # Basic validation of response
        self.assertIsInstance(response, str)
        self.assertTrue(len(response.strip()) > 0)
        # The response should contain 4 somewhere
        self.assertIn("4", response)

    def test_gemini_integration(self):
        """Test Gemini model with actual API calls"""
        if "gemini" not in self.available_models:
            self.skipTest("Gemini API key not configured")

        prompt = "What is 2+2? Answer with just the number."
        response = self.gemini_model.call(prompt)

        # Basic validation of response
        self.assertIsInstance(response, str)
        self.assertTrue(len(response.strip()) > 0)
        # The response should contain 4 somewhere
        self.assertIn("4", response)

    def test_openai_parse_integration(self):
        """Test OpenAI model parse method with actual API calls"""
        if "openai" not in self.available_models:
            self.skipTest("OpenAI API key not configured")

        statement = "The sky is blue."
        citation = "According to meteorological observations, the sky appears blue due to Rayleigh scattering of sunlight."
        prompt = _STRUCTURED_OUTPUT_TEST_PROMPT.format(
            statement=statement, citation=citation
        )

        response = self.openai_model.parse(prompt, CitationSupport, self.model_kwargs)

        self.assertIsInstance(response, CitationSupport)
        self.assertIsInstance(response.support, CitationSupportValues)
        self.assertIn(response.support, CitationSupportValues)

    def test_gemini_parse_integration(self):
        """Test Gemini model parse method with actual API calls"""
        if "gemini" not in self.available_models:
            self.skipTest("Gemini API key not configured")

        statement = "The sky is blue."
        citation = "According to meteorological observations, the sky appears blue due to Rayleigh scattering of sunlight."
        prompt = _STRUCTURED_OUTPUT_TEST_PROMPT.format(
            statement=statement, citation=citation
        )

        response = self.gemini_model.parse(prompt, CitationSupport, self.model_kwargs)

        self.assertIsInstance(response, CitationSupport)
        self.assertIsInstance(response.support, CitationSupportValues)
        self.assertIn(response.support, CitationSupportValues)


if __name__ == "__main__":
    unittest.main()
