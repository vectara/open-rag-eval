from abc import ABC
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
            
        Returns:
            str: The model's response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **model_kwargs                
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error calling OpenAI API: {str(e)}")