from pydantic import BaseModel, Field
from typing import List, Literal
from openai import OpenAI
import os
from enum import Enum

# Define Pydantic model for the structured output
class NuggetImportanceValues(str, Enum):
    VITAL = "vital"
    OKAY = "okay"

class NuggetImportance(BaseModel):
    importance: list[NuggetImportanceValues]


# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create a function that gets the structured output
def get_status_list(prompt: str) -> List[str]:
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",  # Use appropriate model
        messages=[
            {"role": "system", "content": "You are a helpful assistant that categorizes items as either 'vital' or 'okay'."},
            {"role": "user", "content": prompt}
        ],
        response_format=NuggetImportance,       
    )

    message = completion.choices[0].message
    if message.parsed:
        print(message.parsed.importance)
        return message.parsed.importance
    else:
        print(message.refusal)    
        



# Example usage
if __name__ == "__main__":
    prompt = "Categorize these 10 items as 'vital' or 'okay': water, food, shelter, internet, coffee, TV, medicine, video games, first aid kit, social media"
    result = get_status_list(prompt)
    print([str(item.value) for item in result])
