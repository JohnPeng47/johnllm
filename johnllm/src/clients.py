import os
from openai import OpenAI
from pydantic import BaseModel
import instructor

deepseek_client = instructor.from_openai(
    OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
)

def deepseek_cost(usage):
    input_token_cost = usage.prompt_tokens * 2.7e-7
    ouptut_token_cost = (usage.total_tokens - usage.prompt_tokens) * 0.0000011
    return input_token_cost + ouptut_token_cost