from typing import Optional, Any
from pydantic import BaseModel

class CompletionTokensDetails(BaseModel):
    accepted_prediction_tokens: Optional[int] = None
    audio_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    rejected_prediction_tokens: Optional[int] = None

class PromptTokensDetails(BaseModel):
    audio_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None

class CompletionUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    completion_tokens_details: Optional[CompletionTokensDetails]
    prompt_tokens_details: PromptTokensDetails

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)
    
def convert_instructor_usage(raw_response):
    raw_response.usage = CompletionUsage(**raw_response.usage.dict())
    return raw_response