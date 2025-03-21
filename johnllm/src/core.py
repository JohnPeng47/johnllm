from typing import Any, Dict, Optional, Type, List, Tuple, Generic, TypeVar
import inspect
from pathlib import Path
import json
import time
import jinja2
from textwrap import dedent
from openai import OpenAI
import os
from abc import ABC, abstractmethod

from pydantic import BaseModel
import tiktoken

import instructor   
from litellm import completion
from pydantic import BaseModel

from .utils import CompletionUsage
from .ops import OpsList, Op
from .cache import LLMCache

def convert_instructor_usage(raw_response):
    raw_response.usage = CompletionUsage(**raw_response.usage.dict())
    return raw_response

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens

SHORT_NAMES = {
    "gpt-4o" : "gpt-4o",
    "claude" : "claude-3-5-sonnet-20240620",
    "deepseek" : "deepseek/deepseek-chat"
}

### TEMPORARY INSTRUCTOR IMPLEMENATION FOR DEEPSEEK
def deepseek_cost(usage):
    input_token_cost = usage.prompt_tokens * 2.7e-7
    ouptut_token_cost = (usage.total_tokens - usage.prompt_tokens) * 0.0000011
    return input_token_cost + ouptut_token_cost

def deepseek_instructor():
    class DeepseekClient:
        def __init__(self):
            self.chat = self.Chat()

        class Chat:
            def __init__(self):
                self.completions = self.Completions()

            class Completions:
                def create_with_completion(self, 
                                           model: str = "",
                                           messages: List = [] , 
                                           response_model: Type[BaseModel] = None, 
                                           **kwargs) -> Tuple[Any, Any]:
                    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
                    instructor_message = dedent(
                        f"""
                        As a genius expert, your task is to understand the content and provide
                        the parsed objects in json that match the following json_schema:\n

                        {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

                        Make sure to return an instance of the JSON, not the schema itself
                        """
                    )

                    if len(messages) > 1:
                        raise Exception("Only one message is allowed for this task")
                    else:
                        messages[0]["content"] = messages[0]["content"] + "\n\n" + instructor_message

                    res = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=messages,
                    )
                    content = res.choices[0].message.content
                    # Extract JSON between ```json``` markers if present
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]

                    parsed = json.loads(content)
                    return response_model.model_validate(parsed), res

    return DeepseekClient()
###############################################

class LLMVerificationError(Exception):
    pass

class ChatMessage(BaseModel):
    role: str
    content: str

client = instructor.from_litellm(completion)

TEMPLATE_PATH = "llm"

class LLMModel:
    """
    LLMModel that wraps LiteLLM and Instructor for structured output
    Also implements a caching layer
    """
    
    def __init__(
        self,
        use_cache: bool = False,
        configpath: Path = Path(__file__).parent / "cache.yaml",
        dbpath: Path = Path(__file__).parent / "llm_cache.db",
        cache: Optional[LLMCache] = None
    ) -> None:
        """
        Initialize the LLM model with the specified provider and configuration.
        
        Args:
            provider: The name of the model provider (e.g., "openai", "anthropic")
        """
        self.use_cache = use_cache
        self.config = self._read_config(configpath)
        self.cost = 0
        
        # Use provided cache or create new one if use_cache is True
        self.cache = cache if cache is not None else (LLMCache(dbpath) if use_cache else None)
        
        # Add call chain tracking
        self.call_chain = []

    def get_cost(self) -> float:
        return self.cost

    def _read_config(self, fp: Path):
        return {}

    def _get_caller_info(self):
        frame = inspect.currentframe()
        caller_frame = frame.f_back.f_back  # Go back one more frame
        caller_function = caller_frame.f_code.co_name
        caller_filename = caller_frame.f_code.co_filename
        return caller_filename, caller_function

    def _update_llm_stats(self, usage):
        pass

    def _handle_cache(self,
                     prompt: str | List[ChatMessage],
                     model_name: str,
                     response_format: Optional[Type[BaseModel]],
                     use_cache: bool,
                     delete_cache: bool,
                     key: int,
                     caller_function: str) -> Optional[Any]:
        """Handle caching logic for invoke method."""
        if not self.cache or not use_cache:
            return None
            
        if delete_cache:
            self.cache.delete_entry(caller_function, model_name, prompt, key)
            return None
            
        cached_response = self.cache.get_cached_response(
            caller_function,
            model_name,
            prompt,
            response_format, 
            key
        )
        return cached_response

    def invoke(self, 
               prompt: str | List[ChatMessage],
               *, 
               model_name: str = "gpt-4o", 
               response_format: Optional[Type[BaseModel]] = None,
               use_cache: bool = True,
               delete_cache: bool = False,
               key: int = 0,
               **kwargs) -> Any:
        """Modified invoke method with caching."""
        # Use instance default if use_cache is None
        use_cache = self.use_cache if use_cache is None else use_cache
        
        # Track the call
        caller_filename, caller_function = self._get_caller_info()
        self.call_chain.append((caller_filename, caller_function))
        
        cached_response = self._handle_cache(
            prompt,
            model_name,
            response_format,
            use_cache,
            delete_cache,
            key,
            caller_function
        )
        if cached_response is not None:
            return cached_response

        # Format messages
        if isinstance(prompt, str):
            messages = [{
                "role": "user",
                "content": prompt,
            }]
        elif isinstance(prompt, list):
            messages = [m.dict() for m in prompt]
            
        model_name = SHORT_NAMES[model_name]

        # TODO: hack because litellm/deepseek client does not work with instructor
        if model_name == "deepseek/deepseek-chat":
            print("using deepseek client")
            llm_client = deepseek_instructor()
            model_name = "deepseek-chat"
        else:
            llm_client = client

        print(llm_client)
        res, raw_response = llm_client.chat.completions.create_with_completion(
            model=model_name,
            messages=messages,
            response_model=response_format,
            **kwargs
        )
        cost = raw_response.get("_hidden_params", {}).get("response_cost", 0)
        self.cost += cost

        # Cache the response if enabled
        if self.cache and use_cache:
            self.cache.store_cached_response(
                caller_function,
                model_name,
                prompt,
                res,
                key
            )
        
        return res
    
    def __del__(self):
        """Cleanup database connections on object destruction."""
        if self.cache:
            self.cache.close()

# TODO: change to use generic[t]
# DESIGN: not sure how to enforce this but we should only allow JSON serializable
# args to be passed to the model, to be compatible with Braintrust 
T = TypeVar("T")
class LMP(Generic[T]):
    """
    A language model progsram
    """
    prompt: str
    response_format: T
    templates: Dict = {}

    def _prepare_prompt(self, templates={}, **prompt_args) -> str:        
        return jinja2.Template(self.prompt).render(**prompt_args, **templates)

    def _verify_or_raise(self, res, **prompt_args):
        return True

    def _process_result(self, res, **prompt_args) -> Any:
        return res

    def invoke(self, 
               model: LLMModel,
               model_name: str = "claude",
               max_retries: int = 3,
               retry_delay: int = 1,
               use_cache: bool = False,
               # gonna have to manually specify the args to pass into model.invoke
               # or do some arg merging shit here
               prompt_args: Dict = {}) -> Any:
        prompt = self._prepare_prompt(
            templates=self.templates,
            **prompt_args,
        )

        current_retry = 1
        while current_retry <= max_retries:
            try:
                res = model.invoke(prompt, 
                                   model_name=model_name,
                                   response_format=self.response_format,
                                   use_cache=use_cache)
                self._verify_or_raise(res, **prompt_args)
                return self._process_result(res, **prompt_args)
            
            except LLMVerificationError as e:
                current_retry += 1
                
                if current_retry > max_retries:
                    raise e
                
                # Exponential backoff: retry_delay * (2 ^ attempt)
                current_delay = retry_delay * (2 ** (current_retry - 1))
                time.sleep(current_delay)
                print(f"Retry attempt {current_retry}/{max_retries} after error: {str(e)}. Waiting {current_delay}s")
 