from typing import Any, Dict, Optional, Type, List, Tuple, Generic, TypeVar
import inspect
from pathlib import Path
import sqlite3
import hashlib
import json
import functools
import time
import jinja2

from pydantic import BaseModel
import tiktoken

import instructor   
from litellm import completion, completion_cost
from litellm.types.utils import ModelResponse
from pydantic import BaseModel

from .models import CompletionUsage

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
        dbpath: Path = Path(__file__).parent / "llm_cache.db"
    ) -> None:
        """
        Initialize the LLM model with the specified provider and configuration.
        
        Args:
            provider: The name of the model provider (e.g., "openai", "anthropic")
        """
        self.use_cache = use_cache
        self.config = self._read_config(configpath)
        self.cost = 0

        # Initialize cache-related attributes
        self.cache_enabled_functions = self._get_cache_enabled_functions()
        print("Enabled functions: ", "\n".join([f for f, enabled in self.cache_enabled_functions.items() if enabled]))

        self.db_connection = None
        
        # Initialize cache database
        self._initialize_cache(dbpath)
        
        # Add call chain tracking
        self.call_chain = []

    def get_cost(self) -> float:
        return self.cost

    def _read_config(self, fp: Path):
        # with open(fp, "r") as f:
        #     config = yaml.safe_load(f)
        # return config
        # Cache turned off
        return {}

    def _get_caller_info(self):
        frame = inspect.currentframe()
        caller_frame = frame.f_back.f_back  # Go back one more frame
        caller_function = caller_frame.f_code.co_name
        caller_filename = caller_frame.f_code.co_filename

        return caller_filename, caller_function
        
    def _get_cache_enabled_functions(self) -> Dict[str, bool]:
        """Extract function names and their cache states from config."""
        cache_states = {}
        for func_name, settings in self.config.items():
            # Look for cache setting in the list of dictionaries
            cache_setting = next((item.get('cache') 
                                for item in settings 
                                if isinstance(item, dict) and 'cache' in item), 
                               False)
            cache_states[func_name] = cache_setting
        return cache_states

    def _initialize_cache(self, dbpath: Path) -> None:
        """Initialize SQLite connection and create cache table if it doesn't exist."""
        self.db_connection = sqlite3.connect(dbpath)
        cursor = self.db_connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                function_name TEXT,
                model_name TEXT,
                prompt_hash TEXT,
                response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (function_name, model_name, prompt_hash)
            )
        """)
        self.db_connection.commit()

    def _hash_prompt(self, prompt: str, model: str, key: int = 0) -> str:
        """Create a consistent hash of the prompt with the key used for iterative prompts."""
        return hashlib.sha256(prompt.encode() + model.encode() + str(key).encode()).hexdigest()

    def _delete_hash(self, function_name: str, model_name: str, prompt_hash: str) -> None:
        """Delete a specific cache entry by its hash."""
        if not self.db_connection:
            return
            
        cursor = self.db_connection.cursor()
        cursor.execute(
            "DELETE FROM llm_cache WHERE function_name = ? AND model_name = ? AND prompt_hash = ?",
            (function_name, model_name, prompt_hash)
        )
        self.db_connection.commit()

    def _get_cached_response(self, function_name: str, model_name: str, prompt_hash: str) -> Optional[str]:
        """Retrieve cached response if it exists."""
        if not self.db_connection:
            return None
            
        cursor = self.db_connection.cursor()
        cursor.execute(
            "SELECT response FROM llm_cache WHERE function_name = ? AND model_name = ? AND prompt_hash = ?",
            (function_name, model_name, prompt_hash)
        )
        result = cursor.fetchone()
        return json.loads(result[0]) if result else None

    def _cache_response(self, function_name: str, model_name: str, prompt_hash: str, response: Any) -> None:
        """Store response in cache."""
        if not self.db_connection:
            return
                    
        cursor = self.db_connection.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO llm_cache (function_name, model_name, prompt_hash, response) VALUES (?, ?, ?, ?)",
            (function_name, model_name, prompt_hash, json.dumps(response))
        )
        self.db_connection.commit()

    def cache_llm_response(func):
        """Method decorator to handle LLM response caching."""
        @functools.wraps(func)
        def wrapper(self, 
                    prompt: str, 
                    *, 
                    model_name: str = "gpt-4o", 
                    response_format: Optional[Type[BaseModel]] = None, 
                    use_cache: Optional[bool] = None,
                    delete_cache: bool = False,
                    key: int = 0,
                    **kwargs):
            
            # Use instance default if use_cache is None
            use_cache = self.use_cache if use_cache is None else use_cache
            
            # Track the call
            caller_filename, caller_function = self._get_caller_info()
            self.call_chain.append((caller_filename, caller_function))
            
            # Check if caching is enabled for this function
            # cache_enabled = (use_cache and 
            #                caller_function in self.cache_enabled_functions and 
            #                self.cache_enabled_functions[caller_function])
            cache_enabled = use_cache
            
            if delete_cache:
                self._delete_hash(caller_function, model_name, self._hash_prompt(prompt, model_name, key=key))
                
            elif not delete_cache and cache_enabled:
                # Check cache for existing response
                prompt_hash = self._hash_prompt(prompt, model_name, key=key)
                cached_response = self._get_cached_response(caller_function, model_name, prompt_hash)
                                
                if cached_response is not None:
                    caller = inspect.stack()[1]  # Get immediate caller
                    print(f"Returning from cache[LLM]:")
                    print(f"|---> Called from {caller.filename}:{caller.lineno} in {caller.function}")

                    # If response is a Pydantic model, reconstruct it
                    if response_format is not None:
                        return response_format.model_validate(cached_response)
                    
                    return cached_response["content"]
            
            # Get response from LLM
            res = func(self, 
                       prompt, 
                       model_name=model_name, 
                       response_format=response_format, 
                       **kwargs)
            
            # Prepare response for caching
            if isinstance(res, ModelResponse):
                res = res.choices[0].message.content
                cached_response = res
            elif isinstance(res, BaseModel):
                cached_response = res.model_dump()
            elif isinstance(res, Tuple):
                if isinstance(res[0], BaseModel):
                    cached_response = res[0].model_dump()
                else:
                    raise Exception(f"Unsupported return type: {type(res)}")
            else:
                raise Exception(f"Unsupported return type: {type(res)}")
                
            # Cache the response if enabled
            if cache_enabled:
                print("Caching response: ", model_name, prompt[:20], key, prompt_hash[:4])
                self._cache_response(caller_function, model_name, prompt_hash, cached_response)
                
            return res
            
        return wrapper

    def _update_llm_stats(self, usage):
        pass

    @cache_llm_response
    def invoke(self, 
               prompt: str | List[ChatMessage],
               *, 
               model_name: str = "gpt-4o", 
               response_format: Optional[Type[BaseModel]] = None,
               use_cache: bool = True,
               **kwargs) -> Any:
        """Modified invoke method with caching."""

        if isinstance(prompt, str):
            messages = [{
                "role": "user",
                "content": prompt,
            }]
        elif isinstance(prompt, list):
            messages = [m.dict() for m in prompt]
            
        model_name = SHORT_NAMES[model_name]
        res, raw_response = client.chat.completions.create_with_completion(
            model=model_name,
            messages=messages,
            response_model=response_format,
            **kwargs
        )
        self.cost += raw_response._hidden_params["response_cost"]
        print("Cost: ", raw_response._hidden_params["response_cost"])

        return res
    
    def __del__(self):
        """Cleanup database connections on object destruction."""
        # import traceback
        # print("Cleaning up database connection. Called from:")
        # traceback.print_stack()

        if self.db_connection:
            self.db_connection.close()

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
               **prompt_args) -> Any:
        prompt = self._prepare_prompt(
            templates=self.templates,
            **prompt_args
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
    