import sqlite3
import hashlib
import json
import functools
import inspect
from pathlib import Path
from typing import Any, Optional, Type
from pydantic import BaseModel
from litellm.types.utils import ModelResponse

class LLMCache:
    """Handles caching of LLM responses"""
    
    def __init__(self, dbpath: Path) -> None:
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

    def get_cached_response(self, 
                          function_name: str, 
                          model_name: str, 
                          prompt: str, 
                          response_format: Optional[Type[BaseModel]] = None,
                          key: int = 0) -> Optional[Any]:
        """Get cached response if it exists and handle response format conversion."""
        cached_response = self.get_response(function_name, model_name, prompt, key)
                            
        if cached_response is not None:
            caller = inspect.stack()[1]  # Get immediate caller
            print(f"Returning from cache[LLM]:")
            print(f"|---> Called from {caller.filename}:{caller.lineno} in {caller.function}")

            # If response is a Pydantic model, reconstruct it
            if response_format is not None:
                return response_format.model_validate(cached_response)
            
            return cached_response["content"]
        return None

    def store_cached_response(self, 
                            function_name: str, 
                            model_name: str, 
                            prompt: str, 
                            response: Any, 
                            key: int = 0) -> None:
        """Store response in cache with proper formatting."""
        # Prepare response for caching
        if isinstance(response, ModelResponse):
            response = response.choices[0].message.content
            cached_response = response
        elif isinstance(response, BaseModel):
            cached_response = response.model_dump()
        elif isinstance(response, tuple):
            if isinstance(response[0], BaseModel):
                cached_response = response[0].model_dump()
            else:
                raise Exception(f"Unsupported return type: {type(response)}")
        else:
            raise Exception(f"Unsupported return type: {type(response)}")
            
        # Cache the response
        print("Caching response: ", model_name, prompt[:20], key)
        self.store_response(function_name, model_name, prompt, cached_response, key)

    def get_response(self, function_name: str, model_name: str, prompt: str, key: int = 0) -> Optional[Any]:
        """Raw get response from cache."""
        prompt_hash = self._hash_prompt(prompt, model_name, key)
        cursor = self.db_connection.cursor()
        cursor.execute(
            "SELECT response FROM llm_cache WHERE function_name = ? AND model_name = ? AND prompt_hash = ?",
            (function_name, model_name, prompt_hash)
        )
        result = cursor.fetchone()
        return json.loads(result[0]) if result else None

    def store_response(self, function_name: str, model_name: str, prompt: str, response: Any, key: int = 0) -> None:
        """Raw store response in cache."""
        prompt_hash = self._hash_prompt(prompt, model_name, key)
        cursor = self.db_connection.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO llm_cache (function_name, model_name, prompt_hash, response) VALUES (?, ?, ?, ?)",
            (function_name, model_name, prompt_hash, json.dumps(response))
        )
        self.db_connection.commit()

    def delete_entry(self, function_name: str, model_name: str, prompt: str, key: int = 0) -> None:
        """Delete a specific cache entry."""
        prompt_hash = self._hash_prompt(prompt, model_name, key)
        cursor = self.db_connection.cursor()
        cursor.execute(
            "DELETE FROM llm_cache WHERE function_name = ? AND model_name = ? AND prompt_hash = ?",
            (function_name, model_name, prompt_hash)
        )
        self.db_connection.commit()

    def close(self):
        """Close the database connection."""
        if self.db_connection:
            self.db_connection.close()