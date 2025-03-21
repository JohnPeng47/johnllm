from typing import List, Any, Dict, TypeVar, Generic
from abc import ABC, abstractmethod

from pydantic import BaseModel
from ..ops import OpsList, Op
from ..core import LLMModel, LMP

# TODO: maybe should get rid of Item? Its gonna be a string most of the time
# TODO: make this generic
class LMGroupList(BaseModel, ABC):
    @abstractmethod
    def get_item_ids(self) -> List[str]:
        """Returns the prompt ids for each item"""
        pass    

    @abstractmethod
    def get_groups(self):
        pass

class LMGroup(BaseModel, ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def items(self) -> List[str]:
        pass

class LMItem(BaseModel, ABC):
    """
    Represents an object inside of a LLM/LMP. 
    In an LLM, the representation is a string with an ID
    In the LMP code, the representation is *just* the ID
    -> why? gives us an interface to do add/remove operations on the LMItem
    """
    prompt_id: str
    original_id: str
    
    @abstractmethod
    def to_prompt_str(self) -> str:
        """Returns the string representation of the item"""
        pass

    def __eq__(self, other: "LMItem") -> bool:
        return self.prompt_id == other.prompt_id

# Define a type variable for the item type
T = TypeVar('T')

class LMTransform(LMP, ABC, Generic[T]):
    response_format: OpsList
    """
    Implements a generic LLM transformation operation, iteratively transforming items
    with the assumption that some items remain leftover after each iteration
    """
    def __init__(self):
        self._iters = 0
        self._soft_limit = 3 # default number of refinement passes
        self._hard_limit = 10 # just in case our should stop condition fails

        if not issubclass(self.response_format, OpsList):
            raise Exception("response_format must be a subclass of OpsList")

        # TODO: template string checking code is not working as intended
        if "{{items}}" not in self.prompt:
            raise Exception("prompt template must contain '{{items}}' keyword")

        super().__init__()

    @abstractmethod
    def _apply_ops(self, ops: OpsList, items: List[LMItem]) -> List[LMItem]:
        pass
        
    @abstractmethod
    def _convert_to_lmitem(self, items: List[T]) -> List[LMItem]:
        pass

    def _should_stop(self, leftover: List[T]) -> bool:
        """Stopping condition depends on any of items or _num_iters"""
        return self._iters == self._soft_limit
    
    def _prepare_prompt(self, items: List[LMItem], **kwargs) -> str:
        """Prepares the prompt by formatting the items into strings"""
        
        return super()._prepare_prompt(
            items="\n".join([item.to_prompt_str() for item in items]),
            **kwargs
        )
    
    def invoke(self,
               model: LLMModel,
               model_name: str = "claude",
               use_cache: bool = False,
               items: List[T] = []) -> List[T]:
        if not items:
            raise Exception("items must be a non-empty list")

        items = self._convert_to_lmitem(items)
        transformed_items = []
        leftover_items = items
        stop = False
        while not stop:
            prompt = self._prepare_prompt(items=leftover_items)  
            res = model.invoke(prompt,
                             model_name=model_name,
                             response_format=self.response_format, 
                             use_cache=use_cache)
            new_transformed = self._apply_ops(res.ops, leftover_items)
            transformed_items = transformed_items + new_transformed
            print("[TRANSFORMED ITEMS]: ", len(transformed_items))

            leftover_items = [item for item in leftover_items if item not in transformed_items]
            print("[LEFTOVER ITEMS]: ", len(leftover_items))

            stop = self._should_stop(leftover_items)
            self._iters += 1
            if self._iters == self._hard_limit or len(leftover_items) == 0:
                break

        return transformed_items

__all__ = ["LMTransform", "LMItem", "LMGroup", "LMGroupList"]
