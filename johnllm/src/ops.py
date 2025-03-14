from pydantic import BaseModel
from typing import List

class Op(BaseModel):
    name: str

class OpsList(BaseModel):
    ops: List[Op]