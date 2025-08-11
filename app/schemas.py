from pydantic import BaseModel
from typing import List

class IndexRequest(BaseModel):
    documents: List[str]

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str