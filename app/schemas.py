from pydantic import BaseModel
from typing import List

class IndexRequest(BaseModel):
    index_name: str
    documents: List[str]

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    docs: List[str]