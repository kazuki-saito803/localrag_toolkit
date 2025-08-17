from pydantic import BaseModel
from typing import List
from fastapi import UploadFile

class IndexRequest(BaseModel):
    index_name: str
    documents: List[UploadFile]

class QueryRequest(BaseModel):
    question: str
    top_k: int
    index_name: str

class DocItem(BaseModel):
    content: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    docs: List[DocItem]