from pydantic import BaseModel, Field
from typing import List, Optional

class DocumentUpload(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    filename: Optional[str] = "manual_entry.txt"
    content: str = Field(..., min_length=1)
    mime_type: Optional[str] = "text/plain"

class DocumentResponse(BaseModel):
    id: str
    filename: str
    title: str
    mime_type: str
    word_count: int
    total_chunks: int
    created_at: str

class ChunkResponse(BaseModel):
    id: str
    document_id: str
    chunk_index: int
    content: str

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = 4

class CitationItem(BaseModel):
    citation_num: int
    chunk_id: str
    document_id: str
    document_title: str
    snippet: str
    relevance_score: float

class QueryResponse(BaseModel):
    query_id: str
    question: str
    answer: str
    confidence: float
    hallucination_flag: bool
    citations: List[CitationItem]
    chunks_evaluated: int
