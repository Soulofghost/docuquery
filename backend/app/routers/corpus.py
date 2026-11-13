import uuid
import json
from fastapi import APIRouter, HTTPException
from typing import List
from app.database import get_db_connection
from app.schemas import DocumentUpload, DocumentResponse, ChunkResponse
from app.engine.chunker import parse_and_chunk
from app.engine.vector_store import vector_store

router = APIRouter(prefix="/corpus", tags=["Corpus Management"])

@router.get("", response_model=List[DocumentResponse])
def list_documents():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents ORDER BY created_at DESC")
        rows = cursor.fetchall()
        return [
            DocumentResponse(
                id=r["id"],
                filename=r["filename"],
                title=r["title"],
                mime_type=r["mime_type"],
                word_count=r["word_count"],
                total_chunks=r["total_chunks"],
                created_at=r["created_at"]
            ) for r in rows
        ]

@router.post("", response_model=DocumentResponse, status_code=201)
def upload_document(doc: DocumentUpload):
    doc_id = "doc_" + str(uuid.uuid4())[:8]
    chunks = parse_and_chunk(doc.content)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document content cannot be empty")

    word_count = sum(c["word_count"] for c in chunks)

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO documents (id, filename, title, mime_type, word_count, total_chunks)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (doc_id, doc.filename or "document.txt", doc.title, doc.mime_type or "text/plain", word_count, len(chunks)))

        for c in chunks:
            chunk_id = f"{doc_id}_c{c['index']}"
            emb = vector_store.embed(c["content"])
            cursor.execute("""
                INSERT INTO chunks (id, document_id, chunk_index, content, embedding)
                VALUES (?, ?, ?, ?, ?)
            """, (chunk_id, doc_id, c["index"], c["content"], json.dumps(emb)))

        conn.commit()

        cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        saved = cursor.fetchone()
        return DocumentResponse(
            id=saved["id"],
            filename=saved["filename"],
            title=saved["title"],
            mime_type=saved["mime_type"],
            word_count=saved["word_count"],
            total_chunks=saved["total_chunks"],
            created_at=saved["created_at"]
        )

@router.get("/{doc_id}/chunks", response_model=List[ChunkResponse])
def get_document_chunks(doc_id: str):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, document_id, chunk_index, content FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC", (doc_id,))
        rows = cursor.fetchall()
        if not rows:
            raise HTTPException(status_code=404, detail="Document or chunks not found")
        return [
            ChunkResponse(
                id=r["id"],
                document_id=r["document_id"],
                chunk_index=r["chunk_index"],
                content=r["content"]
            ) for r in rows
        ]

@router.delete("/{doc_id}")
def delete_document(doc_id: str):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        cursor.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        conn.commit()
        return {"status": "success", "deleted_id": doc_id}
