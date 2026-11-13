import uuid
from fastapi import APIRouter
from typing import List
from app.database import get_db_connection
from app.schemas import QueryRequest, QueryResponse, CitationItem
from app.engine.rag_pipeline import execute_rag_pipeline

router = APIRouter(prefix="/query", tags=["RAG QA Engine"])

@router.post("", response_model=QueryResponse)
def ask_question(payload: QueryRequest):
    query_id = "qry_" + str(uuid.uuid4())[:8]

    with get_db_connection() as conn:
        rag_res = execute_rag_pipeline(conn, payload.question, top_k=payload.top_k or 4)

        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO queries (id, question, answer, confidence, hallucination_flag)
            VALUES (?, ?, ?, ?, ?)
        """, (query_id, payload.question, rag_res["answer"], rag_res["confidence"], 1 if rag_res["hallucination_flag"] else 0))

        citation_items = []
        for c in rag_res["citations"]:
            cit_id = "cit_" + str(uuid.uuid4())[:8]
            cursor.execute("""
                INSERT INTO citations (id, query_id, chunk_id, document_id, citation_num, snippet, score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (cit_id, query_id, c["chunk_id"], c["document_id"], c["citation_num"], c["snippet"], c["relevance_score"]))

            citation_items.append(CitationItem(
                citation_num=c["citation_num"],
                chunk_id=c["chunk_id"],
                document_id=c["document_id"],
                document_title=c["document_title"],
                snippet=c["snippet"],
                relevance_score=c["relevance_score"]
            ))

        conn.commit()

        return QueryResponse(
            query_id=query_id,
            question=payload.question,
            answer=rag_res["answer"],
            confidence=rag_res["confidence"],
            hallucination_flag=rag_res["hallucination_flag"],
            citations=citation_items,
            chunks_evaluated=rag_res["chunks_evaluated"]
        )

@router.get("/history")
def get_query_history():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT q.id, q.question, q.confidence, q.hallucination_flag, q.created_at,
                   COUNT(c.id) as citation_count
            FROM queries q
            LEFT JOIN citations c ON q.id = c.query_id
            GROUP BY q.id
            ORDER BY q.created_at DESC
            LIMIT 20
        """)
        return cursor.fetchall()
