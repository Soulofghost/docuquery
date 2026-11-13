import sqlite3
import json
from app.engine.vector_store import vector_store
from app.engine.rag_pipeline import execute_rag_pipeline

def setup_in_memory_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE documents (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE queries (
            id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            confidence REAL NOT NULL,
            hallucination_flag INTEGER NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE citations (
            id TEXT PRIMARY KEY,
            query_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            citation_num INTEGER NOT NULL,
            snippet TEXT NOT NULL,
            score REAL NOT NULL
        )
    """)

    # Seed knowledge chunk
    doc_id = "doc_test_1"
    cursor.execute("INSERT INTO documents (id, title) VALUES (?, ?)", (doc_id, "Zero Knowledge Cryptography"))
    chunk_content = "Zero-knowledge proofs allow a prover to demonstrate knowledge of a secret to a verifier without revealing the secret itself. zk-SNARKs and zk-STARKs represent non-interactive implementations widely used in blockchain scaling."
    emb = vector_store.embed(chunk_content)
    cursor.execute("INSERT INTO chunks (id, document_id, chunk_index, content, embedding) VALUES (?, ?, ?, ?, ?)",
                   ("chunk_1", doc_id, 0, chunk_content, json.dumps(emb)))
    conn.commit()
    return conn

def test_rag_pipeline_retrieval_and_citation():
    conn = setup_in_memory_db()
    res = execute_rag_pipeline(conn, "What are zero-knowledge proofs and zk-SNARKs?", top_k=2)

    assert res["hallucination_flag"] is False
    assert res["confidence"] > 0.40
    assert len(res["citations"]) >= 1
    assert "[1]" in res["answer"]
    assert res["citations"][0]["document_title"] == "Zero Knowledge Cryptography"

def test_rag_pipeline_hallucination_guardrail():
    conn = setup_in_memory_db()
    # Ask something completely absent from corpus
    res = execute_rag_pipeline(conn, "What is the capital city of Madagascar?", top_k=2)

    assert res["hallucination_flag"] is True
    assert "does not contain sufficient verified evidence" in res["answer"]
    assert len(res["citations"]) == 0
