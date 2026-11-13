from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_api_health():
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_full_rag_workflow():
    # 1. Upload Document
    doc_payload = {
        "title": "Quantum Error Correction Principles",
        "filename": "quantum_surface_code.md",
        "content": "Surface codes arrange physical qubits on a 2D lattice to protect logical qubits from bit-flip and phase-flip errors. Syringe syndrome extraction circuits measure stabilizer operators repetitively.",
        "mime_type": "text/markdown"
    }
    upload_res = client.post("/api/v1/corpus", json=doc_payload)
    assert upload_res.status_code == 201
    doc_data = upload_res.json()
    doc_id = doc_data["id"]
    assert doc_data["total_chunks"] >= 1

    # 2. Get Chunks
    chunks_res = client.get(f"/api/v1/corpus/{doc_id}/chunks")
    assert chunks_res.status_code == 200
    assert len(chunks_res.json()) >= 1

    # 3. Query RAG
    query_payload = {
        "question": "How do surface codes protect qubits from bit-flip errors?",
        "top_k": 3
    }
    query_res = client.post("/api/v1/query", json=query_payload)
    assert query_res.status_code == 200
    q_data = query_res.json()
    assert q_data["hallucination_flag"] is False
    assert len(q_data["citations"]) >= 1
    assert q_data["citations"][0]["document_id"] == doc_id

    # 4. Check Query History
    hist_res = client.get("/api/v1/query/history")
    assert hist_res.status_code == 200
    assert len(hist_res.json()) >= 1

    # 5. Clean up Document
    del_res = client.delete(f"/api/v1/corpus/{doc_id}")
    assert del_res.status_code == 200
