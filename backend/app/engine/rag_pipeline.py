import json
import sqlite3
import re
from typing import Dict, List, Tuple
from app.engine.vector_store import vector_store
from app.config import SIMILARITY_THRESHOLD

STOP_WORDS = {
    "the", "and", "is", "in", "it", "of", "to", "for", "with", "on", "at", "by", 
    "from", "up", "about", "into", "over", "after", "what", "which", "who", "whom", 
    "this", "that", "these", "those", "am", "are", "was", "were", "be", "been", 
    "being", "have", "has", "had", "do", "does", "did", "can", "could", "shall", 
    "should", "will", "would", "may", "might", "must", "how", "why", "when", "where"
}

def execute_rag_pipeline(conn: sqlite3.Connection, question: str, top_k: int = 4) -> Dict[str, any]:
    """
    Retrieval-Augmented Generation execution with citation grounding and hallucination guardrails.
    """
    question_clean = question.strip()
    q_vec = vector_store.embed(question_clean)
    all_q_tokens = set(re.sub(r'[^a-zA-Z0-9\s]', ' ', question_clean.lower()).split())
    
    # Meaningful content terms excluding stop words
    content_tokens = {t for t in all_q_tokens if t not in STOP_WORDS and len(t) > 2}

    cursor = conn.cursor()
    cursor.execute("""
        SELECT c.id, c.document_id, c.chunk_index, c.content, c.embedding,
               d.title as doc_title
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
    """)
    all_chunks = cursor.fetchall()

    if not all_chunks:
        return {
            "answer": "The knowledge base is currently empty. Please upload source documents to enable question-answering.",
            "confidence": 0.0,
            "hallucination_flag": True,
            "citations": [],
            "chunks_evaluated": 0
        }

    scored_candidates = []
    for row in all_chunks:
        emb = json.loads(row["embedding"])
        sim = vector_store.cosine_similarity(q_vec, emb)

        # Content token overlap
        chunk_words = set(re.sub(r'[^a-zA-Z0-9\s]', ' ', row["content"].lower()).split())
        overlap = len(content_tokens.intersection(chunk_words)) if content_tokens else 0
        overlap_score = (overlap / max(1, len(content_tokens))) if content_tokens else 0.0

        combined_score = (0.7 * sim) + (0.3 * overlap_score)
        if combined_score > 0.05:
            scored_candidates.append({
                "chunk_id": row["id"],
                "document_id": row["document_id"],
                "doc_title": row["doc_title"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "score": combined_score,
                "overlap": overlap
            })

    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = scored_candidates[:top_k]

    # Guardrail Check: Must meet similarity threshold AND have verified content keyword overlap
    has_evidence = (
        top_candidates and 
        top_candidates[0]["score"] >= SIMILARITY_THRESHOLD and 
        top_candidates[0]["overlap"] > 0
    )

    if not has_evidence:
        return {
            "answer": f"The provided corpus does not contain sufficient verified evidence to reliably answer: \"{question_clean}\". To avoid hallucination, no speculative response was generated.",
            "confidence": round(top_candidates[0]["score"] if top_candidates else 0.0, 3),
            "hallucination_flag": True,
            "citations": [],
            "chunks_evaluated": len(all_chunks)
        }

    # Extract sentences for synthesized grounded answer
    citations = []
    answer_sentences = []
    citation_idx = 1

    for cand in top_candidates:
        # Split chunk into sentences
        raw_sentences = re.split(r'(?<=[.!?])\s+', cand["content"])
        best_sentence = None
        best_match_count = -1

        for sent in raw_sentences:
            s_words = set(sent.lower().split())
            m_count = len(content_tokens.intersection(s_words))
            if m_count > best_match_count and len(sent.split()) > 4:
                best_match_count = m_count
                best_sentence = sent.strip()

        if not best_sentence and raw_sentences:
            best_sentence = raw_sentences[0].strip()

        if best_sentence:
            answer_sentences.append(f"{best_sentence} [{citation_idx}]")
            snippet = cand["content"][:240] + ("..." if len(cand["content"]) > 240 else "")
            citations.append({
                "citation_num": citation_idx,
                "chunk_id": cand["chunk_id"],
                "document_id": cand["document_id"],
                "document_title": cand["doc_title"],
                "snippet": snippet,
                "relevance_score": round(cand["score"], 4)
            })
            citation_idx += 1

    synthesized_answer = " ".join(answer_sentences)
    avg_confidence = round(sum(c["score"] for c in top_candidates) / len(top_candidates), 3)

    return {
        "answer": synthesized_answer,
        "confidence": min(0.98, max(0.40, avg_confidence * 1.3)),
        "hallucination_flag": False,
        "citations": citations,
        "chunks_evaluated": len(all_chunks)
    }
