from typing import List, Dict
import re

def parse_and_chunk(text: str, chunk_size: int = 300, overlap: int = 60) -> List[Dict[str, any]]:
    """
    Parses document text into contextual overlapping chunks.
    """
    clean_text = re.sub(r'\r\n', '\n', text)
    clean_text = re.sub(r'[ \t]+', ' ', clean_text).strip()
    if not clean_text:
        return []

    words = clean_text.split()
    total_words = len(words)

    if total_words <= chunk_size:
        return [{
            "index": 0,
            "content": clean_text,
            "word_count": total_words
        }]

    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    idx = 0

    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk_words = words[start:end]
        chunk_content = ' '.join(chunk_words)
        chunks.append({
            "index": idx,
            "content": chunk_content,
            "word_count": len(chunk_words)
        })
        idx += 1
        if end >= total_words:
            break
        start += step

    return chunks
