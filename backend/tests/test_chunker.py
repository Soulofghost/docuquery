from app.engine.chunker import parse_and_chunk

def test_parse_empty_content():
    assert parse_and_chunk("") == []
    assert parse_and_chunk("    \n\t  ") == []

def test_single_chunk_preservation():
    text = "Retrieval-Augmented Generation bridges vector similarity and large language models."
    chunks = parse_and_chunk(text, chunk_size=50, overlap=10)
    assert len(chunks) == 1
    assert chunks[0]["index"] == 0
    assert chunks[0]["content"] == text
    assert chunks[0]["word_count"] == len(text.split())

def test_multi_chunk_overlap():
    words = [f"token_{i}" for i in range(120)]
    text = " ".join(words)
    chunks = parse_and_chunk(text, chunk_size=40, overlap=10)
    assert len(chunks) > 1

    first_words = chunks[0]["content"].split()
    second_words = chunks[1]["content"].split()
    assert first_words[-10:] == second_words[:10]
