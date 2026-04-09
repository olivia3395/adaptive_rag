from app.ingestion.chunker import TextChunker


def test_chunker_splits_long_text():
    chunker = TextChunker(chunk_size=20, chunk_overlap=5)
    chunks = chunker.chunk("abcdefghijklmnopqrstuvwxyz0123456789")
    assert len(chunks) >= 2
