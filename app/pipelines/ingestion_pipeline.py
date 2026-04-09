from __future__ import annotations

from app.core.config import settings
from app.ingestion.loaders import load_text_file
from app.ingestion.chunker import TextChunker
from app.utils.file_utils import list_text_files
from app.utils.id_utils import stable_id
from app.stores.document_store import ChunkRecord


class IngestionPipeline:
    def __init__(self, document_store, embedder, vector_store, graph_index) -> None:
        self.document_store = document_store
        self.embedder = embedder
        self.vector_store = vector_store
        self.graph_index = graph_index
        self.chunker = TextChunker(settings.chunk_size, settings.chunk_overlap)

    def ingest(self, data_dir: str) -> int:
        chunk_ids = []
        documents = []
        metadatas = []

        for path in list_text_files(data_dir):
            text = load_text_file(path)
            chunks = self.chunker.chunk(text)
            for idx, chunk in enumerate(chunks):
                chunk_id = stable_id(f"{path}:{idx}:{chunk[:80]}")
                meta = {"source": str(path), "chunk_index": idx}
                self.document_store.add(ChunkRecord(chunk_id=chunk_id, content=chunk, metadata=meta))
                self.graph_index.add_chunk(chunk_id, chunk)
                chunk_ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append(meta)

        if documents:
            matrix = self.embedder.fit_transform(documents)
            self.vector_store.build(chunk_ids, documents, metadatas, matrix)
        return len(documents)
