"""Normalized bi-encoder retrieval backed by a content-addressed Chroma index."""
from __future__ import annotations

import json
from pathlib import Path
from .ingest import corpus_fingerprint, read_markdown_documents
from .schemas import RetrievedChunk


class DenseRetriever:
    def __init__(self, docs_dir: str | Path = "docs", index_dir: str | Path = ".rag_index",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        import chromadb
        self.docs_dir = Path(docs_dir)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=str(self.index_dir))
        self.collection_name = "northstar_rag_v2"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "description": "Normalized MiniLM vectors"},
        )
        self._ensure_index()

    def _ensure_index(self) -> None:
        manifest = self.index_dir / "fingerprint.json"
        fresh = corpus_fingerprint(self.docs_dir, self.model_name)
        chunks = read_markdown_documents(self.docs_dir)
        cached = {}
        if manifest.exists():
            try:
                cached = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                cached = {}
        if cached.get("hash") == fresh and self.collection.count() == len(chunks):
            return
        # Full rebuild avoids duplicate/upsert drift when documents are removed or edited.
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "description": "Normalized MiniLM vectors"},
        )
        texts = [x.content for x in chunks]
        vectors = self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        # Batches avoid Chroma's platform-dependent max request size.
        for offset in range(0, len(chunks), 50):
            batch = chunks[offset:offset + 50]
            self.collection.upsert(
                ids=[x.chunk_id for x in batch],
                documents=[x.content for x in batch],
                metadatas=[x.metadata | {"title": x.title} for x in batch],
                embeddings=[vec.tolist() for vec in vectors[offset:offset + 50]],
            )
        manifest.write_text(json.dumps({"hash": fresh, "model": self.model_name,
                                        "chunks": len(chunks)}, indent=2), encoding="utf-8")

    def search(self, query: str, k: int = 20) -> list[RetrievedChunk]:
        count = self.collection.count()
        if count == 0:
            return []
        query_vector = self.encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        found = self.collection.query(query_embeddings=[query_vector.tolist()],
                                      n_results=max(1, min(k, count)),
                                      include=["documents", "metadatas", "distances"])
        results = []
        for rank, (doc_id, content, meta, distance) in enumerate(zip(
            found["ids"][0], found["documents"][0],
            found["metadatas"][0], found["distances"][0]), 1):
            results.append(RetrievedChunk(
                chunk_id=doc_id, content=content, title=meta.get("title", doc_id),
                score=1.0 - float(distance), source="internal_doc",
                stage1_rank=rank, metadata={key: value for key, value in meta.items() if value is not None},
            ))
        return results
