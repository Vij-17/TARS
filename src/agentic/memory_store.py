"""Memory store utilities: Chroma-backed vector DB with sentence-transformers embeddings.

This module provides a MemoryStore class with a safe fallback to an in-memory store
when Chroma or sentence-transformers are not available. It exposes simple store
and retrieve operations used by the agent controller.
"""
from typing import Optional, List, Tuple, Dict
import time
import uuid

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None


class MemoryStore:
    def __init__(self, collection_name: str = "tars_memory", embed_model: str = "all-MiniLM-L6-v2"):
        self.collection_name = collection_name
        self._use_chroma = chromadb is not None and SentenceTransformer is not None

        if self._use_chroma:
            # initialize embedder and chroma client
            self.embedder = SentenceTransformer(embed_model)
            # default client; user can configure CHROMA settings via env
            self.client = chromadb.Client()
            self.col = self.client.get_or_create_collection(name=collection_name)
        else:
            # fallback simple in-memory store
            self._mem = []  # list of dicts: {id, ts, summary, orig, metadata}

    def _summarize(self, llm_generate_fn, text: str) -> str:
        # Default simple summarizer: ask the provided LLM if available
        if llm_generate_fn is None:
            # no LLM available; truncate
            return text[:512]
        prompt = "Summarize this text in 2 short bullets:\n\n" + text
        try:
            return llm_generate_fn(prompt, max_new_tokens=128)
        except Exception:
            return text[:512]

    def store_event(self, text: str, metadata: Optional[Dict] = None, summarizer=None, llm_generate_fn=None) -> str:
        summary = text
        if summarizer is not None:
            summary = summarizer(llm_generate_fn, text)
        elif llm_generate_fn is not None:
            summary = self._summarize(llm_generate_fn, text)

        _id = str(uuid.uuid4())
        meta = {"ts": time.time(), **(metadata or {}), "summary": summary, "orig": text}

        if self._use_chroma:
            emb = self.embedder.encode([summary]).tolist()
            self.col.add(ids=[_id], documents=[summary], metadatas=[meta], embeddings=emb)
        else:
            self._mem.append({"id": _id, "summary": summary, "orig": text, "metadata": meta})

        return _id

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, Dict]]:
        """Return list of (summary, metadata) tuples."""
        if self._use_chroma:
            q_emb = self.embedder.encode([query]).tolist()
            res = self.col.query(query_embeddings=q_emb, n_results=k)
            docs = []
            if res and "documents" in res:
                for d, m in zip(res["documents"][0], res["metadatas"][0]):
                    docs.append((d, m))
            return docs
        else:
            # simple substring ranking fallback
            found = []
            for e in self._mem:
                score = 1.0 if query.lower() in e['summary'].lower() or query.lower() in e['orig'].lower() else 0.0
                found.append((score, e))
            found.sort(key=lambda x: x[0], reverse=True)
            return [(e['summary'], e['metadata']) for s, e in found[:k]]


__all__ = ["MemoryStore"]
