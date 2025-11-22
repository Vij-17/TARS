"""RAG Retrieval Tool"""

import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGRetrieval:
    """
    RAG (Retrieval Augmented Generation) system.
    Stores documents in vector database for semantic search.
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2', embedding_dim=384):
        self.embed_model = SentenceTransformer(model_name)
        self.embedding_dim = embedding_dim
        
        # FAISS index for fast similarity search
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents = []
        
    def add_documents(self, documents):
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of strings or dicts with 'text' key
        """
        
        if not documents:
            return
        
        # Extract text
        texts = []
        for doc in documents:
            if isinstance(doc, str):
                texts.append(doc)
            elif isinstance(doc, dict) and 'text' in doc:
                texts.append(doc['text'])
            else:
                texts.append(str(doc))
        
        # Embed
        embeddings = self.embed_model.encode(texts, convert_to_numpy=True)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(documents)
        
        print(f"Added {len(documents)} documents. Total: {len(self.documents)}")
    
    def retrieve(self, query, top_k=5):
        """
        Retrieve most relevant documents.
        
        Args:
            query: Search query string
            top_k: Number of documents to retrieve
        
        Returns:
            list of retrieved documents
        """
        
        if len(self.documents) == 0:
            return ["RAG database is empty. Please add documents first."]
        
        # Embed query
        if isinstance(query, str):
            query_text = query
        else:
            query_text = str(query)
        
        query_embedding = self.embed_model.encode([query_text], convert_to_numpy=True)
        
        # Search
        k = min(top_k, len(self.documents))
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Retrieve documents
        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        
        return results
    
    def clear(self):
        """Clear all documents"""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.documents = []
    
    def __call__(self, query):
        """Make callable"""
        results = self.retrieve(query, top_k=5)
        
        # Format results
        if results:
            formatted = "\n\n---\n\n".join([
                str(r) if isinstance(r, str) else r.get('text', str(r))
                for r in results
            ])
            return formatted
        
        return "No relevant documents found"