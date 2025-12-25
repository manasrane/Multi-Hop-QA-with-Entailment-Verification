from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
import os

class BiEncoderRetriever:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.passages = []
        self.passage_embeddings = None

    def encode_passages(self, passages):
        """Encode all passages and build FAISS index"""
        self.passages = passages
        self.passage_embeddings = self.model.encode(passages, convert_to_numpy=True, show_progress_bar=True)
        
        # Build FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(self.passage_embeddings)

    def retrieve(self, query, top_k=10):
        """Retrieve top-k passages for a query"""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        scores, indices = self.index.search(query_embedding, top_k)
        retrieved_passages = [self.passages[idx] for idx in indices[0]]
        return retrieved_passages, scores[0]

    def save_index(self, path):
        """Save FAISS index and passages"""
        faiss.write_index(self.index, path + '_index.faiss')
        with open(path + '_passages.json', 'w') as f:
            json.dump(self.passages, f)

    def load_index(self, path):
        """Load FAISS index and passages"""
        self.index = faiss.read_index(path + '_index.faiss')
        with open(path + '_passages.json', 'r') as f:
            self.passages = json.load(f)