import faiss
import numpy as np
import json

class FAISSIndex:
    def __init__(self, embedding_dim):
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.passages = []

    def add_embeddings(self, embeddings, passages):
        """Add embeddings and corresponding passages"""
        self.index.add(embeddings)
        self.passages.extend(passages)

    def search(self, query_embedding, top_k):
        """Search for top-k similar passages"""
        scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        retrieved_passages = [self.passages[idx] for idx in indices[0]]
        return retrieved_passages, scores[0]

    def save(self, path):
        """Save index and passages"""
        faiss.write_index(self.index, path + '_index.faiss')
        with open(path + '_passages.json', 'w') as f:
            json.dump(self.passages, f)

    def load(self, path):
        """Load index and passages"""
        self.index = faiss.read_index(path + '_index.faiss')
        with open(path + '_passages.json', 'r') as f:
            self.passages = json.load(f)