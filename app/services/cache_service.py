import os
import pickle
from typing import Optional
from services.ai_service import embedding
from core.logging_config import logger
import numpy as np

# Simple local cache for demonstration
CACHE_FILE = "semantic_cache.pkl"

class SemanticCache:
    def __init__(self, threshold: float = 0.98):
        self.threshold = threshold
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "rb") as f:
                return pickle.load(f)
        return [] # List of (embedding, query, answer)

    def _save_cache(self):
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(self.cache, f)

    def get(self, query: str) -> Optional[str]:
        if not self.cache:
            return None
            
        query_embedding = embedding.embed_query(query)
        
        for cached_emb, cached_query, cached_answer in self.cache:
            # Simple cosine similarity
            similarity = np.dot(query_embedding, cached_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_emb)
            )
            
            if similarity > self.threshold:
                logger.info(f"Semantic Cache Hit! Similarity: {similarity:.4f}")
                return cached_answer
        return None

    def set(self, query: str, answer: str):
        if not answer or len(answer) < 10:
            return 
        query_embedding = embedding.embed_query(query)
        self.cache.append((query_embedding, query, answer))
        self._save_cache()

semantic_cache = SemanticCache()
