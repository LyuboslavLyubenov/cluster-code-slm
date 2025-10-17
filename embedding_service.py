"""
Semantic embedding service using LM Studio embeddings.
Generates embeddings for pattern descriptions for clustering.
"""

import numpy as np
from typing import List

class EmbeddingService:
    """Service for generating semantic embeddings using LM Studio."""
    
    def __init__(self):
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the LM Studio embedding model."""
        try:
            import lmstudio as lms
            self.model = lms.embedding_model("Mungert/Qwen3-Embedding-4B-GGUF")
            print("✅ LM Studio embedding model loaded")
        except ImportError:
            print("⚠️  LM Studio not available, using mock embeddings")
            self.model = None
        except Exception as e:
            print(f"⚠️  Failed to load embedding model: {e}")
            self.model = None
    
    def embed_descriptions(self, descriptions: List[str]) -> np.ndarray:
        """Generate embeddings for a list of descriptions."""
        if not descriptions:
            return np.array([])
        
        try:
            if self.model:
                embeddings = []
                for desc in descriptions:
                    embedding = self.model.embed(desc)
                    embeddings.append(embedding)
                embeddings = np.array(embeddings)
            else:
                embeddings = self._mock_embeddings(descriptions)
            
            return embeddings
            
        except Exception as e:
            print(f"⚠️  Embedding generation failed: {e}")