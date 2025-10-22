"""
Semantic embedding service supporting both LM Studio and llama-cpp-python.
Generates embeddings for pattern descriptions for clustering.
"""

import numpy as np
from typing import List, Literal

class EmbeddingService:
    """Service for generating semantic embeddings with multiple backends."""
    
    def __init__(self, model_path: str, backend: Literal["lmstudio", "llamacpp"] = "lmstudio"):
        self.model = None
        self.backend = backend
        self._initialize_model(model_path=model_path)
    
    def _initialize_model(self, model_path: str):
        """Initialize the embedding model with specified backend."""
        try:
            if self.backend == "lmstudio":
                import lmstudio as lms
                self.model = lms.embedding_model(model_path)
                print("✅ LM Studio embedding model loaded")
            else:  # llamacpp
                from llama_cpp import Llama
                self.model = Llama(model_path=model_path, embedding=True)
                print("✅ llama-cpp-python embedding model loaded")
        except Exception as e:
            print(f"⚠️  Failed to load embedding model: {e}")
            raise e
    
    def embed_descriptions(self, descriptions: List[str]) -> np.ndarray:
        """Generate embeddings for a list of descriptions."""
        if not descriptions:
            return np.array([])
        
        try:
            if self.model:
                embeddings = []
                for desc in descriptions:
                    if self.backend == "lmstudio":
                        embedding = self.model.embed(desc)
                    else:  # llamacpp
                        embedding_result = self.model.create_embedding(desc)
                        embedding = embedding_result["data"][0]["embedding"]
                    embeddings.append(embedding)
                embeddings = np.array(embeddings)
            else:
                embeddings = self._mock_embeddings(descriptions)
            
            return embeddings
            
        except Exception as e:
            print(f"⚠️  Embedding generation failed: {e}")