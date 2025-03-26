"""
FastEmbed model implementations and utilities for vector search.

This module provides:
- FastEmbed integration for text embeddings
- Embedding provider compatible with MLX/Ollama interface
- Utility functions for sparse vector generation
"""

import os
import re
import time
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Iterator, Literal
from collections import Counter

# Check if FastEmbed is available
try:
    from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    HAS_FASTEMBED = True
except ImportError:
    HAS_FASTEMBED = False
    print("Warning: fastembed not available. Install with: pip install fastembed")

# Default constants
DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_FASTEMBED_SPARSE_MODEL = "prithivida/Splade_PP_en_v1"

# Registry of common FastEmbed models and their dimensions
FASTEMBED_MODELS_REGISTRY = {
    "BAAI/bge-small-en-v1.5": {
        "ndim": 384,
        "normalize": True,
        "description": "BGE Small English, good general purpose"
    },
    "BAAI/bge-base-en-v1.5": {
        "ndim": 768,
        "normalize": True,
        "description": "BGE Base English, better quality than small"
    },
    "BAAI/bge-large-en-v1.5": {
        "ndim": 1024,
        "normalize": True,
        "description": "BGE Large English, high quality but slower"
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "ndim": 384,
        "normalize": True,
        "description": "Small all-round model, good for general purpose"
    },
    "Qdrant/flag-embedding": {
        "ndim": 1024,
        "normalize": True,
        "description": "High quality, multipurpose embedding model"
    },
    "thenlper/gte-small": {
        "ndim": 384,
        "normalize": True,
        "description": "General Text Embeddings (GTE) model, small variant"
    },
    "thenlper/gte-base": {
        "ndim": 768,
        "normalize": True,
        "description": "General Text Embeddings (GTE) model, base variant"
    },
    "thenlper/gte-large": {
        "ndim": 1024,
        "normalize": True,
        "description": "General Text Embeddings (GTE) model, large variant"
    },
    "intfloat/multilingual-e5-small": {
        "ndim": 384,
        "normalize": True,
        "description": "Multilingual E5 model, small variant"
    },
    "intfloat/multilingual-e5-base": {
        "ndim": 768,
        "normalize": True,
        "description": "Multilingual E5 model, base variant"
    },
    "intfloat/multilingual-e5-large": {
        "ndim": 1024,
        "normalize": True,
        "description": "Multilingual E5 model, large variant"
    },
    "prithivida/Splade_PP_en_v1": {
        "ndim": 0,  # Special case for sparse model
        "normalize": False,
        "description": "SPLADE++ model for sparse embeddings"
    }
}


class FastEmbedProvider:
    """Provider for embeddings using FastEmbed library"""
    
    def __init__(self, 
                 model_name: str = DEFAULT_FASTEMBED_MODEL,
                 sparse_model_name: str = DEFAULT_FASTEMBED_SPARSE_MODEL,
                 use_gpu: bool = False,
                 batch_size: int = 32,
                 cache_dir: Optional[str] = None,
                 verbose: bool = False):
        """
        Initialize FastEmbedProvider.
        
        Args:
            model_name: Name of the FastEmbed model to use
            sparse_model_name: Name of the sparse model to use
            use_gpu: Whether to use GPU (requires fastembed-gpu)
            batch_size: Batch size for embedding multiple texts
            cache_dir: Directory to cache models
            verbose: Whether to show verbose output
        """
        self.verbose = verbose
        self.model_name = model_name
        self.sparse_model_name = sparse_model_name
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        
        # Set providers for ONNX runtime
        self.providers = ["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        
        self.dense_model = None
        self.sparse_model = None
        self.cross_encoder = None
        
        # Determine vector dimension based on model
        if model_name in FASTEMBED_MODELS_REGISTRY:
            self.ndim = FASTEMBED_MODELS_REGISTRY[model_name]["ndim"]
        else:
            # Default dimension for unknown models
            self.ndim = 384
            if verbose:
                print(f"Unknown model: {model_name}. Using default dimension: {self.ndim}")
        
        # Check if FastEmbed is available
        if not HAS_FASTEMBED:
            if verbose:
                print("FastEmbed not available. Embeddings will fail.")
            return
            
        # Initialize the models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the FastEmbed models"""
        if not HAS_FASTEMBED:
            return
            
        try:
            # Initialize the dense embedding model
            if self.verbose:
                print(f"Initializing FastEmbed dense model: {self.model_name}")
                
            kwargs = {}
            if self.cache_dir:
                kwargs["cache_dir"] = self.cache_dir
                
            self.dense_model = TextEmbedding(
                model_name=self.model_name,
                providers=self.providers,
                **kwargs
            )
            
            if self.verbose:
                print(f"FastEmbed dense model initialized with dimension: {self.ndim}")
                
            # Initialize the sparse embedding model if it's a different model
            if self.sparse_model_name != self.model_name and self.sparse_model_name.lower() != "none":
                if self.verbose:
                    print(f"Initializing FastEmbed sparse model: {self.sparse_model_name}")
                    
                try:
                    self.sparse_model = SparseTextEmbedding(
                        model_name=self.sparse_model_name,
                        providers=self.providers,
                        **kwargs
                    )
                    
                    if self.verbose:
                        print(f"FastEmbed sparse model initialized")
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to initialize sparse model: {e}")
                        print("Will fall back to BOW sparse embeddings")
            
            # Initialize cross-encoder for reranking if available
            try:
                self.cross_encoder = TextCrossEncoder(
                    model_name="Xenova/ms-marco-MiniLM-L-6-v2",
                    providers=self.providers,
                    **kwargs
                )
                
                if self.verbose:
                    print("FastEmbed cross-encoder initialized for reranking")
            except Exception as e:
                if self.verbose:
                    print(f"Failed to initialize cross-encoder: {e}")
                    print("Reranking will not be available")
                    
        except Exception as e:
            if self.verbose:
                print(f"Error initializing FastEmbed models: {e}")
    
    def get_dense_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Get dense embeddings for text(s) using FastEmbed.
        
        Args:
            text: String or list of strings to encode
        
        Returns:
            Numpy array of embeddings with shape (batch_size, dimension)
        """
        if not HAS_FASTEMBED or self.dense_model is None:
            raise RuntimeError("FastEmbed not available or model not initialized")
            
        # Handle input type
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        if not texts:
            if is_single:
                return np.zeros(self.ndim, dtype=np.float32)
            return np.zeros((0, self.ndim), dtype=np.float32)
        
        try:
            # Generate embeddings
            embeddings = list(self.dense_model.embed(texts))
            
            # Return single embedding if input was a string
            if is_single:
                return embeddings[0]
            return np.array(embeddings)
            
        except Exception as e:
            if self.verbose:
                print(f"Error getting embeddings from FastEmbed: {e}")
            
            # Return zeros on error
            if is_single:
                return np.zeros(self.ndim, dtype=np.float32)
            return np.zeros((len(texts), self.ndim), dtype=np.float32)
    
    def get_sparse_embedding(self, text: str) -> Tuple[List[int], List[float]]:
        """
        Generate a sparse embedding for text.
        
        Args:
            text: String to encode
                
        Returns:
            Tuple of (indices, values) for sparse vector representation
        """
        if not text.strip():
            return [0], [0.0]
            
        # Use SPLADE model if available
        if HAS_FASTEMBED and self.sparse_model is not None:
            try:
                sparse_emb = list(self.sparse_model.embed([text]))[0]
                indices = sparse_emb.indices.tolist()
                values = sparse_emb.values.tolist()
                
                # If empty, return a default
                if not indices:
                    return [0], [0.0]
                    
                return indices, values
            except Exception as e:
                if self.verbose:
                    print(f"Failed to generate sparse embedding with SPLADE: {e}")
                    print("Falling back to BOW sparse embeddings")
        
        # Fall back to bag-of-words approach
        return generate_sparse_vector(text)
    
    def rerank_with_fastembed(self, query: str, passages: List[str], top_k: Optional[int] = None) -> List[float]:
        """
        Rerank passages using FastEmbed cross-encoder.
        
        Args:
            query: Original search query
            passages: List of passages to rerank
            top_k: Optional limit on returned results
            
        Returns:
            List of reranking scores
        """
        if not passages:
            return []
            
        if not HAS_FASTEMBED or self.cross_encoder is None:
            # Fall back to embedding-based reranking if cross-encoder not available
            return self._fallback_rerank(query, passages)
        
        try:
            # Use cross-encoder for reranking
            scores = list(self.cross_encoder.rerank(query, passages))
            return scores
            
        except Exception as e:
            if self.verbose:
                print(f"Error in cross-encoder reranking: {e}")
                print("Falling back to embedding-based reranking")
                
            return self._fallback_rerank(query, passages)
    
    def _fallback_rerank(self, query: str, passages: List[str]) -> List[float]:
        """
        Fallback reranking using dense embeddings and cosine similarity.
        
        Args:
            query: Original search query
            passages: List of passages to rerank
            
        Returns:
            List of reranking scores
        """
        # Get dense embeddings for query and passages
        query_embedding = self.get_dense_embedding(query)
        passage_embeddings = self.get_dense_embedding(passages)
        
        # Compute similarity scores
        reranked_scores = []
        
        for passage_emb in passage_embeddings:
            # Cosine similarity
            similarity = np.dot(query_embedding, passage_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(passage_emb)
            )
            reranked_scores.append(float(similarity))
            
        return reranked_scores
    
    def _process_sparse_batch(self, batch: List[str]) -> List[Tuple[List[int], List[float]]]:
        """
        Process a batch of texts for sparse embedding generation.
        This method provides compatibility with the MLX provider interface.
        
        Args:
            batch: List of texts to process
            
        Returns:
            List of (indices, values) tuples
        """
        results = []
        
        for text in batch:
            sparse_embedding = self.get_sparse_embedding(text)
            results.append(sparse_embedding)
        
        return results


# Utility functions that can be used independently of the provider
def generate_sparse_vector(text: str) -> Tuple[List[int], List[float]]:
    """
    Generate a simple sparse vector using term frequencies.
    This is a standalone utility function for sparse representations.
    
    Args:
        text: Text to encode
            
    Returns:
        Tuple of (indices, values) for sparse vector representation
    """
    from collections import Counter
    import re
    
    # Simple tokenization
    tokens = re.findall(r'\b\w+\b', text.lower())
    
    # Remove very common words (simple stopwords)
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'of', 'to', 'in', 'that', 'it', 'with', 'for'}
    tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
    
    # Count terms
    counter = Counter(tokens)
    
    # Convert to sparse vector format (for simplicity, use term hashes as indices)
    indices = []
    values = []
    
    for term, count in counter.items():
        # Simple hash function for terms - use a better one in production
        # Limit to 100000 dimensions
        term_index = hash(term) % 100000
        term_value = count / max(1, len(tokens))  # Normalize by document length, avoid division by zero
        indices.append(term_index)
        values.append(term_value)
    
    # If empty, return a single default dimension
    if not indices:
        return [0], [0.0]
            
    return indices, values