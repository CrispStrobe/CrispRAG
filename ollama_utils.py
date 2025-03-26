"""
Ollama model implementations and utilities for vector search.

This module provides:
- Ollama API client for embeddings
- Embedding provider compatible with the MLX provider interface
- Utility functions for Ollama-based embedding generation
"""

import os
import re
import time
import json
import requests
from typing import List, Dict, Any, Tuple, Optional, Set, Iterator, Union, Literal
import numpy as np
from collections import Counter

# Check if Ollama is available
try:
    import requests
    # Test Ollama connection
    try:
        response = requests.get("http://localhost:11434/api/embeddings")
        if response.status_code == 404:  # API exists but requires model parameter
            HAS_OLLAMA = True
        else:
            HAS_OLLAMA = False
            print("Warning: Ollama API not responding correctly")
    except requests.exceptions.ConnectionError:
        HAS_OLLAMA = False
        print("Warning: Ollama not available at http://localhost:11434")
except ImportError:
    HAS_OLLAMA = False
    print("Warning: requests not available. Install with: pip install requests")

# Default constants
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_TIMEOUT = 30

# Ollama Models Registry - Contains info about known models
OLLAMA_MODELS_REGISTRY = {
    "nomic-embed-text": {
        "ndim": 768,
        "normalize": True,
        "max_length": 512,
    },
    "all-minilm": {
        "ndim": 384,
        "normalize": True,
        "max_length": 512,
    },
    "mxbai-embed-large": {
        "ndim": 1024,
        "normalize": True,
        "max_length": 2048,
    },
    "bge-small": {
        "ndim": 384,
        "normalize": True,
        "max_length": 512,
    },
    "bge-base": {
        "ndim": 768, 
        "normalize": True,
        "max_length": 512,
    },
    "bge-large": {
        "ndim": 1024,
        "normalize": True,
        "max_length": 2048,
    },
    "e5-small": {
        "ndim": 384,
        "normalize": True,
        "max_length": 512,
    }
}


class OllamaEmbeddingProvider:
    """Provider for dense embeddings using Ollama API"""
    
    def __init__(self, 
                 model_name: str = DEFAULT_OLLAMA_EMBED_MODEL,
                 host: str = DEFAULT_OLLAMA_HOST, 
                 normalize: bool = True,
                 timeout: int = DEFAULT_OLLAMA_TIMEOUT,
                 batch_size: int = 8,
                 verbose: bool = False):
        """
        Initialize OllamaEmbeddingProvider.
        
        Args:
            model_name: Name of the Ollama model to use
            host: Ollama API host URL
            normalize: Whether to normalize embeddings
            timeout: API request timeout in seconds
            batch_size: Batch size for embedding multiple texts
            verbose: Whether to show verbose output
        """
        self.verbose = verbose
        self.model_name = model_name
        self.host = host
        self.normalize = normalize
        self.timeout = timeout
        self.batch_size = batch_size
        
        # Determine vector dimension based on model
        if model_name in OLLAMA_MODELS_REGISTRY:
            self.ndim = OLLAMA_MODELS_REGISTRY[model_name]["ndim"]
        else:
            # Default dimension for unknown models
            self.ndim = 768
            if verbose:
                print(f"Unknown model: {model_name}. Using default dimension: {self.ndim}")
        
        # Check if Ollama is available
        if not HAS_OLLAMA:
            if verbose:
                print("Ollama not available. Embeddings will fail.")
        elif verbose:
            print(f"Using Ollama with model: {model_name}, dimension: {self.ndim}")
            
    def get_dense_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Get dense embeddings for text(s) using Ollama.
        
        Args:
            text: String or list of strings to encode
        
        Returns:
            Numpy array of embeddings with shape (batch_size, dimension)
        """
        # Handle input type
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        if not texts:
            if is_single:
                return np.zeros(self.ndim, dtype=np.float32)
            return np.zeros((0, self.ndim), dtype=np.float32)
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            batch_embeddings = self._get_batch_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
            
        embeddings = np.array(all_embeddings, dtype=np.float32)
        
        # Return single embedding if input was a string
        if is_single:
            return embeddings[0]
        return embeddings
        
    def _get_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embeddings for a batch of texts from Ollama API.
        
        Args:
            texts: List of strings to encode
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for text in texts:
            # Skip empty texts
            if not text.strip():
                embeddings.append(np.zeros(self.ndim, dtype=np.float32))
                continue
                
            try:
                # Call Ollama API
                url = f"{self.host}/api/embeddings"
                payload = {
                    "model": self.model_name,
                    "prompt": text
                }
                
                response = requests.post(url, json=payload, timeout=self.timeout)
                
                if response.status_code != 200:
                    if self.verbose:
                        print(f"Ollama API error: {response.status_code} - {response.text}")
                    embeddings.append(np.zeros(self.ndim, dtype=np.float32))
                    continue
                    
                result = response.json()
                embedding = np.array(result.get("embedding", []), dtype=np.float32)
                
                # Check if we got the expected dimension
                if embedding.shape[0] != self.ndim:
                    if self.verbose:
                        print(f"Unexpected embedding dimension: {embedding.shape[0]} != {self.ndim}")
                    
                    # Resize if needed
                    if embedding.shape[0] > self.ndim:
                        embedding = embedding[:self.ndim]
                    else:
                        # Pad with zeros
                        padding = np.zeros(self.ndim - embedding.shape[0], dtype=np.float32)
                        embedding = np.concatenate([embedding, padding])
                
                # Normalize if requested
                if self.normalize:
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                
                embeddings.append(embedding)
                
            except Exception as e:
                if self.verbose:
                    print(f"Error getting embedding from Ollama: {e}")
                embeddings.append(np.zeros(self.ndim, dtype=np.float32))
                
        return embeddings
    
    def get_sparse_embedding(self, text: str) -> Tuple[List[int], List[float]]:
        """
        Generate a sparse embedding for text (fallback implementation).
        
        This uses a bag-of-words approach as Ollama doesn't natively support sparse embeddings.
        
        Args:
            text: String to encode
                
        Returns:
            Tuple of (indices, values) for sparse vector representation
        """
        return generate_sparse_vector(text)
        
    def rerank_with_ollama(self, query: str, passages: List[str], top_k: int = None) -> List[float]:
        """
        Rerank passages using dense embeddings and cosine similarity.
        
        Args:
            query: Original search query
            passages: List of passages to rerank
            top_k: Optional limit on returned results
            
        Returns:
            List of reranking scores
        """
        if not passages:
            return []
            
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
        This is a compatibility method to match the MLX provider interface.
        
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