"""
utils.py:

Consolidated utilities for all functionality.

This module contains all shared functionality including:
- Text processing and result formatting
- Search algorithms
- File processing
- Model utilities
- General utilities

Design principles:
1. Avoids circular imports through provider-agnostic base implementations
2. Uses lazy imports for provider-specific functionality
3. Maintains clear separation of concerns between components
"""

import re
import os
import time
import json
import shutil
import fnmatch
import tempfile
import requests
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set, Iterator, Union, Literal
import numpy as np

# Constants
DEFAULT_COLLECTION = "documents"
DEFAULT_MODEL = "cstr/paraphrase-multilingual-MiniLM-L12-v2-mlx"
DEFAULT_WEIGHTS_PATH = "weights/paraphrase-multilingual-MiniLM-L12-v2.npz"
DEFAULT_WEIGHTS_URL = "https://huggingface.co/cstr/paraphrase-multilingual-MiniLM-L12-v2-mlx/resolve/main/paraphrase-multilingual-MiniLM-L12-v2.npz"
DEFAULT_CONFIG_URL = "https://huggingface.co/cstr/paraphrase-multilingual-MiniLM-L12-v2-mlx/resolve/main/config.json"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".pdf", ".json", ".csv"}
CHUNK_SIZE = 512  # Max tokens per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks
VECTOR_SIZE = 384  # Default vector size for the model

# Comprehensive mapping of model IDs to their corresponding tokenizer IDs
MODEL_TOKENIZER_MAP = {
    # Traditional models
    "cstr/paraphrase-multilingual-MiniLM-L12-v2-mlx": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    
    # MLX models - Dense
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "bge-m3": "BAAI/bge-m3",
    "bge-micro": "TaylorAI/bge-micro-v2",
    
    "minilm-l6": "sentence-transformers/all-MiniLM-L6-v2",
    "minilm-l12": "sentence-transformers/all-MiniLM-L12-v2",
    
    "multilingual-e5-small": "intfloat/multilingual-e5-small",
    "multilingual-e5-base": "intfloat/multilingual-e5-base",
    "multilingual-e5-large": "intfloat/multilingual-e5-large",
    
    "gte-tiny": "TaylorAI/gte-tiny",
    "gte-small": "thenlper/gte-small",
    "gte-base": "thenlper/gte-base",
    "gte-large": "thenlper/gte-large",
    
    "snowflake-xs": "Snowflake/snowflake-arctic-embed-xs",
    "snowflake-lg": "Snowflake/snowflake-arctic-embed-l",
    
    "nomic-text-v1": "nomic-ai/nomic-embed-text-v1",
    "nomic-text-v1.5": "nomic-ai/nomic-embed-text-v1.5",
    
    "mixedbread-large": "mixedbread-ai/mxbai-embed-large-v1",
    
    # MLX models - Sparse
    "distilbert-splade": "raphaelsty/distilbert-splade",
    "bert-base-uncased": "bert-base-uncased",
    "neuralcherche-sparse-embed": "raphaelsty/neural-cherche-sparse-embed",
    "opensearch": "opensearch-project/opensearch-neural-sparse-encoding-doc-v1",
    "naver-splade-distilbert": "naver/splade-v3-distilbert",
    
    # Ollama models
    "nomic-embed-text": "nomic-ai/nomic-embed-text-v1",
    "llama2": "meta-llama/Llama-2-7b-hf",
    "mistral": "mistralai/Mistral-7B-v0.1",
    
    # FastEmbed models
    "BAAI/bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "Qdrant/flag-embedding": "Qdrant/flag-embedding",
    "sentence-transformers/all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2"
}

# Try to import optional dependencies
try:
    import tqdm
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not available. Install with: pip install tqdm")

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("Warning: MLX not available. Will use PyTorch for embeddings if available.")

try:
    from transformers import AutoTokenizer, AutoConfig, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available. Install with: pip install transformers")

try:
    import torch
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    if not HAS_MLX:
        print("Warning: Neither MLX nor PyTorch is available. At least one is required for embedding generation.")

# Lazy imports for provider modules to avoid circular dependencies
def _import_mlx_embedding_models():
    """Safely import mlx_embedding_models if available"""
    try:
        from mlx_embedding_models.embedding import EmbeddingModel, SpladeModel
        HAS_MLX_EMBEDDING_MODELS = True
        return EmbeddingModel, SpladeModel, HAS_MLX_EMBEDDING_MODELS
    except ImportError:
        HAS_MLX_EMBEDDING_MODELS = False
        if HAS_MLX:
            print("Warning: mlx_embedding_models not available. Install with: pip install mlx-embedding-models")
        return None, None, False

def _import_mlx_provider():
    """Safely import MLXEmbeddingProvider"""
    try:
        from mlx_utils import MLXEmbeddingProvider, MLXModel, DEFAULT_DENSE_MODEL, DEFAULT_SPARSE_MODEL
        return MLXEmbeddingProvider, MLXModel, DEFAULT_DENSE_MODEL, DEFAULT_SPARSE_MODEL
    except ImportError:
        return None, None, "bge-small", "distilbert-splade"

def _import_ollama_provider():
    """Safely import OllamaEmbeddingProvider"""
    try:
        from ollama_utils import OllamaEmbeddingProvider, HAS_OLLAMA, DEFAULT_OLLAMA_EMBED_MODEL
        return OllamaEmbeddingProvider, HAS_OLLAMA, DEFAULT_OLLAMA_EMBED_MODEL
    except ImportError:
        return None, False, "nomic-embed-text"

def _import_fastembed_provider():
    """Safely import FastEmbedProvider"""
    try:
        from fastembed_utils import FastEmbedProvider, HAS_FASTEMBED, DEFAULT_FASTEMBED_MODEL, DEFAULT_FASTEMBED_SPARSE_MODEL
        return FastEmbedProvider, HAS_FASTEMBED, DEFAULT_FASTEMBED_MODEL, DEFAULT_FASTEMBED_SPARSE_MODEL
    except ImportError:
        return None, False, "BAAI/bge-small-en-v1.5", "prithivida/Splade_PP_en_v1"

# Check for MLX Embedding Models
_, _, HAS_MLX_EMBEDDING_MODELS = _import_mlx_embedding_models()

# Import Qdrant for availability check
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance
    qdrant_available = True
except ImportError:
    qdrant_available = False
    print("Qdrant client not available. Install with: pip install qdrant-client")


def get_tokenizer_name(model_name):
    """
    Get the appropriate tokenizer name for a given model.
    
    Args:
        model_name: Name of the model to get tokenizer for
        
    Returns:
        Name of the tokenizer to use
    """
    # Check if we have a specific mapping
    if model_name in MODEL_TOKENIZER_MAP:
        return MODEL_TOKENIZER_MAP[model_name]
    
    # For unmapped models, apply some heuristics
    # For known prefix patterns, map to likely namespace
    if model_name.startswith("bge-"):
        return f"BAAI/{model_name}"
    
    if model_name.startswith("e5-"):
        return f"intfloat/{model_name}"
    
    if model_name.startswith("gte-"):
        return f"thenlper/{model_name}"
    
    # Default: just use the name as is, and let HF figure it out
    return model_name


#######################################
# EmbeddingUtils Class
#######################################

class EmbeddingUtils:
    """Utilities for generating embeddings and vector operations"""
    
    @staticmethod
    def generate_sparse_vector(text: str) -> Tuple[List[int], List[float]]:
        """
        Generate a simple sparse vector using term frequencies.
        This is a standalone implementation that doesn't depend on providers.
        
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
        
        # Convert to sparse vector format
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

#######################################
# TextProcessor Class
#######################################

class TextProcessor:
    def __init__(self, 
                    model_name: str, 
                    weights_path: str, 
                    dense_model: str = "bge-small", 
                    sparse_model: str = "distilbert-splade",
                    top_k: int = 64,
                    custom_repo_id: str = None,
                    custom_ndim: int = None,
                    custom_pooling: str = "mean",
                    custom_normalize: bool = True,
                    custom_max_length: int = 512,
                    use_mlx_embedding: bool = False,
                    use_ollama: bool = False,
                    use_fastembed: bool = False,
                    ollama_model: str = None,
                    ollama_host: str = "http://localhost:11434",
                    fastembed_model: str = None,
                    fastembed_sparse_model: str = None,
                    fastembed_use_gpu: bool = False,
                    fastembed_cache_dir: str = None,
                    verbose: bool = False):
        """
        Initialize DocumentProcessor with support for MLX, Ollama, or FastEmbed embedding models.
        
        Args:
            model_name: Original model name (fallback if embedding models not available)
            weights_path: Path to model weights (fallback)
            dense_model: MLX embedding model name for dense vectors
            sparse_model: MLX embedding model name for sparse vectors
            top_k: Top-k tokens to keep in sparse vectors
            custom_repo_id: Custom model repo ID
            custom_ndim: Custom model embedding dimension
            custom_pooling: Custom model pooling strategy
            custom_normalize: Whether to normalize embeddings
            custom_max_length: Maximum sequence length
            use_mlx_embedding: Whether to use mlx_embedding_models
            use_ollama: Whether to use Ollama for embeddings
            use_fastembed: Whether to use FastEmbed for embeddings
            ollama_model: Ollama model to use
            ollama_host: Ollama API host URL
            fastembed_model: FastEmbed model to use
            fastembed_sparse_model: FastEmbed sparse model to use
            fastembed_use_gpu: Whether to use GPU with FastEmbed
            fastembed_cache_dir: Directory to cache FastEmbed models
            verbose: Whether to show verbose output
        """
        self.verbose = verbose
        self.model_name = model_name
        self.weights_path = weights_path
        self.tokenizer = None
        self.pytorch_model = None
        self.mlx_model = None
        self.use_mlx = HAS_MLX
        self.vector_size = VECTOR_SIZE  # Default, will be updated during model initialization
        
        # Use custom_repo_id as the dense model ID if provided
        self.dense_model_id = custom_repo_id if custom_repo_id else dense_model
        self.sparse_model_id = sparse_model
        
        # Flag to track whether we need traditional model
        need_traditional_model = True
        
        # Load default provider constants if not provided
        if ollama_model is None:
            _, _, DEFAULT_OLLAMA_EMBED_MODEL = _import_ollama_provider()
            ollama_model = DEFAULT_OLLAMA_EMBED_MODEL
            
        if fastembed_model is None:
            _, _, DEFAULT_FASTEMBED_MODEL, DEFAULT_FASTEMBED_SPARSE_MODEL = _import_fastembed_provider()
            fastembed_model = DEFAULT_FASTEMBED_MODEL
            fastembed_sparse_model = DEFAULT_FASTEMBED_SPARSE_MODEL
        
        # Determine which embedding provider to use (priority order)
        self.use_fastembed = use_fastembed
        self.use_ollama = use_ollama and not self.use_fastembed
        self.use_mlx_embedding = use_mlx_embedding and not self.use_fastembed and not self.use_ollama
        
        self.mlx_embedding_provider = None
        self.ollama_embedding_provider = None
        self.fastembed_provider = None
        
        # Determine which tokenizer model to use based on the active embedding provider
        tokenizer_model = model_name  # Default fallback
        
        # FastEmbed (highest priority)
        if self.use_fastembed:
            FastEmbedProvider, HAS_FASTEMBED, DEFAULT_FASTEMBED_MODEL, DEFAULT_FASTEMBED_SPARSE_MODEL = _import_fastembed_provider()
            
            if not HAS_FASTEMBED:
                if self.verbose:
                    print("Warning: FastEmbed not available. Falling back to next provider.")
                self.use_fastembed = False
            else:
                if self.verbose:
                    print(f"Using FastEmbed for embeddings with model: {fastembed_model}")
                
                self.fastembed_provider = FastEmbedProvider(
                    model_name=fastembed_model,
                    sparse_model_name=fastembed_sparse_model,
                    use_gpu=fastembed_use_gpu,
                    cache_dir=fastembed_cache_dir,
                    verbose=verbose
                )
                
                # Set vector size and tokenizer model
                if self.fastembed_provider is not None:
                    self.vector_size = self.fastembed_provider.ndim
                    tokenizer_model = fastembed_model
                    if verbose:
                        print(f"Using vector size from FastEmbed model: {self.vector_size}")
                
                # Update model IDs for database storage
                self.dense_model_id = fastembed_model
                self.sparse_model_id = fastembed_sparse_model
                
                # Don't need traditional model
                need_traditional_model = False
        
        # Ollama (second priority)
        if self.use_ollama:
            OllamaEmbeddingProvider, HAS_OLLAMA, DEFAULT_OLLAMA_EMBED_MODEL = _import_ollama_provider()
            
            if not HAS_OLLAMA:
                if self.verbose:
                    print("Warning: Ollama not available. Falling back to next provider.")
                self.use_ollama = False
            else:
                if self.verbose:
                    print(f"Using Ollama for embeddings with model: {ollama_model}")
                
                self.ollama_embedding_provider = OllamaEmbeddingProvider(
                    model_name=ollama_model,
                    host=ollama_host,
                    verbose=verbose
                )
                
                # Set vector size and tokenizer model
                if self.ollama_embedding_provider is not None:
                    self.vector_size = self.ollama_embedding_provider.ndim
                    tokenizer_model = ollama_model
                    if verbose:
                        print(f"Using vector size from Ollama model: {self.vector_size}")
                
                # Update model IDs for database storage
                self.dense_model_id = ollama_model
                self.sparse_model_id = "ollama_sparse"
                
                # Don't need traditional model
                need_traditional_model = False
        
        # MLX embedding models (third priority)
        if self.use_mlx_embedding:
            MLXEmbeddingProvider, MLXModel, DEFAULT_DENSE_MODEL, DEFAULT_SPARSE_MODEL = _import_mlx_provider()
            
            # If a custom repo ID is specified, it should override the dense model
            effective_dense_model = custom_repo_id if custom_repo_id else dense_model
            
            if not HAS_MLX_EMBEDDING_MODELS or MLXEmbeddingProvider is None:
                if self.verbose:
                    print("Warning: MLX embedding models not available. Falling back to traditional model.")
                self.use_mlx_embedding = False
            else:
                if self.verbose:
                    print(f"Using MLX embedding models:")
                    if custom_repo_id:
                        print(f"  - Custom repository: {custom_repo_id}")
                    else:
                        print(f"  - Dense model: {dense_model}")
                    print(f"  - Sparse model: {sparse_model}")
                
                # Create MLX embedding provider using the effective model name
                try:
                    self.mlx_embedding_provider = MLXEmbeddingProvider(
                        dense_model_name=effective_dense_model,  # This can be either custom_repo_id or dense_model
                        sparse_model_name=sparse_model,
                        top_k=top_k,
                        custom_repo_id=custom_repo_id,
                        custom_ndim=custom_ndim,
                        custom_pooling=custom_pooling,
                        custom_normalize=custom_normalize,
                        custom_max_length=custom_max_length,
                        verbose=verbose
                    )
                    self.embedding_model = self.mlx_embedding_provider
                    self.dense_model_id = custom_repo_id if custom_repo_id else dense_model
                    self.sparse_model_id = sparse_model
                    
                    # Verify that dense model was properly loaded
                    if hasattr(self.mlx_embedding_provider, 'dense_model') and self.mlx_embedding_provider.dense_model is not None:
                        if self.verbose:
                            print(f"Successfully initialized MLX embedding provider with dense model")
                            
                        # Set vector size and tokenizer model
                        self.vector_size = self.mlx_embedding_provider.ndim
                        
                        # For tokenizer, always use the effective model
                        tokenizer_model = effective_dense_model
                        
                        if verbose:
                            print(f"Using vector size from MLX embedding model: {self.vector_size}")
                        
                        # Don't need traditional model since MLX embedding provider is working
                        need_traditional_model = False
                    else:
                        if self.verbose:
                            print(f"MLX embedding provider was created but dense_model wasn't properly loaded")
                            print(f"Falling back to traditional model")
                        # We'll need traditional model as a fallback
                        need_traditional_model = True
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to initialize MLX embedding provider: {e}")
                        print("Falling back to traditional model")
                    need_traditional_model = True
        
        # Load the appropriate tokenizer based on the active model
        self.load_tokenizer(tokenizer_model)
        
        # Load the traditional model if needed - ONLY if explicitly needed
        if need_traditional_model:
            if self.verbose:
                print(f"Loading traditional model: {model_name}")
            self.load_model()
        elif self.verbose:
            print(f"Skipping traditional model load - using embedding provider directly")

    @staticmethod
    def create_preview(text, query, context_size=300):
        """
        Create a preview with context around search terms.
        
        Args:
            text: The full text to extract preview from
            query: Search query to look for
            context_size: Maximum context size in characters
        
        Returns:
            A preview string with context around matched terms
        """
        if not text:
            return ""
        
        # If text is shorter than context_size, return the whole text
        if len(text) <= context_size:
            return text
        
        # Find positions of search terms in text
        text_lower = text.lower()
        query_lower = query.lower()
        query_terms = query_lower.split()
        positions = []
        
        # Try complete query first
        pos = text_lower.find(query_lower)
        if pos != -1:
            positions.append(pos)
        
        # Then try individual terms
        for term in query_terms:
            if len(term) > 2:  # Skip very short terms
                term_pos = text_lower.find(term)
                while term_pos != -1:
                    positions.append(term_pos)
                    term_pos = text_lower.find(term, term_pos + 1)
        
        # If no matches found, return the beginning of the text
        if not positions:
            return text[:context_size] + "..."
        
        # Find the best window that contains the most matches
        best_start = 0
        best_end = min(context_size, len(text))
        
        if positions:
            # Sort positions
            positions.sort()
            # Choose the middle position to center around
            middle_pos = positions[len(positions) // 2]
            
            # Center context window around the middle position
            best_start = max(0, middle_pos - context_size // 2)
            best_end = min(len(text), best_start + context_size)
        
        # Adjust window to not cut words
        if best_start > 0:
            while best_start > 0 and text[best_start] != ' ':
                best_start -= 1
            best_start += 1  # Move past the space
        
        if best_end < len(text):
            while best_end < len(text) and text[best_end] != ' ':
                best_end += 1
        
        # Create preview with ellipses if needed
        preview = ""
        if best_start > 0:
            preview += "..."
        
        preview += text[best_start:best_end]
        
        if best_end < len(text):
            preview += "..."
        
        # Highlight search terms with **
        final_preview = preview
        for term in sorted(query_terms, key=len, reverse=True):
            if len(term) > 2:  # Skip very short terms
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                final_preview = pattern.sub(f"**{term}**", final_preview)
        
        return final_preview
    
    def _load_fallback_tokenizer(self):
        """
        Try to load a fallback tokenizer when the primary one fails.
        
        This method attempts multiple common tokenizers in order of reliability.
        """
        fallbacks = [
            "sentence-transformers/all-MiniLM-L6-v2",  # Reliable general tokenizer
            "bert-base-uncased",                      # Basic BERT tokenizer
            "distilbert-base-uncased",                # DistilBERT tokenizer
            "gpt2"                                   # GPT tokenizer as last resort
        ]
        
        for fallback in fallbacks:
            try:
                if self.verbose:
                    print(f"Trying fallback tokenizer: {fallback}")
                self.tokenizer = AutoTokenizer.from_pretrained(fallback)
                if self.verbose:
                    print(f"Successfully loaded fallback tokenizer: {fallback}")
                return
            except Exception as fallback_error:
                if self.verbose:
                    print(f"Fallback tokenizer {fallback} failed: {fallback_error}")
        
        # If all fallbacks failed, warn but continue
        if self.verbose:
            print("Warning: Could not load any tokenizer. Text chunking functionality may be limited.")
    
    def load_tokenizer(self, model_name):
        """
        Load the appropriate tokenizer for the given model.
        
        This method attempts to find the best matching tokenizer for the provided model name,
        using a mapping for known models and intelligent fallbacks for unknown models.
        
        Args:
            model_name: Name of the model to load tokenizer for
        """
        if not HAS_TRANSFORMERS:
            if self.verbose:
                print("Transformers not available, skipping tokenizer loading")
            return
        
        try:
            # First check if we have a specific mapping for this model
            if model_name in MODEL_TOKENIZER_MAP:
                tokenizer_name = MODEL_TOKENIZER_MAP[model_name]
                if self.verbose:
                    print(f"Using mapped tokenizer: {tokenizer_name}")
            else:
                # Using general heuristics for model name
                tokenizer_name = get_tokenizer_name(model_name)
                if self.verbose:
                    print(f"Using direct tokenizer name: {tokenizer_name}")
            
            # Attempt to load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if self.verbose:
                print(f"Successfully loaded tokenizer from {tokenizer_name}")
        
        except Exception as e:
            if self.verbose:
                print(f"Error loading tokenizer for {model_name}: {e}")
                print("Trying fallback options...")
            
            # Try a series of fallbacks
            self._load_fallback_tokenizer()
    
    @staticmethod
    def create_smart_preview(text, query, context_size=300):
        """
        Create a preview with clear highlighting of searched terms.
        
        Args:
            text: The full text to extract preview from
            query: Search query to look for
            context_size: Maximum context size in characters
        
        Returns:
            A preview string with context around matched terms and highlighting
        """
        if not text:
            return ""
        
        # If text is shorter than context_size, return the whole text
        if len(text) <= context_size:
            return TextProcessor.highlight_query_terms(text, query)
        
        # Import re if not already imported
        import re
        
        # Normalize query and text for searching
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Find the exact query in the text
        match_pos = text_lower.find(query_lower)
        
        # If exact query not found, try individual words
        if match_pos == -1:
            # Extract query terms, filtering out short words
            query_terms = [term for term in query_lower.split() if len(term) > 2]
            
            # Find positions of all query terms
            term_positions = []
            for term in query_terms:
                pos = text_lower.find(term)
                if pos != -1:
                    term_positions.append(pos)
            
            # If any terms were found, use the earliest one
            if term_positions:
                match_pos = min(term_positions)
            else:
                # No terms found, return beginning of text
                return text[:context_size] + "..."
        
        # Center the preview around the match position, but use a larger context
        # to show more text before and after the match
        half_context = context_size // 2
        start = max(0, match_pos - half_context)
        end = min(len(text), match_pos + half_context)
        
        # Try to expand to full sentences
        # Look for sentence boundaries before start (up to 200 chars back)
        sentence_start = max(0, start - 200)  # Look back up to 200 chars
        potential_start = text.rfind('. ', sentence_start, start)
        if potential_start != -1:
            start = potential_start + 2  # Move past the period and space
        
        # Look for sentence boundaries after end (up to 200 chars ahead)
        potential_end = text.find('. ', end, min(len(text), end + 200))
        if potential_end != -1:
            end = potential_end + 1  # Include the period
        
        # Extract preview
        preview = text[start:end]
        
        # Add ellipses to indicate truncation
        if start > 0:
            preview = "..." + preview
        if end < len(text):
            preview += "..."
        
        # Highlight the query in the preview
        highlighted_preview = TextProcessor.highlight_query_terms(preview, query)
        
        return highlighted_preview
    
    @staticmethod
    def highlight_query_terms(text, query):
        """
        Highlight query terms in the text using bold markdown syntax.
        
        Args:
            text: Text to highlight terms in
            query: Query string containing terms to highlight
        
        Returns:
            Text with highlighted terms
        """
        if not text or not query:
            return text
        
        import re
        
        # Get both the full query and individual terms
        query_lower = query.lower()
        query_terms = [term for term in query_lower.split() if len(term) > 2]
        
        # Add the full query to the list of terms to highlight
        if len(query) > 2 and query not in query_terms:
            query_terms.append(query_lower)
        
        # Sort terms by length (descending) to avoid partial matches
        query_terms = sorted(set(query_terms), key=len, reverse=True)
        
        # Initialize result text
        result = text
        
        # Loop through each term and highlight it
        for term in query_terms:
            # Skip highlighting for very short terms
            if len(term) <= 2:
                continue
                
            # Create a case-insensitive pattern to find the term
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            
            # Find all occurrences of the term
            matches = list(pattern.finditer(result))
            
            # Process matches from end to start to avoid position issues
            offset = 0
            for match in matches:
                start_pos = match.start() + offset
                end_pos = match.end() + offset
                
                # Extract the original term with original case
                original_term = result[start_pos:end_pos]
                
                # Replace with highlighted version
                result = result[:start_pos] + f"**{original_term}**" + result[end_pos:]
                
                # Update offset for next replacements
                offset += 4  # Length of added characters "**" + "**"
        
        return result
    
    @staticmethod
    def format_search_results(points, query, search_type, processor, context_size=300, 
                         retriever=None, db_type="qdrant"):
        """
        Format search results with improved preview.
        
        Args:
            points: List of search result points
            query: Search query string
            search_type: Type of search that was performed
            processor: Document processor instance
            context_size: Maximum context size for previews
            retriever: Optional function to retrieve context for chunks
            db_type: Type of database that produced the results
            
        Returns:
            Dictionary with formatted search results
        """
        # Get embedder model IDs for metadata
        dense_model_id = getattr(processor, 'dense_model_id', "unknown") if processor else "unknown"
        sparse_model_id = getattr(processor, 'sparse_model_id', "unknown") if processor else "unknown"
        
        if not points or len(points) == 0:
            return {
                "query": query,
                "search_type": search_type,
                "count": 0,
                "embedder_info": {
                    "dense": dense_model_id,
                    "sparse": sparse_model_id
                },
                "results": []
            }
        
        # Adapt all points to a common format
        adapted_points = []
        for point in points:
            # Skip if result is in error format
            if isinstance(point, dict) and "error" in point:
                continue
            adapted_points.append(ResultProcessor.adapt_result(point, db_type))
        
        # Filter out tiny documents (< 20 chars) as they won't have meaningful content
        filtered_points = []
        for point in adapted_points:
            if hasattr(point, "payload") and point.payload and "text" in point.payload:
                if len(point.payload["text"]) >= 20:  # Only include meaningful chunks
                    filtered_points.append(point)
        
        # Use original points if all were filtered out
        if not filtered_points and adapted_points:
            filtered_points = adapted_points
        
        # Format results
        formatted_results = []
        for i, result in enumerate(filtered_points):
            payload = result.payload if hasattr(result, "payload") else {}

            file_path = payload.get("file_path", "")
            cindex = payload.get("chunk_index", 0)

            # Retrieve big context from the DB if a retriever is provided
            big_context = ""
            if retriever:
                big_context = retriever(file_path, cindex, window=1)

            # Get score safely
            score = ResultProcessor.get_score(result)
            text = payload.get("text", "")
            
            # Create preview with context handling
            preview = TextProcessor.create_smart_preview(text, query, context_size)
            
            # Get chunk size information
            chunk_size = {
                "characters": len(text),
                "words": len(text.split()),
                "lines": len(text.splitlines())
            }
            
            # Get embedding information
            embedder_meta = payload.get("metadata", {})
            embedder_info = {
                "dense_embedder": embedder_meta.get("dense_embedder", embedder_meta.get("embedder", "unknown")),
                "sparse_embedder": embedder_meta.get("sparse_embedder", "unknown"),
            }
            
            formatted_results.append({
                "rank": i + 1,
                "score": score,
                "id": getattr(result, "id", f"result_{i}"),
                "file_path": file_path,
                "file_name": payload.get("file_name", ""),
                "chunk_index": cindex,
                "chunk_size": chunk_size,
                "preview": preview,
                "text": text,
                "embedder_info": embedder_info,
                "big_context": big_context
            })
        
        return {
            "query": query,
            "search_type": search_type,
            "count": len(formatted_results),
            "embedder_info": {
                "dense": dense_model_id,
                "sparse": sparse_model_id
            },
            "results": formatted_results
        }
        
    def load_model(self):
        """Load the model (PyTorch or MLX based on availability)"""
        try:
            if self.verbose:
                print(f"Loading model {self.model_name}")
                
            start_time = time.time()
            
            # Tokenizer is already loaded in __init__, so no need to load again
            
            if self.use_mlx:
                # Load MLX model - use lazy import
                if self.verbose:
                    print(f"Using MLX model with weights from {self.weights_path}")
                
                # Import MLXModel only when needed
                _, MLXModel, _, _ = _import_mlx_provider()
                if MLXModel is None:
                    raise ImportError("MLXModel not available")
                    
                self.mlx_model = MLXModel(self.weights_path, self.verbose)
                self.mlx_model.load()
                self.vector_size = self.mlx_model.config.hidden_size
            else:
                # Load PyTorch model
                if self.verbose:
                    print(f"Using PyTorch model from {self.model_name}")
                if not HAS_PYTORCH:
                    raise ImportError("PyTorch is required when MLX is not available. Install with: pip install torch")
                self.pytorch_model = AutoModel.from_pretrained(self.model_name)
                config = AutoConfig.from_pretrained(self.model_name)
                self.vector_size = config.hidden_size
                
            if self.verbose:
                print(f"Model loaded in {time.time() - start_time:.2f} seconds")
                print(f"Model embedding size: {self.vector_size}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def get_sparse_embedding(self, text: str) -> Tuple[List[int], List[float]]:
        """
        Unified method to get sparse embedding from any available provider.
        
        Args:
            text: Text to encode
            
        Returns:
            Tuple of (indices, values) for sparse vector representation
        """
        # Check FastEmbed provider first
        if hasattr(self, 'fastembed_provider') and self.fastembed_provider is not None:
            try:
                return self.fastembed_provider.get_sparse_embedding(text)
            except Exception as e:
                if self.verbose:
                    print(f"FastEmbed sparse embedding failed: {e}")
        
        # Check MLX provider second
        if hasattr(self, 'mlx_embedding_provider') and self.mlx_embedding_provider is not None:
            try:
                return self.mlx_embedding_provider.get_sparse_embedding(text)
            except Exception as e:
                if self.verbose:
                    print(f"MLX sparse embedding failed: {e}")
        
        # Check Ollama provider third
        if hasattr(self, 'ollama_embedding_provider') and self.ollama_embedding_provider is not None:
            try:
                return self.ollama_embedding_provider.get_sparse_embedding(text)
            except Exception as e:
                if self.verbose:
                    print(f"Ollama sparse embedding failed: {e}")
        
        # Fall back to basic implementation
        return EmbeddingUtils.generate_sparse_vector(text)
            
    def get_embedding_pytorch(self, text: str) -> np.ndarray:
        """Get embedding for a text using PyTorch backend"""
        if not HAS_PYTORCH:
            raise RuntimeError("PyTorch is not installed. Cannot compute embeddings.")

        if not text.strip():
            if self.verbose:
                print("[PyTorch] Input text is empty. Returning zero vector.")
            return np.zeros(self.vector_size, dtype=np.float32)

        if self.pytorch_model is None:
            raise RuntimeError("PyTorch model is not loaded.")

        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CHUNK_SIZE
        )

        try:
            with torch.no_grad():
                outputs = self.pytorch_model(**tokens)
                pooled_output = outputs.pooler_output[0].cpu().numpy()

            return pooled_output.astype(np.float32)

        except Exception as e:
            print(f"[PyTorch] Error during PyTorch embedding: {e}")
            return np.zeros(self.vector_size, dtype=np.float32)
        
        
    def get_embedding_mlx(self, text: str) -> np.ndarray:
        """Get embedding for a text using the MLX backend"""
        if not text.strip():
            if self.verbose:
                print("[MLX] Input text is empty. Returning zero vector.")
            return np.zeros(self.vector_size, dtype=np.float32)

        if self.mlx_model is None or not self.mlx_model.loaded:
            raise RuntimeError("MLX model is not loaded. Cannot compute embeddings.")

        tokenized = self.tokenizer(text, return_tensors="np", truncation=True, max_length=CHUNK_SIZE)

        try:
            # Convert input to MLX arrays
            input_ids = mx.array(tokenized["input_ids"])
            attention_mask = mx.array(tokenized["attention_mask"])
            
            # Get embeddings using the model
            _, pooled_output = self.mlx_model.model(
                input_ids, 
                attention_mask=attention_mask
            )
            
            # Convert to numpy
            embedding_np = mx.eval(pooled_output)[0]
            
            return embedding_np.astype(np.float32)

        except Exception as e:
            print(f"[MLX] Error during MLX embedding: {e}")
            return np.zeros(self.vector_size, dtype=np.float32)

            
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text input using preferred backend"""
        text = text.strip()
        if not text:
            return np.zeros(self.vector_size, dtype=np.float32)

        # Try to use FastEmbed first if available
        if hasattr(self, 'fastembed_provider') and self.fastembed_provider is not None:
            try:
                if self.verbose:
                    print(f"Generating embedding with FastEmbed provider")
                return self.fastembed_provider.get_dense_embedding(text)
            except Exception as e:
                if self.verbose:
                    print(f"Error using FastEmbed for embedding: {e}")
                    print("Falling back to next embedding method")

        # Try to use Ollama next if available
        if hasattr(self, 'ollama_embedding_provider') and self.ollama_embedding_provider is not None:
            try:
                if self.verbose:
                    print(f"Generating embedding with Ollama provider")
                return self.ollama_embedding_provider.get_dense_embedding(text)
            except Exception as e:
                if self.verbose:
                    print(f"Error using Ollama for embedding: {e}")
                    print("Falling back to next embedding method")

        # Try to use MLX embedding models next if available
        if hasattr(self, 'mlx_embedding_provider') and self.mlx_embedding_provider is not None:
            # Check if the dense model is properly loaded
            if hasattr(self.mlx_embedding_provider, 'dense_model') and self.mlx_embedding_provider.dense_model is not None:
                try:
                    if self.verbose:
                        print(f"Generating embedding with MLX embedding provider")
                    return self.mlx_embedding_provider.get_dense_embedding(text)
                except Exception as e:
                    if self.verbose:
                        print(f"Error using MLX embedding model: {e}")
                        print("Falling back to traditional embedding method")
            else:
                if self.verbose:
                    print("MLX embedding provider doesn't have a properly initialized dense_model")
                    print("Falling back to traditional embedding method")

        # Use MLX traditional model if available
        if self.use_mlx and self.mlx_model is not None and hasattr(self.mlx_model, 'loaded') and self.mlx_model.loaded:
            if self.verbose:
                print(f"Using traditional MLX model")
            return self.get_embedding_mlx(text)

        # Fall back to PyTorch
        if self.pytorch_model is not None:
            if self.verbose:
                print(f"Using PyTorch model")
            return self.get_embedding_pytorch(text)

        raise RuntimeError("No valid model backend available (FastEmbed, Ollama, MLX, or PyTorch).")

        
    def process_file(self, file_path: str) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """Process a single file into chunks with embeddings (dense only)"""
        if self.verbose:
            print(f"Processing file: {file_path}")
            
        # Extract text from file
        text = TextExtractorUtils.extract_from_file(file_path, self.verbose)
        if not text:
            if self.verbose:
                print(f"No text extracted from {file_path}")
            return []
            
        # Create chunks
        chunks = ChunkUtils.prepare_chunks(text, self.tokenizer)
        if self.verbose:
            print(f"Created {len(chunks)} chunks from {file_path}")
            
        # Calculate embeddings for each chunk
        results = []
        for i, chunk in enumerate(chunks):
            if self.verbose and (i == 0 or (i+1) % 10 == 0 or i == len(chunks) - 1):
                print(f"Calculating embedding for chunk {i+1}/{len(chunks)}")
                
            embedding = self.get_embedding(chunk["text"])
            
            # Create payload
            payload = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "text": chunk["text"],
                "token_count": chunk["token_count"],
                "chunk_index": i,
                "total_chunks": len(chunks),
                "metadata": {
                    "file_ext": os.path.splitext(file_path)[1],
                    "file_size": os.path.getsize(file_path),
                    "created_at": time.time(),
                    "embedder": self.dense_model_id
                }
            }
            
            results.append((embedding, payload))
            
        return results

    def process_file_with_sparse(self, file_path: str) -> List[Tuple[np.ndarray, Dict[str, Any], Tuple[List[int], List[float]]]]:
        """
        Process a file into chunks with both dense and sparse embeddings.
        Now supports FastEmbed, MLX and Ollama providers.
        """
        if self.verbose:
            print(f"Processing file: {file_path}")
            
        # Extract text from file
        text = TextExtractorUtils.extract_from_file(file_path, self.verbose)
        if not text:
            if self.verbose:
                print(f"No text extracted from {file_path}")
            return []
            
        # Create chunks
        chunks = ChunkUtils.prepare_chunks(text, self.tokenizer)
        if self.verbose:
            print(f"Created {len(chunks)} chunks from {file_path}")
        
        # Extract text from chunks for batch processing
        chunk_texts = [chunk["text"] for chunk in chunks]
        
        # Process chunks in batches
        batch_size = min(50, max(1, len(chunks) // 5))  # Max 50, min 1, aim for ~5 batches
        
        # Track results
        results = []
        
        # Track success/failure
        success_count = 0
        error_count = 0
        
        # Process in batches
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_indices = list(range(batch_start, batch_end))
            batch_chunks = [chunks[i] for i in batch_indices]
            batch_texts = [chunk_texts[i] for i in batch_indices]
            
            if self.verbose and len(chunks) > batch_size:
                print(f"Processing batch {batch_start//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} (chunks {batch_start+1}-{batch_end})")
            
            # Get dense embeddings for this batch
            dense_batch_embeddings = []
            try:
                # Try FastEmbed first
                if hasattr(self, 'fastembed_provider') and self.fastembed_provider is not None:
                    dense_batch_embeddings = self.fastembed_provider.get_dense_embedding(batch_texts)
                    if self.verbose:
                        print(f"Generated dense embeddings with FastEmbed for {len(batch_texts)} chunks")
                # Then try Ollama
                elif hasattr(self, 'ollama_embedding_provider') and self.ollama_embedding_provider is not None:
                    dense_batch_embeddings = self.ollama_embedding_provider.get_dense_embedding(batch_texts)
                    if self.verbose:
                        print(f"Generated dense embeddings with Ollama for {len(batch_texts)} chunks")
                # Then try MLX
                elif hasattr(self, 'mlx_embedding_provider') and self.mlx_embedding_provider is not None:
                    dense_batch_embeddings = self.mlx_embedding_provider.get_dense_embedding(batch_texts)
                    if self.verbose:
                        print(f"Generated dense embeddings with MLX for {len(batch_texts)} chunks")
            except Exception as e:
                if self.verbose:
                    print(f"Batch dense embedding failed: {e}")
            
            # Get sparse embeddings for this batch
            sparse_batch_embeddings = []
            try:
                # Try FastEmbed first
                if hasattr(self, 'fastembed_provider') and self.fastembed_provider is not None:
                    sparse_batch_embeddings = self.fastembed_provider._process_sparse_batch(batch_texts)
                    if self.verbose and sparse_batch_embeddings:
                        print(f"Generated sparse embeddings with FastEmbed for {len(sparse_batch_embeddings)} chunks")
                # Then try Ollama
                elif hasattr(self, 'ollama_embedding_provider') and self.ollama_embedding_provider is not None:
                    sparse_batch_embeddings = self.ollama_embedding_provider._process_sparse_batch(batch_texts)
                    if self.verbose and sparse_batch_embeddings:
                        print(f"Generated sparse embeddings with Ollama for {len(sparse_batch_embeddings)} chunks")
                # Then try MLX
                elif hasattr(self, 'mlx_embedding_provider') and self.mlx_embedding_provider is not None:
                    sparse_batch_embeddings = self.mlx_embedding_provider._process_sparse_batch(batch_texts)
                    if self.verbose and sparse_batch_embeddings:
                        print(f"Generated sparse embeddings with MLX for {len(sparse_batch_embeddings)} chunks")
            except Exception as e:
                if self.verbose:
                    print(f"Batch sparse embedding failed: {e}")
            
            # Process each chunk in the batch
            for i, chunk_idx in enumerate(batch_indices):
                chunk = chunks[chunk_idx]
                
                try:
                    # Get dense embedding (from batch or individual)
                    if isinstance(dense_batch_embeddings, list) and len(dense_batch_embeddings) > i:
                        dense_embedding = dense_batch_embeddings[i]
                    elif isinstance(dense_batch_embeddings, np.ndarray) and dense_batch_embeddings.ndim == 2 and dense_batch_embeddings.shape[0] > i:
                        dense_embedding = dense_batch_embeddings[i]
                    else:
                        # Individual fallback
                        dense_embedding = self.get_embedding(chunk["text"])
                    
                    # Get sparse embedding (from batch or individual)
                    if sparse_batch_embeddings and i < len(sparse_batch_embeddings):
                        sparse_embedding = sparse_batch_embeddings[i]
                    else:
                        # Use unified get_sparse_embedding method that handles all fallbacks
                        sparse_embedding = self.get_sparse_embedding(chunk["text"])
                    
                    # Create payload with embedder information
                    payload = {
                        "file_path": file_path,
                        "file_name": os.path.basename(file_path),
                        "text": chunk["text"],
                        "token_count": chunk["token_count"],
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                        "metadata": {
                            "file_ext": os.path.splitext(file_path)[1],
                            "file_size": os.path.getsize(file_path),
                            "created_at": time.time(),
                            "dense_embedder": self.dense_model_id,
                            "sparse_embedder": self.sparse_model_id
                        }
                    }
                    
                    results.append((dense_embedding, payload, sparse_embedding))
                    success_count += 1
                    
                except Exception as e:
                    error_count += 1
                    if self.verbose:
                        print(f"Error processing chunk {chunk_idx+1}: {e}")
        
        if self.verbose:
            print(f"File processing complete: {success_count} chunks successful, {error_count} chunks failed")
            
        return results


#######################################
# ResultProcessor Class
#######################################

class ResultProcessor:
    """Utilities for processing search results"""
    
    @staticmethod
    def get_score(result):
        """Safely get score from a result"""
        if hasattr(result, "score"):
            return getattr(result, "score", 0)
        return 0
    
    @staticmethod
    def adapt_result(result, db_type: str) -> Dict[str, Any]:
        """
        Adapt result object from different databases to a common format.
        
        Args:
            result: Result object from a database search
            db_type: Type of database that produced the result
            
        Returns:
            Standardized result dictionary with common structure
        """
        # Handle None or empty results
        if result is None:
            return {
                "id": "unknown",
                "payload": {
                    "text": "",
                    "file_path": "",
                    "file_name": "",
                    "chunk_index": 0,
                    "metadata": {}
                },
                "score": 0
            }
            
        # Adapt Qdrant results (already in expected format)
        if db_type.lower() == "qdrant":
            # Make sure required fields exist
            if not hasattr(result, "payload"):
                result.payload = {}
                
            # Return as is - already in the right format
            return result
            
        # Adapt LanceDB results
        elif db_type.lower() == "lancedb":
            # Convert from dict to object-like structure for consistency
            class AdaptedResult:
                def __init__(self, id, payload, score):
                    self.id = id
                    self.payload = payload
                    self.score = score
            
            # Extract fields from LanceDB result (dict format)
            result_id = result.get("id", "unknown")
            
            # Build payload from available fields
            payload = {
                "text": result.get("text", ""),
                "file_path": result.get("file_path", ""),
                "file_name": result.get("file_name", ""),
                "chunk_index": result.get("chunk_index", 0),
                "total_chunks": result.get("total_chunks", 0)
            }
            
            # Handle metadata - could be stored as JSON string
            metadata = result.get("metadata", "{}")
            if isinstance(metadata, str):
                try:
                    payload["metadata"] = json.loads(metadata)
                except:
                    payload["metadata"] = {}
            else:
                payload["metadata"] = metadata
                
            # Extract score - could be stored as _distance or score
            score = result.get("_distance", result.get("score", 0))
            
            # Create and return adapted result
            return AdaptedResult(result_id, payload, score)
            
        # Adapt Meilisearch results
        elif db_type.lower() == "meilisearch":
            class AdaptedResult:
                def __init__(self, id, payload, score):
                    self.id = id
                    self.payload = payload
                    self.score = score
            
            result_id = result.get("id", "unknown")
            
            # Build payload - Meilisearch flattens the structure
            payload = {
                "text": result.get("text", ""),
                "file_path": result.get("file_path", ""),
                "file_name": result.get("file_name", ""),
                "chunk_index": result.get("chunk_index", 0)
            }
            
            # Extract metadata if it exists
            metadata = {}
            for key, value in result.items():
                if key.startswith("metadata_"):
                    # Remove metadata_ prefix
                    metadata_key = key[9:]
                    metadata[metadata_key] = value
            
            payload["metadata"] = metadata
            
            # Meilisearch has _rankingScore
            score = result.get("_rankingScore", 0)
            
            return AdaptedResult(result_id, payload, score)
            
        # Adapt Elasticsearch results
        elif db_type.lower() == "elasticsearch":
            class AdaptedResult:
                def __init__(self, id, payload, score):
                    self.id = id
                    self.payload = payload
                    self.score = score
            
            # Extract data from Elasticsearch hit structure
            result_id = result.get("_id", "unknown")
            source = result.get("_source", {})
            
            # Build payload from source fields
            payload = {
                "text": source.get("text", ""),
                "file_path": source.get("file_path", ""),
                "file_name": source.get("file_name", ""),
                "chunk_index": source.get("chunk_index", 0)
            }
            
            # Extract metadata
            payload["metadata"] = source.get("metadata", {})
            
            # Extract score
            score = result.get("_score", 0)
            
            return AdaptedResult(result_id, payload, score)
            
        # Default for unknown database types - try to adapt generically
        else:
            # If it's already an object with id, payload, and score, return as is
            if hasattr(result, "id") and hasattr(result, "payload") and hasattr(result, "score"):
                return result
                
            # If it's a dict, try to adapt
            if isinstance(result, dict):
                class AdaptedResult:
                    def __init__(self, id, payload, score):
                        self.id = id
                        self.payload = payload
                        self.score = score
                
                result_id = result.get("id", "unknown")
                
                # Build payload
                payload = {
                    "text": result.get("text", ""),
                    "file_path": result.get("file_path", ""),
                    "file_name": result.get("file_name", ""),
                    "chunk_index": result.get("chunk_index", 0)
                }
                
                # Extract metadata
                payload["metadata"] = result.get("metadata", {})
                
                # Extract score
                score = result.get("score", 0)
                
                return AdaptedResult(result_id, payload, score)
            
            # If we can't adapt, return as is and hope for the best
            return result

#######################################
# SearchAlgorithms Class
#######################################

class SearchAlgorithms:
    """Common search algorithms that work across different database backends"""
    
    @staticmethod
    def manual_fusion(dense_results, sparse_results, limit: int, fusion_type: str = "rrf"):
        """
        Perform manual fusion of dense and sparse search results
        
        Args:
            dense_results: Results from dense search
            sparse_results: Results from sparse search
            limit: Maximum number of results to return
            fusion_type: Type of fusion to use (rrf or dbsf)
            
        Returns:
            Combined and reranked list of results
        """
        # Safety check for inputs
        if not dense_results and not sparse_results:
            return []
        
        # Helper function to get score safely
        def get_score(item):
            return getattr(item, "score", 0.0)
        
        # Dictionary to store combined results with ID as key
        combined_dict = {}
        
        # Process dense results
        for rank, item in enumerate(dense_results):
            item_id = getattr(item, "id", str(rank))
            combined_dict[item_id] = {
                "item": item,
                "dense_rank": rank + 1,
                "dense_score": get_score(item),
                "sparse_rank": float('inf'),
                "sparse_score": 0.0
            }
        
        # Process sparse results
        for rank, item in enumerate(sparse_results):
            item_id = getattr(item, "id", str(rank))
            if item_id in combined_dict:
                # Update existing entry
                combined_dict[item_id]["sparse_rank"] = rank + 1
                combined_dict[item_id]["sparse_score"] = get_score(item)
            else:
                # Add new entry
                combined_dict[item_id] = {
                    "item": item,
                    "dense_rank": float('inf'),
                    "dense_score": 0.0,
                    "sparse_rank": rank + 1,
                    "sparse_score": get_score(item)
                }
        
        # Apply fusion based on chosen method
        if fusion_type.lower() == "dbsf":
            # Distribution-based Score Fusion
            # Normalize scores within each result set
            dense_max = max([d["dense_score"] for d in combined_dict.values()]) if dense_results else 1.0
            sparse_max = max([d["sparse_score"] for d in combined_dict.values()]) if sparse_results else 1.0
            
            # Calculate combined scores
            for item_id, data in combined_dict.items():
                norm_dense = data["dense_score"] / dense_max if dense_max > 0 else 0
                norm_sparse = data["sparse_score"] / sparse_max if sparse_max > 0 else 0
                
                # DBSF: Weighted sum of normalized scores
                data["combined_score"] = 0.5 * norm_dense + 0.5 * norm_sparse
        else:
            # Reciprocal Rank Fusion (default)
            k = 60  # Constant for RRF
            
            # Calculate combined scores using RRF formula
            for item_id, data in combined_dict.items():
                rrf_dense = 1.0 / (k + data["dense_rank"]) if data["dense_rank"] != float('inf') else 0
                rrf_sparse = 1.0 / (k + data["sparse_rank"]) if data["sparse_rank"] != float('inf') else 0
                
                # RRF: Sum of reciprocal ranks
                data["combined_score"] = rrf_dense + rrf_sparse
        
        # Sort by combined score and convert back to a list
        sorted_results = sorted(
            combined_dict.values(), 
            key=lambda x: x["combined_score"], 
            reverse=True
        )
        
        # Return only the original items, limited to requested count
        return [data["item"] for data in sorted_results[:limit]]

    @staticmethod
    def handle_search_error(db_type: str, search_type: str, error: Exception, verbose: bool = False) -> Dict[str, str]:
        """
        Standardized error handling for search operations.
        
        Args:
            db_type: Type of database that encountered the error
            search_type: Type of search being performed
            error: The exception that occurred
            verbose: Whether to print detailed error information
            
        Returns:
            Error dictionary with standardized format
        """
        if verbose:
            print(f"Error in {search_type} search on {db_type}: {str(error)}")
            import traceback
            traceback.print_exc()
            
        return {
            "error": f"Error in {search_type} search on {db_type}: {str(error)}",
            "db_type": db_type,
            "search_type": search_type
        }
    
    @staticmethod
    def rerank_results(query: str, results, processor: Any, limit: int, verbose: bool = False):
        """
        Rerank results using the best available reranker.
        
        Args:
            query: Original search query
            results: List of search results to rerank
            processor: Document processor with embedding capabilities
            limit: Maximum number of results to return
            verbose: Whether to show verbose output
            
        Returns:
            Reranked list of results
        """
        if not results:
            return []
        
        if verbose:
            print(f"Reranking {len(results)} results")
        
        # Extract text from results
        passages = []
        for result in results:
            if hasattr(result, "payload") and result.payload and "text" in result.payload:
                passages.append(result.payload["text"])
            else:
                # Use empty string if no text available
                passages.append("")
        
        # Try FastEmbed reranker first
        if hasattr(processor, 'fastembed_provider') and processor.fastembed_provider is not None:
            try:
                reranked_scores = processor.fastembed_provider.rerank_with_fastembed(query, passages)
                if reranked_scores and len(reranked_scores) == len(results):
                    if verbose:
                        print("Using FastEmbed cross-encoder reranking")
                    
                    # Create tuples of (result, score) for sorting
                    scored_results = list(zip(results, reranked_scores))
                    
                    # Sort by reranked score
                    reranked_results = [result for result, _ in sorted(
                        scored_results, 
                        key=lambda x: x[1], 
                        reverse=True
                    )]
                    
                    return reranked_results[:limit]
            except Exception as e:
                if verbose:
                    print(f"FastEmbed reranking failed: {e}")
        
        # Try MLX reranker next
        if hasattr(processor, 'mlx_embedding_provider') and processor.mlx_embedding_provider is not None:
            try:
                if hasattr(processor.mlx_embedding_provider, 'rerank_with_mlx'):
                    reranked_scores = processor.mlx_embedding_provider.rerank_with_mlx(query, passages)
                    if reranked_scores and len(reranked_scores) == len(results):
                        if verbose:
                            print("Using MLX cross-encoder reranking")
                        
                        # Create tuples of (result, score) for sorting
                        scored_results = list(zip(results, reranked_scores))
                        
                        # Sort by reranked score
                        reranked_results = [result for result, _ in sorted(
                            scored_results, 
                            key=lambda x: x[1], 
                            reverse=True
                        )]
                        
                        return reranked_results[:limit]
            except Exception as e:
                if verbose:
                    print(f"MLX reranking failed: {e}")
        
        # Try Ollama reranker next
        if hasattr(processor, 'ollama_embedding_provider') and processor.ollama_embedding_provider is not None:
            try:
                if hasattr(processor.ollama_embedding_provider, 'rerank_with_ollama'):
                    reranked_scores = processor.ollama_embedding_provider.rerank_with_ollama(query, passages)
                    if reranked_scores and len(reranked_scores) == len(results):
                        if verbose:
                            print("Using Ollama reranking")
                        
                        # Create tuples of (result, score) for sorting
                        scored_results = list(zip(results, reranked_scores))
                        
                        # Sort by reranked score
                        reranked_results = [result for result, _ in sorted(
                            scored_results, 
                            key=lambda x: x[1], 
                            reverse=True
                        )]
                        
                        return reranked_results[:limit]
            except Exception as e:
                if verbose:
                    print(f"Ollama reranking failed: {e}")
        
        # Fallback: Use ColBERT-style late interaction scoring
        try:
            # Get query embedding
            query_embedding = processor.get_embedding(query)
            
            # Get passage embeddings
            passage_embeddings = []
            for passage in passages:
                embedding = processor.get_embedding(passage)
                passage_embeddings.append(embedding)
            
            # Compute similarity scores
            import numpy as np
            similarity_scores = []
            
            for passage_emb in passage_embeddings:
                # Cosine similarity
                similarity = np.dot(query_embedding, passage_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(passage_emb)
                )
                similarity_scores.append(float(similarity))
            
            # Create tuples of (result, score) for sorting
            scored_results = list(zip(results, similarity_scores))
            
            # Sort by similarity score
            reranked_results = [result for result, _ in sorted(
                scored_results, 
                key=lambda x: x[1], 
                reverse=True
            )]
            
            if verbose:
                print("Using fallback cosine similarity reranking")
            
            return reranked_results[:limit]
        except Exception as e:
            if verbose:
                print(f"Error in fallback reranking: {e}")
            # If reranking fails, keep original order
            return results[:limit]


#######################################
# FileUtils Class
#######################################

class FileUtils:
    """Utilities for file operations"""
    
    @staticmethod
    def get_files_to_process(
        search_dir: str,
        include_patterns: Optional[List[str]] = None,
        exclude_dirs: Optional[List[str]] = None,
        limit: Optional[int] = None,
        verbose: bool = False
    ) -> List[str]:
        """
        Get list of files to process based on patterns and limits.
        
        Args:
            search_dir: Directory to search
            include_patterns: File patterns to include
            exclude_dirs: Directories to exclude
            limit: Maximum number of files to return
            verbose: Whether to show verbose output
            
        Returns:
            List of file paths matching the criteria
        """
        if verbose:
            print(f"Scanning directory: {search_dir}")
            if include_patterns:
                print(f"Include patterns: {include_patterns}")
            if exclude_dirs:
                print(f"Exclude directories: {exclude_dirs}")
            if limit:
                print(f"Limit: {limit} files")
        
        files = []
        exclude_dirs = exclude_dirs or []
        
        # Normalize exclude dirs
        exclude_dirs = [os.path.normpath(d) for d in exclude_dirs]
        
        # Default to all supported extensions if no patterns provided
        if not include_patterns:
            include_patterns = [f"*{ext}" for ext in SUPPORTED_EXTENSIONS]
        
        for root, dirs, filenames in os.walk(search_dir):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if os.path.normpath(os.path.join(root, d)) not in exclude_dirs]
            
            for filename in filenames:
                file_path = os.path.join(root, filename)
                
                # Check if file matches any pattern
                if any(fnmatch.fnmatch(filename, pattern) for pattern in include_patterns):
                    files.append(file_path)
                    
                    if verbose and len(files) % 100 == 0:
                        print(f"Found {len(files)} matching files so far...")
                    
                    # Check limit
                    if limit is not None and len(files) >= limit:
                        if verbose:
                            print(f"Reached limit of {limit} files")
                        return files[:limit]
        
        if verbose:
            print(f"Found {len(files)} files to process")
        
        return files

    @staticmethod
    def download_file(url, local_path, verbose=False, retries=3):
        """
        Download a file with progress bar and retry support.
        
        Args:
            url: URL to download from
            local_path: Path to save the file to
            verbose: Whether to show verbose output
            retries: Number of retries for failed downloads
            
        Raises:
            Exception: If download fails after retries
        """
        import time

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        for attempt in range(retries):
            try:
                if verbose:
                    print(f"Downloading from {url} to {local_path} (attempt {attempt+1})")

                with requests.get(url, stream=True, timeout=30) as response:
                    if response.status_code != 200:
                        raise Exception(f"HTTP {response.status_code} - {response.reason}")

                    file_size = int(response.headers.get('Content-Length', 0))
                    chunk_size = 8192

                    with open(local_path, 'wb') as f:
                        if verbose and HAS_TQDM:
                            progress = tqdm(total=file_size, unit='B', unit_scale=True, desc=os.path.basename(local_path))
                            for chunk in response.iter_content(chunk_size=chunk_size):
                                if chunk:
                                    f.write(chunk)
                                    progress.update(len(chunk))
                            progress.close()
                        else:
                            for chunk in response.iter_content(chunk_size=chunk_size):
                                if chunk:
                                    f.write(chunk)

                # Check for LFS placeholder (small files starting with version info)
                if os.path.getsize(local_path) < 1000:
                    with open(local_path, 'rb') as f:
                        head = f.read(200)
                        if b"oid sha256" in head:
                            raise Exception("Downloaded LFS pointer file instead of real model. Check if LFS is enabled.")
                return  # Success

            except Exception as e:
                if verbose:
                    print(f"Download failed: {e}")
                time.sleep(1)

        raise Exception(f"Failed to download file after {retries} attempts: {url}")


#######################################
# ModelUtils Class
#######################################

class ModelUtils:
    """Utilities for model operations"""
    
    @staticmethod
    def download_model_files(model_name, weights_path, verbose=False):
        """
        Download model weights and config from HuggingFace.
        
        Args:
            model_name: Name of the model
            weights_path: Path to save weights to
            verbose: Whether to show verbose output
            
        Raises:
            ValueError: If model is not supported for direct download
        """
        weights_dir = os.path.dirname(weights_path)
        if not os.path.exists(weights_dir):
            if verbose:
                print(f"Creating weights directory: {weights_dir}")
            os.makedirs(weights_dir, exist_ok=True)

        # Check if we need to download weights
        if not os.path.exists(weights_path):
            if verbose:
                print(f"Model weights not found at {weights_path}. Downloading...")
            
            # If model is hosted on HuggingFace, download direct link
            if model_name.startswith(("cstr/", "sentence-transformers/")):
                weights_url = DEFAULT_WEIGHTS_URL
                FileUtils.download_file(weights_url, weights_path, verbose)
            else:
                # For other models, we'd need to convert them first (not implemented here)
                raise ValueError(f"Direct download only supported for specific models. Please download weights manually for {model_name}")
        elif verbose:
            print(f"Model weights already exist at: {weights_path}")
        
        # Download config if needed
        config_path = os.path.join(os.path.dirname(weights_path), "config.json")
        if not os.path.exists(config_path):
            if verbose:
                print(f"Downloading model config to {config_path}")
            FileUtils.download_file(DEFAULT_CONFIG_URL, config_path, verbose)
        elif verbose:
            print(f"Model config already exists at: {config_path}")
    
    @staticmethod
    def check_qdrant_available(host="localhost", port=6333, verbose=False):
        """
        Check if Qdrant is available (either server or local).
        
        Args:
            host: Qdrant host
            port: Qdrant port
            verbose: Whether to show verbose output
            
        Returns:
            bool: Whether Qdrant is available
        """
        try:
            if not qdrant_available:
                print("Qdrant client not available. Install with: pip install qdrant-client")
                return False
                
            # First try remote connection if specified
            if host != "localhost" or port != 6333:
                import socket
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    s.settimeout(2)
                    s.connect((host, port))
                    s.close()
                    if verbose:
                        print(f"✓ Qdrant server available at {host}:{port}")
                    return True
                except:
                    if verbose:
                        print(f"✗ Remote Qdrant server not available at {host}:{port}")
                        print("Falling back to local mode")
            
            # Try local mode
            try:
                # Create a test client with local storage
                temp_dir = tempfile.mkdtemp(prefix="qdrant_test_")
                
                # Initialize client
                client = QdrantClient(path=temp_dir)
                
                # Create a small test collection to verify functionality
                client.create_collection(
                    collection_name="test_collection",
                    vectors_config=VectorParams(size=4, distance=Distance.COSINE)
                )
                
                # Get collections to verify it worked
                collections = client.get_collections()
                
                # Clean up
                client.delete_collection("test_collection")
                shutil.rmtree(temp_dir, ignore_errors=True)
                
                if verbose:
                    print("✓ Local Qdrant storage mode is available")
                return True
            except Exception as e:
                print(f"Error testing Qdrant in local mode: {str(e)}")
                print("Please install Qdrant client: pip install qdrant-client")
                return False
        except:
            return False

#######################################
# TextExtractorUtils Class
#######################################

class TextExtractorUtils:
    """Utilities for extracting text from various file formats"""
    
    @staticmethod
    def extract_from_file(file_path: str, verbose: bool = False) -> str:
        """
        Extract text from a file based on its extension.
        
        Args:
            file_path: Path to the file
            verbose: Whether to show verbose output
            
        Returns:
            Extracted text
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if verbose:
            print(f"Extracting text from: {file_path} (format: {ext})")
            
        if ext == ".txt" or ext == ".md" or ext == ".html":
            return TextExtractorUtils._extract_from_text_file(file_path)
        elif ext == ".pdf":
            return TextExtractorUtils._extract_from_pdf(file_path)
        elif ext == ".json":
            return TextExtractorUtils._extract_from_json(file_path)
        elif ext == ".csv":
            return TextExtractorUtils._extract_from_csv(file_path)
        else:
            if verbose:
                print(f"Unsupported file format: {ext}. Skipping.")
            return ""
    
    @staticmethod
    def _extract_from_text_file(file_path: str) -> str:
        """Extract text from a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
                return ""
    
    @staticmethod
    def _extract_from_pdf(file_path: str) -> str:
        """Extract text from a PDF file"""
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text() + "\n"
            return text
        except ImportError:
            print("PyPDF2 not installed. Install with: pip install PyPDF2")
            return ""
        except Exception as e:
            print(f"Error extracting text from PDF {file_path}: {str(e)}")
            return ""
    
    @staticmethod
    def _extract_from_json(file_path: str) -> str:
        """Extract text from a JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error extracting text from JSON {file_path}: {str(e)}")
            return ""
    
    @staticmethod
    def _extract_from_csv(file_path: str) -> str:
        """Extract text from a CSV file"""
        try:
            import csv
            text = ""
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    text += " ".join(row) + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from CSV {file_path}: {str(e)}")
            return ""

#######################################
# ChunkUtils Class
#######################################

class ChunkUtils:
    """Utilities for text chunking to strictly respect token limits"""
    
    @staticmethod
    def prepare_chunks(text: str, tokenizer: Any, max_length: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks suitable for embedding.
        
        Args:
            text: Text to split into chunks
            tokenizer: Tokenizer to use for token counting
            max_length: Maximum tokens per chunk (including special tokens)
            overlap: Overlap between chunks in tokens
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text.strip():
            return []
        
        # CRITICAL: Adjust for special tokens
        special_tokens_count = 2  # Most BERT-based models add [CLS] and [SEP]
        effective_max_length = max_length - special_tokens_count
        
        chunks = []
        
        # For very long texts, first split by paragraphs 
        if len(text) > 100000:
            paragraphs = [p for p in text.split("\n\n") if p.strip()]
            
            # Process each paragraph separately
            for i, paragraph in enumerate(paragraphs):
                paragraph_chunks = ChunkUtils._chunk_text(
                    paragraph, tokenizer, effective_max_length, overlap
                )
                
                # Add paragraph index to chunks for reference
                for chunk in paragraph_chunks:
                    chunk["paragraph_idx"] = i
                    chunks.append(chunk)
                    
            return ChunkUtils._verify_chunks(chunks, tokenizer, max_length)
        
        # For shorter texts, process the entire text
        chunks = ChunkUtils._chunk_text(text, tokenizer, effective_max_length, overlap)
        return ChunkUtils._verify_chunks(chunks, tokenizer, max_length)
    
    @staticmethod
    def _chunk_text(text: str, tokenizer: Any, max_length: int = 510, overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Split text into chunks with improved token limit enforcement.
        
        Args:
            text: Text to chunk
            tokenizer: Tokenizer for counting tokens
            max_length: Maximum tokens per chunk (excluding special tokens)
            overlap: Overlap between chunks in tokens
            
        Returns:
            List of chunk dictionaries
        """
        if not text.strip():
            return []

        # Split into sentences for more natural chunking
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        
        chunks = []
        current_chunk_sentences = []
        current_chunk_tokens = []
        
        for sentence in sentences:
            # Get tokens for this sentence (without special tokens)
            # Handle case when tokenizer is not available
            if tokenizer is None:
                # Fallback to rough word count
                sentence_tokens = sentence.split()
            else:
                try:
                    sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
                except Exception:
                    # If tokenizer fails, fall back to rough word count
                    sentence_tokens = sentence.split()
            
            # If a single sentence is too long, split it by words
            if len(sentence_tokens) > max_length:
                # Process any accumulated sentences first
                if current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences)
                    
                    # Calculate token count for the chunk
                    if tokenizer is not None:
                        try:
                            chunk_tokens = tokenizer.encode(chunk_text, add_special_tokens=True)
                            token_count = len(chunk_tokens)
                        except:
                            # Fallback token count
                            token_count = len(chunk_text.split())
                    else:
                        token_count = len(chunk_text.split())
                    
                    chunks.append({
                        "text": chunk_text,
                        "token_count": token_count,
                        "start_idx": 0,
                        "end_idx": 0
                    })
                    
                    current_chunk_sentences = []
                    current_chunk_tokens = []
                
                # Split the long sentence into word-level chunks
                words = sentence.split()
                current_piece = []  # Always strings
                current_piece_token_count = 0  # Count tokens instead of keeping token list
                
                for word in words:
                    if tokenizer is not None:
                        try:
                            word_tokens = tokenizer.encode(word, add_special_tokens=False)
                            word_token_count = len(word_tokens)
                        except:
                            word_token_count = 1  # Fallback: treat as single token
                    else:
                        word_token_count = 1  # Fallback: treat as single token
                    
                    # If adding this word would exceed the limit, save the current piece
                    if current_piece_token_count + word_token_count > max_length:
                        if current_piece:
                            piece_text = " ".join(current_piece)
                            
                            # Add special tokens for final count
                            if tokenizer is not None:
                                try:
                                    piece_tokens = tokenizer.encode(piece_text, add_special_tokens=True)
                                    token_count = len(piece_tokens)
                                except:
                                    # Fallback token count
                                    token_count = len(piece_text.split())
                            else:
                                token_count = len(piece_text.split())
                            
                            chunks.append({
                                "text": piece_text,
                                "token_count": token_count,
                                "start_idx": 0,
                                "end_idx": 0
                            })
                            
                            # Keep some overlap for context
                            overlap_tokens = min(overlap, current_piece_token_count)
                            if overlap_tokens > 0:
                                # Keep last few words for overlap
                                overlap_word_count = max(1, len(current_piece) // 3)
                                current_piece = current_piece[-overlap_word_count:]
                                
                                # Recalculate token count for the overlap portion
                                current_piece_token_count = 0
                                if tokenizer is not None:
                                    try:
                                        overlap_text = " ".join(current_piece)
                                        overlap_tokens = tokenizer.encode(overlap_text, add_special_tokens=False)
                                        current_piece_token_count = len(overlap_tokens)
                                    except:
                                        current_piece_token_count = len(current_piece)  # Fallback
                                else:
                                    current_piece_token_count = len(current_piece)  # Fallback
                            else:
                                current_piece = []
                                current_piece_token_count = 0
                        
                        # If the word itself is too long (rare), we must truncate it
                        if word_token_count > max_length:
                            # Try to truncate intelligently based on token count
                            trunc_len = max(1, int(len(word) * (max_length / max(1, word_token_count)) * 0.8))
                            trunc_word = word[:trunc_len]
                            current_piece.append(trunc_word)
                            current_piece_token_count = 1  # Approximate for truncated word
                        else:
                            current_piece.append(word)
                            current_piece_token_count = word_token_count
                    else:
                        current_piece.append(word)
                        current_piece_token_count += word_token_count
                
                # Add the last piece if not empty
                if current_piece:
                    piece_text = " ".join(current_piece)
                    
                    # Calculate final token count
                    if tokenizer is not None:
                        try:
                            piece_tokens = tokenizer.encode(piece_text, add_special_tokens=True)
                            token_count = len(piece_tokens)
                        except:
                            # Fallback token count
                            token_count = len(piece_text.split())
                    else:
                        token_count = len(piece_text.split())
                    
                    chunks.append({
                        "text": piece_text,
                        "token_count": token_count,
                        "start_idx": 0,
                        "end_idx": 0
                    })
                
                # Continue with the next sentence
                current_chunk_sentences = []
                current_chunk_tokens = []
            
            # For normal-length sentences, add to current chunk if it fits
            elif isinstance(current_chunk_tokens, list) and isinstance(sentence_tokens, list) and len(current_chunk_tokens) + len(sentence_tokens) <= max_length:
                current_chunk_sentences.append(sentence)
                
                # Recalculate tokens to account for spacing and interaction between tokens
                if tokenizer is not None:
                    try:
                        current_chunk_tokens = tokenizer.encode(" ".join(current_chunk_sentences), add_special_tokens=False)
                    except:
                        # Fallback: just count words
                        current_chunk_tokens = []
                        for s in current_chunk_sentences:
                            current_chunk_tokens.extend(s.split())
                else:
                    # Fallback: just count words
                    current_chunk_tokens = []
                    for s in current_chunk_sentences:
                        current_chunk_tokens.extend(s.split())
            else:
                # Current chunk is full, save it
                if current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences)
                    
                    # Calculate final token count
                    if tokenizer is not None:
                        try:
                            chunk_tokens = tokenizer.encode(chunk_text, add_special_tokens=True)
                            token_count = len(chunk_tokens)
                        except:
                            # Fallback token count
                            token_count = len(chunk_text.split())
                    else:
                        token_count = len(chunk_text.split())
                    
                    chunks.append({
                        "text": chunk_text,
                        "token_count": token_count,
                        "start_idx": 0,
                        "end_idx": 0
                    })
                
                # Start a new chunk with overlap
                if overlap > 0 and len(current_chunk_sentences) > 0:
                    # Calculate overlap sentences (keep approximately 1/3 of previous sentences)
                    overlap_sentence_count = max(1, len(current_chunk_sentences) // 3)
                    overlap_sentences = current_chunk_sentences[-overlap_sentence_count:]
                    
                    # Start new chunk with overlap sentences
                    current_chunk_sentences = overlap_sentences + [sentence]
                    
                    # Recalculate tokens
                    if tokenizer is not None:
                        try:
                            current_chunk_tokens = tokenizer.encode(" ".join(current_chunk_sentences), add_special_tokens=False)
                        except:
                            # Fallback: just count words
                            current_chunk_tokens = []
                            for s in current_chunk_sentences:
                                current_chunk_tokens.extend(s.split())
                    else:
                        # Fallback: just count words
                        current_chunk_tokens = []
                        for s in current_chunk_sentences:
                            current_chunk_tokens.extend(s.split())
                else:
                    # No overlap, just start with current sentence
                    current_chunk_sentences = [sentence]
                    if isinstance(sentence_tokens, list):
                        current_chunk_tokens = sentence_tokens
                    else:
                        # Fallback: convert to list if needed
                        current_chunk_tokens = [sentence_tokens]
        
        # Add the last chunk if not empty
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            
            # Calculate final token count
            if tokenizer is not None:
                try:
                    chunk_tokens = tokenizer.encode(chunk_text, add_special_tokens=True)
                    token_count = len(chunk_tokens)
                except:
                    # Fallback token count
                    token_count = len(chunk_text.split())
            else:
                token_count = len(chunk_text.split())
            
            chunks.append({
                "text": chunk_text,
                "token_count": token_count,
                "start_idx": 0,
                "end_idx": 0
            })
        
        return chunks
    
    @staticmethod
    def _verify_chunks(chunks: List[Dict[str, Any]], tokenizer: Any, max_length: int) -> List[Dict[str, Any]]:
        """
        Verify all chunks are under the token limit and fix any that aren't.
        """
        verified_chunks = []
        
        for chunk in chunks:
            # Skip empty chunks
            if not chunk["text"].strip():
                continue
                
            # Re-tokenize to ensure accurate count
            token_count = chunk["token_count"]
            
            # If tokenizer is available, double-check token count
            if tokenizer is not None:
                try:
                    tokens = tokenizer.encode(chunk["text"], add_special_tokens=True)
                    token_count = len(tokens)
                except:
                    # If tokenization fails, use existing count
                    pass
            
            # If chunk is within limit, add it as is
            if token_count <= max_length:
                chunk["token_count"] = token_count  # Update the token count
                verified_chunks.append(chunk)
            else:
                # For chunks that are too big, truncate
                if tokenizer is not None:
                    try:
                        # Tokenize and truncate
                        tokens = tokenizer.encode(chunk["text"], add_special_tokens=True, truncation=True, max_length=max_length)
                        truncated_text = tokenizer.decode(tokens, skip_special_tokens=False)
                        
                        # Create new chunk with truncated text
                        verified_chunks.append({
                            "text": truncated_text,
                            "token_count": len(tokens),
                            "start_idx": chunk.get("start_idx", 0),
                            "end_idx": chunk.get("end_idx", 0),
                            "paragraph_idx": chunk.get("paragraph_idx", None),
                            "truncated": True
                        })
                    except:
                        # If tokenization fails, use character-based truncation
                        ratio = max_length / token_count
                        char_limit = int(len(chunk["text"]) * ratio * 0.9)  # 10% safety margin
                        truncated_text = chunk["text"][:char_limit]
                        
                        verified_chunks.append({
                            "text": truncated_text,
                            "token_count": max_length,  # Estimate
                            "start_idx": chunk.get("start_idx", 0),
                            "end_idx": chunk.get("end_idx", 0),
                            "paragraph_idx": chunk.get("paragraph_idx", None),
                            "truncated": True
                        })
                else:
                    # Without a tokenizer, use character-based truncation
                    ratio = max_length / token_count
                    char_limit = int(len(chunk["text"]) * ratio * 0.9)  # 10% safety margin
                    truncated_text = chunk["text"][:char_limit]
                    
                    verified_chunks.append({
                        "text": truncated_text,
                        "token_count": max_length,  # Estimate
                        "start_idx": chunk.get("start_idx", 0),
                        "end_idx": chunk.get("end_idx", 0),
                        "paragraph_idx": chunk.get("paragraph_idx", None),
                        "truncated": True
                    })
        
        return verified_chunks


#######################################
# GeneralUtils Class
#######################################

class GeneralUtils:
    """General utility functions"""
    
    @staticmethod
    def get_size_str(size_bytes):
        """
        Convert bytes to human-readable format.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Human-readable size string
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    @staticmethod
    def check_mlx_embedding_compatibility():
        """Check MLX embedding models compatibility and available models"""
        try:
            import mlx_embedding_models
            from importlib.metadata import version
            
            # Get the version
            mlx_embed_version = version('mlx_embedding_models')
            print(f"MLX Embedding Models version: {mlx_embed_version}")
            
            # Check supported parameters based on version
            from mlx_embedding_models import EmbeddingModel
            import inspect
            
            # Get supported parameters for from_pretrained
            sig = inspect.signature(EmbeddingModel.from_pretrained)
            params = list(sig.parameters.keys())
            print(f"Supported parameters for from_pretrained: {', '.join(params)}")
            
            # Check if from_registry exists
            if hasattr(EmbeddingModel, 'from_registry'):
                # Get available models in registry
                try:
                    if hasattr(mlx_embedding_models, 'EMBEDDING_REGISTRY'):
                        registry = mlx_embedding_models.EMBEDDING_REGISTRY
                    elif hasattr(mlx_embedding_models.embedding, 'registry'):
                        registry = mlx_embedding_models.embedding.registry
                    else:
                        registry = {}
                    
                    print(f"Available models in registry ({len(registry)} models):")
                    for i, (model_name, config) in enumerate(registry.items()):
                        ndim = config.get('ndim', 'unknown')
                        is_sparse = "Yes" if config.get('lm_head', False) else "No"
                        print(f"  {i+1}. {model_name} - Dim: {ndim}, Sparse: {is_sparse}")
                except Exception as e:
                    print(f"Error accessing model registry: {e}")
            else:
                print("Warning: from_registry method not available")
                
            return True
        except ImportError:
            print("MLX Embedding Models not installed")
            return False
        except Exception as e:
            print(f"Error checking MLX embedding compatibility: {e}")
            return False
    
    @staticmethod
    def check_db_dependencies(db_type: str) -> Tuple[bool, List[str]]:
        """
        Check if required dependencies for a database type are installed.
        
        Args:
            db_type: Database type
            
        Returns:
            Tuple of (is_available, missing_dependencies)
        """
        if db_type.lower() == "qdrant":
            try:
                import qdrant_client
                return True, []
            except ImportError:
                return False, ["qdrant-client"]
        elif db_type.lower() == "lancedb":
            missing = []
            try:
                import lancedb
            except ImportError:
                missing.append("lancedb")
            try:
                import pyarrow
            except ImportError:
                missing.append("pyarrow")
            return len(missing) == 0, missing
        elif db_type.lower() == "meilisearch":
            try:
                import meilisearch
                return True, []
            except ImportError:
                return False, ["meilisearch"]
        elif db_type.lower() == "elasticsearch":
            try:
                import elasticsearch
                return True, []
            except ImportError:
                return False, ["elasticsearch"]
        return True, []  # Default for unknown database types