"""
MLX model implementations and utilities for vector search.

This module provides:
- MLX model classes for embeddings
- BERT transformer implementation
- Embedding provider for dense and sparse embeddings
"""

import os
import re
import time
import json
from typing import List, Dict, Any, Tuple, Optional, Set, Iterator, Union, Literal
import numpy as np
from collections import Counter

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError as e:
    HAS_MLX = False
    print(f"Warning: MLX not available ({e}). Will use PyTorch for embeddings if available.")


# Try to import MLX embedding models
try:
    from mlx_embedding_models.embedding import EmbeddingModel, SpladeModel
    HAS_MLX_EMBEDDING_MODELS = True
except ImportError:
    HAS_MLX_EMBEDDING_MODELS = False
    print("Warning: mlx_embedding_models not available. Install with: pip install mlx-embedding-models")

# MLX Embedding Models Registry
MLX_EMBEDDING_REGISTRY = {
    # 3 layers, 384-dim
    "bge-micro": {
        "repo": "TaylorAI/bge-micro-v2",
        "max_length": 512,
        "pooling_strategy": "mean",
        "normalize": True, 
        "ndim": 384,
    },
    # 6 layers, 384-dim
    "gte-tiny": {
        "repo": "TaylorAI/gte-tiny",
        "max_length": 512,
        "pooling_strategy": "mean",
        "normalize": True,
        "ndim": 384,
    },
    "minilm-l6": {
        "repo": "sentence-transformers/all-MiniLM-L6-v2",
        "max_length": 512,
        "pooling_strategy": "mean",
        "normalize": True,
        "ndim": 384,
    },
    "snowflake-xs": {
        "repo": "Snowflake/snowflake-arctic-embed-xs",
        "max_length": 512,
        "pooling_strategy": "first",
        "normalize": True,
        "ndim": 384,
    },
    # 12 layers, 384-dim
    "minilm-l12": {
        "repo": "sentence-transformers/all-MiniLM-L12-v2",
        "max_length": 512,
        "pooling_strategy": "mean",
        "normalize": True,
        "ndim": 384,
    },
    "bge-small": {
        "repo": "BAAI/bge-small-en-v1.5",
        "max_length": 512,
        "pooling_strategy": "first", # cls token, not pooler output
        "normalize": True,
        "ndim": 384,
    },
    "multilingual-e5-small": {
        "repo": "intfloat/multilingual-e5-small",
        "max_length": 512,
        "pooling_strategy": "mean",
        "normalize": True,
        "ndim": 384,
    },
    # 12 layers, 768-dim
    "bge-base": {
        "repo": "BAAI/bge-base-en-v1.5",
        "max_length": 512,
        "pooling_strategy": "first",
        "normalize": True,
        "ndim": 768,
    },
    "nomic-text-v1": {
        "repo": "nomic-ai/nomic-embed-text-v1",
        "max_length": 2048,
        "pooling_strategy": "mean",
        "normalize": True,
        "ndim": 768,
    },
    "nomic-text-v1.5": {
        "repo": "nomic-ai/nomic-embed-text-v1.5",
        "max_length": 2048,
        "pooling_strategy": "mean",
        "normalize": True,
        "ndim": 768,
        "apply_ln": True,
    },
    # 24 layers, 1024-dim
    "bge-large": {
        "repo": "BAAI/bge-large-en-v1.5",
        "max_length": 512,
        "pooling_strategy": "first",
        "normalize": True,
        "ndim": 1024,
    },
    "snowflake-lg": {
        'repo': 'Snowflake/snowflake-arctic-embed-l',
        'max_length': 512,
        'pooling_strategy': 'first',
        'normalize': True,
        'ndim': 1024,
    },
    "bge-m3": {
        "repo": "BAAI/bge-m3",
        "max_length": 8192,
        "pooling_strategy": "first",
        "normalize": True,
        "ndim": 1024
    },
    "mixedbread-large": {
        "repo": 'mixedbread-ai/mxbai-embed-large-v1',
        "max_length": 512,
        "pooling_strategy": "first",
        "normalize": True,
        "ndim": 1024
    },
    # SPARSE MODELS #
    "distilbert-splade": {
        "repo": "raphaelsty/distilbert-splade",
        "max_length": 512,
        "lm_head": True,
        "pooling_strategy": "max",
        "normalize": False,
        "ndim": 768,
    },
    "neuralcherche-sparse-embed": {
        "repo": "raphaelsty/neural-cherche-sparse-embed",
        "max_length": 512,
        "lm_head": True,
        "pooling_strategy": "max",
        "normalize": False,
        "ndim": 768,
    },
    "opensearch": {
        "repo": "opensearch-project/opensearch-neural-sparse-encoding-doc-v1",
        "max_length": 512,
        "lm_head": True,
        "pooling_strategy": "max",
        "normalize": False,
        "ndim": 768,
    },
    "bert-base-uncased": { # mainly here as a baseline
        "repo": "bert-base-uncased",
        "max_length": 512,
        "pooling_strategy": "max",
        "normalize": False,
        "ndim": 768,
    },
    "naver-splade-distilbert": {
        "repo": "naver/splade-v3-distilbert",
        "max_length": 512,
        "lm_head": True,
        "pooling_strategy": "max",
        "normalize": False,
        "ndim": 768,
    },
    # Custom models
    "cstr-paraphrase-multilingual": {
        "repo": "cstr/paraphrase-multilingual-MiniLM-L12-v2-mlx",
        "max_length": 512,
        "pooling_strategy": "mean",
        "normalize": True,
        "ndim": 384,
    }
}

# Default constants
DEFAULT_DENSE_MODEL = "bge-small"
DEFAULT_SPARSE_MODEL = "distilbert-splade"

class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer implementation for MLX"""
    
    def __init__(
        self,
        dims: int,
        num_heads: int,
        mlp_dims: Optional[int] = None,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        mlp_dims = mlp_dims or dims * 4
        self.attention = nn.MultiHeadAttention(dims, num_heads, bias=True)
        self.ln1 = nn.LayerNorm(dims, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(dims, eps=layer_norm_eps)
        self.linear1 = nn.Linear(dims, mlp_dims)
        self.linear2 = nn.Linear(mlp_dims, dims)
        self.gelu = nn.GELU()

    def __call__(self, x, mask):
        attention_out = self.attention(x, x, x, mask)
        add_and_norm = self.ln1(x + attention_out)
        ff = self.linear1(add_and_norm)
        ff_gelu = self.gelu(ff)
        ff_out = self.linear2(ff_gelu)
        x = self.ln2(ff_out + add_and_norm)
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder stack implementation for MLX"""
    
    def __init__(self, num_layers: int, dims: int, num_heads: int, mlp_dims: Optional[int] = None):
        super().__init__()
        self.layers = [TransformerEncoderLayer(dims, num_heads, mlp_dims) for _ in range(num_layers)]

    def __call__(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class BertEmbeddings(nn.Module):
    """BERT embeddings layer implementation for MLX"""
    
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, input_ids: mx.array, token_type_ids: mx.array = None) -> mx.array:
        words = self.word_embeddings(input_ids)
        position = self.position_embeddings(
            mx.broadcast_to(mx.arange(input_ids.shape[1]), input_ids.shape)
        )

        if token_type_ids is None:
            token_type_ids = mx.zeros_like(input_ids)

        token_types = self.token_type_embeddings(token_type_ids)
        embeddings = position + words + token_types
        return self.norm(embeddings)


class BertModel(nn.Module):
    """BERT model implementation for MLX"""
    
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = TransformerEncoder(
            num_layers=config.num_hidden_layers,
            dims=config.hidden_size,
            num_heads=config.num_attention_heads,
            mlp_dims=config.intermediate_size,
        )
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: mx.array = None,
        attention_mask: mx.array = None,
    ) -> Tuple[mx.array, mx.array]:
        x = self.embeddings(input_ids, token_type_ids)

        if attention_mask is not None:
            attention_mask = mx.log(attention_mask)
            attention_mask = mx.expand_dims(attention_mask, (1, 2))

        y = self.encoder(x, attention_mask)
        pooled = mx.tanh(self.pooler(y[:, 0]))  # CLS token output
        return y, pooled


class MLXEmbeddingProvider:
    """Provider for dense and sparse embedding models from mlx_embedding_models or custom models"""
    
    def __init__(self, 
                dense_model_name: str = DEFAULT_DENSE_MODEL, 
                sparse_model_name: str = DEFAULT_SPARSE_MODEL,
                custom_repo_id: str = None,
                custom_ndim: int = None,
                custom_pooling: str = "mean",
                custom_normalize: bool = True,
                custom_max_length: int = 512,
                top_k: int = 64,
                batch_size: int = 16,
                verbose: bool = False):
        """
        Initialize MLXEmbeddingProvider with specified dense and sparse models.
        """
        # Initialize basic properties
        self.verbose = verbose
        self.dense_model = None
        self.sparse_model = None
        self.dense_model_name = dense_model_name
        self.sparse_model_name = sparse_model_name
        self.custom_repo_id = custom_repo_id
        self.custom_ndim = custom_ndim
        self.custom_pooling = custom_pooling
        self.custom_normalize = custom_normalize
        self.custom_max_length = custom_max_length
        self.batch_size = batch_size
        self.top_k = top_k
        
        # Use default dimension from registry, can be overridden for custom models
        if dense_model_name in MLX_EMBEDDING_REGISTRY:
            self.ndim = MLX_EMBEDDING_REGISTRY[dense_model_name]["ndim"]
        else:
            self.ndim = custom_ndim or 384  # Default if not specified
        
        # Skip model loading if mlx_embedding_models not available
        if not HAS_MLX_EMBEDDING_MODELS:
            if verbose:
                print("mlx_embedding_models not available, skipping model loading")
            return
        
        try:
            # Load dense model
            if verbose:
                print(f"Loading dense embedding model: {dense_model_name}")
            
            # Handle custom model if specified
            if custom_repo_id:
                if verbose:
                    print(f"Using custom model repo: {custom_repo_id}")
                # Create model with custom parameters
                try:
                    self.dense_model = EmbeddingModel.from_pretrained(
                        custom_repo_id,
                        pooling_strategy=custom_pooling,
                        normalize=custom_normalize,
                        max_length=custom_max_length
                    )
                    if custom_ndim:
                        self.ndim = custom_ndim
                    else:
                        # Try to get dimension from model config, safely
                        try:
                            if hasattr(self.dense_model.model, 'config') and hasattr(self.dense_model.model.config, 'hidden_size'):
                                self.ndim = self.dense_model.model.config.hidden_size
                        except Exception as e:
                            if verbose:
                                print(f"Could not determine dimension from model: {e}")
                                print(f"Using default dimension: {self.ndim}")
                except Exception as e:
                    print(f"Error loading custom model {custom_repo_id}: {e}")
                    self.dense_model = None
            else:
                # Load from registry
                try:
                    self.dense_model = EmbeddingModel.from_registry(dense_model_name)
                    # Safely get dimension from model
                    if hasattr(self.dense_model.model, 'config') and hasattr(self.dense_model.model.config, 'hidden_size'):
                        self.ndim = self.dense_model.model.config.hidden_size
                except Exception as e:
                    print(f"Error loading model {dense_model_name} from registry: {e}")
                    self.dense_model = None
                    
            if self.dense_model and verbose:
                print(f"Loaded dense model with dimension: {self.ndim}")
            
            # Load sparse model
            if verbose:
                print(f"Loading sparse embedding model: {sparse_model_name}")
            try:
                self.sparse_model = SpladeModel.from_registry(sparse_model_name, top_k=top_k)
                if verbose:
                    print(f"Loaded SPLADE model with top-k: {top_k}")
            except Exception as e:
                print(f"Error loading SPLADE model: {e}")
                self.sparse_model = None
        except Exception as e:
            print(f"Error loading MLX embedding models: {e}")
    
    def rerank_with_mlx(self, query: str, passages: List[str], top_k: int = None) -> List[float]:
        """
        Rerank passages using a cross-encoder style approach with MLX.
        
        This method interfaces with the common SearchAlgorithms class.
        
        Args:
            query: Original search query
            passages: List of passages to rerank
            top_k: Optional limit on returned results
            
        Returns:
            List of reranking scores
        """
        # Implementation could remain here as it's specific to MLX
        # But should share common code with SearchAlgorithms.rerank_results
        
        # If you're moving the implementation to SearchAlgorithms.rerank_results:
        from .utils import SearchAlgorithms
        
        # Create a simple processor-like object that provides get_embedding
        class SimpleProcessor:
            def __init__(self, provider):
                self.provider = provider
            
            def get_embedding(self, text):
                return self.provider.get_dense_embedding(text)
        
        processor = SimpleProcessor(self)
        
        # Get original results in a format expected by rerank_results
        class DummyResult:
            def __init__(self, payload):
                self.payload = payload
        
        dummy_results = [DummyResult({"text": passage}) for passage in passages]
        
        # Use the common reranking algorithm but keep our results
        reranked = SearchAlgorithms.rerank_results(query, dummy_results, processor, len(passages))
        
        # Extract and return just the scores
        scores = []
        original_indices = {}
        
        # Map original indices to track reordering
        for i, result in enumerate(dummy_results):
            original_indices[id(result)] = i
            
        # Get scores in original order
        for result in reranked:
            idx = original_indices.get(id(result), -1)
            if idx >= 0:
                # Use a high score for matched results, low for unmatched
                scores.append(1.0 - idx/len(passages))
            else:
                scores.append(0.0)
        
        return scores
        
    def get_dense_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Get dense embeddings for text(s).
        
        Args:
            text: String or list of strings to encode
        
        Returns:
            Numpy array of embeddings with shape (batch_size, dimension)
        """
        if self.dense_model is None:
            raise RuntimeError("Dense model not loaded")
        
        # Handle input type
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        # Generate embeddings
        embeddings = self.dense_model.encode(
            texts, 
            batch_size=self.batch_size,
            show_progress=self.verbose
        )
        
        # Return single embedding if input was a string
        if isinstance(text, str):
            return embeddings[0]
        return embeddings
    
    def get_sparse_embedding(self, text: str) -> Tuple[List[int], List[float]]:
        """
        Main entrypoint for sparse embedding generation.
        Handles errors and fallbacks internally.
        
        Args:
            text: String to encode
                
        Returns:
            Tuple of (indices, values) for sparse vector representation
        """
        # Handle empty text
        if not text.strip():
            return [0], [0.0]
            
        # If no SPLADE model, use fallback immediately
        if self.sparse_model is None:
            return self._generate_bow_embedding(text)
        
        # Preprocess text for the model
        processed_text = self._preprocess_text_for_splade(text)
        
        # Try main SPLADE encoding approach
        try:
            return self._generate_splade_embedding(processed_text)
        except Exception as e:
            if self.verbose:
                print(f"Primary SPLADE approach failed: {e}")
                print("Trying alternative SPLADE method...")
            
            # Try alternative direct approach
            try:
                return self._generate_direct_splade_embedding(processed_text)
            except Exception as e2:
                if self.verbose:
                    print(f"Alternative SPLADE approach failed: {e2}")
                    print("Falling back to simple sparse encoding...")
                
                # Fall back to BOW if all else fails
                return self._generate_bow_embedding(text)

    def _preprocess_text_for_splade(self, text: str) -> str:
        """
        Preprocess text for SPLADE model, handling token length issues.
        
        Args:
            text: Original text to preprocess
            
        Returns:
            Processed text ready for SPLADE encoding
        """
        # Skip preprocessing if no model or tokenizer
        if not hasattr(self, 'sparse_model') or not hasattr(self.sparse_model, 'tokenizer'):
            return text
        
        try:
            # Check token length without warning
            tokens = self.sparse_model.tokenizer.encode(text, add_special_tokens=True, truncation=False)
            
            max_len = self.sparse_model.max_length
            
            if len(tokens) > max_len:
                # Only log detailed info in verbose mode
                if self.verbose:
                    print(f"\nTruncating text from {len(tokens)} tokens to {max_len}")
                
                # Intelligent truncation:
                # 1. First try to respect sentence boundaries if possible
                try:
                    # Simple sentence splitting 
                    sentences = re.split(r'(?<=[.!?])\s+', text)
                    
                    # Build up text until we approach the limit
                    truncated_text = ""
                    current_len = 0
                    safe_max = max_len - 5  # Leave some room for special tokens
                    
                    for sentence in sentences:
                        # Check how many new tokens this sentence would add
                        sent_tokens = self.sparse_model.tokenizer.encode(sentence, add_special_tokens=False)
                        
                        # Stop if adding this sentence would exceed the limit
                        if current_len + len(sent_tokens) > safe_max:
                            break
                            
                        truncated_text += sentence + " "
                        current_len += len(sent_tokens)
                    
                    # If we managed to include at least some text, use it
                    if truncated_text and len(truncated_text) > len(text) / 10:  # At least 10% of original
                        text = truncated_text
                    else:
                        # Fall back to simple truncation
                        tokens = tokens[:max_len]
                        text = self.sparse_model.tokenizer.decode(tokens, skip_special_tokens=False)
                except:
                    # Fall back to simple truncation if sentence splitting fails
                    tokens = tokens[:max_len]
                    text = self.sparse_model.tokenizer.decode(tokens, skip_special_tokens=False)
                
                # Verify truncation worked (silently)
                try:
                    new_tokens = self.sparse_model.tokenizer.encode(text, add_special_tokens=True)
                    
                    if len(new_tokens) > max_len:
                        # If still too long, do direct token truncation
                        tokens = new_tokens[:max_len]
                        text = self.sparse_model.tokenizer.decode(tokens, skip_special_tokens=False)
                except:
                    # Last resort: character-based truncation
                    char_ratio = max_len / len(tokens)
                    char_limit = int(len(text) * char_ratio * 0.9)  # 10% safety margin
                    text = text[:char_limit]
        except Exception as e:
            if self.verbose:
                print(f"Token handling error (will continue): {e}")
        
        return text

    def _generate_splade_embedding(self, text: str) -> Tuple[List[int], List[float]]:
        """
        Primary method to generate SPLADE embeddings using the model's encode method
        with patched _sort_inputs.
        
        Args:
            text: Preprocessed text to encode
            
        Returns:
            Tuple of (indices, values) for sparse vector representation
        """
        # Apply a temporary patch to _sort_inputs
        original_sort_inputs = None
        if hasattr(self.sparse_model, '_sort_inputs'):
            original_sort_inputs = self.sparse_model._sort_inputs
            
            def patched_sort_inputs(tokens):
                result = original_sort_inputs(tokens)
                if isinstance(result, tuple) and len(result) == 3:
                    return result[0], result[1]
                return result
                
            self.sparse_model._sort_inputs = patched_sort_inputs
        
        # Patch tokenizer to silence warnings
        original_encode = None
        if hasattr(self.sparse_model, 'tokenizer') and hasattr(self.sparse_model.tokenizer, 'encode'):
            original_encode = self.sparse_model.tokenizer.encode
            
            # Create a patched version that automatically truncates
            def patched_encode(text, *args, **kwargs):
                # If truncation not explicitly set, force it on
                if 'truncation' not in kwargs:
                    kwargs['truncation'] = True
                    # Only set max_length if not already provided
                    if 'max_length' not in kwargs and hasattr(self.sparse_model, 'max_length'):
                        kwargs['max_length'] = self.sparse_model.max_length
                        
                # Call original function with modified args
                return original_encode(text, *args, **kwargs)
            
            # Apply the patch
            self.sparse_model.tokenizer.encode = patched_encode
        
        # Disable tqdm for cleaner output
        import tqdm.auto
        original_tqdm = tqdm.auto.tqdm
        
        # Create a dummy tqdm that does nothing
        class DummyTQDM:
            def __init__(self, *args, **kwargs):
                self.total = kwargs.get('total', 0)
                self.n = 0
            def update(self, n=1): self.n += n
            def close(self): pass
            def __enter__(self): return self
            def __exit__(self, *args, **kwargs): pass
            def set_postfix(self, *args, **kwargs): pass
        
        try:
            # Replace tqdm with our dummy version
            tqdm.auto.tqdm = DummyTQDM
            
            # Now encode with warnings suppressed
            sparse_embedding = self.sparse_model.encode([text], batch_size=1, show_progress=False)
            
            # Process the result
            if isinstance(sparse_embedding, list):
                if len(sparse_embedding) > 0:
                    sparse_embedding = sparse_embedding[0]
                else:
                    return [0], [0.0]
            elif sparse_embedding is None:
                return [0], [0.0]
            
            # Extract non-zero indices and values efficiently
            indices = []
            values = []
            
            if hasattr(sparse_embedding, "shape"):
                import numpy as np
                
                # Flatten if needed
                if len(sparse_embedding.shape) > 1:
                    sparse_embedding = np.ravel(sparse_embedding)
                
                # Get non-zero indices directly
                nonzero_indices = np.nonzero(sparse_embedding > 0)[0]
                for idx in nonzero_indices:
                    try:
                        value = float(sparse_embedding[idx])
                        indices.append(int(idx))
                        values.append(value)
                    except:
                        continue
            elif isinstance(sparse_embedding, tuple) and len(sparse_embedding) == 2:
                indices, values = sparse_embedding
            else:
                if self.verbose:
                    print(f"Unknown sparse embedding format: {type(sparse_embedding)}")
                return [0], [0.0]
            
            # Handle empty results
            if not indices:
                return [0], [0.0]
            
            return indices, values
            
        except Exception as e:
            if self.verbose:
                print(f"SPLADE model error: {e}")
            raise
        finally:
            # Restore the original methods
            if original_sort_inputs is not None:
                self.sparse_model._sort_inputs = original_sort_inputs
                
            # Restore original tokenizer.encode if we patched it
            if original_encode is not None:
                self.sparse_model.tokenizer.encode = original_encode
                
            # Restore original tqdm
            tqdm.auto.tqdm = original_tqdm

    def _generate_direct_splade_embedding(self, text: str) -> Tuple[List[int], List[float]]:
        """
        Alternative method to generate SPLADE embeddings by directly working with the model,
        bypassing the encode method and _sort_inputs.
        
        Args:
            text: Preprocessed text to encode
                
        Returns:
            Tuple of (indices, values) for sparse vector representation
        """
        try:
            # Check if we have the required components
            if not hasattr(self.sparse_model, 'model') or not hasattr(self.sparse_model, 'tokenizer'):
                raise ValueError("SPLADE model is missing required components")
            
            # Manually tokenize the text
            inputs = self.sparse_model.tokenizer(
                [text],
                padding='max_length',
                truncation=True,
                max_length=self.sparse_model.max_length,
                return_tensors='np'
            )
            
            # Prepare batch for model
            batch = {}
            for k, v in inputs.items():
                # Convert numpy arrays to MLX arrays
                import mlx.core as mx
                batch[k] = mx.array(v)
            
            # Run the model manually
            mlm_output, _ = self.sparse_model.model(**batch)
            
            # Apply max pooling over sequence length dimension (dim=1)
            # Multiply by attention mask to ignore padding
            embs = mx.max(mlm_output * mx.expand_dims(batch["attention_mask"], -1), axis=1)
            
            # Apply SPLADE log(1+ReLU(x)) transformation
            embs = mx.log(1 + mx.maximum(embs, 0))
            
            # Apply top-k if needed
            if hasattr(self.sparse_model, 'top_k') and self.sparse_model.top_k > 0 and hasattr(self.sparse_model, '_create_sparse_embedding'):
                embs = self.sparse_model._create_sparse_embedding(embs, self.sparse_model.top_k)
            
            # Convert to numpy
            import numpy as np
            sparse_embs = np.array(mx.eval(embs), copy=False)
            
            # Extract nonzero components (the first result in the batch)
            embedding = sparse_embs[0]
            
            # Convert to sparse representation
            indices = []
            values = []
            
            for idx in range(len(embedding)):
                value = float(embedding[idx])
                if value > 0:
                    indices.append(idx)
                    values.append(value)
            
            # If empty, return default
            if not indices:
                return [0], [0.0]
            
            return indices, values
            
        except Exception as e:
            if self.verbose:
                print(f"Error in direct SPLADE processing: {e}")
            raise
    
    def _generate_bow_embedding_imported(self, text: str) -> Tuple[List[int], List[float]]:
        """
        Generate a simple bag-of-words sparse vector (fallback method).
        
        Args:
            text: Text to encode
                
        Returns:
            Tuple of (indices, values) for sparse representation
        """
        return EmbeddingUtils.generate_sparse_vector(text)

    def _generate_bow_embedding(self, text: str) -> Tuple[List[int], List[float]]:
        """
        Generate a simple bag-of-words sparse vector (fallback method).
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

    def get_sparse_embeddings_batch(self, texts: List[str]) -> List[Tuple[List[int], List[float]]]:
        """
        Get sparse embeddings for multiple texts without nested progress bars.
        
        Args:
            texts: List of strings to encode
            
        Returns:
            List of (indices, values) tuples for sparse vector representation
        """
        # Handle empty input
        if not texts:
            return []
        
        # Process in optimal batch size
        batch_size = min(16, len(texts))
        results = []
        
        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            batch = texts[batch_start:batch_end]
            
            # Process batch with proper error handling
            batch_results = self._process_sparse_batch(batch)
            results.extend(batch_results)
        
        return results

    def _process_sparse_batch(self, batch: List[str]) -> List[Tuple[List[int], List[float]]]:
        """
        Process a batch of texts for sparse embedding generation with unified error handling.
        
        Args:
            batch: List of texts to process
            
        Returns:
            List of (indices, values) tuples
        """
        results = []
        
        for text in batch:
            try:
                # Use the main method that handles all fallbacks internally
                result = self.get_sparse_embedding(text)
                results.append(result)
            except Exception as e:
                if self.verbose:
                    print(f"Error processing text in batch: {e}")
                # Always return something, even if processing fails
                results.append(EmbeddingUtils.generate_sparse_vector(text))
        
        return results

class MLXModel:
    """Wrapper for MLX model operations using custom BertModel"""
    
    def __init__(self, weights_path, verbose=False):
        self.weights_path = weights_path
        self.verbose = verbose
        self.config = None
        self.model = None
        self.loaded = False

    def load(self):
        """Load config and weights into MLX model"""
        if self.loaded:
            return
            
        try:
            # Load configuration
            config_path = os.path.join(os.path.dirname(self.weights_path), "config.json")
            if self.verbose:
                print(f"Loading config from {config_path}")
                
            # Check if there's a local config
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                    
                # Create a config object manually to avoid HuggingFace lookup
                class Config:
                    def __init__(self, **kwargs):
                        for key, value in kwargs.items():
                            setattr(self, key, value)
                
                config = Config(**config_dict)
                
                if self.verbose:
                    print(f"Loaded local config from {config_path}")
            else:
                # If no config available, raise error
                raise ValueError(f"No config file found at {config_path}")
            
            if self.verbose:
                print(f"Creating model with vocab_size={config.vocab_size}, hidden_size={config.hidden_size}")
                
            # Create the model
            self.model = BertModel(config)
            
            if self.verbose:
                print(f"Loading weights from {self.weights_path}")
            
            # Load weights directly
            weights = np.load(self.weights_path)
            params = {k: mx.array(v) for k, v in weights.items()}
            
            # Use MLX's standard method for updating parameters
            self.model.update(params)
            
            if self.verbose:
                print("Weights loaded successfully")
                print("Running dummy forward pass tests...")

                dummy_input = {
                    "input_ids": mx.array([[101, 102]]),         # [CLS] [SEP]
                    "attention_mask": mx.array([[1, 1]])
                }

                try:
                    _, pooled = self.model(**dummy_input)
                    mx.eval(pooled)
                    print("✅ Dummy forward pass successful. Output shape:", pooled.shape)
                except Exception as e:
                    print("❌ Forward pass failed:", str(e))
                
            self.config = config
            self.loaded = True
            
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")

    def get_embedding(self, tokenized_input) -> np.ndarray:
        """Compute embedding from tokenized input"""
        input_ids = mx.array(tokenized_input["input_ids"])
        attention_mask = mx.array(tokenized_input["attention_mask"])
        
        # Forward pass through the model
        _, pooled_output = self.model(
            input_ids,
            attention_mask=attention_mask
        )
        
        # Convert to numpy array
        return mx.eval(pooled_output)[0]

# Utility functions that can be shared across modules
def generate_sparse_vector(text: str) -> Tuple[List[int], List[float]]:
    """
    Generate a simple sparse vector using term frequencies.
    This is a standalone utility function that doesn't require an MLX embedding provider.
    
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