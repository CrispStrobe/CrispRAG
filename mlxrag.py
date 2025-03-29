#!/usr/bin/env python3
"""
mlxrag.py:
Enhanced Qdrant Indexer with MLX and SPLADE support.

This script combines document indexing and search capabilities with flexible model selection:
- MLX embedding models from registry or custom models
- MLX for efficient dense embedding generation
- SPLADE for sparse embedding generation
- Qdrant for vector storage and retrieval with multiple model support
"""

import argparse
import fnmatch
import os
import time
import json
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set, Iterator, Union, Literal
import tempfile
import requests
from collections import Counter
import re
import numpy as np

from mlx_utils import (
    MLXEmbeddingProvider, MLXModel, MLX_EMBEDDING_REGISTRY,
    DEFAULT_DENSE_MODEL, DEFAULT_SPARSE_MODEL, generate_sparse_vector as mlx_generate_sparse_vector,
    HAS_MLX, HAS_MLX_EMBEDDING_MODELS
)

try:
    from ollama_utils import (
        OllamaEmbeddingProvider, generate_sparse_vector as ollama_generate_sparse_vector,
        HAS_OLLAMA, DEFAULT_OLLAMA_EMBED_MODEL, OLLAMA_MODELS_REGISTRY
    )
except ImportError:
    HAS_OLLAMA = False

try:
    from fastembed_utils import (
        FastEmbedProvider, generate_sparse_vector as fastembed_generate_sparse_vector,
        HAS_FASTEMBED, DEFAULT_FASTEMBED_MODEL, DEFAULT_FASTEMBED_SPARSE_MODEL, FASTEMBED_MODELS_REGISTRY
    )
except ImportError:
    HAS_FASTEMBED = False

# Import custom modules
from utils import (
    FileUtils, ModelUtils, GeneralUtils, TextProcessor, 
    ResultProcessor, SearchAlgorithms, ChunkUtils, TextExtractorUtils
)

# Try to import optional dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available. Install with: pip install psutil")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not available. Install with: pip install tqdm")

# Import Qdrant in dual-mode (HTTP + local support)
try:
    from qdrant_client import QdrantClient

    # HTTP-specific models
    from qdrant_client import models as qmodels
    from qdrant_client.http.models import Distance as HTTPDistance, VectorParams as HTTPVectorParams, MatchText
    from qdrant_client.http.exceptions import UnexpectedResponse

    # Local (non-HTTP) models
    from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue, SparseVector
    from qdrant_client.models import models as qdrant_models

    qdrant_available = True
except ImportError:
    qdrant_available = False
    print("Qdrant not available. Install with: pip install qdrant-client")

# Try to import transformers
try:
    from transformers import AutoTokenizer, AutoConfig, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available. Install with: pip install transformers")

# Try to import PyTorch
try:
    import torch
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    if not HAS_MLX:
        print("Warning: Neither MLX nor PyTorch is available. At least one is required for embedding generation.")

# Constants
DEFAULT_COLLECTION = "documents"
DEFAULT_MODEL = "cstr/paraphrase-multilingual-MiniLM-L12-v2-mlx"
DEFAULT_WEIGHTS_PATH = "weights/paraphrase-multilingual-MiniLM-L12-v2.npz"
DEFAULT_WEIGHTS_URL = "https://huggingface.co/cstr/paraphrase-multilingual-MiniLM-L12-v2-mlx/resolve/main/paraphrase-multilingual-MiniLM-L12-v2.npz"
DEFAULT_CONFIG_URL = "https://huggingface.co/cstr/paraphrase-multilingual-MiniLM-L12-v2-mlx/resolve/main/config.json"
MODEL_TOKENIZER_MAP = {
    "cstr/paraphrase-multilingual-MiniLM-L12-v2-mlx": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}

SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".pdf", ".json", ".csv"}
CHUNK_SIZE = 512  # Max tokens per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks
VECTOR_SIZE = 384  # Default vector size for the model


def run_search(args):
    """Run search on the existing collection with modular database support"""
    
    # Get database type from arguments, defaulting to Qdrant
    db_type = args.db_type
    
    # Import the factory
    from vector_db_interface import DBFactory
    
    # Carefully import from utils to avoid circular imports
    from utils import TextProcessor, ResultProcessor, SearchAlgorithms
    
    # 1. Initialize database manager using the factory
    if args.verbose:
        print(f"\n====== Initializing {db_type.capitalize()} Connection ======")
    
    # Collect common parameters for all database types
    db_args = {
        "collection_name": args.collection,
        "storage_path": args.storage_path,  
        "verbose": args.verbose,
        "dense_model_id": args.dense_model,
        "sparse_model_id": args.sparse_model
    }
    
    # Add database-specific parameters based on type
    if db_type.lower() == 'qdrant':
        db_args.update({
            "host": args.host,
            "port": args.port,
        })
    elif db_type.lower() == 'lancedb':
        db_args.update({
            "uri": args.lancedb_uri,
        })
    elif db_type.lower() == 'meilisearch':
        db_args.update({
            "url": args.meilisearch_url,
            "api_key": args.meilisearch_api_key,
        })
    elif db_type.lower() == 'chromadb':
        db_args.update({
            "host": args.chromadb_host,
            "port": args.chromadb_port,
            "use_embedding_function": args.chromadb_use_embedding_function,
            "embedding_function_model": args.chromadb_embedding_model
        })
    elif db_type.lower() == 'elasticsearch':
        db_args.update({
            "hosts": args.es_hosts,
            "api_key": args.es_api_key,
            "username": args.es_username,
            "password": args.es_password,
        })
    elif db_type.lower() == 'milvus':
        db_args.update({
            "host": args.milvus_host,
            "port": args.milvus_port,
            "user": args.milvus_user,
            "password": args.milvus_password,
            "secure": args.milvus_secure,
            "token": args.milvus_token,
        })
        
    # Create the database handler using the factory
    try:
        db_manager = DBFactory.create_db(db_type, **db_args)
    except Exception as e:
        print(f"Error initializing {db_type} database: {e}")
        return
        
    # Check for collection info with proper error handling
    try:
        collection_info = db_manager.get_collection_info()
        if isinstance(collection_info, dict) and "error" in collection_info:
            print(f"Warning: {collection_info['error']}")
    except Exception as e:
        print(f"Warning: Unable to get collection info: {e}")
    
    # 2. Load model for vector search if needed - but NOT for keyword search
    processor = None
    if args.search_type in ["vector", "sparse", "hybrid"]:
        if args.verbose:
            print("\n====== Loading Models for Search ======")
            
        # Initialize document processor with embedding models if available
        processor_args = {
            "model_name": args.model,
            "weights_path": args.weights,
            "verbose": args.verbose
        }
        
        # Check for FastEmbed usage first (highest priority)
        if args.use_fastembed:
            if HAS_FASTEMBED:
                processor_args.update({
                    "use_fastembed": True,
                    "fastembed_model": args.fastembed_model,
                    "fastembed_sparse_model": args.fastembed_sparse_model,
                    "fastembed_use_gpu": args.fastembed_use_gpu,
                    "fastembed_cache_dir": args.fastembed_cache_dir
                })
                if args.verbose:
                    print(f"Using FastEmbed with model: {args.fastembed_model}")
            else:
                print("Warning: FastEmbed not available. Install with: pip install fastembed")
                print("Falling back to Ollama or MLX")
        
        # Check for Ollama usage second
        elif args.use_ollama:
            if HAS_OLLAMA:
                processor_args.update({
                    "use_ollama": True,
                    "ollama_model": args.ollama_model,
                    "ollama_host": args.ollama_host
                })
                if args.verbose:
                    print(f"Using Ollama with model: {args.ollama_model}")
            else:
                print("Warning: Ollama not available. Falling back to MLX or default models.")
        
        # Use MLX embedding models if available and neither FastEmbed nor Ollama used
        if not processor_args.get("use_ollama", False) and args.use_mlx_models and HAS_MLX_EMBEDDING_MODELS:
            processor_args.update({
                "use_mlx_embedding": True,
                "dense_model": args.dense_model,
                "sparse_model": args.sparse_model,
                "top_k": args.top_k,
                "custom_repo_id": args.custom_repo_id,
                "custom_ndim": args.custom_ndim,
                "custom_pooling": args.custom_pooling,
                "custom_normalize": args.custom_normalize,
                "custom_max_length": args.custom_max_length
            })
            
            if args.verbose:
                if args.custom_repo_id:
                    print(f"Using custom model: {args.custom_repo_id}")
                else:
                    print(f"Using MLX embedding models: {args.dense_model} (dense), {args.sparse_model} (sparse)")
        
        processor = TextProcessor(**processor_args)
    
    # 3. Run search with improved relevance 
    if args.verbose:
        print(f"\n====== Running {args.search_type.capitalize()} Search on {db_type.capitalize()} ======")
    
    # Use the search method from the database manager
    results = db_manager.search(
        query=args.query,
        search_type=args.search_type,
        limit=args.limit,
        processor=processor,
        prefetch_limit=args.prefetch_limit,
        fusion_type=args.fusion,
        relevance_tuning=args.relevance_tuning,
        context_size=args.context_size,
        score_threshold=args.score_threshold,
        rerank=args.rerank,
        reranker_type=args.reranker_type
    )
    
    # 4. Display results 
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    # Use terminal colors if enabled
    color_output = not args.no_color
    
    if color_output:
        RESET = "\033[0m"
        BOLD = "\033[1m"
        BLUE = "\033[34m"
        YELLOW = "\033[33m"
        CYAN = "\033[36m"
        HIGHLIGHT = "\033[43m"  # Yellow background
    else:
        RESET = BOLD = BLUE = YELLOW = CYAN = HIGHLIGHT = ""
    
    # Display header
    print(f"\n{BOLD}{BLUE}====== Search Results ======{RESET}\n")
    print(f"Database: {BOLD}{db_type.capitalize()}{RESET}")
    print(f"Query: {BOLD}'{results['query']}'{RESET}")
    print(f"Search type: {results['search_type']}")
    print(f"Using embedders: {results['embedder_info']['dense']} (dense), "
          f"{results['embedder_info']['sparse']} (sparse)")
    print(f"Found {BOLD}{results['count']}{RESET} results\n")
    
    if results['count'] == 0:
        print(f"{YELLOW}No results found for your query.{RESET}")
        return
    
    # Display results
    for result in results['results']:
        print(f"{CYAN}{'=' * 60}{RESET}")
        print(f"{BOLD}Rank: {result['rank']}, Score: {YELLOW}{result['score']:.4f}{RESET}")
        print(f"File: {result['file_name']}")
        print(f"Path: {result['file_path']}")
        print(f"Chunk: {result['chunk_index']}")
        
        # Chunk size information
        print(f"Chunk size: {result['chunk_size']['characters']} chars, "
              f"{result['chunk_size']['words']} words, "
              f"{result['chunk_size']['lines']} lines")
        
        print(f"Embedders: {result['embedder_info']['dense_embedder']} (dense), "
              f"{result['embedder_info']['sparse_embedder']} (sparse)")
        
        # Preview with highlighted terms
        print(f"\n{BOLD}Preview:{RESET}")
        
        # Convert markdown ** to terminal formatting
        preview = result['preview']
        if color_output:
            preview = preview.replace('**', HIGHLIGHT)
            # Fix uneven highlights by adding RESET
            if preview.count(HIGHLIGHT) % 2 != 0:
                preview += RESET
            else:
                preview = preview.replace(HIGHLIGHT + HIGHLIGHT, HIGHLIGHT)
                
            # Make sure highlighting is reset at the end
            preview += RESET
        
        print(preview)
        print()
    
    # Display performance metrics if available
    try:
        hit_rates = db_manager.get_hit_rates()
        if hit_rates:
            print(f"\n{BOLD}Search Performance Metrics:{RESET}")
            for search_type, hit_rate in hit_rates.items():
                print(f"{search_type}: {hit_rate*100:.1f}% hit rate")
    except:
        # If get_hit_rates doesn't exist or fails, just skip it
        pass


def run_indexing(args):
    """Main function to run the indexing process with modular database support"""

    # First check if the requested database backend is available
    db_type = args.db_type
    
    # Import the factory and interface
    from vector_db_interface import DBFactory

    # For Qdrant, use the specific availability checker
    if db_type.lower() == 'qdrant':
        if not ModelUtils.check_qdrant_available(args.host, args.port, args.verbose):
            print("ERROR: Qdrant not available in either server or local mode")
            return
        
    start_time = time.time()
    
    # 1. Download model if needed (only needed for fallback model)
    if args.verbose:
        print("\n====== STEP 1: Downloading Model (if needed) ======")
        print(f"args.use_mlx_models: {args.use_mlx_models}, HAS_MLX_EMBEDDING_MODELS: {HAS_MLX_EMBEDDING_MODELS}.")
        
    # Only download legacy model files if not using MLX embedding models
    if not (args.use_mlx_models and HAS_MLX_EMBEDDING_MODELS):
        ModelUtils.download_model_files(args.model, args.weights, args.verbose)
    
    # 2. Initialize document processor
    if args.verbose:
        print("\n====== STEP 2: Initializing Document Processor ======")
    
    # Base processor arguments
    processor_args = {
        "model_name": args.model,
        "weights_path": args.weights,
        "verbose": args.verbose
    }
    
    # Store the actual model ID for database (prioritize custom_repo_id if available)
    effective_dense_model = args.custom_repo_id if args.custom_repo_id else args.dense_model
    
    # Check for FastEmbed usage first (highest priority)
    if args.use_fastembed:
        if HAS_FASTEMBED:
            processor_args.update({
                "use_fastembed": True,
                "fastembed_model": args.fastembed_model,
                "fastembed_sparse_model": args.fastembed_sparse_model,
                "fastembed_use_gpu": args.fastembed_use_gpu,
                "fastembed_cache_dir": args.fastembed_cache_dir
            })
            if args.verbose:
                print(f"Using FastEmbed with model: {args.fastembed_model}")
        else:
            print("Warning: FastEmbed not available. Install with: pip install fastembed")
            print("Falling back to Ollama or MLX")
    
    # Check for Ollama usage second
    elif args.use_ollama:
        if HAS_OLLAMA:
            processor_args.update({
                "use_ollama": True,
                "ollama_model": args.ollama_model,
                "ollama_host": args.ollama_host
            })
            if args.verbose:
                print(f"Using Ollama with model: {args.ollama_model}")
        else:
            print("Warning: Ollama not available. Falling back to MLX or default models.")
    
    # Use MLX embedding models if available and neither FastEmbed nor Ollama used
    if not processor_args.get("use_ollama", False) and args.use_mlx_models and HAS_MLX_EMBEDDING_MODELS:
        if args.verbose:
            if args.custom_repo_id:
                print(f"Using MLX embedding models with custom repository: {args.custom_repo_id}")
            else:
                print(f"Using MLX with dense model: {args.dense_model}")
        
        processor_args.update({
            "use_mlx_embedding": True,
            "dense_model": args.dense_model,
            "sparse_model": args.sparse_model,
            "top_k": args.top_k,
            "custom_repo_id": args.custom_repo_id,
            "custom_ndim": args.custom_ndim,
            "custom_pooling": args.custom_pooling,
            "custom_normalize": args.custom_normalize,
            "custom_max_length": args.custom_max_length
        })
    
    processor = TextProcessor(**processor_args)
    
    # 3. Initialize database manager using the factory
    if args.verbose:
        print(f"\n====== STEP 3: Setting up {db_type.capitalize()} Database ======")

    try:
        # Collect common parameters for all database types
        db_args = {
            "collection_name": args.collection,
            "vector_size": processor.vector_size,
            "storage_path": args.storage_path,
            "verbose": args.verbose,
            "dense_model_id": processor.dense_model_id,  # Use the processor's dense_model_id which handles custom_repo_id
            "sparse_model_id": processor.sparse_model_id
        }

        if args.verbose:
            print(f"Database vector parameters:")
            print(f"  - Vector size: {processor.vector_size}")
            print(f"  - Dense model: {processor.dense_model_id}")
            print(f"  - Sparse model: {processor.sparse_model_id}")
        
        # Add database-specific parameters based on type
        if db_type.lower() == 'qdrant':
            db_args.update({
                "host": args.host,
                "port": args.port,
            })
        elif db_type.lower() == 'lancedb':
            db_args.update({
                "uri": args.lancedb_uri,
            })
        elif db_type.lower() == 'meilisearch':
            db_args.update({
                "url": args.meilisearch_url,
                "api_key": args.meilisearch_api_key,
            })
        elif db_type.lower() == 'chromadb':
            db_args.update({
                "host": args.chromadb_host,
                "port": args.chromadb_port,
                "use_embedding_function": args.chromadb_use_embedding_function,
                "embedding_function_model": args.chromadb_embedding_model
            })
        elif db_type.lower() == 'elasticsearch':
            db_args.update({
                "hosts": args.es_hosts,
                "api_key": args.es_api_key,
                "username": args.es_username,
                "password": args.es_password,
            })
        elif db_type.lower() == 'milvus':
            db_args.update({
                "host": args.milvus_host,
                "port": args.milvus_port,
                "user": args.milvus_user,
                "password": args.milvus_password,
                "secure": args.milvus_secure,
                "token": args.milvus_token,
            })
            
        # Create the database handler using the factory
        if args.verbose:
            print("Creating database handler...")

        db_manager = DBFactory.create_db(db_type, **db_args)
        
        # Create collection
        if args.verbose:
            print("Creating collection...")

        db_manager.create_collection(recreate=args.recreate)
    except Exception as e:
        print(f"Error setting up {db_type} database: {e}")
        return
    
    # 4. Get files to process
    if args.verbose:
        print("\n====== STEP 4: Finding Files to Process ======")
    include_patterns = args.include.split() if args.include else None
    files = FileUtils.get_files_to_process(
        args.directory,
        include_patterns=include_patterns,
        limit=args.limit,
        verbose=args.verbose
    )
    
    if not files:
        print("No files found to process")
        return
    
    # 5. Process files and index them
    if args.verbose:
        print(f"\n====== STEP 5: Processing and Indexing {len(files)} Files ======")
    
    # Safely get collection info, handling errors gracefully
    try:
        collection_info_before = db_manager.get_collection_info()
        if isinstance(collection_info_before, dict):
            points_before = collection_info_before.get("points_count", 0)
            if isinstance(points_before, dict) and "error" in points_before:
                points_before = 0
        else:
            points_before = 0
    except Exception as e:
        print(f"Warning: Unable to get initial collection info: {e}")
        points_before = 0
    
    total_chunks = 0
    total_files_processed = 0
    successful_files = 0
    
    # Use tqdm for progress bar 
    if HAS_TQDM:
        file_iterator = tqdm(files, desc="Processing files")
    else:
        file_iterator = files
    
    # Determine whether to use regular or sparse processing
    use_sparse = (
        args.use_mlx_models and hasattr(processor, 'mlx_embedding_provider')
    ) or (
        args.use_fastembed and hasattr(processor, 'fastembed_provider')
    ) or (
        args.use_ollama and hasattr(processor, 'ollama_embedding_provider')
    )
    
    for file_path in file_iterator:
        try:
            # Show progress details when verbose
            if args.verbose and not HAS_TQDM:
                print(f"\nProcessing file {total_files_processed + 1}/{len(files)}: {file_path}")
                
                # Show system stats periodically
                if total_files_processed % 10 == 0 and HAS_PSUTIL:
                    mem = psutil.virtual_memory()
                    cpu = psutil.cpu_percent()
                    print(f"System stats - CPU: {cpu}%, RAM: {mem.percent}% ({GeneralUtils.get_size_str(mem.used)}/{GeneralUtils.get_size_str(mem.total)})")
            
            # Process file with appropriate method (with or without sparse)
            if use_sparse:
                # Process with sparse embeddings
                try:
                    results = processor.process_file_with_sparse(file_path)
                    
                    if results:
                        # Insert embeddings with sparse vectors
                        db_manager.insert_embeddings_with_sparse(results)
                        total_chunks += len(results)
                        successful_files += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            else:
                # Process with dense embeddings only
                try:
                    results = processor.process_file(file_path)
                    
                    if results:
                        # Insert embeddings
                        db_manager.insert_embeddings(results)
                        total_chunks += len(results)
                        successful_files += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            total_files_processed += 1
            
            # Show progress update
            if args.verbose and not HAS_TQDM:
                print(f"Completed {total_files_processed}/{len(files)} files, {total_chunks} chunks indexed")
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 6. Show final statistics
    try:
        collection_info_after = db_manager.get_collection_info()
        if isinstance(collection_info_after, dict):
            points_after = collection_info_after.get("points_count", 0)
            if isinstance(points_after, dict) and "error" in points_after:
                points_after = total_chunks
        else:
            points_after = total_chunks
        points_added = points_after - points_before
    except Exception as e:
        print(f"Warning: Unable to get final collection info: {e}")
        points_after = total_chunks
        points_added = total_chunks
    
    print("\n====== Indexing Complete ======")
    print(f"Database type: {db_type.capitalize()}")
    print(f"Processed {total_files_processed} files")
    print(f"Successfully indexed {successful_files} files")
    print(f"Added {points_added} chunks to database")
    print(f"Total chunks in collection: {points_after}")
    
    # Safely display collection info
    try:
        if isinstance(collection_info_after, dict):
            if "disk_usage" in collection_info_after and collection_info_after["disk_usage"]:
                print(f"Collection size on disk: {GeneralUtils.get_size_str(collection_info_after['disk_usage'])}")
            
            # Show vector configurations if available
            if "vector_configs" in collection_info_after and collection_info_after["vector_configs"]:
                print("\nVector configurations:")
                for name, config in collection_info_after["vector_configs"].items():
                    print(f"  - {name}: {config}")
                    
            if "sparse_vector_configs" in collection_info_after and collection_info_after["sparse_vector_configs"]:
                print("\nSparse vector configurations:")
                for name, config in collection_info_after["sparse_vector_configs"].items():
                    print(f"  - {name}: {config}")
    except Exception as e:
        print(f"Warning: Error displaying collection info details: {e}")
    
    print(f"Total time: {time.time() - start_time:.2f} seconds")


def main():
    """Main function with support for different vector database backends"""
    import argparse
    
    # Create the main parser with consistent argument style
    parser = argparse.ArgumentParser(description="CrispRAG: Vector Database Indexer & Search with Multiple Backend Support")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--storage-path", help="Path to store database data")
    
    # Add database selection option
    parser.add_argument("--db-type", choices=["qdrant", "lancedb", "meilisearch", "elasticsearch", "chromadb", "milvus"], 
                    default="qdrant", help="Vector database backend to use")

    # Create embedding provider group with mutually exclusive options
    embedding_group = parser.add_argument_group("Embedding options")
    embedding_provider = embedding_group.add_mutually_exclusive_group()
    embedding_provider.add_argument("--use-mlx-models", dest="use_mlx_models", action="store_true", 
                                   help="Use mlx_embedding_models for embeddings")
    embedding_provider.add_argument("--use-ollama", dest="use_ollama", action="store_true", 
                                   help="Use Ollama for embeddings")
    embedding_provider.add_argument("--use-fastembed", dest="use_fastembed", action="store_true", 
                                   help="Use FastEmbed for embeddings")
    
    # MLX-specific arguments
    mlx_group = parser.add_argument_group("MLX embedding options")
    mlx_group.add_argument("--dense-model", dest="dense_model", type=str, default="bge-small", 
                          help="MLX dense embedding model name")
    mlx_group.add_argument("--sparse-model", dest="sparse_model", type=str, default="distilbert-splade", 
                          help="MLX sparse embedding model name")
    mlx_group.add_argument("--top-k", dest="top_k", type=int, default=64, 
                          help="Top-k tokens to keep in sparse vectors")
    mlx_group.add_argument("--custom-repo-id", dest="custom_repo_id", type=str, 
                          help="Custom model HuggingFace repo ID")
    mlx_group.add_argument("--custom-ndim", dest="custom_ndim", type=int, 
                          help="Custom model embedding dimension")
    mlx_group.add_argument("--custom-pooling", dest="custom_pooling", type=str, 
                          choices=["mean", "first", "max"], default="mean", 
                          help="Custom model pooling strategy")
    mlx_group.add_argument("--custom-normalize", dest="custom_normalize", action="store_true", default=True, 
                          help="Normalize embeddings")
    mlx_group.add_argument("--custom-max-length", dest="custom_max_length", type=int, default=512, 
                          help="Custom model max sequence length")
    
    # Ollama-specific arguments
    ollama_group = parser.add_argument_group("Ollama embedding options")
    ollama_group.add_argument("--ollama-model", dest="ollama_model", type=str, default=DEFAULT_OLLAMA_EMBED_MODEL, 
                             help="Ollama model name for embeddings")
    ollama_group.add_argument("--ollama-host", dest="ollama_host", type=str, default="http://localhost:11434", 
                             help="Ollama API host URL")
    
    # FastEmbed-specific arguments
    fastembed_group = parser.add_argument_group("FastEmbed options")
    fastembed_group.add_argument("--fastembed-model", dest="fastembed_model", type=str, default=DEFAULT_FASTEMBED_MODEL,
                                help="FastEmbed model name for dense embeddings")
    fastembed_group.add_argument("--fastembed-sparse-model", dest="fastembed_sparse_model", type=str, 
                                default=DEFAULT_FASTEMBED_SPARSE_MODEL,
                                help="FastEmbed model name for sparse embeddings")
    fastembed_group.add_argument("--fastembed-use-gpu", dest="fastembed_use_gpu", action="store_true",
                                help="Use GPU with FastEmbed (requires fastembed-gpu package)")
    fastembed_group.add_argument("--fastembed-cache-dir", dest="fastembed_cache_dir", type=str,
                                help="Directory to cache FastEmbed models")
    
    # Add subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument("directory", help="Directory containing documents to index")
    index_parser.add_argument("--include", type=str, help="File patterns to include (space separated, e.g. '*.txt *.pdf')")
    index_parser.add_argument("--limit", type=int, help="Maximum number of files to index")
    index_parser.add_argument("--collection", type=str, default="documents", help="Collection name")
    index_parser.add_argument("--model", type=str, default="cstr/paraphrase-multilingual-MiniLM-L12-v2-mlx", 
                             help="Hugging Face model name (fallback)")
    index_parser.add_argument("--weights", type=str, default="weights/paraphrase-multilingual-MiniLM-L12-v2.npz", 
                             help="Path to store/load MLX weights (fallback)")
    index_parser.add_argument("--recreate", action="store_true", help="Recreate collection if it exists")
    
    # Add database-specific parameters for Qdrant
    qdrant_group = index_parser.add_argument_group("Qdrant options")
    qdrant_group.add_argument("--host", type=str, default="localhost", help="Qdrant host")
    qdrant_group.add_argument("--port", type=int, default=6333, help="Qdrant port")
    
    # Add database-specific parameters for LanceDB
    lancedb_group = index_parser.add_argument_group("LanceDB options")
    lancedb_group.add_argument("--lancedb-uri", dest="lancedb_uri", type=str, 
                              help="LanceDB URI (if not using local storage)")
    
    # Add database-specific parameters for Meilisearch
    meilisearch_group = index_parser.add_argument_group("Meilisearch options")
    meilisearch_group.add_argument("--meilisearch-url", dest="meilisearch_url", type=str, 
                                  default="http://localhost:7700", help="Meilisearch URL")
    meilisearch_group.add_argument("--meilisearch-api-key", dest="meilisearch_api_key", type=str, 
                                  help="Meilisearch API key")
    
    # Add database-specific parameters for Elasticsearch
    es_group = index_parser.add_argument_group("Elasticsearch options")
    es_group.add_argument("--es-hosts", dest="es_hosts", type=str, nargs="+", 
                         default=["http://localhost:9200"], help="Elasticsearch hosts")
    es_group.add_argument("--es-api-key", dest="es_api_key", type=str, help="Elasticsearch API key")
    es_group.add_argument("--es-username", dest="es_username", type=str, help="Elasticsearch username")
    es_group.add_argument("--es-password", dest="es_password", type=str, help="Elasticsearch password")

    # Add database-specific parameters for Milvus for the index_parser
    milvus_group = index_parser.add_argument_group("Milvus options")
    milvus_group.add_argument("--milvus-host", dest="milvus_host", type=str, 
                            default="localhost", help="Milvus host")
    milvus_group.add_argument("--milvus-port", dest="milvus_port", type=str, 
                            default="19530", help="Milvus port")
    milvus_group.add_argument("--milvus-user", dest="milvus_user", type=str, 
                            default="", help="Milvus username for authentication")
    milvus_group.add_argument("--milvus-password", dest="milvus_password", type=str, 
                            default="", help="Milvus password for authentication")
    milvus_group.add_argument("--milvus-secure", dest="milvus_secure", action="store_true",
                            help="Use secure connection to Milvus")
    milvus_group.add_argument("--milvus-token", dest="milvus_token", type=str,
                            help="Milvus auth token (alternative to username/password)")

    # Add database-specific parameters for ChromaDB in the index_parser
    chromadb_group = index_parser.add_argument_group("ChromaDB options")
    chromadb_group.add_argument("--chromadb-host", dest="chromadb_host", type=str, 
                            help="ChromaDB host (for remote API)")
    chromadb_group.add_argument("--chromadb-port", dest="chromadb_port", type=int, 
                            help="ChromaDB port (for remote API)")
    chromadb_group.add_argument("--chromadb-use-embedding-function", dest="chromadb_use_embedding_function", 
                           action="store_true", help="Use ChromaDB's built-in embedding functions")
    chromadb_group.add_argument("--chromadb-embedding-model", dest="chromadb_embedding_model", type=str,
                            default="sentence-transformers/all-MiniLM-L6-v2",
                            help="Model to use with ChromaDB's embedding function")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--search-type", dest="search_type", choices=["keyword", "vector", "sparse", "hybrid"], 
                              default="hybrid", help="Type of search to perform")
    search_parser.add_argument("--limit", type=int, default=10, help="Maximum number of results to return")
    search_parser.add_argument("--collection", type=str, default="documents", help="Collection name")
    search_parser.add_argument("--model", type=str, default="cstr/paraphrase-multilingual-MiniLM-L12-v2-mlx", 
                              help="Hugging Face model name (fallback)")
    search_parser.add_argument("--weights", type=str, default="weights/paraphrase-multilingual-MiniLM-L12-v2.npz", 
                              help="Path to MLX weights (fallback)")
    search_parser.add_argument("--prefetch-limit", dest="prefetch_limit", type=int, default=50, 
                              help="Prefetch limit for hybrid search")
    search_parser.add_argument("--fusion", choices=["rrf", "dbsf", "linear"], default="rrf", 
                              help="Fusion strategy for hybrid search")
    search_parser.add_argument("--relevance-tuning", dest="relevance_tuning", action="store_true", default=True, 
                              help="Apply relevance tuning to hybrid search")
    search_parser.add_argument("--context-size", dest="context_size", type=int, default=300, 
                              help="Size of context window for preview text")
    search_parser.add_argument("--score-threshold", dest="score_threshold", type=float,
                              help="Minimum score threshold for results (0.0-1.0)")
    search_parser.add_argument("--debug", action="store_true", 
                              help="Show detailed debug information")
    search_parser.add_argument("--no-color", dest="no_color", action="store_true", 
                              help="Disable colored output")
    search_parser.add_argument("--rerank", action="store_true",
                              help="Apply reranking to improve result quality")
    search_parser.add_argument("--reranker-type", dest="reranker_type", 
                              choices=["cross", "colbert", "cohere", "jina", "rrf", "linear"], 
                              help="Type of reranker to use")

    # Don't duplicate these args since they're already in the parent parser
    # Just ensure the search_parser gets a reference to them when needed

    # Add database-specific parameters for search with Qdrant
    qdrant_search_group = search_parser.add_argument_group("Qdrant options")
    qdrant_search_group.add_argument("--host", type=str, default="localhost", help="Qdrant host")
    qdrant_search_group.add_argument("--port", type=int, default=6333, help="Qdrant port")
    
    # Add database-specific parameters for search with LanceDB
    lancedb_search_group = search_parser.add_argument_group("LanceDB options")
    lancedb_search_group.add_argument("--lancedb-uri", dest="lancedb_uri", type=str, 
                                     help="LanceDB URI (if not using local storage)")
    
    # Add database-specific parameters for search with Meilisearch
    meilisearch_search_group = search_parser.add_argument_group("Meilisearch options")
    meilisearch_search_group.add_argument("--meilisearch-url", dest="meilisearch_url", type=str, 
                                         default="http://localhost:7700", help="Meilisearch URL")
    meilisearch_search_group.add_argument("--meilisearch-api-key", dest="meilisearch_api_key", type=str, 
                                         help="Meilisearch API key")
    
    # Add database-specific parameters for search with Elasticsearch
    es_search_group = search_parser.add_argument_group("Elasticsearch options")
    es_search_group.add_argument("--es-hosts", dest="es_hosts", type=str, nargs="+", 
                                default=["http://localhost:9200"], help="Elasticsearch hosts")
    es_search_group.add_argument("--es-api-key", dest="es_api_key", type=str, help="Elasticsearch API key")
    es_search_group.add_argument("--es-username", dest="es_username", type=str, help="Elasticsearch username")
    es_search_group.add_argument("--es-password", dest="es_password", type=str, help="Elasticsearch password")

    # Add database-specific parameters for search with Milvus
    milvus_search_group = search_parser.add_argument_group("Milvus options")
    milvus_search_group.add_argument("--milvus-host", dest="milvus_host", type=str, 
                                default="localhost", help="Milvus host")
    milvus_search_group.add_argument("--milvus-port", dest="milvus_port", type=str, 
                                default="19530", help="Milvus port")
    milvus_search_group.add_argument("--milvus-user", dest="milvus_user", type=str, 
                                default="", help="Milvus username for authentication")
    milvus_search_group.add_argument("--milvus-password", dest="milvus_password", type=str, 
                                default="", help="Milvus password for authentication")
    milvus_search_group.add_argument("--milvus-secure", dest="milvus_secure", action="store_true",
                                help="Use secure connection to Milvus")
    milvus_search_group.add_argument("--milvus-token", dest="milvus_token", type=str,
                                help="Milvus auth token (alternative to username/password)")

    # add ChromaDB options to the search_parser
    chromadb_search_group = search_parser.add_argument_group("ChromaDB options")
    chromadb_search_group.add_argument("--chromadb-host", dest="chromadb_host", type=str, 
                                    help="ChromaDB host (for remote API)")
    chromadb_search_group.add_argument("--chromadb-port", dest="chromadb_port", type=int, 
                                    help="ChromaDB port (for remote API)")
    chromadb_search_group.add_argument("--chromadb-use-embedding-function", dest="chromadb_use_embedding_function", 
                                  action="store_true", help="Use ChromaDB's built-in embedding functions")
    chromadb_search_group.add_argument("--chromadb-embedding-model", dest="chromadb_embedding_model", type=str,
                                    default="sentence-transformers/all-MiniLM-L6-v2",
                                    help="Model to use with ChromaDB's embedding function")

    
    # List models command
    list_parser = subparsers.add_parser("list-models", help="List available MLX embedding models")

    # List databases command - new command to list available database backends
    list_db_parser = subparsers.add_parser("list-dbs", help="List available vector database backends")

    args = parser.parse_args()

    if args.verbose:
        print(f"Command: {args.command}")
        print(f"Arguments: {args}")
        print("\n====== Checking MLX Embedding Models Compatibility ======")
        GeneralUtils.check_mlx_embedding_compatibility()
        print("========================================================\n")

    if args.command == "index":
        is_available, missing_deps = GeneralUtils.check_db_dependencies(args.db_type)
        if not is_available:
            print(f"Error: Missing required dependencies for {args.db_type}: {', '.join(missing_deps)}")
            print(f"Please install the required packages with: pip install {' '.join(missing_deps)}")
            return
            
        if args.use_mlx_models and not HAS_MLX_EMBEDDING_MODELS:
            print("Warning: mlx_embedding_models not available. Install with: pip install mlx-embedding-models")
            print("Falling back to traditional model")
        
        run_indexing(args)
    elif args.command == "search":
        if not args.query:
            print("Error: Search query cannot be empty")
            return
        
        # Check if required dependencies are installed for the selected database
        is_available, missing_deps = GeneralUtils.check_db_dependencies(args.db_type)
        if not is_available:
            print(f"Error: Missing required dependencies for {args.db_type}: {', '.join(missing_deps)}")
            print(f"Please install the required packages with: pip install {' '.join(missing_deps)}")
            return
        
        # Check database-specific dependencies
        if args.db_type == "qdrant":
            try:
                import qdrant_client
            except ImportError:
                print("Error: qdrant-client not installed. Please install with: pip install qdrant-client")
                return
        elif args.db_type == "lancedb":
            try:
                import lancedb
                import pyarrow
            except ImportError:
                print("Error: lancedb and/or pyarrow not installed. Please install with: pip install lancedb pyarrow")
                return
        elif args.db_type == "meilisearch":
            try:
                import meilisearch
            except ImportError:
                print("Error: meilisearch not installed. Please install with: pip install meilisearch")
                return
        elif args.db_type == "elasticsearch":
            try:
                import elasticsearch
            except ImportError:
                print("Error: elasticsearch not installed. Please install with: pip install elasticsearch")
                return
        elif args.db_type == "milvus":
            try:
                import pymilvus
            except ImportError:
                print("Error: pymilvus not installed. Please install with: pip install pymilvus")
                # also "pymilvus[model]"
                return
            
        run_search(args)
    
    elif args.command == "list-models":
        models_available = False
        
        # List FastEmbed models if available
        if HAS_FASTEMBED:
            models_available = True
            print("\n====== Available FastEmbed Models ======")
            
            print("\nDense Models:")
            for i, (model_name, info) in enumerate(FASTEMBED_MODELS_REGISTRY.items()):
                if info.get("ndim", 0) > 0:  # Skip sparse models
                    dim = info.get("ndim", "unknown")
                    normalize = "Yes" if info.get("normalize", False) else "No"
                    description = info.get("description", "")
                    print(f"{i+1:2d}. {model_name:40s} - Dim: {dim:4d}, Normalize: {normalize}")
                    if description:
                        print(f"    {description}")
                
            print("\nSparse Models:")
            for i, (model_name, info) in enumerate(FASTEMBED_MODELS_REGISTRY.items()):
                if info.get("ndim", 1) == 0:  # Only sparse models
                    description = info.get("description", "")
                    print(f"{i+1:2d}. {model_name:40s}")
                    if description:
                        print(f"    {description}")
                
            print("\nNote: Install fastembed with: pip install fastembed")
            print("For GPU support: pip install fastembed-gpu")
        
        # List Ollama models if available
        if HAS_OLLAMA:
            models_available = True
            print("\n====== Available Ollama Embedding Models ======")
            
            print("\nPre-registered Models:")
            for i, (model_name, info) in enumerate(OLLAMA_MODELS_REGISTRY.items()):
                dim = info.get("ndim", "unknown")
                normalize = "Yes" if info.get("normalize", False) else "No"
                max_length = info.get("max_length", "unknown")
                print(f"{i+1:2d}. {model_name:30s} - Dim: {dim:4d}, Normalize: {normalize}, Max Length: {max_length}")
        
        # List MLX models if available
        if HAS_MLX_EMBEDDING_MODELS:
            models_available = True
            print("\n====== Available MLX Embedding Models ======")
            
            print("\nDense Models:")
            dense_models = [k for k, v in MLX_EMBEDDING_REGISTRY.items() if not v.get("lm_head")]
            for i, model in enumerate(dense_models):
                dim = MLX_EMBEDDING_REGISTRY[model].get("ndim", "unknown")
                repo = MLX_EMBEDDING_REGISTRY[model].get("repo", "unknown")
                print(f"{i+1:2d}. {model:30s} - Dim: {dim:4d}, Repo: {repo}")
                
            print("\nSparse Models (SPLADE):")
            sparse_models = [k for k, v in MLX_EMBEDDING_REGISTRY.items() if v.get("lm_head")]
            for i, model in enumerate(sparse_models):
                dim = MLX_EMBEDDING_REGISTRY[model].get("ndim", "unknown")
                repo = MLX_EMBEDDING_REGISTRY[model].get("repo", "unknown")
                print(f"{i+1:2d}. {model:30s} - Dim: {dim:4d}, Repo: {repo}")
                
            print("\nNote: You can also use custom models with --custom-repo-id parameter")
            
        if not models_available:
            print("Error: No embedding models available. Please install at least one of:")
            print("- FastEmbed: pip install fastembed")
            print("- Ollama: Download from https://ollama.ai")
            print("- MLX Embedding Models: pip install mlx-embedding-models")
            return
            
    elif args.command == "list-dbs":
        # List available database backends
        try:
            from vector_db_interface import DBFactory
            
            print("\n====== Available Vector Database Backends ======")
            # Get available backends from DBFactory
            available_dbs = []
            try:
                # Try to import all available backends
                from . import AVAILABLE_DBS
                available_dbs = list(AVAILABLE_DBS.keys())
            except (ImportError, AttributeError):
                # Fallback to hard-coded list
                available_dbs = ["qdrant", "lancedb", "meilisearch", "elasticsearch"]
                
            for i, db_name in enumerate(available_dbs):
                print(f"{i+1}. {db_name}")
                
            # Check availability of each database
            print("\nDatabase Availability:")
            for db_name in available_dbs:
                is_available, missing = GeneralUtils.check_db_dependencies(db_name)
                status = "✅ Available" if is_available else f"❌ Missing dependencies: {', '.join(missing)}"
                print(f"{db_name}: {status}")
                
            print("\nTo use a specific database backend, use --db-type [name] when running index or search commands")
        except ImportError:
            print("Error: Could not import vector_db module")
    elif args.command is None:
        parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        if "--verbose" in sys.argv:
            import traceback
            traceback.print_exc()