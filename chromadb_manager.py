import os
import time
import json
import shutil
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set, Iterator, Union, Literal
import numpy as np

from vector_db_interface import VectorDBInterface
from utils import TextProcessor, ResultProcessor, SearchAlgorithms, EmbeddingUtils, GeneralUtils

# Try to import Chroma client
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    chromadb_available = True
except ImportError:
    chromadb_available = False
    print("Warning: ChromaDB client not available. Install with: pip install chromadb")


class ChromaDBManager(VectorDBInterface):
    """Manager for ChromaDB vector database operations with semantic search capabilities"""
    
    def __init__(self, 
                host: str = None, 
                port: int = None, 
                collection_name: str = "documents",
                vector_size: int = 384,
                storage_path: str = None,
                verbose: bool = False,
                dense_model_id: str = "bge-small",
                sparse_model_id: str = "distilbert-splade"):
        """
        Initialize ChromaDBManager with model-specific vector configuration.
        
        Args:
            host: ChromaDB host (for remote API)
            port: ChromaDB port (for remote API)
            collection_name: Collection name
            vector_size: Vector dimension
            storage_path: Storage path for local mode
            verbose: Verbose output
            dense_model_id: ID of dense model for naming vectors
            sparse_model_id: ID of sparse model for naming vectors
        """
        if not chromadb_available:
            raise ImportError("ChromaDB client not available. Install with: pip install chromadb")
            
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_dim = vector_size
        self.storage_path = storage_path
        self.verbose = verbose
        self.client = None
        self.collection = None
        self.is_remote = host is not None and port is not None
        
        # Store the original model IDs
        self.dense_model_id = dense_model_id
        self.sparse_model_id = sparse_model_id
        
        # Performance tracking
        self._hit_rates = {}
        
        self.connect()

    def connect(self) -> None:
        """Connect to ChromaDB using either remote or local mode"""
        try:
            if self.is_remote:
                # Connect to remote ChromaDB server
                if self.verbose:
                    print(f"Connecting to ChromaDB server at {self.host}:{self.port}")
                    
                self.client = chromadb.HttpClient(host=self.host, port=self.port)
                
                if self.verbose:
                    print(f"Connected to remote ChromaDB server")
            else:
                # Use local mode with storage path
                if not self.storage_path:
                    # Create a persistent directory in the current working directory
                    self.storage_path = os.path.join(os.getcwd(), "chromadb_storage")
                
                # Make sure storage path exists
                os.makedirs(self.storage_path, exist_ok=True)
                
                if self.verbose:
                    print(f"Using local ChromaDB storage at: {self.storage_path}")
                
                # Initialize local client
                self.client = chromadb.PersistentClient(path=self.storage_path)
                
                if self.verbose:
                    print(f"Connected to local ChromaDB storage")
            
            # Check if collection exists and get or create it
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                if self.verbose:
                    print(f"Found existing collection '{self.collection_name}'")
            except Exception as e:
                if self.verbose:
                    print(f"Collection '{self.collection_name}' does not exist yet")
                self.collection = None
                
        except Exception as e:
            print(f"Error connecting to ChromaDB: {str(e)}")
            raise

    def create_collection(self, recreate: bool = False) -> None:
        """Create ChromaDB collection with appropriate settings"""
        try:
            # Check if collection exists
            try:
                existing_collection = self.client.get_collection(name=self.collection_name)
                exists = True
                if self.verbose:
                    print(f"Collection '{self.collection_name}' exists")
            except Exception:
                exists = False
                if self.verbose:
                    print(f"Collection '{self.collection_name}' does not exist")
            
            # Delete collection if it exists and recreate=True
            if exists and recreate:
                if self.verbose:
                    print(f"Deleting existing collection '{self.collection_name}'")
                self.client.delete_collection(name=self.collection_name)
                exists = False
            
            # Create collection if it doesn't exist
            if not exists:
                if self.verbose:
                    print(f"Creating collection '{self.collection_name}'")
                
                # Create the metadata to store model information
                metadata = {
                    "vector_size": self.vector_dim,
                    "dense_model_id": self.dense_model_id,
                    "sparse_model_id": self.sparse_model_id
                }
                
                # Create the collection (ChromaDB handles vector dimensions automatically)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata=metadata
                )
                
                if self.verbose:
                    print(f"Collection '{self.collection_name}' created successfully")
            else:
                # Get existing collection
                self.collection = self.client.get_collection(name=self.collection_name)
                
                if self.verbose:
                    print(f"Using existing collection '{self.collection_name}'")
                    
        except Exception as e:
            print(f"Error creating collection: {str(e)}")
            raise

    def search_keyword(self, query: str, limit: int = 10, score_threshold: float = None,
                      rerank: bool = False, reranker_type: str = None):
        """
        Perform keyword-based search using ChromaDB's built-in functionality.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            score_threshold: Minimum score threshold
            rerank: Whether to apply reranking
            reranker_type: Type of reranker to use
            
        Returns:
            List of search results
        """
        if not query.strip():
            return {"error": "Empty query"}
            
        try:
            if self.verbose:
                print(f"Executing keyword search for query: '{query}'")
                
            # Search using query text
            results = self.collection.query(
                query_texts=query,
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            points = []
            
            # Check if we have any results
            if not results['ids'][0]:
                if self.verbose:
                    print("No results found for keyword search")
                return []
            
            # Extract results
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                document = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                # Convert distance to score (ChromaDB returns L2 distance, lower is better)
                # Transform to a 0-1 score where 1 is best
                score = 1.0 / (1.0 + distance)
                
                # Apply score threshold if provided
                if score_threshold is not None and score < score_threshold:
                    continue
                
                # Build result object with both document and metadata
                result = {
                    "id": doc_id,
                    "text": document,  # Original document text
                    "score": score
                }
                
                # Add metadata fields
                for key, value in metadata.items():
                    # Skip sparse vector data in metadata for cleaner results
                    if key not in ['sparse_indices', 'sparse_values']:
                        result[key] = value
                
                # Add payload for compatibility
                result["payload"] = {
                    "id": doc_id,
                    "text": document,
                    "score": score,
                    **{k: v for k, v in metadata.items() if k not in ['sparse_indices', 'sparse_values']}
                }
                
                points.append(result)
            
            if self.verbose:
                print(f"Keyword search returned {len(points)} results")
            
            # Apply reranking if requested
            if rerank and points and processor:
                if self.verbose:
                    print(f"Applying {reranker_type or 'default'} reranking")
                
                points = SearchAlgorithms.rerank_results(query, points, processor, limit, self.verbose)
            
            # Record metrics if ground truth is available
            true_context = getattr(processor, 'expected_context', None) if rerank and processor else None
            if true_context:
                hit = any(hasattr(p, "payload") and p.payload and p.payload.get("text") == true_context for p in points)
                self._record_hit("keyword", hit)
            
            return points[:limit]
            
        except Exception as e:
            print(f"Error in keyword search: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": f"Error in keyword search: {str(e)}"}

    def search_hybrid(self, query: str, processor: Any, limit: int, 
                    prefetch_limit: int = 50, fusion_type: str = "rrf",
                    score_threshold: float = None, rerank: bool = False,
                    reranker_type: str = None):
        """
        Perform hybrid search combining dense and sparse search results.
        
        Args:
            query: Search query string
            processor: Document processor with embedding capabilities
            limit: Maximum number of results to return
            prefetch_limit: Number of results to prefetch for fusion
            fusion_type: Type of fusion to use (rrf or dbsf)
            score_threshold: Minimum score threshold
            rerank: Whether to apply reranking
            reranker_type: Type of reranker to use
            
        Returns:
            List of search results
        """
        if processor is None:
            return {"error": "Hybrid search requires an embedding model"}
        
        try:
            # Get more results than needed for fusion
            fetch_limit = prefetch_limit
            
            # Generate query embedding
            query_vector = processor.get_embedding(query)
            
            if self.verbose:
                print(f"Executing hybrid search for query: '{query}'")
                print(f"Using fusion type: {fusion_type}")
            
            # Step 1: Get dense search results
            dense_results = self.collection.query(
                query_embeddings=query_vector.tolist(),
                n_results=fetch_limit,
                include=["documents", "metadatas", "distances"]
            )
            
            # Step 2: Get keyword search results
            keyword_results = self.collection.query(
                query_texts=query,
                n_results=fetch_limit,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process dense results
            dense_points = []
            if dense_results['ids'][0]:
                for i in range(len(dense_results['ids'][0])):
                    doc_id = dense_results['ids'][0][i]
                    document = dense_results['documents'][0][i]
                    metadata = dense_results['metadatas'][0][i]
                    distance = dense_results['distances'][0][i]
                    
                    # Convert distance to score (ChromaDB returns L2 distance, lower is better)
                    score = 1.0 / (1.0 + distance)
                    
                    # Build result object
                    result = {
                        "id": doc_id,
                        "text": document,
                        "score": score
                    }
                    
                    # Add metadata fields
                    for key, value in metadata.items():
                        if key not in ['sparse_indices', 'sparse_values']:
                            result[key] = value
                    
                    # Add payload for compatibility
                    result["payload"] = {
                        "id": doc_id,
                        "text": document,
                        "score": score,
                        **{k: v for k, v in metadata.items() if k not in ['sparse_indices', 'sparse_values']}
                    }
                    
                    dense_points.append(result)
            
            # Process keyword results
            keyword_points = []
            if keyword_results['ids'][0]:
                for i in range(len(keyword_results['ids'][0])):
                    doc_id = keyword_results['ids'][0][i]
                    document = keyword_results['documents'][0][i]
                    metadata = keyword_results['metadatas'][0][i]
                    distance = keyword_results['distances'][0][i]
                    
                    # Convert distance to score (ChromaDB returns L2 distance, lower is better)
                    score = 1.0 / (1.0 + distance)
                    
                    # Build result object
                    result = {
                        "id": doc_id,
                        "text": document,
                        "score": score
                    }
                    
                    # Add metadata fields
                    for key, value in metadata.items():
                        if key not in ['sparse_indices', 'sparse_values']:
                            result[key] = value
                    
                    # Add payload for compatibility
                    result["payload"] = {
                        "id": doc_id,
                        "text": document,
                        "score": score,
                        **{k: v for k, v in metadata.items() if k not in ['sparse_indices', 'sparse_values']}
                    }
                    
                    keyword_points.append(result)
            
            if self.verbose:
                print(f"Dense search returned {len(dense_points)} results")
                print(f"Keyword search returned {len(keyword_points)} results")
            
            # Apply fusion to combine results
            combined_points = SearchAlgorithms.manual_fusion(
                dense_points, 
                keyword_points, 
                fetch_limit, 
                fusion_type
            )
            
            if self.verbose:
                print(f"Combined hybrid search returned {len(combined_points)} results")
            
            # Apply reranking if requested
            if rerank and combined_points and processor:
                if self.verbose:
                    print(f"Applying {reranker_type or 'default'} reranking")
                
                combined_points = SearchAlgorithms.rerank_results(query, combined_points, processor, limit, self.verbose)
            
            # Record metrics if ground truth is available
            true_context = getattr(processor, 'expected_context', None)
            if true_context:
                hit = any(hasattr(p, "payload") and p.payload and p.payload.get("text") == true_context for p in combined_points)
                self._record_hit("hybrid", hit)
            
            return combined_points[:limit]
            
        except Exception as e:
            print(f"Error in hybrid search: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": f"Error in hybrid search: {str(e)}"}
    
    def search_sparse(self, query: str, processor: Any, limit: int, score_threshold: float = None,
                    rerank: bool = False, reranker_type: str = None):
        """
        Perform sparse vector search in ChromaDB.
        ChromaDB doesn't natively support sparse vectors, so we simulate it using metadata.
        
        Args:
            query: Search query string
            processor: Document processor with embedding capabilities
            limit: Maximum number of results to return
            score_threshold: Minimum score threshold
            rerank: Whether to apply reranking
            reranker_type: Type of reranker to use
            
        Returns:
            List of search results
        """
        if processor is None:
            return {"error": "Sparse search requires an embedding model"}
        
        try:
            if self.verbose:
                print(f"Executing sparse search for query: '{query}'")
            
            # Generate sparse vector for query
            sparse_indices, sparse_values = processor.get_sparse_embedding(query)
            
            if self.verbose:
                print(f"Generated sparse vector with {len(sparse_indices)} dimensions")
            
            # Since ChromaDB doesn't support sparse vectors natively,
            # we'll do a full-text search first, then re-rank based on sparse vector similarity
            
            # Get more results than needed
            fetch_limit = limit * 5
            
            # First, do a full-text search to get candidates
            results = self.collection.query(
                query_texts=query,
                n_results=fetch_limit,
                include=["documents", "metadatas", "distances"]
            )
            
            # Check if we have any results
            if not results['ids'][0]:
                if self.verbose:
                    print("No results found for full-text search")
                return []
            
            # Extract and re-score results based on sparse vector similarity
            points = []
            
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                document = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                
                # Check if document has sparse vector info in metadata
                if 'sparse_indices' in metadata and 'sparse_values' in metadata:
                    # Extract sparse vector from metadata
                    doc_sparse_indices = [int(idx) for idx in metadata['sparse_indices']]
                    doc_sparse_values = [float(val) for val in metadata['sparse_values']]
                    
                    # Calculate sparse similarity (dot product)
                    similarity = 0.0
                    query_sparse_dict = {idx: val for idx, val in zip(sparse_indices, sparse_values)}
                    for idx, val in zip(doc_sparse_indices, doc_sparse_values):
                        if idx in query_sparse_dict:
                            similarity += val * query_sparse_dict[idx]
                    
                    # Normalize similarity score to 0-1 range
                    score = min(1.0, similarity)
                else:
                    # If document doesn't have sparse vector info, assign lower score
                    score = 0.5
                
                # Apply score threshold if provided
                if score_threshold is not None and score < score_threshold:
                    continue
                
                # Build result object
                result = {
                    "id": doc_id,
                    "text": document,
                    "score": score
                }
                
                # Add metadata fields
                for key, value in metadata.items():
                    if key not in ['sparse_indices', 'sparse_values']:  # Skip sparse vector data
                        result[key] = value
                
                # Add payload for compatibility
                result["payload"] = {
                    "id": doc_id,
                    "text": document,
                    "score": score,
                    **{k: v for k, v in metadata.items() if k not in ['sparse_indices', 'sparse_values']}
                }
                
                points.append(result)
            
            # Sort by score (descending)
            points.sort(key=lambda x: x["score"], reverse=True)
            
            if self.verbose:
                print(f"Sparse search returned {len(points)} results")
            
            # Apply reranking if requested
            if rerank and points and processor:
                if self.verbose:
                    print(f"Applying {reranker_type or 'default'} reranking")
                
                points = SearchAlgorithms.rerank_results(query, points, processor, limit, self.verbose)
            
            # Record metrics if ground truth is available
            true_context = getattr(processor, 'expected_context', None)
            if true_context:
                hit = any(hasattr(p, "payload") and p.payload and p.payload.get("text") == true_context for p in points)
                self._record_hit("sparse", hit)
            
            return points[:limit]
            
        except Exception as e:
            print(f"Error in sparse search: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": f"Error in sparse search: {str(e)}"}

    def search(self, query: str, search_type: str = "hybrid", limit: int = 10,
            processor: Any = None, prefetch_limit: int = 50, fusion_type: str = "rrf",
            relevance_tuning: bool = True, context_size: int = 300, 
            score_threshold: float = None, rerank: bool = False,
            reranker_type: str = None):
        """
        Search with various options.
        
        Args:
            query: Search query string
            search_type: Type of search to perform (hybrid, vector, sparse, keyword)
            limit: Maximum number of results to return
            processor: Document processor with embedding capabilities
            prefetch_limit: Number of results to prefetch for fusion
            fusion_type: Type of fusion to use (rrf or dbsf)
            relevance_tuning: Whether to apply relevance tuning
            context_size: Size of context window for preview
            score_threshold: Minimum score threshold
            rerank: Whether to apply reranking as a third step
            reranker_type: Type of reranker to use
            
        Returns:
            Dictionary with search results and metadata
        """
        try:
            query = query.strip()
            if not query:
                return {"error": "Empty query"}
            
            if not self.collection:
                return {"error": f"Collection '{self.collection_name}' not found"}
            
            # Determine correct search method based on type
            if search_type.lower() in ["vector", "dense"]:
                if processor is None:
                    return {"error": "Vector search requires an embedding model"}
                points = self.search_dense(query, processor, limit, score_threshold, rerank, reranker_type)
            elif search_type.lower() == "sparse":
                if processor is None:
                    return {"error": "Sparse search requires an embedding model"}
                points = self.search_sparse(query, processor, limit, score_threshold, rerank, reranker_type)
            elif search_type.lower() in ["keyword", "fts"]:
                points = self.search_keyword(query, limit, score_threshold, rerank, reranker_type)
            else:  # Default to hybrid
                if processor is None:
                    return {"error": "Hybrid search requires an embedding model"}
                points = self.search_hybrid(query, processor, limit, prefetch_limit,
                                        fusion_type, score_threshold, rerank, reranker_type)
            
            # Check for errors
            if isinstance(points, dict) and "error" in points:
                return points
            
            # Create a retriever function for context
            def context_retriever(file_path, chunk_index, window=1):
                return self._retrieve_context_for_chunk(file_path, chunk_index, window)
            
            # Format results with improved preview
            return TextProcessor.format_search_results(
                points, query, search_type, processor, context_size,
                retriever=context_retriever,
                db_type="chromadb"
            )
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    # Implementation for retrieving context for a chunk
    def _retrieve_context_for_chunk(
        self, 
        file_path: str, 
        chunk_index: int, 
        window: int = 1
    ) -> str:
        """
        For a given chunk (identified by its file_path and chunk_index),
        retrieve the neighboring chunks from ChromaDB (e.g. chunk_index-1 and +1),
        and return a combined text string.
        """
        try:
            # Calculate range of chunk indices to retrieve
            min_idx = max(0, chunk_index - window)
            max_idx = chunk_index + window
            
            # Create a metadata filter for the file path
            where_filter = {
                "$and": [
                    {"file_path": file_path},
                    {"chunk_index": {"$gte": min_idx}},
                    {"chunk_index": {"$lte": max_idx}}
                ]
            }
            
            # Query for chunks within the window
            results = self.collection.query(
                query_texts=None,
                where=where_filter,
                n_results=window * 2 + 1,  # Max possible chunks in window
                include=["documents", "metadatas"]
            )
            
            # Check if we have any results
            if not results['ids'][0]:
                return ""
            
            # Extract document texts and their chunk indices
            chunks = []
            for i in range(len(results['ids'][0])):
                document = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                chunks.append((metadata.get('chunk_index', 0), document))
            
            # Sort chunks by chunk index
            chunks.sort(key=lambda x: x[0])
            
            # Combine text from all chunks
            combined_text = "\n".join([chunk[1] for chunk in chunks])
            
            return combined_text
            
        except Exception as e:
            if self.verbose:
                print(f"Error retrieving context: {str(e)}")
            return ""

    def cleanup(self, remove_storage: bool = False) -> None:
        """Clean up resources"""
        try:
            # Close the client if it exists
            if self.client:
                # ChromaDB doesn't have an explicit close method
                
                # Remove the storage directory only if requested
                if remove_storage and self.storage_path and os.path.exists(self.storage_path):
                    if self.verbose:
                        print(f"Removing storage directory: {self.storage_path}")
                    shutil.rmtree(self.storage_path, ignore_errors=True)
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    def search_dense(self, query: str, processor: Any, limit: int, score_threshold: float = None,
                   rerank: bool = False, reranker_type: str = None):
        """
        Perform dense vector search in ChromaDB.
        
        Args:
            query: Search query string
            processor: Document processor with embedding capabilities
            limit: Maximum number of results to return
            score_threshold: Minimum score threshold
            rerank: Whether to apply reranking
            reranker_type: Type of reranker to use
            
        Returns:
            List of search results
        """
        if processor is None:
            return {"error": "Vector search requires an embedding model"}
        
        try:
            # Generate query embedding
            query_vector = processor.get_embedding(query)
            
            if self.verbose:
                print(f"Executing vector search with {len(query_vector)}-dimensional query vector")
            
            # Get more results than needed if reranking
            fetch_limit = limit * 3 if rerank else limit
            
            # Search using query embedding
            results = self.collection.query(
                query_embeddings=query_vector.tolist(),
                n_results=fetch_limit,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            points = []
            
            # Check if we have any results
            if not results['ids'][0]:
                if self.verbose:
                    print("No results found for vector search")
                return []
            
            # Extract results
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                document = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                # Convert distance to score (ChromaDB returns L2 distance, lower is better)
                # Transform to a 0-1 score where 1 is best
                score = 1.0 / (1.0 + distance)
                
                # Apply score threshold if provided
                if score_threshold is not None and score < score_threshold:
                    continue
                
                # Build result object with both document and metadata
                result = {
                    "id": doc_id,
                    "text": document,  # Original document text
                    "score": score
                }
                
                # Add metadata fields
                for key, value in metadata.items():
                    result[key] = value
                
                # Add payload for compatibility
                result["payload"] = {
                    "id": doc_id,
                    "text": document,
                    "score": score,
                    **metadata
                }
                
                points.append(result)
            
            if self.verbose:
                print(f"Vector search returned {len(points)} results")
            
            # Apply reranking if requested
            if rerank and points and processor:
                if self.verbose:
                    print(f"Applying {reranker_type or 'default'} reranking")
                
                points = SearchAlgorithms.rerank_results(query, points, processor, limit, self.verbose)
                
                # Record metrics if ground truth is available
                true_context = getattr(processor, 'expected_context', None)
                if true_context:
                    hit = any(hasattr(p, "payload") and p.payload and p.payload.get("text") == true_context for p in points)
                    self._record_hit(f"vector_{reranker_type or 'default'}" if rerank else "vector", hit)
            
            return points[:limit]
            
        except Exception as e:
            print(f"Error in vector search: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": f"Error in vector search: {str(e)}"}

    def insert_embeddings(self, embeddings_with_payloads: List[Tuple[np.ndarray, Dict[str, Any]]]) -> None:
        """
        Insert embeddings into ChromaDB.
        
        Args:
            embeddings_with_payloads: List of (embedding, payload) tuples
        """
        if not embeddings_with_payloads:
            return
            
        try:
            if not self.collection:
                raise ValueError(f"Collection '{self.collection_name}' not initialized")
            
            # Prepare documents for insertion
            embeddings = []
            documents = []
            metadatas = []
            ids = []
            
            for i, (embedding, payload) in enumerate(embeddings_with_payloads):
                # Generate ID if not provided
                if 'id' not in payload:
                    doc_id = str(uuid.uuid4())
                else:
                    doc_id = str(payload['id'])  # Ensure ID is string
                
                # Get text content
                if 'content' in payload:
                    text = payload['content']
                elif 'text' in payload:
                    text = payload['text']
                else:
                    text = ""
                
                # Prepare metadata from payload
                metadata = {k: v for k, v in payload.items() if k not in ['content', 'text', 'id']}
                
                # Add documents
                embeddings.append(embedding.tolist())
                documents.append(text)
                metadatas.append(metadata)
                ids.append(doc_id)
            
            if self.verbose:
                print(f"Inserting {len(documents)} documents with embeddings")
            
            # Insert in batches to handle large datasets
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                end = min(i + batch_size, len(documents))
                
                if self.verbose and len(documents) > batch_size:
                    print(f"Inserting batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
                
                # Insert the batch
                self.collection.add(
                    embeddings=embeddings[i:end],
                    documents=documents[i:end],
                    metadatas=metadatas[i:end],
                    ids=ids[i:end]
                )
            
            if self.verbose:
                print(f"Successfully inserted {len(documents)} documents")
                
        except Exception as e:
            print(f"Error inserting embeddings: {str(e)}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            if not self.client or not self.collection:
                return {"error": "Not connected to ChromaDB or collection not found"}
                
            # Get collection details and count
            count = self.collection.count()
            
            # Get metadata if available
            try:
                metadata = self.collection.metadata
                if not metadata:
                    metadata = {}
            except:
                metadata = {}
            
            # Format vector configuration based on metadata
            vector_configs = {}
            if "dense_model_id" in metadata:
                vector_configs[metadata["dense_model_id"]] = {
                    "size": metadata.get("vector_size", self.vector_dim),
                    "model": metadata.get("dense_model_id", self.dense_model_id)
                }
            
            # Format sparse vector configuration
            sparse_vector_configs = {}
            if "sparse_model_id" in metadata:
                sparse_vector_configs[metadata["sparse_model_id"]] = {
                    "model": metadata.get("sparse_model_id", self.sparse_model_id)
                }
            
            # Calculate disk usage if storage_path is available
            disk_usage = None
            if self.storage_path and os.path.exists(self.storage_path):
                try:
                    chroma_dir = Path(self.storage_path)
                    disk_usage = sum(f.stat().st_size for f in chroma_dir.glob('**/*') if f.is_file())
                except Exception as e:
                    if self.verbose:
                        print(f"Could not calculate disk usage: {str(e)}")
            
            # Include performance data if available
            performance = {}
            if hasattr(self, '_hit_rates') and self._hit_rates:
                performance["hit_rates"] = self._hit_rates
            
            return {
                "name": self.collection_name,
                "points_count": count,
                "disk_usage": disk_usage,
                "vector_configs": vector_configs,
                "sparse_vector_configs": sparse_vector_configs,
                "metadata": metadata,
                "performance": performance
            }
        except Exception as e:
            if self.verbose:
                print(f"Error getting collection info: {str(e)}")
            return {"error": str(e)}
    
    def _record_hit(self, search_type: str, hit: bool):
        """Record hit for performance tracking"""
        if not hasattr(self, '_hit_rates'):
            self._hit_rates = {}
            
        if search_type not in self._hit_rates:
            self._hit_rates[search_type] = {"hits": 0, "total": 0}
            
        self._hit_rates[search_type]["total"] += 1
        if hit:
            self._hit_rates[search_type]["hits"] += 1
    
    def get_hit_rates(self) -> Dict[str, float]:
        """Get hit rates for different search types"""
        if not hasattr(self, '_hit_rates'):
            return {}
            
        result = {}
        for search_type, stats in self._hit_rates.items():
            if stats["total"] > 0:
                hit_rate = stats["hits"] / stats["total"]
                result[search_type] = hit_rate
        return result

    def insert_embeddings_with_sparse(self, embeddings_with_sparse: List[Tuple[np.ndarray, Dict[str, Any], Tuple[List[int], List[float]]]]) -> None:
        """
        Insert embeddings with sparse vectors into ChromaDB.
        ChromaDB doesn't natively support sparse vectors, so we store them in metadata.
        
        Args:
            embeddings_with_sparse: List of (dense_embedding, payload, sparse_vector) tuples
        """
        if not embeddings_with_sparse:
            return
            
        try:
            if not self.collection:
                raise ValueError(f"Collection '{self.collection_name}' not initialized")
            
            # Prepare documents for insertion
            embeddings = []
            documents = []
            metadatas = []
            ids = []
            
            for i, (dense_embedding, payload, sparse_vector) in enumerate(embeddings_with_sparse):
                # Generate ID if not provided
                if 'id' not in payload:
                    doc_id = str(uuid.uuid4())
                else:
                    doc_id = str(payload['id'])  # Ensure ID is string
                
                # Get text content
                if 'content' in payload:
                    text = payload['content']
                elif 'text' in payload:
                    text = payload['text']
                else:
                    text = ""
                
                # Prepare metadata from payload
                metadata = {k: v for k, v in payload.items() if k not in ['content', 'text', 'id']}
                
                # Add sparse vector information to metadata
                # ChromaDB doesn't support sparse vectors natively, so we store them in metadata
                sparse_indices, sparse_values = sparse_vector
                
                # Store only non-zero values from sparse vector to save space
                # Convert to strings for serialization
                metadata["sparse_indices"] = [str(idx) for idx in sparse_indices]
                metadata["sparse_values"] = [str(val) for val in sparse_values]
                
                # Add documents
                embeddings.append(dense_embedding.tolist())
                documents.append(text)
                metadatas.append(metadata)
                ids.append(doc_id)
            
            if self.verbose:
                print(f"Inserting {len(documents)} documents with embeddings and sparse vectors")
            
            # Insert in batches to handle large datasets
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                end = min(i + batch_size, len(documents))
                
                if self.verbose and len(documents) > batch_size:
                    print(f"Inserting batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
                
                # Insert the batch
                self.collection.add(
                    embeddings=embeddings[i:end],
                    documents=documents[i:end],
                    metadatas=metadatas[i:end],
                    ids=ids[i:end]
                )
            
            if self.verbose:
                print(f"Successfully inserted {len(documents)} documents with sparse vectors")
                
        except Exception as e:
            print(f"Error inserting embeddings with sparse vectors: {str(e)}")
            raise