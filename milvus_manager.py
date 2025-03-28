# milvus_manager.py
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

# Try to import Milvus client
try:
    from pymilvus import (
        connections, 
        utility,
        Collection,
        FieldSchema, 
        CollectionSchema, 
        DataType,
        Index,
        IndexType,
        SearchResult
    )
    milvus_available = True
except ImportError:
    milvus_available = False
    print("Warning: Milvus client not available. Install with: pip install pymilvus")


class MilvusManager(VectorDBInterface):
    """Manager for Milvus vector database operations with hybrid search capabilities"""
    
    def __init__(self, 
                host: str = "localhost", 
                port: str = "19530",
                collection_name: str = "documents",
                vector_size: int = 384,
                storage_path: str = None,
                verbose: bool = False,
                dense_model_id: str = "bge-small",
                sparse_model_id: str = "distilbert-splade",
                user: str = "",
                password: str = "",
                secure: bool = False,
                token: str = None):
        """
        Initialize MilvusManager with model-specific vector configuration.
        
        Args:
            host: Milvus host
            port: Milvus port
            collection_name: Collection name
            vector_size: Vector dimension
            storage_path: Storage path (not used for Milvus but kept for API compatibility)
            verbose: Verbose output
            dense_model_id: ID of dense model for naming vectors
            sparse_model_id: ID of sparse model for naming vectors
            user: Username for authentication
            password: Password for authentication
            secure: Whether to use SSL connection
            token: Auth token for Milvus (alternative to username/password)
        """
        if not milvus_available:
            raise ImportError("Milvus client not available. Install with: pip install pymilvus")
            
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_dim = vector_size
        self.storage_path = storage_path  # Not used but kept for compatibility
        self.verbose = verbose
        self.collection = None
        self.index_created = False
        
        # Authentication parameters
        self.user = user
        self.password = password
        self.secure = secure
        self.token = token
        
        # Store the original model IDs
        self.dense_model_id = dense_model_id
        self.sparse_model_id = sparse_model_id
        
        # Field names for vectors
        self.dense_field = self._sanitize_field_name(dense_model_id)
        self.sparse_indices_field = f"sparse_{self._sanitize_field_name(sparse_model_id)}_indices"
        self.sparse_values_field = f"sparse_{self._sanitize_field_name(sparse_model_id)}_values"
        
        # Performance tracking
        self._hit_rates = {}
        
        self.connect()

    def connect(self) -> None:
        """Connect to Milvus server"""
        try:
            # Set up connection parameters
            conn_params = {
                "host": self.host,
                "port": self.port
            }
            
            # Add authentication if provided
            if self.user and self.password:
                conn_params["user"] = self.user
                conn_params["password"] = self.password
                
            if self.secure:
                conn_params["secure"] = self.secure
                
            if self.token:
                conn_params["token"] = self.token
                
            if self.verbose:
                print(f"Connecting to Milvus at {self.host}:{self.port}")
                
            # Connect to Milvus
            connections.connect("default", **conn_params)
            
            # Try to get the collection if it exists
            try:
                if utility.has_collection(self.collection_name):
                    self.collection = Collection(self.collection_name)
                    if self.verbose:
                        print(f"Found existing collection '{self.collection_name}'")
                        
                    # Check if collection has an index
                    self._check_index()
                else:
                    if self.verbose:
                        print(f"Collection '{self.collection_name}' does not exist yet")
                    self.collection = None
            except Exception as e:
                if self.verbose:
                    print(f"Error checking for collection: {e}")
                self.collection = None
                
        except Exception as e:
            print(f"Error connecting to Milvus: {str(e)}")
            raise
        
    def _sanitize_field_name(self, field_name):
        """Sanitize model name for use as a field name in Milvus"""
        # Extract just the final part of the model name if it has slashes
        if "/" in field_name:
            field_name = field_name.split("/")[-1]
            
        # Replace any non-alphanumeric characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9]', '_', field_name)
        
        # Ensure the name isn't too long
        if len(sanitized) > 40:
            sanitized = sanitized[:40]
            
        return sanitized
        
    def _check_index(self):
        """Check if collection has an index and load it if needed"""
        try:
            if self.collection:
                # Get index information
                index_info = self.collection.index()
                if index_info:
                    if self.verbose:
                        print(f"Collection has {len(index_info)} indexes")
                    self.index_created = True
                    
                    # Load collection to memory for faster searches
                    self.collection.load()
                else:
                    if self.verbose:
                        print("Collection has no indexes")
                    self.index_created = False
        except Exception as e:
            if self.verbose:
                print(f"Error checking index: {e}")
            self.index_created = False
    
    def create_collection(self, recreate: bool = False) -> None:
        """Create Milvus collection with vector search capabilities"""
        try:
            # Check if collection exists
            exists = utility.has_collection(self.collection_name)
            
            # Recreate if requested
            if exists and recreate:
                if self.verbose:
                    print(f"Dropping existing collection '{self.collection_name}'")
                utility.drop_collection(self.collection_name)
                exists = False
                
            # Create collection if it doesn't exist
            if not exists:
                if self.verbose:
                    print(f"Creating collection '{self.collection_name}' with vector size {self.vector_dim}")
                
                # Define fields for the schema
                fields = [
                    # Primary key field
                    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                    
                    # Metadata fields
                    FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=500),
                    FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="file_type", dtype=DataType.VARCHAR, max_length=20),
                    FieldSchema(name="file_size", dtype=DataType.INT64),
                    FieldSchema(name="created_at", dtype=DataType.INT64),
                    FieldSchema(name="modified_at", dtype=DataType.INT64),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),  # Text content
                    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
                    
                    # Chunking fields
                    FieldSchema(name="is_chunk", dtype=DataType.BOOL),
                    FieldSchema(name="is_parent", dtype=DataType.BOOL),
                    FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="chunk_index", dtype=DataType.INT64),
                    FieldSchema(name="total_chunks", dtype=DataType.INT64),
                    
                    # Vector fields
                    FieldSchema(name=self.dense_field, dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
                    
                    # Sparse vector fields (as arrays)
                    FieldSchema(name=self.sparse_indices_field, dtype=DataType.ARRAY, element_type=DataType.INT64, max_capacity=1000),
                    FieldSchema(name=self.sparse_values_field, dtype=DataType.ARRAY, element_type=DataType.FLOAT, max_capacity=1000)
                ]
                
                # Create schema
                schema = CollectionSchema(fields=fields, description=f"Document collection for {self.dense_model_id}")
                
                # Create collection
                self.collection = Collection(name=self.collection_name, schema=schema)
                
                if self.verbose:
                    print(f"Created collection '{self.collection_name}'")
                    
                # Create indexes for vector fields
                self._create_indexes()
            else:
                # Collection exists, get it
                self.collection = Collection(self.collection_name)
                if self.verbose:
                    print(f"Collection '{self.collection_name}' already exists")
                    
                # Check if indexes exist
                self._check_index()
                
                # Create indexes if they don't exist
                if not self.index_created:
                    self._create_indexes()
                
        except Exception as e:
            print(f"Error creating collection: {str(e)}")
            raise
            
    def _create_indexes(self):
        """Create indexes for the collection"""
        try:
            if not self.collection:
                raise ValueError("Collection is not initialized")
                
            if self.verbose:
                print("Creating indexes...")
                
            # Create index for dense vector field
            index_params = {
                "metric_type": "COSINE",  # Use COSINE similarity for dense vectors
                "index_type": "HNSW",     # Hierarchical Navigable Small World graph
                "params": {
                    "M": 16,              # Maximum number of edges per node
                    "efConstruction": 200 # Construction-time control for accuracy vs. time
                }
            }
            
            # Create index
            self.collection.create_index(
                field_name=self.dense_field,
                index_params=index_params
            )
            
            # Create scalar field indexes for faster filtering
            for field_name in ["file_path", "file_type", "parent_id"]:
                try:
                    self.collection.create_index(
                        field_name=field_name,
                        index_name=f"{field_name}_idx"
                    )
                except Exception as e:
                    if self.verbose:
                        print(f"Error creating index for {field_name}: {e}")
            
            self.index_created = True
            
            # Load collection to memory for faster searches
            self.collection.load()
            
            if self.verbose:
                print("Indexes created and collection loaded")
                
        except Exception as e:
            print(f"Error creating indexes: {e}")
            self.index_created = False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            if not self.collection:
                return {"error": "Collection not initialized"}
                
            # Get collection stats
            stats = self.collection.stats
            
            # Get row count
            row_count = stats.get("row_count", 0)
            
            # Get index information
            index_info = self.collection.index()
            
            # Format index information in a way that matches our interface
            vector_configs = {}
            if index_info:
                for idx in index_info:
                    if "field" in idx and "index_type" in idx and "params" in idx:
                        field_name = idx["field"]
                        vector_configs[field_name] = {
                            "type": idx["index_type"],
                            "metric": idx.get("params", {}).get("metric_type", "unknown"),
                            "params": idx.get("params", {})
                        }
            
            # Include performance data if available
            performance = {}
            if self._hit_rates:
                performance["hit_rates"] = self._hit_rates
            
            return {
                "name": self.collection_name,
                "points_count": row_count,
                "vector_configs": vector_configs,
                "sparse_vector_configs": {
                    "indices_field": self.sparse_indices_field,
                    "values_field": self.sparse_values_field
                },
                "performance": performance
            }
        except Exception as e:
            if self.verbose:
                print(f"Error getting collection info: {str(e)}")
            return {"error": str(e)}
    
    def _record_hit(self, search_type: str, hit: bool):
        """Record hit for performance tracking"""
        if search_type not in self._hit_rates:
            self._hit_rates[search_type] = {"hits": 0, "total": 0}
            
        self._hit_rates[search_type]["total"] += 1
        if hit:
            self._hit_rates[search_type]["hits"] += 1
    
    def get_hit_rates(self) -> Dict[str, float]:
        """Get hit rates for different search types"""
        result = {}
        for search_type, stats in self._hit_rates.items():
            if stats["total"] > 0:
                hit_rate = stats["hits"] / stats["total"]
                result[search_type] = hit_rate
        return result
    
    def insert_embeddings(self, embeddings_with_payloads: List[Tuple[np.ndarray, Dict[str, Any]]]) -> None:
        """Insert embeddings into Milvus"""
        if not embeddings_with_payloads:
            return
            
        try:
            if not self.collection:
                raise ValueError(f"Collection '{self.collection_name}' not initialized")
                
            # Prepare data for insertion in columnar format (Milvus prefers this)
            data = {
                "id": [],
                "file_path": [],
                "file_name": [],
                "file_type": [],
                "file_size": [],
                "created_at": [],
                "modified_at": [],
                "content": [],
                "title": [],
                "is_chunk": [],
                "is_parent": [],
                "parent_id": [],
                "chunk_index": [],
                "total_chunks": [],
                self.dense_field: [],
                self.sparse_indices_field: [],
                self.sparse_values_field: []
            }
            
            for i, (embedding, payload) in enumerate(embeddings_with_payloads):
                # Generate a unique ID if not provided
                if "id" not in payload:
                    doc_id = str(uuid.uuid4())
                else:
                    doc_id = str(payload["id"])
                
                # Generate default sparse vector for compatibility
                sparse_indices = [0]
                sparse_values = [0.0]
                
                # Add data to columns
                data["id"].append(doc_id)
                data["file_path"].append(payload.get("file_path", ""))
                data["file_name"].append(payload.get("file_name", ""))
                data["file_type"].append(payload.get("fileType", ""))
                data["file_size"].append(payload.get("fileSize", 0))
                data["created_at"].append(int(payload.get("createdAt", time.time())))
                data["modified_at"].append(int(payload.get("modifiedAt", time.time())))
                data["content"].append(payload.get("content", "")[:65000])  # Limit size for VARCHAR
                data["title"].append(payload.get("title", "")[:490])        # Limit size for VARCHAR
                data["is_chunk"].append(bool(payload.get("is_chunk", False)))
                data["is_parent"].append(bool(payload.get("is_parent", False)))
                data["parent_id"].append(payload.get("parent_id", ""))
                data["chunk_index"].append(int(payload.get("chunk_index", 0)))
                data["total_chunks"].append(int(payload.get("total_chunks", 0)))
                data[self.dense_field].append(embedding.tolist())
                data[self.sparse_indices_field].append(sparse_indices)
                data[self.sparse_values_field].append(sparse_values)
            
            if self.verbose:
                print(f"Inserting {len(data['id'])} documents into collection '{self.collection_name}'")
                
            # Insert data in batches to handle large datasets
            batch_size = 100
            for i in range(0, len(data["id"]), batch_size):
                end_idx = min(i + batch_size, len(data["id"]))
                
                # Prepare batch data
                batch_data = {k: v[i:end_idx] for k, v in data.items()}
                
                if self.verbose and len(data["id"]) > batch_size:
                    print(f"Inserting batch {i//batch_size + 1}/{(len(data['id'])-1)//batch_size + 1} ({end_idx-i} documents)")
                
                # Insert batch
                self.collection.insert(batch_data)
                
            # Create index if not already created
            if not self.index_created:
                self._create_indexes()
            else:
                # Make sure collection is loaded
                self.collection.load()
                
            if self.verbose:
                print(f"Successfully inserted {len(data['id'])} documents")
                
        except Exception as e:
            print(f"Error inserting embeddings: {str(e)}")
            raise
    
    def insert_embeddings_with_sparse(self, embeddings_with_sparse: List[Tuple[np.ndarray, Dict[str, Any], Tuple[List[int], List[float]]]]) -> None:
        """
        Insert embeddings with sparse vectors into Milvus.
        
        Args:
            embeddings_with_sparse: List of (dense_embedding, payload, sparse_embedding) tuples
        """
        if not embeddings_with_sparse:
            return
            
        try:
            if not self.collection:
                raise ValueError(f"Collection '{self.collection_name}' not initialized")
                
            # Prepare data for insertion in columnar format
            data = {
                "id": [],
                "file_path": [],
                "file_name": [],
                "file_type": [],
                "file_size": [],
                "created_at": [],
                "modified_at": [],
                "content": [],
                "title": [],
                "is_chunk": [],
                "is_parent": [],
                "parent_id": [],
                "chunk_index": [],
                "total_chunks": [],
                self.dense_field: [],
                self.sparse_indices_field: [],
                self.sparse_values_field: []
            }
            
            for i, (embedding, payload, sparse_vector) in enumerate(embeddings_with_sparse):
                # Generate a unique ID if not provided
                if "id" not in payload:
                    doc_id = str(uuid.uuid4())
                else:
                    doc_id = str(payload["id"])
                    
                # Extract sparse indices and values
                sparse_indices, sparse_values = sparse_vector
                
                # Ensure indices and values are not empty (Milvus requires non-empty arrays)
                if not sparse_indices:
                    sparse_indices = [0]
                    sparse_values = [0.0]
                    
                # Check array size limits (max_capacity was set to 1000)
                if len(sparse_indices) > 1000:
                    sparse_indices = sparse_indices[:1000]
                    sparse_values = sparse_values[:1000]
                
                # Add data to columns
                data["id"].append(doc_id)
                data["file_path"].append(payload.get("file_path", ""))
                data["file_name"].append(payload.get("file_name", ""))
                data["file_type"].append(payload.get("fileType", ""))
                data["file_size"].append(payload.get("fileSize", 0))
                data["created_at"].append(int(payload.get("createdAt", time.time())))
                data["modified_at"].append(int(payload.get("modifiedAt", time.time())))
                data["content"].append(payload.get("content", "")[:65000])  # Limit size for VARCHAR
                data["title"].append(payload.get("title", "")[:490])        # Limit size for VARCHAR
                data["is_chunk"].append(bool(payload.get("is_chunk", False)))
                data["is_parent"].append(bool(payload.get("is_parent", False)))
                data["parent_id"].append(payload.get("parent_id", ""))
                data["chunk_index"].append(int(payload.get("chunk_index", 0)))
                data["total_chunks"].append(int(payload.get("total_chunks", 0)))
                data[self.dense_field].append(embedding.tolist())
                data[self.sparse_indices_field].append(sparse_indices)
                data[self.sparse_values_field].append(sparse_values)
            
            if self.verbose:
                print(f"Inserting {len(data['id'])} documents with sparse vectors into collection '{self.collection_name}'")
                
            # Insert data in batches to handle large datasets
            batch_size = 100
            for i in range(0, len(data["id"]), batch_size):
                end_idx = min(i + batch_size, len(data["id"]))
                
                # Prepare batch data
                batch_data = {k: v[i:end_idx] for k, v in data.items()}
                
                if self.verbose and len(data["id"]) > batch_size:
                    print(f"Inserting batch {i//batch_size + 1}/{(len(data['id'])-1)//batch_size + 1} ({end_idx-i} documents)")
                
                # Insert batch
                self.collection.insert(batch_data)
            
            # Create index if not already created
            if not self.index_created:
                self._create_indexes()
            else:
                # Make sure collection is loaded
                self.collection.load()
                
            if self.verbose:
                print(f"Successfully inserted {len(data['id'])} documents with sparse vectors")
                
        except Exception as e:
            print(f"Error inserting embeddings with sparse vectors: {str(e)}")
            raise
    
    def search_hybrid(self, query: str, processor: Any, limit: int, 
                    prefetch_limit: int = 50, fusion_type: str = "rrf",
                    score_threshold: float = None, rerank: bool = False,
                    reranker_type: str = None):
        """
        Perform a hybrid search combining both dense vectors and sparse vectors.
        
        Args:
            query: Search query string
            processor: Document processor with embedding capabilities
            limit: Number of results to return
            prefetch_limit: Number of results to prefetch for fusion
            fusion_type: Type of fusion to use (rrf or dbsf)
            score_threshold: Minimum score threshold
            rerank: Whether to apply reranking after fusion
            reranker_type: Type of reranker to use
            
        Returns:
            List of search results
        """
        if processor is None:
            return {"error": "Hybrid search requires an embedding model"}
        
        try:
            # Milvus doesn't have a native hybrid search like Meilisearch,
            # so we need to implement it manually by combining dense and sparse search results
            
            # 1. Get dense vector results
            dense_results = self.search_dense(query, processor, prefetch_limit, score_threshold, False, None)
            
            # 2. Get sparse vector results
            sparse_results = self.search_sparse(query, processor, prefetch_limit, score_threshold, False, None)
            
            # 3. Combine results using fusion algorithm
            if isinstance(dense_results, dict) and "error" in dense_results:
                dense_results = []
                
            if isinstance(sparse_results, dict) and "error" in sparse_results:
                sparse_results = []
                
            # Ensure results are in the right format for fusion
            if len(dense_results) > 0 or len(sparse_results) > 0:
                # Use SearchAlgorithms to perform fusion
                fused_results = SearchAlgorithms.manual_fusion(
                    dense_results, sparse_results, prefetch_limit, fusion_type
                )
                
                # 4. Apply reranking if requested
                if rerank and len(fused_results) > 0:
                    reranked_results = SearchAlgorithms.rerank_results(
                        query, fused_results, processor, limit, self.verbose
                    )
                    
                    # Record hit rates if ground truth is available
                    true_context = getattr(processor, 'expected_context', None)
                    if true_context:
                        hit = any(hasattr(p, "payload") and p.payload and p.payload.get("content") == true_context 
                                for p in reranked_results)
                        search_key = f"hybrid_{fusion_type}"
                        if rerank:
                            search_key += f"_{reranker_type or 'default'}"
                        self._record_hit(search_key, hit)
                    
                    return reranked_results[:limit]
                
                # Record hit rates if ground truth is available
                true_context = getattr(processor, 'expected_context', None)
                if true_context:
                    hit = any(hasattr(p, "payload") and p.payload and p.payload.get("content") == true_context 
                            for p in fused_results)
                    search_key = f"hybrid_{fusion_type}"
                    self._record_hit(search_key, hit)
                
                return fused_results[:limit]
            else:
                # If both searches failed, return an empty list
                return []
                
        except Exception as e:
            if self.verbose:
                print(f"Error in hybrid search: {e}")
                import traceback
                traceback.print_exc()
            return {"error": f"Error in hybrid search: {str(e)}"}
    
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
            rerank: Whether to apply reranking
            reranker_type: Type of reranker to use
            
        Returns:
            Dictionary with search results and metadata
        """
        try:
            query = query.strip()
            if not query:
                return {"error": "Empty query"}
            
            # Make sure collection is loaded
            if not self.collection:
                return {"error": f"Collection '{self.collection_name}' not found"}
                
            self.collection.load()
            
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
                db_type="milvus"
            )
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def _adapt_milvus_result(self, result):
        """Adapt a Milvus search result to match the format expected by other parts of the code"""
        # Create a point-like object with appropriate attributes
        class Point:
            def __init__(self, entity, distance):
                self.id = entity.get("id", "")
                self.score = 1.0 - distance  # Convert distance to similarity score (0-1)
                
                # All other fields go in payload
                self.payload = {
                    "content": entity.get("content", ""),
                    "file_path": entity.get("file_path", ""),
                    "file_name": entity.get("file_name", ""),
                    "fileType": entity.get("file_type", ""),
                    "fileSize": entity.get("file_size", 0),
                    "createdAt": entity.get("created_at", 0),
                    "modifiedAt": entity.get("modified_at", 0),
                    "title": entity.get("title", ""),
                    "is_chunk": entity.get("is_chunk", False),
                    "is_parent": entity.get("is_parent", False),
                    "parent_id": entity.get("parent_id", ""),
                    "chunk_index": entity.get("chunk_index", 0),
                    "total_chunks": entity.get("total_chunks", 0)
                }
        
        return Point(result[0], result[1])
    
    def search_dense(self, query: str, processor: Any, limit: int, score_threshold: float = None,
                    rerank: bool = False, reranker_type: str = None):
        """
        Perform dense vector search with consistent handling
        
        Args:
            query: Search query string
            processor: Document processor with embedding capabilities
            limit: Number of results to return
            score_threshold: Minimum score threshold
            rerank: Whether to apply reranking
            reranker_type: Type of reranker to use
            
        Returns:
            List of search results
        """
        if processor is None:
            return {"error": "Dense search requires an embedding model"}
        
        try:
            # Generate dense embedding
            query_vector = processor.get_embedding(query)
            
            # Build search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10},  # Number of clusters to search
            }
            
            if score_threshold is not None:
                search_params["params"]["ef"] = 64  # Higher values give more accurate results
                
            # Execute vector search
            results = self.collection.search(
                data=[query_vector.tolist()],  # List of query vectors
                anns_field=self.dense_field,   # Vector field to search
                param=search_params,          # Search parameters
                limit=limit * 3,              # Get more results for filtering
                expr=None,                    # No filtering expression
                output_fields=["id", "content", "file_path", "file_name", "file_type", 
                                "file_size", "created_at", "modified_at", "title", 
                                "is_chunk", "is_parent", "parent_id", "chunk_index", "total_chunks"]
            )
            
            if self.verbose:
                print(f"Dense search returned {len(results[0])} results")
            
            # Convert results to point objects
            points = []
            for hit in results[0]:  # results[0] is for the first query
                entity = {}
                for field in hit.entity.fields:
                    entity[field] = hit.entity.get(field)
                
                # Create point object
                point = self._adapt_milvus_result((entity, hit.distance))
                
                # Filter by score threshold if needed
                if score_threshold is not None and point.score < score_threshold:
                    continue
                    
                points.append(point)
            
            # Apply reranking if requested
            if rerank and len(points) > 0:
                reranked_points = SearchAlgorithms.rerank_results(
                    query, points, processor, limit, self.verbose
                )
                
                # Record hit rates if ground truth is available
                true_context = getattr(processor, 'expected_context', None)
                if true_context:
                    hit = any(p.payload.get("content") == true_context for p in reranked_points)
                    search_key = "vector"
                    if rerank:
                        search_key += f"_{reranker_type or 'default'}"
                    self._record_hit(search_key, hit)
                
                return reranked_points[:limit]
            
            # Record hit rates if ground truth is available
            true_context = getattr(processor, 'expected_context', None)
            if true_context:
                hit = any(p.payload.get("content") == true_context for p in points)
                self._record_hit("vector", hit)
            
            return points[:limit]
            
        except Exception as e:
            if self.verbose:
                print(f"Error in dense search: {e}")
            return {"error": f"Error in dense search: {str(e)}"}
    
    def search_sparse(self, query: str, processor: Any, limit: int, score_threshold: float = None,
                    rerank: bool = False, reranker_type: str = None):
        """
        Perform sparse vector search.
        
        Milvus doesn't have native sparse vector search, so we implement a 
        simple approximation using the stored sparse vectors and BM25-like scoring.
        
        Args:
            query: Search query string
            processor: Document processor with embedding capabilities
            limit: Number of results to return
            score_threshold: Minimum score threshold
            rerank: Whether to apply reranking
            reranker_type: Type of reranker to use
            
        Returns:
            List of search results
        """
        if processor is None:
            return {"error": "Sparse search requires an embedding model"}
        
        try:
            # Generate sparse vector from query
            sparse_indices, sparse_values = processor.get_sparse_embedding(query)
            
            # Since Milvus doesn't have direct sparse vector search,
            # we'll fall back to a keyword search for now
            
            # For more advanced implementations, we could use an expression 
            # to perform sparse matching in Milvus, but that would be complex
            
            if self.verbose:
                print("Sparse search not directly supported in Milvus, falling back to keyword search")
            
            return self.search_keyword(query, limit, score_threshold, rerank, reranker_type)
            
        except Exception as e:
            if self.verbose:
                print(f"Error in sparse search: {e}")
            return {"error": f"Error in sparse search: {str(e)}"}
    
    def search_keyword(self, query: str, limit: int = 10, score_threshold: float = None,
                      rerank: bool = False, reranker_type: str = None):
        """
        Perform a keyword-based search.
        
        Milvus does not have great text search capabilities, so this is a basic implementation
        that searches for exact matches in the content and title fields.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            score_threshold: Minimum score threshold
            rerank: Whether to apply reranking (only works if processor is provided)
            reranker_type: Type of reranker to use
            
        Returns:
            List of search results
        """
        try:
            if not query.strip():
                return {"error": "Empty query"}
                
            # Build expression to search in content and title
            # This is a very basic approach and won't match on synonyms, etc.
            # For better text search, consider using a dedicated text search engine
            
            # Escape quotes in query
            safe_query = query.replace("'", "\\'")
            
            # Build expression for partial matches
            expr = f"content like '%{safe_query}%' or title like '%{safe_query}%'"
            
            # Execute search (with no vectors, just filtering)
            results = self.collection.query(
                expr=expr,
                output_fields=["id", "content", "file_path", "file_name", "file_type", 
                                "file_size", "created_at", "modified_at", "title", 
                                "is_chunk", "is_parent", "parent_id", "chunk_index", "total_chunks"],
                limit=limit * 3
            )
            
            if self.verbose:
                print(f"Keyword search returned {len(results)} results")
                
            # Check if we have any results
            if not results:
                return []
                
            # Create point objects with simulated scores
            points = []
            for i, entity in enumerate(results):
                # Compute a score based on result position and exact match presence
                exact_match_bonus = 0.2 if query.lower() in entity.get("content", "").lower() else 0
                title_match_bonus = 0.3 if query.lower() in entity.get("title", "").lower() else 0
                position_score = max(0.1, 1.0 - (i / len(results)))  # Higher rank = higher score
                
                # Combined score (normalized to 0-1 range)
                sim_score = min(1.0, position_score + exact_match_bonus + title_match_bonus)
                
                # Create point object
                point = self._adapt_milvus_result((entity, 1.0 - sim_score))  # Convert to distance format
                
                # Apply score threshold if needed
                if score_threshold is not None and point.score < score_threshold:
                    continue
                    
                points.append(point)
            
            # Apply reranking if requested - but this requires a processor
            if rerank and len(points) > 0 and 'processor' in locals() and locals()['processor'] is not None:
                processor = locals()['processor']
                reranked_points = SearchAlgorithms.rerank_results(
                    query, points, processor, limit, self.verbose
                )
                return reranked_points[:limit]
            
            return points[:limit]
            
        except Exception as e:
            if self.verbose:
                print(f"Error in keyword search: {e}")
            return {"error": f"Error in keyword search: {str(e)}"}
    
    def _retrieve_context_for_chunk(self, file_path: str, chunk_index: int, window: int = 1) -> str:
        """
        Retrieve context surrounding a chunk.
        
        Args:
            file_path: Path to the file
            chunk_index: Index of the chunk
            window: Number of chunks to include before and after
            
        Returns:
            Context text
        """
        try:
            if not self.collection:
                return "[No collection available for context retrieval]"
            
            # Define the minimum and maximum chunk indices to retrieve
            min_idx = max(0, chunk_index - window)
            max_idx = chunk_index + window
            
            # Build query expression to find chunks in the same file with neighboring indices
            expr = f"file_path == '{file_path}' and chunk_index >= {min_idx} and chunk_index <= {max_idx}"
            
            # Query for the chunks
            results = self.collection.query(
                expr=expr,
                output_fields=["content", "chunk_index"],
                limit=2 * window + 1  # Max number of chunks to retrieve
            )
            
            if not results:
                return f"[No chunks found for file_path={file_path}, chunk_index={chunk_index}]"
            
            # Sort chunks by index
            sorted_chunks = sorted(results, key=lambda x: x.get("chunk_index", 0))
            
            # Combine the content
            combined_text = "\n\n".join(chunk.get("content", "") for chunk in sorted_chunks)
            
            return combined_text
            
        except Exception as e:
            if self.verbose:
                print(f"Error retrieving context: {e}")
            return f"[Error retrieving context: {str(e)}]"
    
    def update_embeddings(self, id_embedding_payload_tuples: List[Tuple[str, np.ndarray, Dict[str, Any]]]) -> None:
        """
        Update existing embeddings with new values.
        
        Args:
            id_embedding_payload_tuples: List of (id, embedding, payload) tuples to update
        """
        if not id_embedding_payload_tuples:
            return
            
        try:
            for doc_id, embedding, payload in id_embedding_payload_tuples:
                # Prepare update data
                update_data = {
                    self.dense_field: embedding.tolist()
                }
                
                # Add other fields from payload if provided
                for key, value in payload.items():
                    if key != "id":
                        update_data[key] = value
                
                # Execute update
                self.collection.update(
                    expr=f"id == '{doc_id}'",
                    data=update_data
                )
                
            # Flush to ensure updates are committed
            self.collection.flush()
            
            if self.verbose:
                print(f"Successfully updated {len(id_embedding_payload_tuples)} embeddings")
                
        except Exception as e:
            print(f"Error updating embeddings: {str(e)}")
            raise
            
    def delete_by_filter(self, filter_condition: str) -> int:
        """
        Delete entries matching a filter condition.
        
        Args:
            filter_condition: Filter condition in Milvus expression format
            
        Returns:
            Number of deleted entries
        """
        try:
            if not self.collection:
                return 0
                
            # Count matching entities first
            matching_count = self.collection.query(
                expr=filter_condition,
                output_fields=["id"],
                limit=1000000  # Large limit to count all matches
            )
            
            count = len(matching_count)
            
            if count == 0:
                if self.verbose:
                    print(f"No entities match the filter: {filter_condition}")
                return 0
                
            if self.verbose:
                print(f"Deleting {count} entities matching filter: {filter_condition}")
                
            # Execute deletion
            self.collection.delete(filter_condition)
            
            if self.verbose:
                print(f"Deleted {count} entities")
                
            return count
                
        except Exception as e:
            print(f"Error deleting by filter: {str(e)}")
            return 0
    
    def cleanup(self, remove_storage: bool = False) -> None:
        """
        Clean up resources.
        
        Args:
            remove_storage: Whether to remove the collection
        """
        try:
            # Release collection from memory
            if self.collection:
                try:
                    self.collection.release()
                except:
                    pass
                
                # Drop collection if requested
                if remove_storage:
                    try:
                        if self.verbose:
                            print(f"Dropping collection '{self.collection_name}'")
                        utility.drop_collection(self.collection_name)
                    except Exception as e:
                        print(f"Error dropping collection: {e}")
                
            # Disconnect from Milvus
            try:
                connections.disconnect("default")
            except:
                pass
                
            self.collection = None
                
        except Exception as e:
            print(f"Error during cleanup: {e}")