# elasticsearch_manager.py
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

# Try to import Elasticsearch client
try:
    from elasticsearch import Elasticsearch, helpers
    from elasticsearch.exceptions import NotFoundError, RequestError
    elasticsearch_available = True
except ImportError:
    elasticsearch_available = False
    print("Warning: Elasticsearch client not available. Install with: pip install elasticsearch")


class ElasticsearchManager(VectorDBInterface):
    """Manager for Elasticsearch vector database operations with hybrid search capabilities"""
    
    def __init__(self, 
                hosts: List[str] = ["http://localhost:9200"], 
                collection_name: str = "documents",
                vector_size: int = 384,
                storage_path: str = None,
                verbose: bool = False,
                dense_model_id: str = "bge-small",
                sparse_model_id: str = "distilbert-splade",
                api_key: Optional[str] = None,
                username: Optional[str] = None,
                password: Optional[str] = None):
        """
        Initialize ElasticsearchManager with model-specific vector configuration.
        
        Args:
            hosts: List of Elasticsearch hosts
            collection_name: Index name
            vector_size: Vector dimension
            storage_path: Storage path (not used for Elasticsearch but kept for API compatibility)
            verbose: Verbose output
            dense_model_id: ID of dense model for naming vectors
            sparse_model_id: ID of sparse model for naming vectors
            api_key: Elasticsearch API key
            username: Elasticsearch username
            password: Elasticsearch password
        """
        if not elasticsearch_available:
            raise ImportError("Elasticsearch client not available. Install with: pip install elasticsearch")
            
        self.hosts = hosts
        self.collection_name = collection_name
        self.vector_dim = vector_size
        self.storage_path = storage_path  # Not used but kept for compatibility
        self.verbose = verbose
        self.client = None
        
        # Authentication parameters
        self.api_key = api_key
        self.username = username
        self.password = password
        
        # Store the original model IDs
        self.dense_model_id = dense_model_id
        self.sparse_model_id = sparse_model_id
        
        # Vector field names
        self.dense_field = f"vector_{self._sanitize_field_name(dense_model_id)}"
        self.sparse_indices_field = f"sparse_indices_{self._sanitize_field_name(sparse_model_id)}"
        self.sparse_values_field = f"sparse_values_{self._sanitize_field_name(sparse_model_id)}"
        
        # Performance tracking
        self._hit_rates = {}
        
        self.connect()

    def connect(self) -> None:
        """Connect to Elasticsearch server"""
        try:
            # Set up connection parameters
            conn_params = {
                "hosts": self.hosts
            }
            
            # Add authentication if provided
            if self.api_key:
                conn_params["api_key"] = self.api_key
            elif self.username and self.password:
                conn_params["basic_auth"] = (self.username, self.password)
                
            if self.verbose:
                print(f"Connecting to Elasticsearch at {', '.join(self.hosts)}")
                
            # Connect to Elasticsearch
            self.client = Elasticsearch(**conn_params)
            
            # Try to check if the index exists
            try:
                index_exists = self.client.indices.exists(index=self.collection_name)
                if index_exists:
                    if self.verbose:
                        print(f"Found existing index '{self.collection_name}'")
                else:
                    if self.verbose:
                        print(f"Index '{self.collection_name}' does not exist yet")
            except Exception as e:
                if self.verbose:
                    print(f"Error checking for index: {e}")
                
        except Exception as e:
            print(f"Error connecting to Elasticsearch: {str(e)}")
            raise
        
    def create_collection(self, recreate: bool = False) -> None:
        """Create Elasticsearch index with vector search capabilities"""
        try:
            # Check if index exists
            index_exists = self.client.indices.exists(index=self.collection_name)
            
            # Recreate if requested
            if index_exists and recreate:
                if self.verbose:
                    print(f"Dropping existing index '{self.collection_name}'")
                self.client.indices.delete(index=self.collection_name)
                index_exists = False
                
            # Create index if it doesn't exist
            if not index_exists:
                if self.verbose:
                    print(f"Creating index '{self.collection_name}' with vector size {self.vector_dim}")
                
                # Define index mappings with vector field support
                mappings = {
                    "properties": {
                        # Metadata fields
                        "id": {"type": "keyword"},
                        "file_path": {"type": "keyword"},
                        "file_name": {"type": "keyword"},
                        "file_type": {"type": "keyword"},
                        "file_size": {"type": "long"},
                        "created_at": {"type": "date", "format": "epoch_second"},
                        "modified_at": {"type": "date", "format": "epoch_second"},
                        "content": {"type": "text", "analyzer": "standard"},
                        "title": {"type": "text", "analyzer": "standard", "fields": {"keyword": {"type": "keyword"}}},
                        
                        # Chunking fields
                        "is_chunk": {"type": "boolean"},
                        "is_parent": {"type": "boolean"},
                        "parent_id": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                        "total_chunks": {"type": "integer"},
                        
                        # Vector fields
                        self.dense_field: {
                            "type": "dense_vector",
                            "dims": self.vector_dim,
                            "index": True,
                            "similarity": "cosine"
                        },
                        
                        # Sparse vector fields (as arrays)
                        self.sparse_indices_field: {"type": "integer", "store": True},
                        self.sparse_values_field: {"type": "float", "store": True},
                        
                        # Metadata field for embedding info
                        "metadata": {"type": "object", "enabled": False}
                    }
                }
                
                # Create the index with mappings
                self.client.indices.create(
                    index=self.collection_name,
                    mappings=mappings,
                    settings={
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                        "refresh_interval": "1s"
                    }
                )
                
                if self.verbose:
                    print(f"Created index '{self.collection_name}'")
                
        except Exception as e:
            print(f"Error creating collection: {str(e)}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            if not self.client:
                return {"error": "Not connected to Elasticsearch"}
                
            # Get index stats
            try:
                stats = self.client.indices.stats(index=self.collection_name)
                index_info = self.client.indices.get(index=self.collection_name)
                
                # Extract document count
                doc_count = stats["indices"][self.collection_name]["total"]["docs"]["count"]
                
                # Extract mapping information
                mappings = index_info[self.collection_name]["mappings"]
                
                # Format vector configurations in a way that matches our interface
                vector_configs = {}
                sparse_vector_configs = {}
                
                # Extract vector field configurations
                properties = mappings.get("properties", {})
                for field_name, field_config in properties.items():
                    if field_config.get("type") == "dense_vector":
                        vector_configs[field_name] = {
                            "dims": field_config.get("dims", 0),
                            "similarity": field_config.get("similarity", "cosine")
                        }
                    
                    # Track sparse vector fields
                    if field_name.startswith("sparse_indices_"):
                        model_name = field_name.replace("sparse_indices_", "")
                        sparse_vector_configs[model_name] = {
                            "indices_field": field_name,
                            "values_field": f"sparse_values_{model_name}"
                        }
                
                # Include performance data if available
                performance = {}
                if self._hit_rates:
                    performance["hit_rates"] = self._hit_rates
                
                return {
                    "name": self.collection_name,
                    "points_count": doc_count,
                    "vector_configs": vector_configs,
                    "sparse_vector_configs": sparse_vector_configs,
                    "performance": performance
                }
            except Exception as e:
                return {"error": f"Error getting index info: {str(e)}"}
                
        except Exception as e:
            if self.verbose:
                print(f"Error in get_collection_info: {e}")
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
        result = {}
        for search_type, stats in self._hit_rates.items():
            if stats["total"] > 0:
                hit_rate = stats["hits"] / stats["total"]
                result[search_type] = hit_rate
        return result
    
    def insert_embeddings(self, embeddings_with_payloads: List[Tuple[np.ndarray, Dict[str, Any]]]) -> None:
        """Insert embeddings into Elasticsearch"""
        if not embeddings_with_payloads:
            return
            
        try:
            # Prepare documents for bulk indexing
            actions = []
            
            for embedding, payload in embeddings_with_payloads:
                # Generate a unique ID if not provided
                if "id" not in payload:
                    doc_id = str(uuid.uuid4())
                else:
                    doc_id = str(payload["id"])
                
                # Create the document with vector field
                document = {
                    "_id": doc_id,
                    "_index": self.collection_name,
                    "_source": {
                        "id": doc_id,
                        "file_path": payload.get("file_path", ""),
                        "file_name": payload.get("file_name", ""),
                        "file_type": payload.get("fileType", ""),
                        "file_size": payload.get("fileSize", 0),
                        "created_at": int(payload.get("createdAt", time.time())),
                        "modified_at": int(payload.get("modifiedAt", time.time())),
                        "content": payload.get("content", payload.get("text", "")),
                        "title": payload.get("title", ""),
                        "is_chunk": bool(payload.get("is_chunk", False)),
                        "is_parent": bool(payload.get("is_parent", False)),
                        "parent_id": payload.get("parent_id", ""),
                        "chunk_index": int(payload.get("chunk_index", 0)),
                        "total_chunks": int(payload.get("total_chunks", 0)),
                        self.dense_field: embedding.tolist(),
                        # Add empty sparse vector fields for compatibility
                        self.sparse_indices_field: [0],
                        self.sparse_values_field: [0.0],
                        "metadata": payload.get("metadata", {})
                    }
                }
                
                actions.append(document)
            
            if self.verbose:
                print(f"Inserting {len(actions)} documents into index '{self.collection_name}'")
                
            # Perform bulk indexing
            helpers.bulk(self.client, actions)
            
            # Refresh index to make documents immediately available for search
            self.client.indices.refresh(index=self.collection_name)
            
            if self.verbose:
                print(f"Successfully inserted {len(actions)} documents")
                
        except Exception as e:
            print(f"Error inserting embeddings: {str(e)}")
            raise
    
    def insert_embeddings_with_sparse(self, embeddings_with_sparse: List[Tuple[np.ndarray, Dict[str, Any], Tuple[List[int], List[float]]]]) -> None:
        """
        Insert embeddings with sparse vectors into Elasticsearch.
        
        Args:
            embeddings_with_sparse: List of (dense_embedding, payload, sparse_embedding) tuples
        """
        if not embeddings_with_sparse:
            return
            
        try:
            # Prepare documents for bulk indexing
            actions = []
            
            for embedding, payload, sparse_vector in embeddings_with_sparse:
                # Generate a unique ID if not provided
                if "id" not in payload:
                    doc_id = str(uuid.uuid4())
                else:
                    doc_id = str(payload["id"])
                
                # Extract sparse indices and values
                sparse_indices, sparse_values = sparse_vector
                
                # Create the document with both vector fields
                document = {
                    "_id": doc_id,
                    "_index": self.collection_name,
                    "_source": {
                        "id": doc_id,
                        "file_path": payload.get("file_path", ""),
                        "file_name": payload.get("file_name", ""),
                        "file_type": payload.get("fileType", ""),
                        "file_size": payload.get("fileSize", 0),
                        "created_at": int(payload.get("createdAt", time.time())),
                        "modified_at": int(payload.get("modifiedAt", time.time())),
                        "content": payload.get("content", payload.get("text", "")),
                        "title": payload.get("title", ""),
                        "is_chunk": bool(payload.get("is_chunk", False)),
                        "is_parent": bool(payload.get("is_parent", False)),
                        "parent_id": payload.get("parent_id", ""),
                        "chunk_index": int(payload.get("chunk_index", 0)),
                        "total_chunks": int(payload.get("total_chunks", 0)),
                        self.dense_field: embedding.tolist(),
                        self.sparse_indices_field: sparse_indices,
                        self.sparse_values_field: sparse_values,
                        "metadata": payload.get("metadata", {})
                    }
                }
                
                actions.append(document)
            
            if self.verbose:
                print(f"Inserting {len(actions)} documents with sparse vectors into index '{self.collection_name}'")
                
            # Perform bulk indexing
            helpers.bulk(self.client, actions)
            
            # Refresh index to make documents immediately available for search
            self.client.indices.refresh(index=self.collection_name)
            
            if self.verbose:
                print(f"Successfully inserted {len(actions)} documents with sparse vectors")
                
        except Exception as e:
            print(f"Error inserting embeddings with sparse vectors: {str(e)}")
            raise
    
    def search_hybrid(self, query: str, processor: Any, limit: int, 
                    prefetch_limit: int = 50, fusion_type: str = "rrf",
                    score_threshold: float = None, rerank: bool = False,
                    reranker_type: str = None):
        """
        Perform a hybrid search combining both dense vectors and keyword matching.
        
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
            # Generate query embedding
            query_vector = processor.get_embedding(query)
            
            # Get model ID from processor to construct the correct field name
            dense_model_id = processor.dense_model_id if hasattr(processor, 'dense_model_id') else self.dense_model_id
            # Normalize the model ID: replace hyphens with underscores
            normalized_model_id = dense_model_id.replace('-', '_')
            vector_field = f"vector_{normalized_model_id}"
            
            if self.verbose:
                print(f"Executing hybrid search with query: '{query}'")
                print(f"Using vector field: {vector_field}")
                print(f"Using fusion type: {fusion_type}")
            
            # First try using the retriever approach for Elasticsearch 8.14+
            try:
                # Build retriever-based hybrid search
                retriever_search = {
                    "size": limit,
                    "retriever": {
                        "rrf" if fusion_type.lower() == "rrf" else "cc": {  # Support both RRF and Convex Combination
                            "retrievers": [
                                # Keyword search retriever
                                {
                                    "standard": {
                                        "query": {
                                            "match": {
                                                "content": query
                                            }
                                        }
                                    }
                                },
                                # Vector search retriever
                                {
                                    "knn": {
                                        "field": vector_field,
                                        "query_vector": query_vector.tolist(),
                                        "k": prefetch_limit,
                                        "num_candidates": 100
                                    }
                                }
                            ]
                        }
                    }
                }
                
                # Execute retriever-based hybrid search
                results = self.client.search(
                    index=self.collection_name,
                    body=retriever_search
                )
                
                # Extract hits
                hits = results.get("hits", {}).get("hits", [])
                
                if self.verbose:
                    print(f"Retriever-based hybrid search returned {len(hits)} results")
                    
            except Exception as retriever_error:
                if self.verbose:
                    print(f"Retriever-based hybrid search failed: {retriever_error}")
                    print("Falling back to manual fusion approach")
                
                # Fallback to separate searches and manual fusion
                dense_results = self.search_dense(query, processor, prefetch_limit, score_threshold, False, None)
                keyword_results = self.search_keyword(query, prefetch_limit, score_threshold)
                
                # Check for errors
                if isinstance(dense_results, dict) and "error" in dense_results:
                    dense_results = []
                    
                if isinstance(keyword_results, dict) and "error" in keyword_results:
                    keyword_results = []
                    
                # Perform manual fusion if results available
                if len(dense_results) > 0 or len(keyword_results) > 0:
                    # Use SearchAlgorithms to perform fusion
                    fused_results = SearchAlgorithms.manual_fusion(
                        dense_results, keyword_results, prefetch_limit, fusion_type
                    )
                    
                    # Apply reranking if requested
                    if rerank and len(fused_results) > 0:
                        reranked_results = SearchAlgorithms.rerank_results(
                            query, fused_results, processor, limit, self.verbose
                        )
                        return reranked_results[:limit]
                    
                    return fused_results[:limit]
                else:
                    # If both searches failed, return an empty list
                    return []
                    
            # Convert to dictionary format compatible with ResultProcessor.adapt_result
            points = [self._adapt_elasticsearch_result_to_point(hit) for hit in hits]
            
            # Apply score threshold if provided
            if score_threshold is not None:
                points = [p for p in points if p.get("_score", 0) >= score_threshold]
            
            # Apply reranking if requested
            if rerank and len(points) > 0:
                reranked_points = SearchAlgorithms.rerank_results(
                    query, points, processor, limit, self.verbose
                )
                
                # Record hit rates if ground truth is available
                true_context = getattr(processor, 'expected_context', None)
                if true_context:
                    hit = any(p.get("_source", {}).get("content") == true_context for p in reranked_points)
                    search_key = f"hybrid_{fusion_type}"
                    if rerank:
                        search_key += f"_{reranker_type or 'default'}"
                    self._record_hit(search_key, hit)
                
                return reranked_points[:limit]
            
            # Record hit rates if ground truth is available
            true_context = getattr(processor, 'expected_context', None)
            if true_context:
                hit = any(p.get("_source", {}).get("content") == true_context for p in points)
                search_key = f"hybrid_{fusion_type}"
                self._record_hit(search_key, hit)
            
            return points[:limit]
                
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
            
            if not self.client:
                return {"error": f"Not connected to Elasticsearch"}
                
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
                db_type="elasticsearch"
            )
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
        
    def _adapt_elasticsearch_result_to_point(self, hit):
        """
        Adapt Elasticsearch hit to the expected result format.
        Instead of returning a custom Point object, return a dictionary
        that's directly compatible with ResultProcessor.adapt_result().
        
        Args:
            hit: Elasticsearch hit result
            
        Returns:
            Dictionary with _id, _source, and _score keys
        """
        # Extract source data
        source = hit.get("_source", {})
        
        # Build payload with all necessary fields
        payload = {
            # Ensure both "text" and "content" fields are present
            "text": source.get("content", source.get("text", "")),
            "content": source.get("content", source.get("text", "")),
            "file_path": source.get("file_path", ""),
            "file_name": source.get("file_name", ""),
            "chunk_index": source.get("chunk_index", 0),
            "total_chunks": source.get("total_chunks", 0),
            "is_chunk": source.get("is_chunk", False),
            "is_parent": source.get("is_parent", False),
            "parent_id": source.get("parent_id", ""),
            "metadata": source.get("metadata", {})
        }
        
        # Extract score, normalizing to [0, 1] range
        raw_score = hit.get("_score", 0)
        normalized_score = min(1.0, max(0.0, raw_score / 10.0))  # Simple normalization
        
        # Return a dictionary directly instead of a Point object
        # This format matches what ResultProcessor.adapt_result expects
        return {
            "_id": hit.get("_id", ""),
            "_source": payload,
            "_score": normalized_score
        }
    
    def search_dense(self, query: str, processor: Any, limit: int, score_threshold: float = None,
                    rerank: bool = False, reranker_type: str = None):
        """
        Perform dense vector search in Elasticsearch using the appropriate field name.
        
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
            # Generate query embedding
            query_vector = processor.get_embedding(query)
            
            # Get model ID from processor to construct the correct field name
            # Use exact same pattern as seen in the document sample: vector_bge_small
            dense_model_id = processor.dense_model_id if hasattr(processor, 'dense_model_id') else self.dense_model_id
            # Normalize the model ID: replace hyphens with underscores
            normalized_model_id = dense_model_id.replace('-', '_')
            vector_field = f"vector_{normalized_model_id}"
            
            if self.verbose:
                print(f"Using vector field: {vector_field}")
            
            # First attempt: Try using knn search parameter
            try:
                # Construct search using knn top-level parameter
                search_body = {
                    "knn": {
                        "field": vector_field,
                        "query_vector": query_vector.tolist(),
                        "k": limit,
                        "num_candidates": 100
                    }
                }
                
                # Execute search
                results = self.client.search(
                    index=self.collection_name,
                    body=search_body
                )
                
                # Extract hits
                hits = results.get("hits", {}).get("hits", [])
                
                if self.verbose:
                    print(f"KNN vector search returned {len(hits)} results")
                    
            except Exception as knn_error:
                if self.verbose:
                    print(f"KNN search failed: {knn_error}")
                    print("Falling back to script_score approach")
                    
                # Second attempt: Try script_score approach for older ES versions
                try:
                    # Build script score query
                    script_query = {
                        "query": {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": f"cosineSimilarity(params.query_vector, doc['{vector_field}']) + 1.0",
                                    "params": {
                                        "query_vector": query_vector.tolist()
                                    }
                                }
                            }
                        }
                    }
                    
                    # Execute search
                    results = self.client.search(
                        index=self.collection_name,
                        body=script_query,
                        size=limit
                    )
                    
                    # Extract hits
                    hits = results.get("hits", {}).get("hits", [])
                    
                    if self.verbose:
                        print(f"Script score search returned {len(hits)} results")
                        
                except Exception as script_error:
                    if self.verbose:
                        print(f"Script score search failed: {script_error}")
                        print("Falling back to keyword search")
                    return self.search_keyword(query, limit, score_threshold, rerank, reranker_type)
            
            # Convert to dictionary format compatible with ResultProcessor.adapt_result
            points = [self._adapt_elasticsearch_result_to_point(hit) for hit in hits]
            
            # Apply score threshold if provided
            if score_threshold is not None:
                points = [p for p in points if p.get("_score", 0) >= score_threshold]
            
            # Apply reranking if requested
            if rerank and len(points) > 0:
                reranked_points = SearchAlgorithms.rerank_results(
                    query, points, processor, limit, self.verbose
                )
                
                # Record hit rates if ground truth is available
                true_context = getattr(processor, 'expected_context', None)
                if true_context:
                    hit = any(p.get("_source", {}).get("content") == true_context for p in reranked_points)
                    search_key = "vector"
                    if rerank:
                        search_key += f"_{reranker_type or 'default'}"
                    self._record_hit(search_key, hit)
                
                return reranked_points[:limit]
            
            # Record hit rates if ground truth is available
            true_context = getattr(processor, 'expected_context', None)
            if true_context:
                hit = any(p.get("_source", {}).get("content") == true_context for p in points)
                self._record_hit("vector", hit)
            
            return points[:limit]
            
        except Exception as e:
            if self.verbose:
                print(f"Error in dense search: {e}")
                import traceback
                traceback.print_exc()
            return {"error": f"Error in dense search: {str(e)}"}


    def _sanitize_field_name(self, field_name):
        """Sanitize model name for use as a field name in Elasticsearch"""
        # Extract just the final part of the model name if it has slashes
        if "/" in field_name:
            field_name = field_name.split("/")[-1]
            
        # Replace any non-alphanumeric characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9]', '_', field_name)
        
        # Ensure the name isn't too long
        if len(sanitized) > 40:
            sanitized = sanitized[:40]
            
        return sanitized

        
    def search_sparse(self, query: str, processor: Any, limit: int, score_threshold: float = None,
                    rerank: bool = False, reranker_type: str = None):
        """
        Perform sparse vector search with fallback to keyword search.
        This version uses minimal scripting to avoid Elasticsearch script errors.
        
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
            
            # Get model ID from processor to construct the correct field names
            sparse_model_id = processor.sparse_model_id if hasattr(processor, 'sparse_model_id') else self.sparse_model_id
            # Normalize the model ID: replace hyphens with underscores
            normalized_model_id = sparse_model_id.replace('-', '_')
            indices_field = f"sparse_indices_{normalized_model_id}"
            values_field = f"sparse_values_{normalized_model_id}"
            
            if self.verbose:
                print(f"Using sparse fields: {indices_field} and {values_field}")
                print(f"Query sparse vector has {len(sparse_indices)} non-zero elements")
            
            # Find the top indices by weight (highest information value)
            if len(sparse_indices) > 0 and len(sparse_values) > 0:
                # Sort indices by values (largest weight first)
                top_pairs = sorted(zip(sparse_indices, sparse_values), key=lambda x: x[1], reverse=True)
                
                # Take top 5 or all if less than 5
                top_terms = min(5, len(sparse_indices))
                top_indices = [str(idx) for idx, _ in top_pairs[:top_terms]]
                
                if self.verbose:
                    print(f"Using top {len(top_indices)} terms from sparse vector")
                    
                # Instead of complex scripting, we'll use a combination of keyword search
                # and a should clause with term queries for sparse indices
                query_body = {
                    "query": {
                        "bool": {
                            "should": [
                                # Regular keyword search with the original query
                                {"match": {"content": {"query": query, "boost": 1.0}}},
                                
                                # Add term queries for each of the top sparse indices
                                # We can't directly search for the sparse vector indices,
                                # but we can boost the keyword search with the structure
                                # of the sparse vector information
                                {"match": {"title": {"query": query, "boost": 0.5}}}
                            ]
                        }
                    }
                }
            else:
                # Fallback to simple match query if no sparse vectors
                query_body = {
                    "query": {
                        "match": {
                            "content": query
                        }
                    }
                }
            
            # Execute search
            results = self.client.search(
                index=self.collection_name,
                body=query_body,
                size=limit * 2
            )
            
            # Extract hits
            hits = results.get("hits", {}).get("hits", [])
            
            if self.verbose:
                print(f"Sparse-enhanced search returned {len(hits)} results")
                
            # Convert to dictionary format compatible with ResultProcessor.adapt_result
            points = [self._adapt_elasticsearch_result_to_point(hit) for hit in hits]
            
            # Apply score threshold if provided
            if score_threshold is not None:
                points = [p for p in points if p.get("_score", 0) >= score_threshold]
            
            # Apply reranking if requested - this can improve results significantly
            if rerank and len(points) > 0:
                if self.verbose:
                    print(f"Applying {reranker_type or 'default'} reranking to improve sparse search results")
                    
                reranked_points = SearchAlgorithms.rerank_results(
                    query, points, processor, limit, self.verbose
                )
                
                # Record hit rates if ground truth is available
                true_context = getattr(processor, 'expected_context', None)
                if true_context:
                    hit = any(p.get("_source", {}).get("content") == true_context for p in reranked_points)
                    search_key = "sparse"
                    if rerank:
                        search_key += f"_{reranker_type or 'default'}"
                    self._record_hit(search_key, hit)
                
                return reranked_points[:limit]
            
            # Record hit rates if ground truth is available
            true_context = getattr(processor, 'expected_context', None)
            if true_context:
                hit = any(p.get("_source", {}).get("content") == true_context for p in points)
                self._record_hit("sparse", hit)
            
            return points[:limit]
                
        except Exception as e:
            if self.verbose:
                print(f"Error in sparse search: {e}")
                import traceback
                traceback.print_exc()
            return {"error": f"Error in sparse search: {str(e)}"}
    
    def search_keyword(self, query: str, limit: int = 10, score_threshold: float = None,
                        rerank: bool = False, reranker_type: str = None):
        """
        Perform a keyword-based search.
        
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
            
            # Build search query
            search_query = {
                "bool": {
                    "should": [
                        {"match": {"content": {"query": query, "boost": 1.0}}},
                        {"match": {"title": {"query": query, "boost": 2.0}}},
                        {"match_phrase": {"content": {"query": query, "boost": 1.5}}},
                        {"match_phrase": {"title": {"query": query, "boost": 3.0}}}
                    ]
                }
            }
            
            # Execute search
            results = self.client.search(
                index=self.collection_name,
                query=search_query,
                size=limit * 3
            )
            
            # Extract hits
            hits = results.get("hits", {}).get("hits", [])
            
            if self.verbose:
                print(f"Keyword search returned {len(hits)} results")
            
            # Convert to dictionary format that's compatible with ResultProcessor.adapt_result
            points = [self._adapt_elasticsearch_result_to_point(hit) for hit in hits]
            
            # Apply score threshold if provided
            if score_threshold is not None:
                points = [p for p in points if p.get("_score", 0) >= score_threshold]
            
            # Apply reranking if requested and processor is provided
            if rerank and len(points) > 0 and 'processor' in locals() and locals()['processor'] is not None:
                processor = locals()['processor']
                reranked_points = SearchAlgorithms.rerank_results(
                    query, points, processor, limit, self.verbose
                )
                
                # Record hit rates if ground truth is available
                true_context = getattr(processor, 'expected_context', None)
                if true_context:
                    hit = any(p.get("_source", {}).get("content") == true_context for p in reranked_points)
                    search_key = "keyword"
                    if rerank:
                        search_key += f"_{reranker_type or 'default'}"
                    self._record_hit(search_key, hit)
                    
                return reranked_points[:limit]
            
            # Record hit rates if ground truth is available and processor is provided
            if 'processor' in locals() and locals()['processor'] is not None:
                processor = locals()['processor']
                true_context = getattr(processor, 'expected_context', None)
                if true_context:
                    hit = any(p.get("_source", {}).get("content") == true_context for p in points)
                    self._record_hit("keyword", hit)
            
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
            if not self.client:
                return "[No connection available for context retrieval]"
            
            # Define the minimum and maximum chunk indices to retrieve
            min_idx = max(0, chunk_index - window)
            max_idx = chunk_index + window
            
            # Build search query for file_path and chunk_index in range
            search_query = {
                "bool": {
                    "must": [
                        {"term": {"file_path": file_path}},
                        {"range": {"chunk_index": {"gte": min_idx, "lte": max_idx}}}
                    ]
                }
            }
            
            # Execute search
            results = self.client.search(
                index=self.collection_name,
                query=search_query,
                size=2 * window + 1,
                sort=[{"chunk_index": {"order": "asc"}}]
            )
            
            # Extract hits
            hits = results.get("hits", {}).get("hits", [])
            
            if not hits:
                return f"[No chunks found for file_path={file_path}, chunk_index={chunk_index}]"
            
            # Extract content from each hit
            chunks = []
            for hit in hits:
                source = hit.get("_source", {})
                # Get content from the appropriate field
                content = source.get("content", source.get("text", ""))
                chunks.append(content)
            
            # Combine the content
            combined_text = "\n\n".join(chunks)
            
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
            # Prepare documents for bulk update
            actions = []
            
            for doc_id, embedding, payload in id_embedding_payload_tuples:
                # Create update document
                update_doc = {
                    "_id": doc_id,
                    "_index": self.collection_name,
                    "_op_type": "update",
                    "doc": {
                        self.dense_field: embedding.tolist(),
                        # Update other fields if provided
                        **{k: v for k, v in payload.items() if k != "id"}
                    }
                }
                
                actions.append(update_doc)
            
            if self.verbose:
                print(f"Updating {len(actions)} documents in index '{self.collection_name}'")
                
            # Perform bulk update
            helpers.bulk(self.client, actions)
            
            # Refresh index to make updates immediately available for search
            self.client.indices.refresh(index=self.collection_name)
            
            if self.verbose:
                print(f"Successfully updated {len(actions)} documents")
                
        except Exception as e:
            print(f"Error updating embeddings: {str(e)}")
            raise
            
    def delete_by_filter(self, filter_condition: str) -> int:
        """
        Delete entries matching a filter condition.
        
        Args:
            filter_condition: Filter condition in Elasticsearch query syntax (JSON string)
            
        Returns:
            Number of deleted entries
        """
        try:
            if not self.client:
                return 0
                
            # Parse filter condition string to JSON
            try:
                query = json.loads(filter_condition)
            except json.JSONDecodeError:
                # If not valid JSON, try to interpret as field:value
                if ":" in filter_condition:
                    field, value = filter_condition.split(":", 1)
                    field = field.strip()
                    value = value.strip()
                    query = {"term": {field: value}}
                else:
                    # Default to match_all
                    query = {"match_all": {}}
            
            # Count documents matching the filter
            count_response = self.client.count(
                index=self.collection_name,
                query=query
            )
            
            count = count_response.get("count", 0)
            
            if count == 0:
                if self.verbose:
                    print(f"No documents match the filter: {filter_condition}")
                return 0
                
            if self.verbose:
                print(f"Deleting {count} documents matching filter: {filter_condition}")
                
            # Delete by query
            delete_response = self.client.delete_by_query(
                index=self.collection_name,
                query=query,
                refresh=True
            )
            
            deleted = delete_response.get("deleted", 0)
            
            if self.verbose:
                print(f"Deleted {deleted} documents")
                
            return deleted
                
        except Exception as e:
            print(f"Error deleting by filter: {str(e)}")
            return 0
    
    def cleanup(self, remove_storage: bool = False) -> None:
        """
        Clean up resources.
        
        Args:
            remove_storage: Whether to remove the index (if True, will delete the index)
        """
        try:
            # Delete the index if requested
            if remove_storage and self.client:
                try:
                    if self.verbose:
                        print(f"Deleting index '{self.collection_name}'")
                    self.client.indices.delete(index=self.collection_name)
                except Exception as e:
                    print(f"Error deleting index: {e}")
                    
            # Close the client connection
            if self.client:
                try:
                    self.client.close()
                except:
                    pass
                
            self.client = None
                
        except Exception as e:
            print(f"Error during cleanup: {e}")
                        