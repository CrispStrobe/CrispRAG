import os
import time
import json
import shutil
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set, Iterator, Union, Literal
import numpy as np

from vector_db_interface import VectorDBInterface
from utils import TextProcessor, ResultProcessor, SearchAlgorithms, EmbeddingUtils, GeneralUtils
from mlx_utils import generate_sparse_vector

# Try to import Qdrant client
try:
    print("Importing QdrantClient...")

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


class QdrantManager(VectorDBInterface):
    """Manager for Qdrant vector database operations with enhanced search capabilities"""
    
    def __init__(self, 
                host: str = "localhost", 
                port: int = 6333, 
                collection_name: str = "documents",
                vector_size: int = 384,
                storage_path: str = None,
                verbose: bool = False,
                dense_model_id: str = "bge-small",
                sparse_model_id: str = "distilbert-splade"):
        """
        Initialize QdrantManager with model-specific vector configuration.
        
        Args:
            host: Qdrant host
            port: Qdrant port
            collection_name: Collection name
            vector_size: Vector dimension
            storage_path: Storage path for local mode
            verbose: Verbose output
            dense_model_id: ID of dense model for naming vectors
            sparse_model_id: ID of sparse model for naming vectors
        """
        if not qdrant_available:
            raise ImportError("Qdrant client not available. Install with: pip install qdrant-client")
            
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_dim = vector_size
        self.storage_path = storage_path
        self.verbose = verbose
        self.client = None
        self.is_remote = False
        
        # Store the original model IDs without sanitizing
        self.dense_model_id = dense_model_id
        self.sparse_model_id = sparse_model_id
        
        # Performance tracking
        self._hit_rates = {}
        
        self.connect()

    def is_local(self):
        """Check if we're running in local mode"""
        return not self.is_remote and self.storage_path is not None
    
    def _get_vector_names(self, dense_model_id=None, sparse_model_id=None):
        """Get standardized vector names for the given model IDs"""
        dense_id = dense_model_id or self.dense_model_id
        sparse_id = sparse_model_id or self.sparse_model_id
        
        # Extract just the final part of the model name if it has slashes
        # This gives us a more concise name without losing the essential identifier
        if "/" in dense_id:
            # For models like "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            # Just use the part after the last slash
            dense_base = dense_id.split("/")[-1]
        else:
            dense_base = dense_id
            
        if "/" in sparse_id:
            sparse_base = sparse_id.split("/")[-1]
        else:
            sparse_base = sparse_id
        
        # Now sanitize the base names
        dense_base = re.sub(r'[^a-zA-Z0-9]', '_', dense_base)
        sparse_base = re.sub(r'[^a-zA-Z0-9]', '_', sparse_base)
        
        # Ensure names don't exceed reasonable lengths
        max_name_length = 40
        if len(dense_base) > max_name_length:
            dense_base = dense_base[:max_name_length]
        if len(sparse_base) > max_name_length:
            sparse_base = sparse_base[:max_name_length]
        
        dense_vector_name = f"dense_{dense_base}"
        sparse_vector_name = f"sparse_{sparse_base}"
        
        if self.verbose:
            print(f"Created vector names from models:")
            print(f"  Dense model: {dense_id} → {dense_vector_name}")
            print(f"  Sparse model: {sparse_id} → {sparse_vector_name}")
        
        return dense_vector_name, sparse_vector_name
        
    def connect(self):
        """Connect to Qdrant using either local storage or remote server"""
        try:
            # Determine if we're using a remote server or local storage
            self.is_remote = self.host != "localhost" or self.port != 6333
            
            if self.is_remote:
                # Connect to remote Qdrant server
                if self.verbose:
                    print(f"Connecting to Qdrant server at {self.host}:{self.port}")
                    
                self.client = QdrantClient(host=self.host, port=self.port)
                
                # Test the connection
                try:
                    collections = self.client.get_collections()
                    if self.verbose:
                        print(f"Connected to Qdrant server. Collections: {[c.name for c in collections.collections]}")
                except Exception as e:
                    print(f"Error connecting to Qdrant server: {str(e)}")
                    
                    # Only fall back to local mode if storage_path is provided
                    if self.storage_path:
                        print(f"Falling back to local storage mode")
                        self.is_remote = False
                    else:
                        raise  # Re-raise the exception if we can't fall back
            
            # If not using remote (either by choice or after fallback)
            if not self.is_remote:
                # Check if storage path is provided
                if self.storage_path:
                    # Use provided storage path
                    storage_path = self.storage_path
                else:
                    # Create a persistent directory in the current working directory
                    storage_path = os.path.join(os.getcwd(), "qdrant_storage")
                
                # Ensure directory exists
                os.makedirs(storage_path, exist_ok=True)
                
                if self.verbose:
                    print(f"Using local storage at: {storage_path}")
                    
                # Initialize client with path for local mode
                self.client = QdrantClient(path=storage_path)
                
                # Store the storage path being used
                self.storage_path = storage_path
                
                # Test the client by getting collections
                collections = self.client.get_collections()
                
                if self.verbose:
                    print(f"Connected to local Qdrant storage. Collections: {[c.name for c in collections.collections]}")
        except Exception as e:
            print(f"Error initializing Qdrant: {str(e)}")
            print("You may need to install Qdrant client: pip install qdrant-client")
            raise
        
    def create_collection(self, recreate: bool = False) -> None:
        """Create Qdrant collection with support for both dense and sparse vectors"""
        try:
            # Import Qdrant models
            from qdrant_client.models import VectorParams, Distance, SparseVectorParams, SparseIndexParams
            
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)

            # Create vector name based on model ID
            dense_vector_name, sparse_vector_name = self._get_vector_names()
            
            if self.verbose:
                print(f"Using vector names: {dense_vector_name} (dense) and {sparse_vector_name} (sparse)")

            if collection_exists:
                if recreate:
                    if self.verbose:
                        print(f"Recreating collection '{self.collection_name}'...")
                    self.client.delete_collection(self.collection_name)
                else:
                    if self.verbose:
                        print(f"Collection '{self.collection_name}' already exists.")
                    return

            if self.verbose:
                print(f"Creating collection '{self.collection_name}' with vector size {self.vector_dim}")

            # Create collection with dense vectors
            vectors_config = {
                dense_vector_name: VectorParams(
                    size=self.vector_dim,
                    distance=Distance.COSINE
                )
            }
            
            # Create sparse vectors config
            sparse_vectors_config = {
                sparse_vector_name: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            }
            
            # Create collection with both dense and sparse vectors
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vectors_config,
                    sparse_vectors_config=sparse_vectors_config
                )
                
                if self.verbose:
                    print(f"Successfully created collection with dense and sparse vectors")
            except Exception as e:
                # Fallback for compatibility with older clients
                if self.verbose:
                    print(f"Error with integrated creation: {e}")
                    print("Falling back to two-step creation")
                    
                # Create with just dense vectors first
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vectors_config
                )
                
                # Then add sparse vectors separately
                try:
                    self.client.create_sparse_vector(
                        collection_name=self.collection_name,
                        vector_name=sparse_vector_name,
                        on_disk=False
                    )
                    if self.verbose:
                        print(f"Added sparse vector configuration: {sparse_vector_name}")
                except Exception as e2:
                    # Handle older clients that might not have create_sparse_vector
                    if "AttributeError" in str(e2):
                        try:
                            # Try alternative approach for older clients
                            self.client.update_collection(
                                collection_name=self.collection_name,
                                sparse_vectors_config=sparse_vectors_config
                            )
                            if self.verbose:
                                print(f"Added sparse vector configuration using update_collection")
                        except Exception as e3:
                            print(f"Warning: Could not add sparse vectors: {e3}")
                            print("Sparse search may not work properly.")
                    else:
                        print(f"Warning: Could not add sparse vectors: {e2}")
                        print("Sparse search may not work properly.")

            # Create payload indexes for better filtering performance
            try:
                # Create text index for full-text search
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="text",
                    field_schema="text"  # Simplified schema that works across versions
                )
                
                # Create keyword indexes for exact matching
                for field in ["file_name", "file_path"]:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema="keyword"
                    )
                    
                if self.verbose:
                    print(f"✅ Collection '{self.collection_name}' created successfully with indexes")
                    
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Could not create payload indexes: {e}")
                    print(f"✅ Collection '{self.collection_name}' created successfully (without indexes)")

        except Exception as e:
            print(f"❌ Error creating collection '{self.collection_name}': {e}")
            raise

    # Implementation for retrieving context for a chunk
    def _retrieve_context_for_chunk(
        self, 
        file_path: str, 
        chunk_index: int, 
        window: int = 1
    ) -> str:
        """
        For a given chunk (identified by its file_path and chunk_index),
        retrieve the neighboring chunks from Qdrant (e.g. chunk_index-1 and +1),
        and return a combined text string.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

        # Build a filter for same file_path and chunk_index in [chunk_index - window, chunk_index + window]
        min_idx = max(0, chunk_index - window)
        max_idx = chunk_index + window

        query_filter = Filter(
            must=[
                FieldCondition(
                    key="file_path",
                    match=MatchValue(value=file_path)
                ),
                FieldCondition(
                    key="chunk_index",
                    range=Range(gte=min_idx, lte=max_idx)
                )
            ]
        )

        # Use scroll to get matching points
        scroll_result = self.client.scroll(
            collection_name=self.collection_name,
            limit=50,  # enough to get neighboring chunks
            scroll_filter=query_filter,
            with_payload=True
        )
        points = scroll_result[0]

        # Sort by chunk_index to ensure correct order
        sorted_points = sorted(
            points,
            key=lambda p: p.payload.get("chunk_index", 0)
        )

        # Concatenate the text from all chunks
        combined_text = ""
        for p in sorted_points:
            txt = p.payload.get("text", "")
            if txt.strip():
                if combined_text:
                    combined_text += "\n"
                combined_text += txt

        return combined_text

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            if not self.client:
                return {"error": "Not connected to Qdrant"}
                
            collection_info = self.client.get_collection(self.collection_name)
            points_count = self.client.count(self.collection_name).count
            
            # Get disk usage if possible
            disk_usage = None
            try:
                if self.storage_path:
                    collection_path = Path(f"{self.storage_path}/collections/{self.collection_name}")
                    if collection_path.exists():
                        disk_usage = sum(f.stat().st_size for f in collection_path.glob('**/*') if f.is_file())
            except Exception as e:
                if self.verbose:
                    print(f"Could not calculate disk usage: {str(e)}")
            
            # Safely get vector configurations
            vector_configs = {}
            try:
                if hasattr(collection_info.config.params, 'vectors'):
                    # Safely get vectors dictionary using model_dump() instead of dict()
                    if hasattr(collection_info.config.params.vectors, 'model_dump'):
                        vectors_dict = collection_info.config.params.vectors.model_dump()
                    elif hasattr(collection_info.config.params.vectors, 'dict'):
                        vectors_dict = collection_info.config.params.vectors.dict()
                    else:
                        vectors_dict = {}
                        
                    for name, config in vectors_dict.items():
                        # Safely access configuration
                        if isinstance(config, dict):
                            vector_configs[name] = {
                                "size": config.get("size"),
                                "distance": config.get("distance")
                            }
                        else:
                            vector_configs[name] = {"info": str(config)}
            except Exception as e:
                if self.verbose:
                    print(f"Error extracting vector configs: {e}")
            
            # Safely get sparse vector configurations
            sparse_vector_configs = {}
            try:
                if hasattr(collection_info.config.params, 'sparse_vectors'):
                    if hasattr(collection_info.config.params.sparse_vectors, 'model_dump'):
                        sparse_vectors_dict = collection_info.config.params.sparse_vectors.model_dump()
                    elif hasattr(collection_info.config.params.sparse_vectors, 'dict'):
                        sparse_vectors_dict = collection_info.config.params.sparse_vectors.dict()
                    else:
                        sparse_vectors_dict = {}
                    
                    sparse_vector_configs = {
                        name: {"type": "sparse"}
                        for name in sparse_vectors_dict.keys()
                    }
            except Exception as e:
                if self.verbose:
                    print(f"Error extracting sparse vector configs: {e}")
            
            # Include performance data if available
            performance = {}
            if hasattr(self, '_hit_rates') and self._hit_rates:
                performance["hit_rates"] = self._hit_rates
            
            return {
                "name": self.collection_name,
                "points_count": points_count,
                "disk_usage": disk_usage,
                "vector_configs": vector_configs,
                "sparse_vector_configs": sparse_vector_configs,
                "performance": performance
            }
        except Exception as e:
            if self.verbose:
                print(f"Error getting collection info: {str(e)}")
                import traceback
                traceback.print_exc()
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
    
    def insert_embeddings(self, embeddings_with_payloads: List[Tuple[np.ndarray, Dict[str, Any]]]) -> None:
        """Insert embeddings into Qdrant (dense only, with generated sparse)"""
        if not embeddings_with_payloads:
            return
            
        try:
            # Determine vector name based on model ID (fallback to default if not in payload)
            first_payload = embeddings_with_payloads[0][1]
            dense_model_id = first_payload.get("metadata", {}).get("embedder", self.dense_model_id)
            dense_vector_name, sparse_vector_name = self._get_vector_names(dense_model_id, self.sparse_model_id)
            
            if self.verbose:
                print(f"Using vector names: {dense_vector_name} (dense) and {sparse_vector_name} (sparse)")
            
            # Prepare points for insertion
            points = []
            for i, (embedding, payload) in enumerate(embeddings_with_payloads):
                # Generate a UUID for the point
                import uuid
                point_id = str(uuid.uuid4())
                
                # Generate sparse vector for the text
                text = payload.get("text", "")
                sparse_indices, sparse_values = generate_sparse_vector(text)
                
                # Create point with vectors in the correct format
                vector_dict = {
                    dense_vector_name: embedding.tolist(),
                    sparse_vector_name: {
                        "indices": sparse_indices,
                        "values": sparse_values
                    }
                }
                
                point = PointStruct(
                    id=point_id,
                    vector=vector_dict,  # Include both dense and sparse in vector dictionary
                    payload=payload
                )
                points.append(point)
            
            if self.verbose:
                print(f"Inserting {len(points)} points into collection '{self.collection_name}'")
                
            # Insert points in batches
            BATCH_SIZE = 100
            for i in range(0, len(points), BATCH_SIZE):
                batch = points[i:i+BATCH_SIZE]
                if self.verbose and len(points) > BATCH_SIZE:
                    print(f"Inserting batch {i//BATCH_SIZE + 1}/{(len(points)-1)//BATCH_SIZE + 1} ({len(batch)} points)")
                
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                
            if self.verbose:
                print(f"Successfully inserted {len(points)} points")
        except Exception as e:
            print(f"Error inserting embeddings: {str(e)}")
            raise

    def insert_embeddings_with_sparse(self, embeddings_with_sparse: List[Tuple[np.ndarray, Dict[str, Any], Tuple[List[int], List[float]]]]) -> None:
        """
        Insert embeddings with sparse vectors into Qdrant following documentation format.
        
        Args:
            embeddings_with_sparse: List of (dense_embedding, payload, sparse_embedding) tuples
        """
        if not embeddings_with_sparse:
            return
            
        try:
            # Import UUID for generating point IDs
            import uuid
            
            # Get model IDs from the first payload, but sanitize them for consistency
            first_payload = embeddings_with_sparse[0][1]
            dense_model_id = first_payload.get("metadata", {}).get("dense_embedder", self.dense_model_id)
            sparse_model_id = first_payload.get("metadata", {}).get("sparse_embedder", self.sparse_model_id)
            dense_vector_name, sparse_vector_name = self._get_vector_names(dense_model_id, sparse_model_id)
            
            if self.verbose:
                print(f"Using vector names: {dense_vector_name} (dense) and {sparse_vector_name} (sparse)")
                
            # Prepare points for insertion
            points = []
            for embedding, payload, sparse_vector in embeddings_with_sparse:
                # Generate a UUID for the point
                point_id = str(uuid.uuid4())
                
                # Extract sparse indices and values
                sparse_indices, sparse_values = sparse_vector
                
                # Format vector dictionary
                vector_dict = {
                    dense_vector_name: embedding.tolist(),
                    sparse_vector_name: {
                        "indices": sparse_indices,
                        "values": sparse_values
                    }
                }
                
                # Create point with the combined vector dictionary
                point = PointStruct(
                    id=point_id,
                    vector=vector_dict,
                    payload=payload
                )
                
                points.append(point)
            
            if self.verbose:
                print(f"Inserting {len(points)} points with sparse vectors into collection '{self.collection_name}'")
                
            # Insert points in batches
            BATCH_SIZE = 100
            for i in range(0, len(points), BATCH_SIZE):
                batch = points[i:i+BATCH_SIZE]
                if self.verbose and len(points) > BATCH_SIZE:
                    print(f"Inserting batch {i//BATCH_SIZE + 1}/{(len(points)-1)//BATCH_SIZE + 1} ({len(batch)} points)")
                
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                
            if self.verbose:
                print(f"Successfully inserted {len(points)} points with sparse vectors")
                    
        except Exception as e:
            print(f"Error during embeddings insertion: {str(e)}")
            import traceback
            traceback.print_exc()

    def _get_reranker(self, reranker_type: str):
        """
        Get a reranker for Qdrant results
        
        Since Qdrant doesn't have built-in rerankers like LanceDB, we'll use SearchAlgorithms helpers
        for reranking after we get the initial results.
        
        Args:
            reranker_type: Type of reranker to use ('rrf', 'linear', 'dbsf', 'colbert', 'cohere', 'jina', 'cross')
            
        Returns:
            Reranker function
        """
        # We don't implement actual rerankers here, just return the type for later processing
        return reranker_type.lower()

    def search_hybrid(self, query: str, processor: Any, limit: int, 
                    prefetch_limit: int = 50, fusion_type: str = "rrf",
                    score_threshold: float = None, rerank: bool = False,
                    reranker_type: str = None):
        """
        Perform a hybrid search combining both dense and sparse vectors with optional reranking
        
        Args:
            query: Original search query
            processor: Document processor with embedding capabilities
            limit: Maximum number of results to return
            prefetch_limit: Number of results to prefetch for fusion
            fusion_type: Type of fusion to use (rrf or dbsf)
            score_threshold: Minimum score threshold
            rerank: Whether to apply reranking as a third step
            reranker_type: Type of reranker to use (not directly used in Qdrant, but tracked for metrics)
        """
        if processor is None:
            return {"error": "Hybrid search requires an embedding model"}
        
        try:
            # Generate query vector
            query_vector = processor.get_embedding(query)
            
            # Generate sparse vector for the query
            sparse_indices, sparse_values = processor.get_sparse_embedding(query)
            
            # Get vector names
            dense_vector_name, sparse_vector_name = self._get_vector_names()
            
            # Calculate oversampling factor
            oversample_factor = 3 if rerank else 1
            fetch_limit = min(prefetch_limit, limit * oversample_factor)
            
            if self.verbose:
                print(f"Performing hybrid search for: '{query}'")
                if rerank:
                    print(f"Will oversample by factor {oversample_factor} for reranking")
            
            # Perform hybrid search using native fusion if available
            try:
                from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
                
                # Create prefetch list
                prefetch_list = [
                    # Dense vector prefetch
                    Prefetch(
                        query=query_vector.tolist(),
                        using=dense_vector_name,
                        limit=fetch_limit,
                    ),
                    # Sparse vector prefetch 
                    Prefetch(
                        query=SparseVector(
                            indices=sparse_indices,
                            values=sparse_values
                        ),
                        using=sparse_vector_name,
                        limit=fetch_limit,
                    )
                ]
                
                # Choose fusion method
                fusion_enum = Fusion.DBSF if fusion_type.lower() == "dbsf" else Fusion.RRF
                
                # Hybrid query
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=prefetch_list,
                    query=FusionQuery(fusion=fusion_enum),
                    limit=fetch_limit,
                    with_payload=True
                )
                
                points = response.points
                
                if self.verbose:
                    print(f"Hybrid search with fusion returned {len(points)} results")
                    
            except Exception as e:
                if self.verbose:
                    print(f"Modern fusion-based hybrid search failed: {e}")
                    print("Using manual hybrid search approach")
                
                # Perform separate searches and combine manually
                dense_results = self.search_dense(query, processor, fetch_limit, score_threshold)
                if isinstance(dense_results, dict) and "error" in dense_results:
                    dense_results = []
                
                sparse_results = self.search_sparse(query, processor, fetch_limit, score_threshold)
                if isinstance(sparse_results, dict) and "error" in sparse_results:
                    sparse_results = []
                    
                # Combine and rerank results using appropriate fusion strategy
                points = SearchAlgorithms.manual_fusion(
                    dense_results, 
                    sparse_results, 
                    fetch_limit, 
                    fusion_type
                )
            
            # Apply reranking if requested
            if rerank and len(points) > 0:
                if self.verbose:
                    print(f"Applying {reranker_type or 'default'} reranking to {len(points)} results")
                
                points = SearchAlgorithms.rerank_results(query, points, processor, limit, self.verbose)
                
                # Record metrics if ground truth is available
                true_context = getattr(processor, 'expected_context', None)
                if true_context:
                    hit = any(hasattr(p, "payload") and p.payload and p.payload.get("text") == true_context for p in points)
                    search_key = f"hybrid_{fusion_type}"
                    if rerank:
                        search_key += f"_{reranker_type or 'default'}"
                    self._record_hit(search_key, hit)
                
            # Apply score threshold if not already applied
            if score_threshold is not None:
                points = [p for p in points if ResultProcessor.get_score(p) >= score_threshold]
                
            # Limit to requested number
            points = points[:limit]
            
            return points
        
        except Exception as e:
            print(f"Error in hybrid search: {str(e)}")
            import traceback
            traceback.print_exc()
            return SearchAlgorithms.handle_search_error("qdrant", "hybrid", e, self.verbose)

    def search(self, query: str, search_type: str = "hybrid", limit: int = 10,
            processor: Any = None, prefetch_limit: int = 50, fusion_type: str = "rrf",
            relevance_tuning: bool = True, context_size: int = 300, 
            score_threshold: float = None, rerank: bool = False,
            reranker_type: str = None):
        """
        Enhanced search using dense, sparse, or hybrid search with improved relevance
        and optional reranking stage.
        
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
            reranker_type: Type of reranker to use (not directly used in Qdrant, but tracked for metrics)
            
        Returns:
            Dictionary with search results and metadata
        """
        try:
            query = query.strip()
            if not query:
                return {"error": "Empty query"}
            
            if self.verbose:
                print(f"Searching for '{query}' using {search_type} search" + 
                    (" with reranking" if rerank else ""))
            
            # Check if collection exists
            collections = self.client.get_collections().collections
            if not any(c.name == self.collection_name for c in collections):
                return {"error": f"Collection {self.collection_name} does not exist"}
            
            # Calculate oversampling factor
            oversample_factor = 3 if rerank else 1
            
            # Determine correct search method based on type
            if search_type.lower() in ["vector", "dense"]:
                # Check if processor is available for vector search
                if processor is None:
                    return {"error": "Vector search requires an embedding model"}
                points = self.search_dense(query, processor, limit * oversample_factor if rerank else limit, score_threshold)
            elif search_type.lower() == "sparse":
                # Check if processor is available for sparse search
                if processor is None:
                    return {"error": "Sparse search requires an embedding model"}
                points = self.search_sparse(query, processor, limit * oversample_factor if rerank else limit, score_threshold)
            elif search_type.lower() in ["keyword", "fts"]:
                # Keyword search doesn't require a processor
                points = self.search_keyword(query, limit * oversample_factor if rerank else limit, score_threshold)
            else:  # Default to hybrid
                # Check if processor is available for hybrid search
                if processor is None:
                    return {"error": "Hybrid search requires an embedding model"}
                points = self.search_hybrid(query, processor, limit, prefetch_limit, 
                                        fusion_type, score_threshold, rerank, reranker_type)
            
            # Check for errors
            if isinstance(points, dict) and "error" in points:
                return points
                
            # Apply reranking as a separate step if requested and not already done in the search method
            if rerank and search_type.lower() != "hybrid" and not isinstance(points, dict) and processor is not None:
                if self.verbose:
                    print(f"Applying {reranker_type or 'default'} reranking to {len(points)} results")
                
                points = SearchAlgorithms.rerank_results(query, points, processor, limit, self.verbose)
                
                # Record metrics if ground truth is available
                true_context = getattr(processor, 'expected_context', None)
                if true_context:
                    hit = any(hasattr(p, "payload") and p.payload and p.payload.get("text") == true_context for p in points)
                    search_key = f"{search_type}"
                    if rerank:
                        search_key += f"_{reranker_type or 'default'}"
                    self._record_hit(search_key, hit)
            
            # Format results with improved preview
            return self._format_search_results(points, query, search_type, processor, context_size)
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def search_dense(self, query: str, processor: Any, limit: int, score_threshold: float = None):
        """Perform dense vector search with consistent handling"""
        if processor is None:
            return {"error": "Dense search requires an embedding model"}
        
        try:
            # Get dense vector name with consistent sanitization
            dense_model_id = getattr(processor, 'dense_model_id', self.dense_model_id)
            dense_vector_name, _ = self._get_vector_names(dense_model_id, None)
            
            if self.verbose:
                print(f"Using vector name for dense search: {dense_vector_name}")
            
            # Generate dense embedding
            query_vector = processor.get_embedding(query)
            
            # Attempt to use different search methods in order of preference
            try:
                # Modern approach with specific vector name
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=(dense_vector_name, query_vector.tolist()),
                    limit=limit * 3,  # Get more results than needed for filtering
                    with_payload=True,
                    score_threshold=score_threshold
                )
                
                # Filter out problematic tiny chunks (less than 20 characters)
                filtered_results = []
                for point in search_result:
                    if hasattr(point, "payload") and point.payload and "text" in point.payload:
                        text = point.payload.get("text", "")
                        # Only include chunks with meaningful content
                        if len(text) >= 20:
                            filtered_results.append(point)
                
                if self.verbose:
                    print(f"Dense search returned {len(search_result)} raw results")
                    print(f"After filtering tiny chunks: {len(filtered_results)} results")
                
                # Record metrics if ground truth is available
                true_context = getattr(processor, 'expected_context', None)
                if true_context:
                    hit = any(hasattr(p, "payload") and p.payload and p.payload.get("text") == true_context for p in filtered_results)
                    self._record_hit("vector", hit)
                
                # Limit to requested number
                return filtered_results[:limit]
                
            except Exception as e:
                if self.verbose:
                    print(f"Standard dense search failed: {e}")
                    print("Trying alternative dense search methods...")
                
                # Try with query_points API
                try:
                    result = self.client.query_points(
                        collection_name=self.collection_name,
                        query=query_vector.tolist(),
                        using=dense_vector_name,
                        limit=limit * 3,  # Get more results than needed for filtering
                        with_payload=True,
                        score_threshold=score_threshold
                    )
                    
                    # Filter out problematic tiny chunks
                    filtered_results = []
                    for point in result.points:
                        if hasattr(point, "payload") and point.payload and "text" in point.payload:
                            text = point.payload.get("text", "")
                            # Only include chunks with meaningful content
                            if len(text) >= 20:
                                filtered_results.append(point)
                    
                    if self.verbose:
                        print(f"Dense search with query_points returned {len(result.points)} raw results")
                        print(f"After filtering tiny chunks: {len(filtered_results)} results")
                    
                    # Record metrics if ground truth is available
                    true_context = getattr(processor, 'expected_context', None)
                    if true_context:
                        hit = any(hasattr(p, "payload") and p.payload and p.payload.get("text") == true_context for p in filtered_results)
                        self._record_hit("vector", hit)
                    
                    # Limit to requested number
                    return filtered_results[:limit]
                    
                except Exception as e2:
                    # Last resort - try without named vectors
                    try:
                        search_result = self.client.search(
                            collection_name=self.collection_name,
                            query_vector=query_vector.tolist(),
                            limit=limit * 3,  # Get more results than needed for filtering
                            with_payload=True,
                            score_threshold=score_threshold
                        )
                        
                        # Filter out problematic tiny chunks
                        filtered_results = []
                        for point in search_result:
                            if hasattr(point, "payload") and point.payload and "text" in point.payload:
                                text = point.payload.get("text", "")
                                # Only include chunks with meaningful content
                                if len(text) >= 20:
                                    filtered_results.append(point)
                        
                        if self.verbose:
                            print(f"Dense search with legacy method returned {len(search_result)} raw results")
                            print(f"After filtering tiny chunks: {len(filtered_results)} results")
                        
                        # Record metrics if ground truth is available
                        true_context = getattr(processor, 'expected_context', None)
                        if true_context:
                            hit = any(hasattr(p, "payload") and p.payload and p.payload.get("text") == true_context for p in filtered_results)
                            self._record_hit("vector", hit)
                        
                        # Limit to requested number
                        return filtered_results[:limit]
                        
                    except Exception as e3:
                        return {"error": f"All dense search methods failed: {e3}"}
        
        except Exception as e:
            print(f"Error in dense search: {str(e)}")
            return {"error": f"Error in dense search: {str(e)}"}

    def search_sparse(self, query: str, processor: Any, limit: int, score_threshold: float = None):
        """
        Perform sparse vector search according to Qdrant documentation format
        """
        if processor is None:
            return {"error": "Sparse search requires an embedding model"}
        
        # Get vector names
        has_mlx_provider = (
            processor is not None
            and hasattr(processor, 'mlx_embedding_provider')
            and processor.mlx_embedding_provider is not None
        )
        dense_model_id = getattr(processor, 'dense_model_id', self.dense_model_id)
        sparse_model_id = getattr(processor, 'sparse_model_id', self.sparse_model_id)
        _, sparse_vector_name = self._get_vector_names(dense_model_id, sparse_model_id)
        
        if self.verbose:
            print(f"[DEBUG] Sparse search using vector name '{sparse_vector_name}'")

        try:
            # Generate sparse vector
            if has_mlx_provider:
                sparse_indices, sparse_values = processor.mlx_embedding_provider.get_sparse_embedding(query)
            else:
                sparse_indices, sparse_values = processor.get_sparse_embedding(query)
            
            # Warn if this appears to be just a bag of words
            if len(sparse_indices) < 5 or max(sparse_values) < 0.1:
                print(f"⚠️ WARNING: Sparse vector appears to be low quality (possibly just a bag of words).")
                print(f"This might significantly reduce search quality.")
                
            if self.verbose:
                print(f"[DEBUG] Created sparse vector with {len(sparse_indices)} non-zero terms")
                if len(sparse_indices) > 0:
                    print(f"Sample indices: {sparse_indices[:5]}")
                    print(f"Sample values: {sparse_values[:5]}")
            
            # Perform search according to Qdrant documentation format
            try:
                # Format the query according to Qdrant documentation
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=(sparse_vector_name, {
                        "indices": sparse_indices,
                        "values": sparse_values
                    }),
                    limit=limit,
                    with_payload=True,
                    score_threshold=score_threshold
                )
                
                if self.verbose:
                    print(f"[DEBUG] Sparse search returned {len(search_result)} results")
                
                # Record metrics if ground truth is available
                true_context = getattr(processor, 'expected_context', None)
                if true_context:
                    hit = any(hasattr(p, "payload") and p.payload and p.payload.get("text") == true_context for p in search_result)
                    self._record_hit("sparse", hit)
                
                return search_result
                
            except Exception as e:
                if self.verbose:
                    print(f"[ERROR] Standard sparse search failed: {e}")
                    print("Trying alternative sparse search methods...")
                
                # Try with NamedSparseVector format (newer Qdrant versions)
                try:
                    from qdrant_client.models import NamedSparseVector, SparseVector
                    search_result = self.client.search(
                        collection_name=self.collection_name,
                        query_vector=NamedSparseVector(
                            name=sparse_vector_name,
                            vector=SparseVector(
                                indices=sparse_indices,
                                values=sparse_values
                            )
                        ),
                        limit=limit,
                        with_payload=True,
                        score_threshold=score_threshold
                    )
                    
                    if self.verbose:
                        print(f"[DEBUG] Alternative sparse search returned {len(search_result)} results")
                    
                    # Record metrics if ground truth is available
                    true_context = getattr(processor, 'expected_context', None)
                    if true_context:
                        hit = any(hasattr(p, "payload") and p.payload and p.payload.get("text") == true_context for p in search_result)
                        self._record_hit("sparse", hit)
                    
                    return search_result
                    
                except Exception as e2:
                    # Last resort - try with query_points
                    try:
                        from qdrant_client.models import SparseVector
                        result = self.client.query_points(
                            collection_name=self.collection_name,
                            query=SparseVector(
                                indices=sparse_indices,
                                values=sparse_values
                            ),
                            using=sparse_vector_name,
                            limit=limit,
                            with_payload=True,
                            score_threshold=score_threshold
                        )
                        
                        if self.verbose:
                            print(f"[DEBUG] query_points sparse search returned {len(result.points)} results")
                        
                        # Record metrics if ground truth is available
                        true_context = getattr(processor, 'expected_context', None)
                        if true_context:
                            hit = any(hasattr(p, "payload") and p.payload and p.payload.get("text") == true_context for p in result.points)
                            self._record_hit("sparse", hit)
                        
                        return result.points
                    except Exception as e3:
                        return {"error": f"All sparse search methods failed: {e3}"}
        
        except Exception as e:
            print(f"Error generating sparse vectors: {e}")
            return SearchAlgorithms.handle_search_error("qdrant", "sparse", e, self.verbose)

    def search_keyword(self, query: str, limit: int = 10, score_threshold: float = None):
        """
        Perform a keyword-based search optimized for local Qdrant installations.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            score_threshold: Minimum score threshold
            
        Returns:
            List of search results
        """
        try:
            if not query.strip():
                return {"error": "Empty query"}
                
            if self.verbose:
                print(f"Performing keyword search for: '{query}'")
            
            # Create a payload filter for text matches
            try:
                from qdrant_client.models import Filter, FieldCondition, MatchText
                
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="text",
                            match=MatchText(text=query)
                        )
                    ]
                )
                
                # Detect if we're running in local mode
                is_local_mode = not self.is_remote and self.storage_path is not None
                
                # Try different search methods in order of preference
                matched_points = []
                
                # For local mode, try the direct scroll method first as it's most reliable
                if is_local_mode:
                    if self.verbose:
                        print("Running in local mode, starting with direct scroll method...")
                    
                    try:
                        # Direct scroll with manual filtering
                        all_results = self.client.scroll(
                            collection_name=self.collection_name,
                            limit=1000,  # Get a larger batch
                            with_payload=True
                        )
                        
                        all_points = all_results[0]
                        
                        if self.verbose:
                            print(f"Retrieved {len(all_points)} points for manual filtering")
                            
                        # Manually filter by keyword
                        query_lower = query.lower()
                        matched_points = []
                        
                        for point in all_points:
                            if hasattr(point, "payload") and point.payload and "text" in point.payload:
                                text = point.payload["text"].lower()
                                if query_lower in text:
                                    matched_points.append(point)
                        
                        if self.verbose:
                            print(f"Manual keyword filtering found {len(matched_points)} matches")
                            
                        # If we found matches, skip other methods
                        if matched_points:
                            if self.verbose:
                                print("Using results from direct scroll method")
                        else:
                            if self.verbose:
                                print("No matches found via direct scroll, trying other methods...")
                    
                    except Exception as e:
                        if self.verbose:
                            print(f"Direct scroll method failed: {e}")
                            print("Trying other search methods...")
                
                # If we're not in local mode or direct scroll didn't work, try the other methods
                if not matched_points:
                    # Method 1: Use scroll API with filter
                    if self.verbose:
                        print("Attempting scroll-based keyword search...")
                        
                    try:
                        # First try with filter parameter
                        try:
                            results = self.client.scroll(
                                collection_name=self.collection_name,
                                filter=query_filter,
                                limit=limit * 3,
                                with_payload=True
                            )
                            points = results[0]
                            if self.verbose:
                                print(f"Keyword search using scroll with filter= returned {len(points)} results")
                            matched_points = points
                        except Exception as scroll_error:
                            # Check if it's the specific error about unknown arguments
                            if "Unknown arguments" in str(scroll_error) and "filter" in str(scroll_error):
                                if self.verbose:
                                    print(f"Scroll with filter= parameter failed: {scroll_error}")
                                    print("Trying with scroll_filter= parameter...")
                                
                                # Try with scroll_filter parameter (newer Qdrant versions)
                                results = self.client.scroll(
                                    collection_name=self.collection_name,
                                    scroll_filter=query_filter,
                                    limit=limit * 3,
                                    with_payload=True
                                )
                                points = results[0]
                                if self.verbose:
                                    print(f"Keyword search using scroll with scroll_filter= returned {len(points)} results")
                                matched_points = points
                            else:
                                # Re-raise if it's a different error
                                raise
                            
                    except Exception as e:
                        if self.verbose:
                            print(f"Scroll-based keyword search failed: {e}")
                            print("Trying query_points method...")
                        
                        # Method 2: Try query_points
                        try:
                            try:
                                # First try with query_filter parameter
                                results = self.client.query_points(
                                    collection_name=self.collection_name,
                                    query_filter=query_filter,
                                    limit=limit * 3,
                                    with_payload=True
                                )
                                
                                points = results.points
                                
                                if self.verbose:
                                    print(f"Keyword search using query_points with query_filter= returned {len(points)} results")
                                    
                                matched_points = points
                            except Exception as query_error:
                                # Try with filter parameter if query_filter fails
                                if "Unknown arguments" in str(query_error) and "query_filter" in str(query_error):
                                    if self.verbose:
                                        print(f"query_points with query_filter= parameter failed: {query_error}")
                                        print("Trying with filter= parameter...")
                                    
                                    results = self.client.query_points(
                                        collection_name=self.collection_name,
                                        filter=query_filter,
                                        limit=limit * 3,
                                        with_payload=True
                                    )
                                    
                                    points = results.points
                                    
                                    if self.verbose:
                                        print(f"Keyword search using query_points with filter= returned {len(points)} results")
                                        
                                    matched_points = points
                                else:
                                    # Re-raise if it's a different error
                                    raise
                                
                        except Exception as e2:
                            if self.verbose:
                                print(f"Query-points keyword search failed: {e2}")
                                print("Trying search with filter method...")
                            
                            # Method 3: Last resort with search + filter
                            try:
                                # Get vector size for dummy vector
                                vector_size = self.vector_dim
                                
                                # Create a dummy vector (all zeros)
                                dummy_vector = [0.0] * vector_size
                                
                                # Get vector name - use first available
                                collection_info = self.client.get_collection(self.collection_name)
                                vector_names = []
                                
                                if hasattr(collection_info.config.params, 'vectors'):
                                    if hasattr(collection_info.config.params.vectors, 'keys'):
                                        vector_names = list(collection_info.config.params.vectors.keys())
                                    elif hasattr(collection_info.config.params.vectors, '__dict__'):
                                        vector_names = list(collection_info.config.params.vectors.__dict__.keys())
                                
                                if not vector_names:
                                    dense_vector_name, _ = self._get_vector_names()
                                    vector_names = [dense_vector_name]
                                vector_name = vector_names[0]
                                
                                try:
                                    # Try with query_filter parameter
                                    results = self.client.search(
                                        collection_name=self.collection_name,
                                        query_vector=(vector_name, dummy_vector),
                                        query_filter=query_filter,
                                        limit=limit * 3,
                                        with_payload=True
                                    )
                                    
                                    points = results
                                    
                                    if self.verbose:
                                        print(f"Keyword search using search+query_filter returned {len(points)} results")
                                        
                                    matched_points = points
                                except Exception as search_error:
                                    # Try with filter parameter if query_filter fails
                                    if "Unknown arguments" in str(search_error) and "query_filter" in str(search_error):
                                        if self.verbose:
                                            print(f"search with query_filter= parameter failed: {search_error}")
                                            print("Trying with filter= parameter...")
                                        
                                        results = self.client.search(
                                            collection_name=self.collection_name,
                                            query_vector=(vector_name, dummy_vector),
                                            filter=query_filter,
                                            limit=limit * 3,
                                            with_payload=True
                                        )
                                        
                                        points = results
                                        
                                        if self.verbose:
                                            print(f"Keyword search using search+filter returned {len(points)} results")
                                            
                                        matched_points = points
                                    else:
                                        # Re-raise if it's a different error
                                        raise
                            
                            except Exception as e3:
                                if self.verbose:
                                    print(f"All keyword search methods failed: {e3}")
                                return {"error": f"All keyword search methods failed: {e3}"}
                
                # If we have no points to process, return empty results
                if not matched_points:
                    if self.verbose:
                        print("No matches found for keyword search")
                    return []
                    
                # Define a simpler Point class for our processed results
                class ScoredPoint:
                    def __init__(self, id, payload, score):
                        self.id = id
                        self.payload = payload
                        self.score = score
                        self.version = 0  # Default version
                
                # Apply post-processing to improve scores and validate matches
                processed_points = []
                query_lower = query.lower()
                query_terms = query_lower.split()
                
                for point in matched_points:
                    # Extract the necessary attributes from the original point
                    point_id = getattr(point, "id", None)
                    payload = getattr(point, "payload", {})
                    
                    # Skip invalid points
                    if point_id is None or not payload:
                        continue
                    
                    # Get text content and validate the match
                    text = payload.get("text", "").lower()
                    if not text or query_lower not in text:
                        # Skip points that don't actually contain the query
                        if self.verbose:
                            print(f"Skipping point {point_id}: Does not actually contain '{query}'")
                        continue
                    
                    # Calculate a better relevance score
                    score = 0.0
                    
                    # Calculate score based on exact match, frequency, and position
                    if query_lower in text:
                        # Base score for exact match
                        score = 0.7
                        
                        # Boost for match at beginning
                        position = text.find(query_lower) / max(1, len(text))
                        position_boost = max(0, 0.3 - position * 0.5)  # Higher boost for earlier positions
                        score += position_boost
                        
                        # Frequency bonus (multiple occurrences)
                        freq = text.count(query_lower)
                        freq_boost = min(0.2, 0.05 * (freq - 1))  # Up to 0.2 for frequency
                        score += freq_boost
                    else:
                        # This shouldn't happen due to our earlier check, but just in case
                        continue
                    
                    # Create a new ScoredPoint object
                    processed_point = ScoredPoint(
                        id=point_id,
                        payload=payload,
                        score=score
                    )
                    
                    processed_points.append(processed_point)
                
                # Sort by score (descending)
                processed_points.sort(key=lambda p: p.score, reverse=True)
                
                # Apply score threshold if provided
                if score_threshold is not None:
                    processed_points = [p for p in processed_points if p.score >= score_threshold]
                
                # Limit to requested number
                points = processed_points[:limit]
                
                if self.verbose:
                    print(f"Final result: {len(points)} validated keyword matches")
                
                # Record metrics if ground truth is available - FIXED: only use if processor is provided
                # This was causing the error
                # if true_context := getattr(processor, 'expected_context', None):
                #     hit = any(hasattr(p, "payload") and p.payload and p.payload.get("text") == true_context for p in points)
                #     self._record_hit("keyword", hit)
                    
                return points
                
            except Exception as e:
                return {"error": f"Error creating keyword filter: {e}"}
        
        except Exception as e:
            print(f"Error in keyword search: {str(e)}")
            import traceback
            traceback.print_exc()
            from utils import SearchAlgorithms
            return SearchAlgorithms.handle_search_error("qdrant", "keyword", e, self.verbose)

    def _format_search_results(self, points, query, search_type, processor, context_size=300):
        """Format search results with improved preview using TextProcessor"""
        # Create a retriever function to pass to the formatter
        def context_retriever(file_path, chunk_index, window=1):
            return self._retrieve_context_for_chunk(file_path, chunk_index, window)
        
        return TextProcessor.format_search_results(
            points, 
            query, 
            search_type, 
            processor, 
            context_size,
            retriever=context_retriever,
            db_type="qdrant"  # Pass the db_type
        )

    def cleanup(self, remove_storage=False):
        """Clean up resources"""
        # Close the client if it exists
        if hasattr(self, 'client') and self.client:
            try:
                if hasattr(self.client, 'close'):
                    self.client.close()
            except:
                pass
                
        # Remove the storage directory only if requested
        if remove_storage and hasattr(self, 'storage_path') and self.storage_path and os.path.exists(self.storage_path):
            try:
                if self.verbose:
                    print(f"Removing storage directory: {self.storage_path}")
                shutil.rmtree(self.storage_path, ignore_errors=True)
            except Exception as e:
                print(f"Error cleaning up storage directory: {e}")