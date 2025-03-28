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
        
    def _get_embedder_name(self, model_id=None):
        """
        Get standardized embedder name for the given model ID, adapted for Meilisearch requirements.
        """
        model_id = model_id or self.dense_model_id
        
        # Extract just the final part of the model name if it has slashes
        # This gives us a more concise name without losing the essential identifier
        if "/" in model_id:
            # For models like "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            # Just use the part after the last slash
            model_base = model_id.split("/")[-1]
        else:
            model_base = model_id
            
        # Sanitize the base name
        model_base = re.sub(r'[^a-zA-Z0-9]', '_', model_base)
        
        # Ensure name doesn't exceed reasonable length
        max_name_length = 40
        if len(model_base) > max_name_length:
            model_base = model_base[:max_name_length]
        
        embedder_name = f"dense_{model_base}"
        
        if self.verbose:
            print(f"Created embedder name from model: {model_id} → {embedder_name}")
        
        return embedder_name

    def create_collection(self, recreate: bool = False) -> None:
        """Create Meilisearch index with vector search capabilities"""
        try:
            # Check if the index already exists (using more robust approach)
            exists = False
            try:
                # Directly try to get the index
                try:
                    self.client.get_index(self.collection_name)
                    exists = True
                    if self.verbose:
                        print(f"Index '{self.collection_name}' exists")
                except Exception as e:
                    if "index not found" in str(e).lower() or "not found" in str(e).lower():
                        exists = False
                        if self.verbose:
                            print(f"Index '{self.collection_name}' does not exist")
                    else:
                        # Try with get_indexes as fallback
                        indexes_response = self.client.get_indexes()
                        
                        if isinstance(indexes_response, dict) and 'results' in indexes_response:
                            exists = any(index.get('uid') == self.collection_name for index in indexes_response['results'])
                        else:
                            # Try as object
                            results = getattr(indexes_response, 'results', None)
                            if results:
                                exists = any(getattr(index, 'uid', None) == self.collection_name for index in results)
            except Exception as e:
                if self.verbose:
                    print(f"Error checking for index existence: {e}")
                    
            # If index exists and recreate is True, delete it
            if exists and recreate:
                if self.verbose:
                    print(f"Deleting existing index '{self.collection_name}'")
                try:
                    task = self.client.delete_index(self.collection_name)
                    # Wait for the task to complete if possible
                    self._wait_for_task(task)
                    exists = False
                except Exception as e:
                    print(f"Error deleting index: {e}")
                    
            # Create index if it doesn't exist
            if not exists:
                if self.verbose:
                    print(f"Creating index '{self.collection_name}' with primary key 'id'")
                task = self.client.create_index(self.collection_name, {'primaryKey': 'id'})
                # Wait for the task to complete
                self._wait_for_task(task)
                
            # Get the index
            self.index = self.client.index(self.collection_name)
            
            # Configure index settings to match the working implementation in search_04.py
            settings = {
                'searchableAttributes': [
                    'title', 
                    'content', 
                    'path', 
                    'filename',
                    'sparse_terms',  # For sparse vector simulation
                    'text'           # Add 'text' attribute to searchable attributes
                ],
                'filterableAttributes': [
                    'fileType', 
                    'extension', 
                    'path', 
                    'directory', 
                    'lastIndexed',
                    'is_chunk', 
                    'is_parent', 
                    'parent_id', 
                    'chunk_index'
                ],
                'sortableAttributes': [
                    'createdAt', 
                    'fileSize', 
                    'lastIndexed', 
                    'modifiedAt', 
                    'chunk_index'
                ],
                'rankingRules': [
                    'words',
                    'typo',
                    'proximity',
                    'attribute',
                    'sort',
                    'exactness'
                ]
            }
            
            # Update settings
            task = self.index.update_settings(settings)
            self._wait_for_task(task)
            
            # Configure vector settings for Meilisearch if we have the vector dimension
            if self.vector_dim > 0:
                # Use a consistent embedder naming approach
                embedder_name = self._get_embedder_name(self.dense_model_id)
                
                # Configure the embedder with userProvided source and dimensions
                # This is required by Meilisearch regardless of naming
                embedder_settings = {
                    embedder_name: {
                        'source': 'userProvided',
                        'dimensions': self.vector_dim
                    }
                }
                
                try:
                    if self.verbose:
                        print(f"Configuring embedder with settings: {embedder_settings}")
                        
                    # Try the dedicated method first (newer Meilisearch versions)
                    try:
                        task = self.index.update_embedders(embedder_settings)
                        self._wait_for_task(task)
                        if self.verbose:
                            print(f"Vector search enabled with {self.vector_dim} dimensions using embedder: {embedder_name}")
                    except AttributeError:
                        # Fall back to direct API call for older clients
                        import requests
                        response = requests.patch(
                            f"{self.url}/indexes/{self.collection_name}/settings/embedders",
                            headers={
                                "Content-Type": "application/json",
                                "Authorization": f"Bearer {self.api_key}" if self.api_key else None
                            },
                            json=embedder_settings
                        )
                        
                        if response.status_code == 202:
                            if self.verbose:
                                print(f"Vector search enabled via REST API with embedder: {embedder_name}")
                        else:
                            print(f"Failed to enable vector search via REST API: {response.text}")
                except Exception as e:
                    print(f"Error configuring vector search: {e}")
                    print("Documents will be indexed without vector search capabilities")
                    
            if self.verbose:
                print(f"Index '{self.collection_name}' configured successfully")
                
        except Exception as e:
            print(f"Error creating collection: {str(e)}")
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
        """
        Insert documents with dense vector embeddings into Meilisearch.
        
        Args:
            embeddings_with_payloads: List of (embedding, payload) tuples
        """
        if not embeddings_with_payloads:
            return
        
        try:
            if not self.index:
                raise ValueError(f"Index '{self.collection_name}' not initialized")
            
            # Get embedder name from the first payload
            first_payload = embeddings_with_payloads[0][1]
            dense_model_id = first_payload.get("metadata", {}).get("embedder", self.dense_model_id)
            embedder_name = self._get_embedder_name(dense_model_id)
            
            if self.verbose:
                print(f"Using embedder name: {embedder_name}")
            
            # Prepare documents for insertion with proper _vectors field
            documents = []
            for embedding, payload in embeddings_with_payloads:
                # Generate ID if not provided
                if 'id' not in payload:
                    doc_id = str(uuid.uuid4())
                else:
                    doc_id = payload['id']
                
                # Create document with all payload fields
                document = {
                    'id': doc_id,
                    **payload  # Include all payload fields
                }
                
                # Add vector embedding in the _vectors field with named embedder format
                document['_vectors'] = {
                    embedder_name: embedding.tolist()  # Use the model-based embedder name
                }
                
                documents.append(document)
            
            if self.verbose:
                print(f"Inserting {len(documents)} documents with vector embeddings")
            
            # Insert documents in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                
                if self.verbose and len(documents) > batch_size:
                    print(f"Inserting batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
                
                # Add documents
                task = self.index.add_documents(batch)
                self._wait_for_task(task)
                
            if self.verbose:
                print(f"Successfully inserted {len(documents)} documents with vector embeddings")
                
        except Exception as e:
            print(f"Error inserting documents with vectors: {str(e)}")
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
        Perform hybrid search combining dense vectors and keywords.
        """
        if processor is None:
            return {"error": "Hybrid search requires an embedding model"}
        
        try:
            # Generate query embedding
            query_vector = processor.get_embedding(query)
            
            # Get embedder name using the consistent naming approach
            dense_model_id = getattr(processor, 'dense_model_id', self.dense_model_id)
            embedder_name = self._get_embedder_name(dense_model_id)
            
            # Determine semantic ratio based on fusion type
            semantic_ratio = 0.5  # Default balanced ratio
            if fusion_type.lower() == 'vector':
                semantic_ratio = 0.9  # Mostly vector search
            elif fusion_type.lower() == 'keyword':
                semantic_ratio = 0.1  # Mostly keyword search
            
            if self.verbose:
                print(f"Executing hybrid search with semanticRatio={semantic_ratio}")
                print(f"Using embedder name: {embedder_name}")
            
            # Build search parameters using the model-based embedder name
            search_params = {
                'limit': limit * 2,
                'attributesToRetrieve': ['*'],
                'vector': query_vector.tolist(),
                'hybrid': {
                    'embedder': embedder_name,   # Use consistent model-based embedder name
                    'semanticRatio': semantic_ratio
                }
            }
            
            if score_threshold is not None:
                search_params['scoreThreshold'] = score_threshold
            
            # Execute hybrid search with both query text and vector
            results = self.index.search(query, search_params)
            
            # Extract hits
            hits = []
            if hasattr(results, 'hits'):
                hits = results.hits
            elif isinstance(results, dict) and 'hits' in results:
                hits = results['hits']
            
            if self.verbose:
                print(f"Hybrid search returned {len(hits)} results")
            
            # Format results
            formatted_results = []
            for hit in hits[:limit]:
                result = {}
                
                # Copy all fields
                if isinstance(hit, dict):
                    for k, v in hit.items():
                        result[k] = v
                else:
                    for attr in dir(hit):
                        if not attr.startswith('_') and not callable(getattr(hit, attr)):
                            result[attr] = getattr(hit, attr)
                
                # Get semantic score if available
                if isinstance(hit, dict) and '_semanticScore' in hit:
                    result['score'] = hit['_semanticScore']
                elif hasattr(hit, '_semanticScore'):
                    result['score'] = hit._semanticScore
                else:
                    result['score'] = 0.8  # Default score
                
                # Ensure content field exists
                if 'text' in result and 'content' not in result:
                    result['content'] = result['text']
                
                # Add payload field for compatibility
                result['payload'] = {}
                for k, v in result.items():
                    if k != 'payload':
                        result['payload'][k] = v
                
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            if self.verbose:
                print(f"Error in hybrid search: {e}")
                print("Falling back to keyword search")
            return self.search_keyword(query, limit)


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

    def search_dense(self, query: str, processor: Any, limit: int, score_threshold: float = None,
                    rerank: bool = False, reranker_type: str = None):
        """
        Perform vector search according to Meilisearch API requirements.
        """
        if processor is None:
            return {"error": "Vector search requires an embedding model"}
        
        try:
            # Generate query embedding
            query_vector = processor.get_embedding(query)
            
            # Get embedder name using the consistent naming approach
            dense_model_id = getattr(processor, 'dense_model_id', self.dense_model_id)
            embedder_name = self._get_embedder_name(dense_model_id)
            
            if self.verbose:
                print(f"Executing vector search with {len(query_vector)}-dimensional query vector")
                print(f"Using embedder name: {embedder_name}")
            
            # Use the model-based embedder name consistently
            search_params = {
                'limit': limit * 2,
                'attributesToRetrieve': ['*'],
                'vector': query_vector.tolist(),
                'hybrid': {
                    'embedder': embedder_name,  # Use consistent model-based embedder name
                    'semanticRatio': 1.0        # Pure semantic search (100% vector)
                }
            }
            
            if score_threshold is not None:
                search_params['scoreThreshold'] = score_threshold
            
            # Run search with empty query for pure vector search
            results = self.index.search('', search_params)
            
            # Extract hits
            hits = []
            if hasattr(results, 'hits'):
                hits = results.hits
            elif isinstance(results, dict) and 'hits' in results:
                hits = results['hits']
            
            if self.verbose:
                print(f"Vector search returned {len(hits)} results")
                
            # Format results
            formatted_results = []
            for hit in hits[:limit]:
                result = {}
                
                # Copy all fields
                if isinstance(hit, dict):
                    for k, v in hit.items():
                        result[k] = v
                else:
                    for attr in dir(hit):
                        if not attr.startswith('_') and not callable(getattr(hit, attr)):
                            result[attr] = getattr(hit, attr)
                
                # Get semantic score if available
                if isinstance(hit, dict) and '_semanticScore' in hit:
                    result['score'] = hit['_semanticScore']
                elif hasattr(hit, '_semanticScore'):
                    result['score'] = hit._semanticScore
                else:
                    result['score'] = 0.8  # Default score
                
                # Ensure content field exists
                if 'text' in result and 'content' not in result:
                    result['content'] = result['text']
                
                # Add payload field for compatibility
                result['payload'] = {}
                for k, v in result.items():
                    if k != 'payload':
                        result['payload'][k] = v
                
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            if self.verbose:
                print(f"Error in vector search: {e}")
                print("Falling back to keyword search")
            return self.search_keyword(query, limit)

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