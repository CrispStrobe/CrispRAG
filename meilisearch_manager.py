# meilisearch_manager.py
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

# Try to import Meilisearch client
try:
    import meilisearch
    from meilisearch.errors import MeilisearchApiError
    meilisearch_available = True
except ImportError:
    meilisearch_available = False
    print("Warning: Meilisearch client not available. Install with: pip install meilisearch")


class MeilisearchManager(VectorDBInterface):
    """Manager for Meilisearch vector database operations with semantic search capabilities"""
    
    def __init__(self, 
                url: str = "http://localhost:7700", 
                api_key: str = None,
                collection_name: str = "documents",
                vector_size: int = 384,
                storage_path: str = None,
                verbose: bool = False,
                dense_model_id: str = "bge-small",
                sparse_model_id: str = "distilbert-splade"):
        """
        Initialize MeilisearchManager with model-specific vector configuration.
        
        Args:
            url: Meilisearch URL
            api_key: Meilisearch API key
            collection_name: Index name
            vector_size: Vector dimension
            storage_path: Storage path (not used for Meilisearch but kept for API compatibility)
            verbose: Verbose output
            dense_model_id: ID of dense model for naming vectors
            sparse_model_id: ID of sparse model for naming vectors
        """
        if not meilisearch_available:
            raise ImportError("Meilisearch client not available. Install with: pip install meilisearch")
            
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.vector_dim = vector_size
        self.storage_path = storage_path  # Not used but kept for compatibility
        self.verbose = verbose
        self.client = None
        self.index = None
        
        # Store the original model IDs
        self.dense_model_id = dense_model_id
        self.sparse_model_id = sparse_model_id
        
        # Performance tracking
        self._hit_rates = {}
        
        self.connect()

    def connect(self) -> None:
        """Connect to Meilisearch server"""
        try:
            # Initialize client configuration
            client_config = {'url': self.url}
            if self.api_key:
                client_config['api_key'] = self.api_key
                
            if self.verbose:
                print(f"Connecting to Meilisearch at {self.url}")
                
            self.client = meilisearch.Client(**client_config)
            
            # Test connection with a health check
            health = self.client.health()
            if health['status'] != 'available':
                raise ConnectionError(f"Meilisearch server is not available: {health}")
                
            if self.verbose:
                print(f"Connected to Meilisearch at {self.url} (version: {health.get('version', 'unknown')})")
                
            # Get the index (this doesn't create it)
            self.index = self.client.index(self.collection_name)
            
            # Check if index exists by trying to get its stats
            try:
                self.index.get_stats()
                if self.verbose:
                    print(f"Found existing index '{self.collection_name}'")
            except Exception:
                if self.verbose:
                    print(f"Index '{self.collection_name}' does not exist yet")
                self.index = None
                
        except Exception as e:
            print(f"Error connecting to Meilisearch: {str(e)}")
            raise


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
                    'sparse_terms'  # For sparse vector simulation
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
                # Use the exact ranking rules from the working implementation
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
                # Create a unique vector name based on the model ID
                vector_name = self._sanitize_model_name(self.dense_model_id)
                
                # Configure the embedder
                try:
                    # Use correct embedder configuration parameters
                    embedder_settings = {
                        vector_name: {
                            'source': 'ollama',
                            'url': 'http://localhost:11434/api/embeddings',
                            'model': self.dense_model_id,
                            'documentTemplate': '{{doc.content}}'  # Properly formatted for Ollama
                        }
                    }
                    
                    try:
                        # Try the dedicated method first (newer Meilisearch versions)
                        task = self.index.update_embedders(embedder_settings)
                        self._wait_for_task(task)
                        if self.verbose:
                            print(f"Vector search enabled with {self.vector_dim} dimensions using embedder: {self.dense_model_id}")
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
                                print(f"Vector search enabled via REST API")
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
    
    def _wait_for_task(self, task):
        """Wait for a Meilisearch task to complete"""
        if not task or not self.client:
            return
            
        try:
            # Extract task ID - handle both dict and TaskInfo object
            task_id = None
            if isinstance(task, dict):
                # Old client version - dictionary response
                task_id = task.get('taskUid', task.get('uid'))
            else:
                # New client version - TaskInfo object
                task_id = getattr(task, 'task_uid', None)
                if task_id is None:
                    # Try alternative attribute names
                    task_id = getattr(task, 'uid', None)
                    
            if not task_id:
                if self.verbose:
                    print("Warning: Could not extract task ID from task object")
                return
                
            if self.verbose:
                print(f"Waiting for task {task_id} to complete...")
                
            if hasattr(self.client, 'wait_for_task'):
                self.client.wait_for_task(task_id)
            else:
                # Manual implementation if wait_for_task is not available
                max_retries = 50
                retry_delay = 0.2  # Start with 200ms
                
                for i in range(max_retries):
                    try:
                        task_status = self.client.get_task(task_id)
                        
                        # Handle both dict and object response
                        if isinstance(task_status, dict):
                            status = task_status.get('status')
                        else:
                            status = getattr(task_status, 'status', None)
                        
                        if status == 'succeeded':
                            if self.verbose:
                                print(f"Task {task_id} completed successfully")
                            return
                        elif status in ['failed', 'cancelled']:
                            # Try to extract error message
                            error_msg = "Unknown error"
                            if isinstance(task_status, dict):
                                error = task_status.get('error', {})
                                error_msg = error.get('message', error_msg)
                            else:
                                error = getattr(task_status, 'error', None)
                                if error:
                                    error_msg = getattr(error, 'message', error_msg)
                                    
                            raise Exception(f"Task failed: {error_msg}")
                            
                        # Still processing, wait with exponential backoff
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 1.5, 1.0)  # Cap at 1 second
                    except Exception as e:
                        if 'Task not found' in str(e):
                            # Task might be processed too quickly and removed
                            return
                        raise
                
                # If we get here, we've exceeded retries
                print(f"Warning: Timed out waiting for task {task_id}")
        except Exception as e:
            if self.verbose:
                print(f"Error waiting for task: {e}")
    
    def _sanitize_model_name(self, model_name):
        """Sanitize model name for use as a vector name in Meilisearch"""
        # Extract just the final part of the model name if it has slashes
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
            
        # Replace any non-alphanumeric characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9]', '_', model_name)
        
        # Ensure the name isn't too long
        if len(sanitized) > 40:
            sanitized = sanitized[:40]
            
        return sanitized
    
    def get_collection_info(self) -> Dict[str, Any]:
        try:
            if not self.client or not self.index:
                return {"error": "Not connected to Meilisearch or index not found"}
                
            # Get index stats
            try:
                # Get stats object
                stats = self.index.get_stats()
                
                # Handle both dictionary and object response format
                # Modern Meilisearch clients return objects, older ones return dictionaries
                num_docs = 0
                num_embedded_docs = 0
                num_embeddings = 0
                field_distribution = {}
                is_indexing = False
                
                if isinstance(stats, dict):
                    # Dictionary response (older client)
                    num_docs = stats.get("numberOfDocuments", 0)
                    num_embedded_docs = stats.get("numberOfEmbeddedDocuments", 0)
                    num_embeddings = stats.get("numberOfEmbeddings", 0)
                    field_distribution = stats.get("fieldDistribution", {})
                    is_indexing = stats.get("isIndexing", False)
                else:
                    # Object response (newer client)
                    num_docs = getattr(stats, "number_of_documents", 0)
                    num_embedded_docs = getattr(stats, "number_of_embedded_documents", 0)
                    num_embeddings = getattr(stats, "number_of_embeddings", 0)
                    field_distribution = getattr(stats, "field_distribution", {})
                    is_indexing = getattr(stats, "is_indexing", False)
                
                # Get index info including primary key
                index_info = {}
                try:
                    index_info_resp = self.client.get_index(self.collection_name)
                    if isinstance(index_info_resp, dict):
                        index_info = index_info_resp
                    else:
                        # Convert object to dictionary
                        index_info = {
                            "uid": getattr(index_info_resp, "uid", self.collection_name),
                            "primaryKey": getattr(index_info_resp, "primary_key", "id"),
                            "createdAt": getattr(index_info_resp, "created_at", None),
                            "updatedAt": getattr(index_info_resp, "updated_at", None)
                        }
                except Exception as e:
                    if self.verbose:
                        print(f"Error getting index info: {e}")
                    index_info = {"primaryKey": "id"}  # Default
                
                # Check for embedder settings (vector search)
                vector_configs = {}
                try:
                    # Try different ways to access embedder settings
                    embedders = None
                    
                    # Method 1: Direct method call (newer client)
                    try:
                        embedders = self.index.get_embedders()
                        # Convert to dictionary if it's an object
                        if embedders and not isinstance(embedders, dict):
                            embedders = {name: {"model": config.model, "source": config.source} 
                                        for name, config in embedders.items()}
                    except AttributeError:
                        pass
                    
                    # Method 2: REST API call (fallback for older clients)
                    if not embedders:
                        try:
                            import requests
                            response = requests.get(
                                f"{self.url}/indexes/{self.collection_name}/settings/embedders",
                                headers={
                                    "Authorization": f"Bearer {self.api_key}" if self.api_key else None
                                }
                            )
                            
                            if response.status_code == 200:
                                embedders = response.json()
                        except Exception as rest_err:
                            if self.verbose:
                                print(f"Error with REST API call: {rest_err}")
                    
                    # Process embedders if found
                    if embedders and isinstance(embedders, dict):
                        for name, config in embedders.items():
                            if isinstance(config, dict):
                                vector_configs[name] = {
                                    "model": config.get("model", "unknown"),
                                    "source": config.get("source", "unknown")
                                }
                            else:
                                # Handle object config
                                vector_configs[name] = {
                                    "model": getattr(config, "model", "unknown"),
                                    "source": getattr(config, "source", "unknown")
                                }
                except Exception as e:
                    if self.verbose:
                        print(f"Error getting embedder settings: {e}")
                
                # Construct result with comprehensive stats
                result = {
                    "name": self.collection_name,
                    "points_count": num_docs,
                    "embedded_docs_count": num_embedded_docs,
                    "embeddings_count": num_embeddings,
                    "is_indexing": is_indexing,
                    "field_distribution": field_distribution,
                    "vector_configs": vector_configs,
                    "sparse_vector_configs": {},  # Meilisearch doesn't have separate sparse configs
                    "primaryKey": index_info.get("primaryKey", "id")
                }
                
                # Include performance data if available
                if hasattr(self, '_hit_rates') and self._hit_rates:
                    result["performance"] = {"hit_rates": self._hit_rates}
                
                return result
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
        if not hasattr(self, '_hit_rates'):
            return {}
            
        result = {}
        for search_type, stats in self._hit_rates.items():
            if stats["total"] > 0:
                hit_rate = stats["hits"] / stats["total"]
                result[search_type] = hit_rate
        return result
        
    def _extract_document_count_from_stats(self, stats):
        """
        Safely extract document count from stats object or dictionary.
        """
        try:
            # For dictionary responses (most common)
            if isinstance(stats, dict):
                if "numberOfDocuments" in stats:
                    return stats["numberOfDocuments"]
                if "number_of_documents" in stats:
                    return stats["number_of_documents"]
                return 0
                
            # For object responses - this seems to be our case
            if hasattr(stats, "number_of_documents"):
                return getattr(stats, "number_of_documents")
                
            # Print debug info about the stats object structure
            if self.verbose:
                print(f"Stats type: {type(stats)}")
                print(f"Stats dir: {dir(stats)}")
                
                # Try to get the attribute directly - might be a naming issue
                for attr in dir(stats):
                    if 'document' in attr.lower() and 'number' in attr.lower():
                        value = getattr(stats, attr)
                        print(f"Found potential doc count attribute: {attr} = {value}")
                        if isinstance(value, (int, float)) and value > 0:
                            return value
                            
            # Return 0 if we couldn't extract the count
            return 0
        except Exception as e:
            if self.verbose:
                print(f"Error extracting document count: {e}")
            return 0

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
                
                # Add vector embedding in the _vectors field format per Meilisearch docs
                document['_vectors'] = embedding.tolist()
                
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
        Insert documents with both dense and sparse vector embeddings.
        """
        if not embeddings_with_sparse:
            return
        
        try:
            if not self.index:
                raise ValueError(f"Index '{self.collection_name}' not initialized")
            
            # Prepare documents with both dense and sparse embeddings
            documents = []
            for dense_embedding, payload, sparse_vector in embeddings_with_sparse:
                # Generate ID if not provided
                if 'id' not in payload:
                    doc_id = str(uuid.uuid4())
                else:
                    doc_id = payload['id']
                
                # Extract sparse indices and values
                sparse_indices, sparse_values = sparse_vector
                
                # Create a sparse terms field to help with text search
                sparse_terms = []
                for idx, val in zip(sparse_indices, sparse_values):
                    if val > 0.1:  # Only include significant terms
                        sparse_terms.append(f"term_{idx}")
                
                # Create document with all fields
                document = {
                    'id': doc_id,
                    'sparse_terms': " ".join(sparse_terms),
                    **payload  # Include all payload fields
                }
                
                # Add vector embedding in the _vectors field with the embedder name
                # The error suggests we need to specify the embedder name
                document['_vectors'] = {
                    'default': dense_embedding.tolist()  # Use 'default' as the embedder name
                }
                
                documents.append(document)
            
            if self.verbose:
                print(f"Inserting {len(documents)} documents with properly formatted vector embeddings")
            
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
            
            # Determine semantic ratio based on fusion type
            semantic_ratio = 0.5  # Default balanced ratio
            if fusion_type.lower() == 'vector':
                semantic_ratio = 0.9  # Mostly vector search
            elif fusion_type.lower() == 'keyword':
                semantic_ratio = 0.1  # Mostly keyword search
            
            if self.verbose:
                print(f"Executing hybrid search with semanticRatio={semantic_ratio}")
            
            # Build search parameters based on error messages and documentation
            search_params = {
                'limit': limit * 2,
                'attributesToRetrieve': ['*'],
                'vector': query_vector.tolist(),
                'hybrid': {
                    'embedder': 'default',      # Add the required embedder field
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
        Search with various options.
        """
        try:
            query = query.strip()
            if not query:
                return {"error": "Empty query"}
            
            if not self.index:
                return {"error": f"Index '{self.collection_name}' not found"}
            
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
            
            # Ensure points have scores - handle both object and dictionary points
            if points:
                for i in range(len(points)):
                    # Handle dictionary case (most common)
                    if isinstance(points[i], dict):
                        if 'score' not in points[i] or points[i]['score'] is None:
                            points[i]['score'] = 0.75  # Default score
                        
                        # Set score in multiple places for compatibility
                        points[i]['_rankingScore'] = points[i]['score'] 
                        points[i]['_score'] = points[i]['score']
                        points[i]['rankingScore'] = points[i]['score']
                        
                        # Ensure payload exists and has score
                        if 'payload' not in points[i]:
                            points[i]['payload'] = {}
                            
                        points[i]['payload']['score'] = points[i]['score']
                        points[i]['payload']['_rankingScore'] = points[i]['score']
                        points[i]['payload']['_score'] = points[i]['score']
                    elif hasattr(points[i], 'score'):
                        # Object case
                        if points[i].score is None:
                            points[i].score = 0.75
                            
                        # Add other score fields as attributes
                        if not hasattr(points[i], '_rankingScore'):
                            setattr(points[i], '_rankingScore', points[i].score)
                        if not hasattr(points[i], '_score'):
                            setattr(points[i], '_score', points[i].score)
            
            # Format results with improved preview
            return TextProcessor.format_search_results(
                points, query, search_type, processor, context_size,
                retriever=context_retriever,
                db_type="meilisearch"
            )
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def _adapt_meilisearch_result(self, hit, query=None):
        """
        Create a dictionary-based result that's compatible with ResultProcessor.
        """
        # Start with an empty result dictionary
        result = {}
        
        # For dictionary hits
        if isinstance(hit, dict):
            # Copy all fields from hit to result
            for k, v in hit.items():
                result[k] = v
                
            # Extract or calculate score
            score = 0.75  # Default score
            
            if '_rankingScore' in hit and hit['_rankingScore']:
                score = hit['_rankingScore']
                if self.verbose:
                    print(f"Using existing ranking score: {score}")
            elif query and 'text' in hit:
                text = hit['text'].lower()
                query_lower = query.lower()
                
                # Check for exact query match first
                if query_lower in text:
                    # Count occurrences
                    count = text.count(query_lower)
                    score = min(0.99, 0.75 + (count * 0.03))
                    if self.verbose:
                        print(f"Calculated score {score} based on {count} occurrences")
                # Check for word forms (e.g., "Gottes" is a form of "Gott")
                elif query_lower.rstrip('s') in text or query_lower + 'es' in text or query_lower + 's' in text:
                    score = 0.65  # Lower score for variant forms
                    if self.verbose:
                        print(f"Assigned score {score} for word variant of '{query}'")
            
            # Set score in multiple places for visibility
            result['score'] = score
            result['_rankingScore'] = score
            result['_score'] = score
            result['rankingScore'] = score
            
            # Ensure content field exists
            if 'text' in hit and 'content' not in result:
                result['content'] = hit['text']
                if self.verbose:
                    print("Copied 'text' to 'content' field")
        
        # For object hits (similar approach as dictionary hits)
        else:
            # Copy all attributes to dictionary
            if hasattr(hit, '__dict__'):
                attrs = hit.__dict__
                for k, v in attrs.items():
                    result[k] = v
            else:
                for attr in dir(hit):
                    if not attr.startswith('_') and not callable(getattr(hit, attr)):
                        result[attr] = getattr(hit, attr)
            
            # Extract or calculate score (similar logic as for dictionaries)
            score = 0.75  # Default score
            
            if hasattr(hit, '_rankingScore') and getattr(hit, '_rankingScore'):
                score = getattr(hit, '_rankingScore')
            elif query and hasattr(hit, 'text'):
                text = getattr(hit, 'text', '').lower()
                query_lower = query.lower()
                
                if query_lower in text:
                    count = text.count(query_lower)
                    score = min(0.99, 0.75 + (count * 0.03))
                elif query_lower.rstrip('s') in text or query_lower + 'es' in text or query_lower + 's' in text:
                    score = 0.65  # Lower score for variant forms
            
            # Set score in multiple places for visibility
            result['score'] = score
            result['_rankingScore'] = score
            result['_score'] = score
            result['rankingScore'] = score
                    
            # Ensure content field exists
            if hasattr(hit, 'text') and 'content' not in result:
                result['content'] = getattr(hit, 'text')
        
        # Create a separate payload dictionary (not a circular reference)
        result['payload'] = {}
        for k, v in result.items():
            if k != 'payload':  # Avoid recursion
                result['payload'][k] = v
        
        # Add scores to payload too
        result['payload']['score'] = score
        result['payload']['_rankingScore'] = score
        result['payload']['_score'] = score
        result['payload']['rankingScore'] = score
        
        return result
    
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
            
            if self.verbose:
                print(f"Executing vector search with {len(query_vector)}-dimensional query vector")
            
            # Based on the error messages, Meilisearch requires a 'hybrid' parameter 
            # with an 'embedder' field when using vector search
            search_params = {
                'limit': limit * 2,
                'attributesToRetrieve': ['*'],
                'vector': query_vector.tolist(),
                'hybrid': {
                    'embedder': 'default',  # Use the default embedder name
                    'semanticRatio': 1.0    # Pure semantic search (100% vector)
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

    def _vector_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        
        # Convert to numpy arrays if not already
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if norm_a == 0 or norm_b == 0:
            return 0
        
        return dot_product / (norm_a * norm_b)
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        
        # Convert to numpy arrays if they aren't already
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if norm_a == 0 or norm_b == 0:
            return 0
        
        return dot_product / (norm_a * norm_b)
    
    def search_sparse(self, query: str, processor: Any, limit: int, score_threshold: float = None,
                    rerank: bool = False, reranker_type: str = None):
        """
        Perform sparse vector search.
        """
        if processor is None:
            return {"error": "Sparse search requires an embedding model"}
        
        try:
            # Generate sparse vector from query
            sparse_indices, sparse_values = processor.get_sparse_embedding(query)
            
            if self.verbose:
                print(f"Executing sparse search with {len(sparse_indices)} sparse dimensions")
            
            # Generate sparse terms for the query
            sparse_terms = []
            for idx, val in zip(sparse_indices, sparse_values):
                if val > 0.1:  # Only use significant terms
                    sparse_terms.append(f"term_{idx}")
            
            if not sparse_terms:
                if self.verbose:
                    print("No significant sparse terms, falling back to keyword search")
                return self.search_keyword(query, limit)
            
            # Build search using the sparse terms - this simulates sparse vector matching
            sparse_query = " ".join(sparse_terms)
            
            search_params = {
                'limit': limit * 2,
                'attributesToRetrieve': ['*'],
                'attributesToSearchOn': ['sparse_terms'],
            }
            
            if score_threshold is not None:
                search_params['scoreThreshold'] = score_threshold
            
            # Execute search on sparse terms
            results = self.index.search(sparse_query, search_params)
            
            # Extract hits
            hits = []
            if hasattr(results, 'hits'):
                hits = results.hits
            elif isinstance(results, dict) and 'hits' in results:
                hits = results['hits']
            
            if self.verbose:
                print(f"Sparse search returned {len(hits)} results")
            
            # If we have hits, calculate relevance scores
            formatted_results = []
            
            for hit in hits:
                result = {}
                
                # Copy all fields
                if isinstance(hit, dict):
                    for k, v in hit.items():
                        result[k] = v
                else:
                    for attr in dir(hit):
                        if not attr.startswith('_') and not callable(getattr(hit, attr)):
                            result[attr] = getattr(hit, attr)
                
                # Try to calculate sparse similarity
                result_sparse_terms = None
                if isinstance(hit, dict) and 'sparse_terms' in hit:
                    result_sparse_terms = hit['sparse_terms']
                elif hasattr(hit, 'sparse_terms'):
                    result_sparse_terms = hit.sparse_terms
                
                if result_sparse_terms:
                    # Simple score based on term overlap
                    query_terms = set(sparse_terms)
                    doc_terms = set(result_sparse_terms.split())
                    overlap = len(query_terms.intersection(doc_terms))
                    total = len(query_terms.union(doc_terms))
                    
                    if total > 0:
                        similarity = overlap / total
                        result['score'] = min(0.99, 0.5 + similarity * 0.5)
                    else:
                        result['score'] = 0.5
                else:
                    result['score'] = 0.5
                
                # Ensure content field exists
                if 'text' in result and 'content' not in result:
                    result['content'] = result['text']
                
                # Add payload
                result['payload'] = {}
                for k, v in result.items():
                    if k != 'payload':
                        result['payload'][k] = v
                
                formatted_results.append(result)
            
            # Sort by score
            formatted_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            return formatted_results[:limit]
            
        except Exception as e:
            if self.verbose:
                print(f"Error in sparse search: {e}")
                print("Falling back to keyword search")
            return self.search_keyword(query, limit)
        
    def debug_index_content(self):
        """
        Debug method to check what's actually in the index.
        """
        try:
            if not self.index:
                print("No index available")
                return
                
            print("Analyzing index content:")
            
            # Get index stats
            try:
                stats = self.index.get_stats()
                
                # Try to get the document count
                doc_count = 0
                if hasattr(stats, "number_of_documents"):
                    doc_count = stats.number_of_documents
                    print(f"Index contains {doc_count} documents")
                else:
                    print("Could not find document count attribute")
                    print(f"Available attributes: {dir(stats)}")
            except Exception as e:
                print(f"Error getting stats: {e}")
                
            # Try to get random samples from the index to check content
            try:
                # Search with empty query to get some documents
                results = self.index.search('', {'limit': 5})
                
                # Extract hits
                hits = []
                if hasattr(results, 'hits'):
                    hits = results.hits
                elif isinstance(results, dict) and 'hits' in results:
                    hits = results['hits']
                    
                if hits and len(hits) > 0:
                    print(f"Found {len(hits)} sample documents")
                    
                    # Show attributes in the first document
                    if isinstance(hits[0], dict):
                        print(f"Sample document attributes: {list(hits[0].keys())}")
                        
                        # Check for content and sparse_terms
                        if 'content' in hits[0]:
                            content_preview = hits[0]['content'][:100] + "..." if len(hits[0]['content']) > 100 else hits[0]['content']
                            print(f"Content preview: {content_preview}")
                            
                        if 'sparse_terms' in hits[0]:
                            sparse_preview = hits[0]['sparse_terms'][:100] + "..." if len(hits[0]['sparse_terms']) > 100 else hits[0]['sparse_terms']
                            print(f"Sparse terms preview: {sparse_preview}")
                    else:
                        print(f"Sample document type: {type(hits[0])}")
                else:
                    print("No sample documents found")
            except Exception as e:
                print(f"Error getting sample documents: {e}")
                
            # Check searchable attributes configuration
            try:
                settings = self.index.get_settings()
                
                if isinstance(settings, dict):
                    if 'searchableAttributes' in settings:
                        print(f"Searchable attributes: {settings['searchableAttributes']}")
                else:
                    if hasattr(settings, 'searchable_attributes'):
                        print(f"Searchable attributes: {settings.searchable_attributes}")
                    elif hasattr(settings, 'searchableAttributes'):
                        print(f"Searchable attributes: {settings.searchableAttributes}")
            except Exception as e:
                print(f"Error getting settings: {e}")
                
        except Exception as e:
            print(f"Overall error in debug method: {e}")
    
    def search_keyword(self, query: str, limit: int = 10, score_threshold: float = None,
                    rerank: bool = False, reranker_type: str = None):
        """
        Keyword search with improved scoring.
        """
        try:
            if not query.strip():
                return {"error": "Empty query"}
            
            # First, ensure the text field is searchable
            if self.verbose:
                print("\n=== FIXING INDEX CONFIGURATION ===")
                try:
                    # Get current settings
                    settings = self.index.get_settings()
                    searchable_attrs = []
                    
                    if isinstance(settings, dict) and 'searchableAttributes' in settings:
                        searchable_attrs = settings['searchableAttributes']
                    elif hasattr(settings, 'searchableAttributes'):
                        searchable_attrs = settings.searchableAttributes
                    elif hasattr(settings, 'searchable_attributes'):
                        searchable_attrs = settings.searchable_attributes
                        
                    # Check if 'text' is already in searchableAttributes
                    if 'text' not in searchable_attrs and '*' not in searchable_attrs:
                        print("Adding 'text' to searchable attributes")
                        
                        # Add 'text' to searchable attributes
                        if isinstance(searchable_attrs, list):
                            searchable_attrs.append('text')
                        else:
                            # Convert to list if needed
                            searchable_attrs = list(searchable_attrs)
                            searchable_attrs.append('text')
                        
                        # Update the settings
                        if isinstance(settings, dict):
                            settings['searchableAttributes'] = searchable_attrs
                            task = self.index.update_settings(settings)
                            print("Index settings updated - waiting for task to complete")
                            self._wait_for_task(task)
                            print("Index settings update completed")
                        else:
                            # Handle object-style settings
                            settings_dict = {'searchableAttributes': searchable_attrs}
                            task = self.index.update_settings(settings_dict)
                            print("Index settings updated - waiting for task to complete")
                            self._wait_for_task(task)
                            print("Index settings update completed")
                    else:
                        print("'text' is already searchable or all attributes are searchable")
                except Exception as e:
                    print(f"Error updating index settings: {e}")
                    print("Proceeding with search using current configuration")
                    
                print("=== END INDEX CONFIGURATION ===\n")
            
            # Now search directly in the text field
            if self.verbose:
                print(f"Executing search for '{query}' using 'text' field")
                
            text_params = {
                'limit': limit * 3,
                'attributesToRetrieve': ['*'],
                'attributesToSearchOn': ['text'],
                'attributesToHighlight': ['text'],
                'highlightPreTag': '<<<HIGHLIGHT>>>',
                'highlightPostTag': '<<<END_HIGHLIGHT>>>',
                'showMatchesPosition': True,
                'showRankingScore': True
            }
            
            try:
                results = self.index.search(query, text_params)
                
                # Extract hits
                hits = []
                if hasattr(results, 'hits'):
                    hits = results.hits
                elif isinstance(results, dict) and 'hits' in results:
                    hits = results['hits']
                    
                if hits and len(hits) > 0:
                    if self.verbose:
                        print(f"Found {len(hits)} results searching in 'text' field")
                        
                    # Adapt results with proper scoring
                    adapted_hits = [self._adapt_meilisearch_result(hit, query) for hit in hits]
                    
                    # Sort by score
                    adapted_hits.sort(key=lambda x: x.get('score', 0), reverse=True)
                    
                    return adapted_hits[:limit]
            except Exception as e:
                if self.verbose:
                    print(f"Error during search: {e}")
                    
            return []
            
        except Exception as e:
            if self.verbose:
                print(f"Error in keyword search: {e}")
                import traceback
                traceback.print_exc()
            return {"error": f"Error in keyword search: {str(e)}"}
        
    def _check_embedder_configured(self, embedder_name):
        """Check if an embedder is configured in the index"""
        try:
            # Get current settings
            settings = self.index.get_settings()
            
            # Check if embedders are configured
            if isinstance(settings, dict):
                if 'embedders' in settings:
                    return embedder_name in settings['embedders']
            elif hasattr(settings, 'embedders'):
                embedders = settings.embedders
                return embedder_name in embedders
                
            return False
        except Exception as e:
            if self.verbose:
                print(f"Error checking embedder configuration: {e}")
            return False
        
    def _format_final_results(self, results, query=None):
        """
        Format the final results to ensure scores are properly displayed.
        This method ensures a consistent structure across all search types.
        """
        formatted = []
        
        for result in results:
            # Create a new clean result dictionary
            formatted_result = {}
            
            # Copy ID (handle both dict and object cases)
            if isinstance(result, dict):
                formatted_result['id'] = result.get('id', '')
            else:
                formatted_result['id'] = getattr(result, 'id', '')
            
            # Copy or set score (handle both dict and object cases)
            if isinstance(result, dict):
                formatted_result['score'] = result.get('score', 0.75)
            else:
                formatted_result['score'] = getattr(result, 'score', 0.75)
            
            # Copy content/text field (handle both dict and object cases)
            if isinstance(result, dict):
                if 'text' in result:
                    formatted_result['text'] = result['text']
                    # Also copy to content for compatibility
                    formatted_result['content'] = result['text']
                elif 'content' in result:
                    formatted_result['content'] = result['content']
                    # Also copy to text for compatibility
                    formatted_result['text'] = result['content']
            else:
                if hasattr(result, 'text'):
                    formatted_result['text'] = result.text
                    formatted_result['content'] = result.text
                elif hasattr(result, 'content'):
                    formatted_result['content'] = result.content
                    formatted_result['text'] = result.content
            
            # Copy all other fields (handle both dict and object cases)
            if isinstance(result, dict):
                for k, v in result.items():
                    if k not in formatted_result and k != 'payload':
                        formatted_result[k] = v
            else:
                for attr in dir(result):
                    if not attr.startswith('_') and not callable(getattr(result, attr)) and attr not in formatted_result and attr != 'payload':
                        formatted_result[attr] = getattr(result, attr)
            
            # Set score in multiple places for compatibility
            formatted_result['_rankingScore'] = formatted_result['score']
            formatted_result['_score'] = formatted_result['score']
            formatted_result['rankingScore'] = formatted_result['score']
            
            # Create payload with all fields
            formatted_result['payload'] = {}
            for k, v in formatted_result.items():
                if k != 'payload':
                    formatted_result['payload'][k] = v
            
            # Add directly without filtering
            formatted.append(formatted_result)
        
        # Sort by score for consistent ranking
        formatted.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        if self.verbose:
            print(f"Final formatted results: {len(formatted)}")
            for i, r in enumerate(formatted[:3]):
                print(f"  Top {i+1}: score={r.get('score', 0)}, id={r.get('id', '')[:20]}...")
                
        return formatted
        
    def _configure_embedder(self, embedder_name, model_id):
        """Configure an embedder for the index"""
        try:
            # Create embedder configuration
            embedder_settings = {
                embedder_name: {
                    'source': 'ollama',
                    'url': 'http://localhost:11434/api/embeddings',
                    'model': model_id,
                    'documentTemplate': '{{doc.text}}'  # Use text field, not content
                }
            }
            
            # Try to update embedders
            try:
                task = self.index.update_embedders(embedder_settings)
                self._wait_for_task(task)
                return True
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
                
                return response.status_code in (200, 202)
                
        except Exception as e:
            if self.verbose:
                print(f"Error configuring embedder: {e}")
            return False

        
    def _rerank_results(self, query: str, hits: List[Dict], processor: Any, limit: int, reranker_type: str = None):
        """
        Rerank search results using the given processor.
        
        Args:
            query: Original search query
            hits: List of search hits from Meilisearch
            processor: Document processor with embedding capabilities
            limit: Number of results to return
            reranker_type: Type of reranker to use
            
        Returns:
            Reranked list of hits
        """
        try:
            # Convert hits to format expected by reranker
            adapted_hits = [self._adapt_meilisearch_result(hit) for hit in hits]
            
            # Use the common reranking algorithm
            reranked_hits = SearchAlgorithms.rerank_results(query, adapted_hits, processor, limit, self.verbose)
            
            # Return the reranked hit objects
            return reranked_hits[:limit]
        except Exception as e:
            if self.verbose:
                print(f"Error during reranking: {e}")
            # Fall back to original hits if reranking fails
            return hits[:limit]
    
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
            if not self.index:
                return "[No index available for context retrieval]"
            
            # Define the minimum and maximum chunk indices to retrieve
            min_idx = max(0, chunk_index - window)
            max_idx = chunk_index + window
            
            # Build search filter for parent_id or file_path and chunk_index in range
            filter_expr = f"path = '{file_path}' AND chunk_index >= {min_idx} AND chunk_index <= {max_idx}"
            
            # Get chunks that match the file path and are within the chunk index range
            search_params = {
                'limit': 2 * window + 1,  # Maximum number of chunks to retrieve
                'filter': filter_expr,
                'sort': ['chunk_index:asc'],  # Sort by chunk index
                'attributesToRetrieve': ['content', 'chunk_index']  # Only need content and index
            }
            
            # Execute search with empty query to just filter
            results = self.index.search('', search_params)
            
            # Extract chunks
            chunks = results.get('hits', [])
            
            if not chunks:
                return f"[No chunks found for file_path={file_path}, chunk_index={chunk_index}]"
            
            # Sort by chunk index just to be sure
            chunks.sort(key=lambda x: x.get('chunk_index', 0))
            
            # Combine the content from all chunks
            combined_text = "\n\n".join(chunk.get('content', '') for chunk in chunks)
            
            return combined_text
            
        except Exception as e:
            if self.verbose:
                print(f"Error retrieving context: {e}")
            return f"[Error retrieving context: {str(e)}]"
    
    def cleanup(self, remove_storage: bool = False) -> None:
        """
        Clean up resources.
        
        Args:
            remove_storage: Whether to remove the index (ignored for Meilisearch)
        """
        # Close the client connection if needed
        self.client = None
        self.index = None
        
        # For Meilisearch, we don't actually remove anything
        # since it's a standalone server that persists data
        # The "remove_storage" parameter is ignored
        if remove_storage and self.verbose:
            print("Note: remove_storage parameter is ignored for Meilisearch")
            print("To remove an index, use the Meilisearch API directly")

    def update_embeddings(self, id_embedding_payload_tuples: List[Tuple[str, np.ndarray, Dict[str, Any]]]) -> None:
        """
        Update existing embeddings with new values.
        
        Args:
            id_embedding_payload_tuples: List of (id, embedding, payload) tuples to update
        """
        if not id_embedding_payload_tuples:
            return
            
        try:
            if not self.index:
                raise ValueError(f"Index '{self.collection_name}' not initialized")
                
            # Prepare documents for update
            documents = []
            for doc_id, embedding, payload in id_embedding_payload_tuples:
                # Create document with updated data
                document = {
                    'id': doc_id,
                    **payload  # Include all payload fields
                }
                
                # Add embedding as _vectors field for Meilisearch
                vector_name = self._sanitize_model_name(self.dense_model_id)
                document['_vectors'] = {
                    vector_name: embedding.tolist()
                }
                
                documents.append(document)
            
            if self.verbose:
                print(f"Updating {len(documents)} documents in index '{self.collection_name}'")
                
            # Update documents in batches to handle large datasets
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                
                if self.verbose and len(documents) > batch_size:
                    print(f"Updating batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ({len(batch)} documents)")
                
                # Update documents
                task = self.index.update_documents(batch)
                self._wait_for_task(task)
            
            if self.verbose:
                print(f"Successfully updated {len(documents)} documents")
                
        except Exception as e:
            print(f"Error updating embeddings: {str(e)}")
            raise

    def inspect_document_content(self):
        """
        Inspect the actual content of documents in the index.
        """
        try:
            if not self.index:
                print("No index available")
                return
                
            # First check if there are any documents
            try:
                stats = self.index.get_stats()
                doc_count = 0
                
                if hasattr(stats, "number_of_documents"):
                    doc_count = stats.number_of_documents
                elif isinstance(stats, dict) and "numberOfDocuments" in stats:
                    doc_count = stats["numberOfDocuments"]
                    
                print(f"Index contains {doc_count} documents")
                
                if doc_count == 0:
                    print("Index is empty - no documents to inspect")
                    return
            except Exception as e:
                print(f"Error getting document count: {e}")
                
            # Get some documents from the index
            try:
                # Get documents without any query (should return all documents)
                results = self.index.search('', {
                    'limit': 5,
                    'attributesToRetrieve': ['id', 'title', 'content', 'path']
                })
                
                hits = []
                if hasattr(results, 'hits'):
                    hits = results.hits
                elif isinstance(results, dict) and 'hits' in results:
                    hits = results['hits']
                    
                if not hits or len(hits) == 0:
                    print("Could not retrieve any documents from the index")
                    return
                    
                print(f"\nFound {len(hits)} documents to inspect:")
                
                for i, hit in enumerate(hits):
                    print(f"\n--- Document {i+1} ---")
                    
                    # Print document ID or title
                    if isinstance(hit, dict):
                        print(f"ID: {hit.get('id', 'Unknown')}")
                        print(f"Title: {hit.get('title', 'Unknown')}")
                        
                        # Check if content exists and is not empty
                        if 'content' in hit and hit['content']:
                            content_preview = hit['content'][:200] + "..." if len(hit['content']) > 200 else hit['content']
                            print(f"Content preview: {content_preview}")
                            print(f"Content length: {len(hit['content'])} characters")
                            
                            # Look for the test word in content
                            test_words = ['Gott', 'Heidenthum']
                            for word in test_words:
                                if word.lower() in hit['content'].lower():
                                    print(f"✓ Content contains the word '{word}'")
                                else:
                                    print(f"✗ Content does NOT contain the word '{word}'")
                        else:
                            print("Document has no content or empty content!")
                    else:
                        print(f"Document is not a dictionary: {type(hit)}")
                        
                # Try a direct database query
                print("\nTrying to search for specific content with database query...")
                
                # Get documents that might contain the target words
                for word in ['Gott', 'Heidenthum', 'die', 'und']:
                    results = self.index.search(word, {
                        'limit': 1,
                        'attributesToSearchOn': ['content']
                    })
                    
                    hits = []
                    if hasattr(results, 'hits'):
                        hits = results.hits
                    elif isinstance(results, dict) and 'hits' in results:
                        hits = results['hits']
                        
                    print(f"Search for '{word}' returned {len(hits)} results")
            except Exception as e:
                print(f"Error inspecting documents: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"Overall error in inspection: {e}")
            
    def delete_by_filter(self, filter_condition: str) -> int:
        """
        Delete entries matching a filter condition.
        
        Args:
            filter_condition: Filter condition in Meilisearch filter syntax (e.g., "fileType = 'pdf'")
            
        Returns:
            Number of deleted entries
        """
        try:
            if not self.index:
                raise ValueError(f"Index '{self.collection_name}' not initialized")
                
            # First count matching documents
            search_params = {
                'filter': filter_condition,
                'limit': 1  # We just need to know if there are matches
            }
            
            # Search with empty query to just apply the filter
            count_results = self.index.search('', search_params)
            total_hits = count_results.get('estimatedTotalHits', 0)
            
            if total_hits == 0:
                return 0  # Nothing to delete
                
            if self.verbose:
                print(f"Found {total_hits} documents matching filter: {filter_condition}")
                
            # Meilisearch doesn't have a direct "delete by filter" API,
            # so we need to get all matching document IDs and delete them
            
            # Get all document IDs matching the filter
            # We might need to do this in batches for large result sets
            search_params = {
                'filter': filter_condition,
                'limit': 1000,  # Get more IDs per batch
                'attributesToRetrieve': ['id']  # Only need the ID
            }
            
            all_ids = []
            offset = 0
            
            # Loop until we've collected all IDs
            while True:
                search_params['offset'] = offset
                batch_results = self.index.search('', search_params)
                batch_hits = batch_results.get('hits', [])
                
                if not batch_hits:
                    break
                    
                # Extract IDs
                batch_ids = [hit['id'] for hit in batch_hits]
                all_ids.extend(batch_ids)
                
                # Move to next batch
                offset += len(batch_hits)
                
                # Break if we've got all results
                if offset >= total_hits:
                    break
            
            if self.verbose:
                print(f"Collected {len(all_ids)} document IDs for deletion")
                
            # Delete documents by ID
            if all_ids:
                task = self.index.delete_documents(all_ids)
                self._wait_for_task(task)
                
                if self.verbose:
                    print(f"Deleted {len(all_ids)} documents")
                    
                return len(all_ids)
            
            return 0
                
        except Exception as e:
            print(f"Error deleting by filter: {str(e)}")
            return 0
            
    def search_by_id(self, doc_id: str) -> Dict[str, Any]:
        """
        Get a document by its ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document data or empty dict if not found
        """
        try:
            if not self.index:
                return {}
                
            try:
                # Get document by ID
                document = self.index.get_document(doc_id)
                if document:
                    return document
            except Exception as e:
                if self.verbose:
                    print(f"Document not found: {e}")
                
            return {}
                
        except Exception as e:
            if self.verbose:
                print(f"Error getting document: {e}")
            return {}