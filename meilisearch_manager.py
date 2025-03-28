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
            # Check if the index already exists
            exists = False
            try:
                indexes = self.client.get_indexes()
                if 'results' in indexes:
                    exists = any(index['uid'] == self.collection_name for index in indexes['results'])
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
            
            # Configure index settings
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
                # This API might differ between Meilisearch versions
                try:
                    # Newer API with dedicated embedder settings
                    embedder_settings = {
                        vector_name: {
                            'source': 'ollama',
                            'url': 'http://localhost:11434/api/embeddings',
                            'model': self.dense_model_id,
                            'documentTemplate': '{{doc.content}}'
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
            if 'taskUid' in task or 'uid' in task:
                task_id = task.get('taskUid', task.get('uid'))
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
                            status = task_status.get('status')
                            
                            if status == 'succeeded':
                                if self.verbose:
                                    print(f"Task {task_id} completed successfully")
                                return
                            elif status in ['failed', 'cancelled']:
                                error = task_status.get('error', {})
                                raise Exception(f"Task failed: {error.get('message', 'Unknown error')}")
                                
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
        """Get information about the Meilisearch index"""
        try:
            if not self.client or not self.index:
                return {"error": "Not connected to Meilisearch or index not found"}
                
            # Get index stats
            try:
                stats = self.index.get_stats()
                # Get primary key
                index_info = self.client.get_index(self.collection_name)
                
                # Check for embedder settings (vector search)
                vector_configs = {}
                try:
                    # Different ways to access embedder settings depending on client version
                    try:
                        embedders = self.index.get_embedders()
                        for name, config in embedders.items():
                            vector_configs[name] = {
                                "model": config.get("model", "unknown"),
                                "source": config.get("source", "unknown")
                            }
                    except AttributeError:
                        # Try with direct API call for older clients
                        import requests
                        response = requests.get(
                            f"{self.url}/indexes/{self.collection_name}/settings/embedders",
                            headers={
                                "Authorization": f"Bearer {self.api_key}" if self.api_key else None
                            }
                        )
                        
                        if response.status_code == 200:
                            embedders = response.json()
                            for name, config in embedders.items():
                                vector_configs[name] = {
                                    "model": config.get("model", "unknown"),
                                    "source": config.get("source", "unknown")
                                }
                except Exception as e:
                    if self.verbose:
                        print(f"Error getting embedder settings: {e}")
                
                # For compatibility with other managers, create a result structure
                # similar to what other vector databases would return
                result = {
                    "name": self.collection_name,
                    "points_count": stats.get("numberOfDocuments", 0),
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
    
    def insert_embeddings(self, embeddings_with_payloads: List[Tuple[np.ndarray, Dict[str, Any]]]) -> None:
        """Insert embeddings into Meilisearch"""
        if not embeddings_with_payloads:
            return
            
        try:
            if not self.index:
                raise ValueError(f"Index '{self.collection_name}' not initialized")
                
            # Prepare documents for insertion
            documents = []
            for i, (embedding, payload) in enumerate(embeddings_with_payloads):
                # Generate a unique ID if not provided
                if 'id' not in payload:
                    doc_id = str(uuid.uuid4())
                else:
                    doc_id = payload['id']
                
                # Create document with embedding
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
                print(f"Inserting {len(documents)} documents into index '{self.collection_name}'")
                
            # Insert documents in batches to handle large datasets
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                
                if self.verbose and len(documents) > batch_size:
                    print(f"Inserting batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ({len(batch)} documents)")
                
                # Add documents with error handling
                try:
                    task = self.index.add_documents(batch)
                    self._wait_for_task(task)
                except Exception as e:
                    print(f"Error inserting batch: {e}")
                    raise
            
            if self.verbose:
                print(f"Successfully inserted {len(documents)} documents")
                
        except Exception as e:
            print(f"Error inserting embeddings: {str(e)}")
            raise
    
    def insert_embeddings_with_sparse(self, embeddings_with_sparse: List[Tuple[np.ndarray, Dict[str, Any], Tuple[List[int], List[float]]]]) -> None:
        """
        Insert embeddings with sparse vectors into Meilisearch.
        
        Note: Meilisearch doesn't natively support sparse vectors, but we simulate them by
        adding terms from the sparse vector to a 'sparse_terms' field for keyword search.
        
        Args:
            embeddings_with_sparse: List of (dense_embedding, payload, sparse_embedding) tuples
        """
        if not embeddings_with_sparse:
            return
            
        try:
            if not self.index:
                raise ValueError(f"Index '{self.collection_name}' not initialized")
                
            # Prepare documents for insertion
            documents = []
            for embedding, payload, sparse_vector in embeddings_with_sparse:
                # Generate a unique ID if not provided
                if 'id' not in payload:
                    doc_id = str(uuid.uuid4())
                else:
                    doc_id = payload['id']
                
                # Extract sparse indices and values
                sparse_indices, sparse_values = sparse_vector
                
                # Create a sparse terms field for keyword searching
                # This is a workaround since Meilisearch doesn't natively support sparse vectors
                sparse_terms = []
                for idx, val in zip(sparse_indices, sparse_values):
                    # Only include high-value terms
                    if val > 0.1:
                        # Use the index as a term (not ideal but workable)
                        sparse_terms.append(f"term_{idx}")
                
                # Create document with both embeddings
                document = {
                    'id': doc_id,
                    'sparse_terms': " ".join(sparse_terms),  # For keyword search
                    **payload  # Include all payload fields
                }
                
                # Add dense embedding as _vectors field for Meilisearch
                vector_name = self._sanitize_model_name(self.dense_model_id)
                document['_vectors'] = {
                    vector_name: embedding.tolist()
                }
                
                documents.append(document)
            
            if self.verbose:
                print(f"Inserting {len(documents)} documents with sparse vectors into index '{self.collection_name}'")
                
            # Insert documents in batches to handle large datasets
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                
                if self.verbose and len(documents) > batch_size:
                    print(f"Inserting batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ({len(batch)} documents)")
                
                # Add documents with error handling
                try:
                    task = self.index.add_documents(batch)
                    self._wait_for_task(task)
                except Exception as e:
                    print(f"Error inserting batch: {e}")
                    raise
            
            if self.verbose:
                print(f"Successfully inserted {len(documents)} documents with sparse vectors")
                
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
            
            # Get vector name based on model
            vector_name = self._sanitize_model_name(processor.dense_model_id)
            
            # Build search parameters for hybrid search
            search_params = {
                'limit': prefetch_limit,  # Get more than we need for potential reranking
                'attributesToRetrieve': ['*'],  # Get all fields
                'attributesToHighlight': ['content'],  # Highlight content for previews
                'highlightPreTag': '<<<HIGHLIGHT>>>',  # Custom highlight tags for consistent formatting
                'highlightPostTag': '<<<END_HIGHLIGHT>>>',
                'showMatchesPosition': True,  # Show where matches occur
                'showRankingScore': True  # Show scores for debugging
            }
            
            # Configure hybrid search 
            # The semanticRatio determines weight between vector and keyword search
            semantic_ratio = 0.5  # Default balanced ratio
            if fusion_type.lower() == 'vector':
                semantic_ratio = 0.9  # Mostly vector search
            elif fusion_type.lower() == 'keyword':
                semantic_ratio = 0.1  # Mostly keyword search
                
            # Add vector to search params
            if query_vector is not None:
                search_params['vector'] = query_vector.tolist()
                search_params['hybrid'] = {
                    'semanticRatio': semantic_ratio,
                    'embedder': vector_name
                }
            
            if score_threshold is not None:
                search_params['scoreThreshold'] = score_threshold
            
            if self.verbose:
                print(f"Executing hybrid search with semanticRatio={semantic_ratio}")
            
            # Execute search
            results = self.index.search(query, search_params)
            
            # Extract hits
            hits = results.get('hits', [])
            
            if self.verbose:
                print(f"Hybrid search returned {len(hits)} results")
            
            # Apply reranking if requested
            if rerank and len(hits) > 0 and processor is not None:
                hits = self._rerank_results(query, hits, processor, limit, reranker_type)
            else:
                # Just limit to requested number
                hits = hits[:limit]
            
            # Record hit rates if we have ground truth
            true_context = getattr(processor, 'expected_context', None)
            if true_context:
                hit = any(hit.get('content') == true_context for hit in hits)
                search_key = f"hybrid_{fusion_type}"
                if rerank:
                    search_key += f"_{reranker_type or 'default'}"
                self._record_hit(search_key, hit)
            
            # Convert hits to the format expected by the rest of the code
            formatted_hits = []
            for hit in hits:
                formatted_hits.append(self._adapt_meilisearch_result(hit))
                
            return formatted_hits
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
    
    def _adapt_meilisearch_result(self, hit):
        """Adapt a Meilisearch hit to match the format expected by other parts of the code"""
        # Create a proper point object with the right structure
        class Point:
            def __init__(self, hit_dict):
                self.id = hit_dict.get('id', '')
                self.score = hit_dict.get('_rankingScore', 0.0)
                
                # All other fields go in payload
                self.payload = {k: v for k, v in hit_dict.items() if not k.startswith('_')}
                
                # Special _formatted field for highlighted content
                if '_formatted' in hit_dict:
                    self._formatted = hit_dict['_formatted']
        
        return Point(hit)
    
    def search_dense(self, query: str, processor: Any, limit: int, score_threshold: float = None,
                    rerank: bool = False, reranker_type: str = None):
        """
        Perform dense vector search in Meilisearch.
        
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
            
            # Get vector name based on model
            vector_name = self._sanitize_model_name(processor.dense_model_id)
            
            # Build search parameters for pure vector search
            search_params = {
                'limit': limit * 3,  # Get more than we need for potential filtering/reranking
                'attributesToRetrieve': ['*'],  # Get all fields
                'showRankingScore': True  # Show scores for debugging
            }
            
            # Add vector with high semantic ratio (mostly vector search)
            if query_vector is not None:
                search_params['vector'] = query_vector.tolist()
                # Set empty query to force pure vector search
                search_params['q'] = ''
                # Still use hybrid search but with very high semanticRatio
                search_params['hybrid'] = {
                    'semanticRatio': 0.99,  # Almost pure vector search
                    'embedder': vector_name
                }
            else:
                return {"error": "Failed to generate query embedding"}
            
            if score_threshold is not None:
                search_params['scoreThreshold'] = score_threshold
            
            if self.verbose:
                print(f"Executing dense vector search with embedder '{vector_name}'")
            
            # Execute search
            # We still use normal search but with empty query and high semantic ratio
            results = self.index.search('', search_params)
            
            # Extract hits
            hits = results.get('hits', [])
            
            if self.verbose:
                print(f"Dense search returned {len(hits)} results")
            
            # Apply reranking if requested
            if rerank and len(hits) > 0:
                adapted_hits = [self._adapt_meilisearch_result(hit) for hit in hits]
                reranked_hits = SearchAlgorithms.rerank_results(query, adapted_hits, processor, limit, self.verbose)
                
                # Convert back to original format
                hits = [getattr(hit, 'payload', {}) for hit in reranked_hits]
                for i, hit in enumerate(hits):
                    hit['_rankingScore'] = getattr(reranked_hits[i], 'score', 0.0)
            else:
                # Just limit to requested number
                hits = hits[:limit]
            
            # Record hit rates if we have ground truth
            true_context = getattr(processor, 'expected_context', None)
            if true_context:
                hit = any(hit.get('content') == true_context for hit in hits)
                search_key = "vector"
                if rerank:
                    search_key += f"_{reranker_type or 'default'}"
                self._record_hit(search_key, hit)
            
            # Adapt to the format expected by TextProcessor
            return [self._adapt_meilisearch_result(hit) for hit in hits]
            
        except Exception as e:
            if self.verbose:
                print(f"Error in dense search: {e}")
            return {"error": f"Error in dense search: {str(e)}"}
    
    def search_sparse(self, query: str, processor: Any, limit: int, score_threshold: float = None,
                    rerank: bool = False, reranker_type: str = None):
        """
        Perform sparse vector search in Meilisearch.
        
        Since Meilisearch doesn't natively support sparse vectors, we use the sparse_terms field
        that was created during document insertion as a proxy.
        
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
            
            # Since Meilisearch doesn't support sparse vectors natively,
            # we'll simulate it by searching on the sparse_terms field
            sparse_terms = []
            for idx, val in zip(sparse_indices, sparse_values):
                if val > 0.1:  # Only include significant terms
                    sparse_terms.append(f"term_{idx}")
            
            # If we have sparse terms, use them for search
            if sparse_terms:
                # Build the search query targeting sparse_terms field
                sparse_query = " ".join(sparse_terms)
                
                # Build search parameters
                search_params = {
                    'limit': limit * 3,  # Get more than we need for potential filtering/reranking
                    'attributesToRetrieve': ['*'],  # Get all fields
                    'attributesToSearchOn': ['sparse_terms'],  # Only search on sparse terms
                    'attributesToHighlight': ['content'],  # Still highlight in content
                    'highlightPreTag': '<<<HIGHLIGHT>>>',
                    'highlightPostTag': '<<<END_HIGHLIGHT>>>',
                    'showMatchesPosition': True,
                    'showRankingScore': True
                }
                
                if score_threshold is not None:
                    search_params['scoreThreshold'] = score_threshold
                
                if self.verbose:
                    print(f"Executing sparse search with {len(sparse_terms)} sparse terms")
                
                # Execute search using the sparse terms query
                results = self.index.search(sparse_query, search_params)
                
                # Extract hits
                hits = results.get('hits', [])
                
                if self.verbose:
                    print(f"Sparse search returned {len(hits)} results")
                
                # Apply reranking if requested
                if rerank and len(hits) > 0:
                    adapted_hits = [self._adapt_meilisearch_result(hit) for hit in hits]
                    reranked_hits = SearchAlgorithms.rerank_results(query, adapted_hits, processor, limit, self.verbose)
                    
                    # Return the reranked results
                    return reranked_hits[:limit]
                
                # Adapt to the format expected by TextProcessor
                return [self._adapt_meilisearch_result(hit) for hit in hits[:limit]]
            else:
                # Fallback to keyword search if no sparse terms
                if self.verbose:
                    print("No sparse terms extracted, falling back to keyword search")
                return self.search_keyword(query, limit, score_threshold, rerank, reranker_type)
                
        except Exception as e:
            if self.verbose:
                print(f"Error in sparse search: {e}")
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
            
            # Build search parameters
            search_params = {
                'limit': limit * 3,  # Get more than we need for potential filtering/reranking
                'attributesToRetrieve': ['*'],  # Get all fields
                'attributesToSearchOn': ['content', 'title', 'filename'],  # Search in these fields
                'attributesToHighlight': ['content'],  # Highlight content for previews
                'highlightPreTag': '<<<HIGHLIGHT>>>',  # Custom highlight tags
                'highlightPostTag': '<<<END_HIGHLIGHT>>>',
                'showMatchesPosition': True,  # Show where matches occur
                'showRankingScore': True  # Show scores for debugging
            }
            
            if score_threshold is not None:
                search_params['scoreThreshold'] = score_threshold
            
            if self.verbose:
                print(f"Executing keyword search for: '{query}'")
            
            # Execute search
            results = self.index.search(query, search_params)
            
            # Extract hits
            hits = results.get('hits', [])
            
            if self.verbose:
                print(f"Keyword search returned {len(hits)} results")
            
            # Apply reranking if requested
            # Note: Reranking requires a processor which might not be available for keyword search
            if rerank and len(hits) > 0 and 'processor' in locals():
                processor = locals()['processor']
                if processor is not None:
                    adapted_hits = [self._adapt_meilisearch_result(hit) for hit in hits]
                    reranked_hits = SearchAlgorithms.rerank_results(query, adapted_hits, processor, limit, self.verbose)
                    
                    # Return the reranked results
                    return reranked_hits[:limit]
            
            # Adapt to the format expected by TextProcessor
            return [self._adapt_meilisearch_result(hit) for hit in hits[:limit]]
            
        except Exception as e:
            if self.verbose:
                print(f"Error in keyword search: {e}")
            return {"error": f"Error in keyword search: {str(e)}"}
    
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