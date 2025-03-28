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
        
        # Create dictionary for tracking embedder name mappings
        # This is crucial for maintaining consistency across operations
        self.embedder_name_map = {}
        
        # Performance tracking
        self._hit_rates = {}
        
        # Initialize name map with default model IDs
        self._register_embedder_name(self.dense_model_id)
        
        self.connect()

    def _register_embedder_name(self, model_id):
        """
        Register a model ID and generate its standardized embedder name.
        Store in the mapping dictionary for consistent reference.
        
        Args:
            model_id: The model identifier to register
            
        Returns:
            The standardized embedder name for this model ID
        """
        if model_id in self.embedder_name_map:
            return self.embedder_name_map[model_id]
            
        # Extract just the final part of the model name if it has slashes
        if "/" in model_id:
            model_base = model_id.split("/")[-1]
        else:
            model_base = model_id
            
        # Sanitize the base name for use as an embedder name
        embedder_name = re.sub(r'[^a-zA-Z0-9]', '_', model_base)
        
        # Ensure name doesn't exceed reasonable length
        max_name_length = 40
        if len(embedder_name) > max_name_length:
            embedder_name = embedder_name[:max_name_length]
            
        # Add prefix to make it clear this is a dense embedder
        embedder_name = f"dense_{embedder_name}"
        
        # Store mapping for future reference
        self.embedder_name_map[model_id] = embedder_name
        
        if self.verbose:
            print(f"Registered model ID '{model_id}' with embedder name '{embedder_name}'")
            
        return embedder_name
        
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection including vector configurations"""
        try:
            if not self.client:
                return {"error": "Not connected to Meilisearch"}
                
            collection_info = {}
            
            try:
                # Get index stats
                stats = self.index.get_stats()
                
                # Extract document count
                doc_count = 0
                if hasattr(stats, "number_of_documents"):
                    doc_count = stats.number_of_documents
                elif isinstance(stats, dict) and "numberOfDocuments" in stats:
                    doc_count = stats["numberOfDocuments"]
                    
                collection_info["points_count"] = doc_count
                
                # Get embedders information
                embedders = self._get_existing_embedders()
                
                # Format vector configurations
                vector_configs = {}
                for name, config in embedders.items():
                    vector_configs[name] = {
                        "dimensions": config.get("dimensions", 0),
                        "source": config.get("source", "unknown")
                    }
                    
                collection_info["vector_configs"] = vector_configs
                
                # Sparse vectors are simulated in Meilisearch
                collection_info["sparse_vector_configs"] = {}
                
                # Add embedder name mapping
                collection_info["embedder_name_map"] = self.embedder_name_map
                
                # Include performance data if available
                if hasattr(self, '_hit_rates') and self._hit_rates:
                    collection_info["performance"] = {"hit_rates": self._hit_rates}
                
                return collection_info
                
            except Exception as e:
                if self.verbose:
                    print(f"Error getting collection info: {e}")
                return {"error": f"Error getting collection info: {str(e)}"}
                
        except Exception as e:
            if self.verbose:
                print(f"Error in get_collection_info: {e}")
            return {"error": str(e)}
        
    def get_embedder_name(self, model_id=None):
        """
        Get the standardized embedder name for a model ID, creating it if needed.
        
        Args:
            model_id: Model ID to get embedder name for (default: self.dense_model_id)
            
        Returns:
            Standardized embedder name
        """
        model_id = model_id or self.dense_model_id
        
        # Check if we already have a mapping
        if model_id in self.embedder_name_map:
            embedder_name = self.embedder_name_map[model_id]
            if self.verbose:
                print(f"Using existing embedder name '{embedder_name}' for model '{model_id}'")
            return embedder_name
            
        # Create new mapping
        embedder_name = self._register_embedder_name(model_id)
        return embedder_name

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

    def _get_existing_embedders(self):
        """
        Get the existing embedders configuration from the Meilisearch index.
        
        Returns:
            Dictionary of embedder configurations, or empty dict if no embedders exist
        """
        try:
            # Try different methods to get embedders configuration
            embedders = {}
            
            # Method 1: Direct method call (newer client)
            try:
                embedders = self.index.get_embedders()
                # Convert object to dict if needed
                if embedders and not isinstance(embedders, dict):
                    embedders = {name: {"dimensions": config.dimensions, "source": config.source} 
                               for name, config in embedders.items()}
            except AttributeError:
                # Method 2: REST API call (fallback for older clients)
                if self.verbose:
                    print("Falling back to REST API to get embedders")
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
            
            if self.verbose:
                if embedders:
                    print(f"Found {len(embedders)} existing embedders:")
                    for name, config in embedders.items():
                        print(f"  - {name}: {config}")
                else:
                    print("No existing embedders found")
                    
            return embedders
        except Exception as e:
            if self.verbose:
                print(f"Error getting existing embedders: {e}")
            return {}

    def create_collection(self, recreate: bool = False) -> None:
        """Create Meilisearch index with vector search capabilities"""
        try:
            # Check if the index already exists
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
                # Get default embedder name for the primary model - always using "default" for simplicity
                primary_embedder_name = "default"
                
                # First, check for existing embedders
                existing_embedders = self._get_existing_embedders()
                
                if self.verbose:
                    print(f"Current embedders: {existing_embedders}")
                
                # Create the proper embedder configuration
                # CRITICAL: In Meilisearch v1.6+, user-provided embedders MUST use the 'userProvided' source
                embedder_settings = {
                    primary_embedder_name: {
                        'source': 'userProvided',
                        'dimensions': self.vector_dim
                    }
                }
                
                if self.verbose:
                    print(f"Configuring embedder with settings: {embedder_settings}")
                    
                # Update embedders with clear error handling
                try:
                    # Try the dedicated method first (newer Meilisearch versions)
                    try:
                        task = self.index.update_embedders(embedder_settings)
                        self._wait_for_task(task)
                        if self.verbose:
                            print(f"Vector search enabled with {self.vector_dim} dimensions using embedder: {primary_embedder_name}")
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
    
    def _update_embedders(self, embedder_settings):
        """
        Update embedders with robust error handling and multiple approaches.
        
        Args:
            embedder_settings: Dictionary of embedder configurations to add/update
        """
        if self.verbose:
            print(f"Updating embedders with: {embedder_settings}")
            
        try:
            # Try the dedicated method first (newer Meilisearch versions)
            task = self.index.update_embedders(embedder_settings)
            self._wait_for_task(task)
            if self.verbose:
                print(f"Vector search enabled successfully using update_embedders method")
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
                raise Exception(f"Failed to enable vector search via REST API: {response.text}")
        except Exception as e:
            raise Exception(f"Error updating embedders: {e}")
    
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

    def insert_embeddings(self, embeddings_with_payloads: List[Tuple[np.ndarray, Dict[str, Any]]]) -> None:
        """
        Insert documents with dense vector embeddings into Meilisearch.
        COMPLETELY rewritten to ensure proper vector formatting.
        
        Args:
            embeddings_with_payloads: List of (embedding, payload) tuples
        """
        if not embeddings_with_payloads:
            print("No embeddings provided to insert")
            return
        
        try:
            if not self.index:
                raise ValueError(f"Index '{self.collection_name}' not initialized")
            
            # CRITICAL: For Meilisearch v1.6+, embedder name must be "default" 
            # and must match exactly what's configured in the index
            embedder_name = "default"
            
            # Verify embedder exists
            existing_embedders = self._get_existing_embedders()
            
            if self.verbose:
                print(f"Current embedders configuration: {existing_embedders}")
                
            # Configure embedder if needed
            if embedder_name not in existing_embedders:
                if self.verbose:
                    print(f"Embedder '{embedder_name}' not configured. Setting it up now...")
                
                # Configure the embedder with proper dimensions
                embedder_settings = {
                    embedder_name: {
                        'source': 'userProvided',
                        'dimensions': self.vector_dim
                    }
                }
                
                try:
                    # Update embedders setting
                    self._update_embedders(embedder_settings)
                    existing_embedders = self._get_existing_embedders()
                    
                    if self.verbose:
                        print(f"Updated embedders configuration: {existing_embedders}")
                except Exception as e:
                    print(f"Failed to configure embedder: {e}")
                    print("Cannot proceed without a properly configured embedder")
                    return
            
            # Create a new list of validated documents
            validated_documents = []
            skipped_count = 0
            
            for i, (embedding, payload) in enumerate(embeddings_with_payloads):
                # STEP 1: Validate embedding
                if embedding is None:
                    if self.verbose:
                        print(f"Skipping document #{i+1}: embedding is None")
                    skipped_count += 1
                    continue
                    
                if not isinstance(embedding, np.ndarray):
                    if self.verbose:
                        print(f"Skipping document #{i+1}: embedding is not a numpy array (type: {type(embedding)})")
                    skipped_count += 1
                    continue
                    
                if embedding.size == 0:
                    if self.verbose:
                        print(f"Skipping document #{i+1}: embedding array is empty")
                    skipped_count += 1
                    continue
                    
                # STEP 2: Convert embedding to proper format
                try:
                    # Convert numpy array to regular list
                    vector_list = embedding.tolist()
                    
                    # Validate the list
                    if not isinstance(vector_list, list):
                        if self.verbose:
                            print(f"Skipping document #{i+1}: embedding.tolist() did not return a list (type: {type(vector_list)})")
                        skipped_count += 1
                        continue
                        
                    if len(vector_list) == 0:
                        if self.verbose:
                            print(f"Skipping document #{i+1}: embedding list is empty after conversion")
                        skipped_count += 1
                        continue
                    
                    # Check dimensions - must match configured embedder
                    embedder_config = existing_embedders.get(embedder_name, {})
                    expected_dim = embedder_config.get('dimensions', self.vector_dim)
                    
                    if len(vector_list) != expected_dim:
                        if self.verbose:
                            print(f"Warning: Document #{i+1} has vector dimension {len(vector_list)}, " 
                                f"expected {expected_dim}. Will try to fix.")
                        
                        # Try to fix dimensions
                        if len(vector_list) > expected_dim:
                            # Truncate
                            vector_list = vector_list[:expected_dim]
                        else:
                            # Pad with zeros
                            vector_list.extend([0.0] * (expected_dim - len(vector_list)))
                except Exception as e:
                    if self.verbose:
                        print(f"Skipping document #{i+1}: error converting embedding: {e}")
                    skipped_count += 1
                    continue
                
                # STEP 3: Create document with proper vector format
                try:
                    # Generate document ID if not provided
                    if 'id' not in payload:
                        doc_id = str(uuid.uuid4())
                    else:
                        doc_id = payload['id']
                    
                    # Create a new clean document
                    document = {
                        'id': doc_id
                    }
                    
                    # Copy all payload fields except reserved ones
                    for key, value in payload.items():
                        if key not in ['_vectors', 'id']:
                            document[key] = value
                    
                    # CRITICAL: Format vectors according to Meilisearch v1.6+ requirements
                    # Must be exactly: '_vectors': {'default': [values]}
                    document['_vectors'] = {
                        embedder_name: vector_list
                    }
                    
                    # Add to validated documents
                    validated_documents.append(document)
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Skipping document #{i+1}: error creating document: {e}")
                    skipped_count += 1
                    continue
            
            if skipped_count > 0:
                print(f"Warning: Skipped {skipped_count} documents due to invalid embeddings or errors")
                
            if not validated_documents:
                print("Error: No valid documents to insert after validation")
                return
                
            if self.verbose:
                print(f"Inserting {len(validated_documents)} validated documents")
                
                # Debug the first document
                first_doc = validated_documents[0]
                print(f"Sample document:")
                print(f"  ID: {first_doc['id']}")
                
                # Check content
                if 'content' in first_doc:
                    content = first_doc['content']
                    print(f"  Content: {content[:50]}..." if len(content) > 50 else content)
                elif 'text' in first_doc:
                    text = first_doc['text']
                    print(f"  Text: {text[:50]}..." if len(text) > 50 else text)
                    
                # Check vector format
                if '_vectors' in first_doc:
                    vectors = first_doc['_vectors']
                    print(f"  _vectors type: {type(vectors)}")
                    
                    if embedder_name in vectors:
                        vector_data = vectors[embedder_name]
                        print(f"  Vector data type: {type(vector_data)}")
                        print(f"  Vector dimensions: {len(vector_data)}")
                        print(f"  First 5 values: {vector_data[:5]}")
                    else:
                        print(f"  ERROR: embedder '{embedder_name}' not found in _vectors")
                else:
                    print(f"  ERROR: _vectors field missing")
            
            # Insert documents in small batches
            batch_size = 10  # Smaller batches for better error isolation
            successful_count = 0
            
            for i in range(0, len(validated_documents), batch_size):
                batch = validated_documents[i:i+batch_size]
                
                if self.verbose:
                    print(f"Inserting batch {i//batch_size + 1}/{(len(validated_documents)-1)//batch_size + 1} ({len(batch)} documents)")
                
                try:
                    # Insert the batch
                    task = self.index.add_documents(batch)
                    self._wait_for_task(task)
                    successful_count += len(batch)
                    
                    if self.verbose:
                        print(f"Successfully inserted batch of {len(batch)} documents")
                        
                except Exception as e:
                    error_message = str(e)
                    print(f"Error inserting batch: {error_message}")
                    
                    if "no vectors provided" in error_message.lower():
                        print("Trying one document at a time to identify problematic ones...")
                        
                        # Try inserting documents one by one
                        for doc in batch:
                            try:
                                # Double-check vector format for this specific document
                                if '_vectors' not in doc:
                                    print(f"Skipping document {doc['id']}: missing _vectors field")
                                    continue
                                    
                                if not isinstance(doc['_vectors'], dict):
                                    print(f"Skipping document {doc['id']}: _vectors is not a dictionary (type: {type(doc['_vectors'])})")
                                    continue
                                    
                                if embedder_name not in doc['_vectors']:
                                    print(f"Skipping document {doc['id']}: missing embedder '{embedder_name}' in _vectors")
                                    continue
                                    
                                vector_data = doc['_vectors'][embedder_name]
                                if not isinstance(vector_data, list):
                                    print(f"Skipping document {doc['id']}: vector data is not a list (type: {type(vector_data)})")
                                    continue
                                    
                                if len(vector_data) == 0:
                                    print(f"Skipping document {doc['id']}: vector list is empty")
                                    continue
                                
                                # Try to insert just this one document
                                task = self.index.add_documents([doc])
                                self._wait_for_task(task)
                                successful_count += 1
                                
                                if self.verbose:
                                    print(f"Successfully inserted document {doc['id']}")
                                    
                            except Exception as doc_e:
                                print(f"Error inserting document {doc['id']}: {doc_e}")
                    else:
                        # Other type of error - might not be related to vector format
                        print(f"Unknown error type, cannot recover automatically.")
            
            print(f"Successfully inserted {successful_count}/{len(validated_documents)} documents")
            
        except Exception as e:
            print(f"Critical error in insert_embeddings: {e}")
            import traceback
            traceback.print_exc()

    def search_hybrid(self, query: str, processor: Any, limit: int, 
                    prefetch_limit: int = 50, fusion_type: str = "rrf",
                    score_threshold: float = None, rerank: bool = False,
                    reranker_type: str = None):
        """
        Perform hybrid search combining dense vectors and keywords with improved debugging.
        """
        if processor is None:
            return {"error": "Hybrid search requires an embedding model"}
        
        try:
            if self.verbose:
                print(f"\n===== HYBRID SEARCH DETAILS =====")
                print(f"Query: '{query}'")
                print(f"Fusion type: {fusion_type}")
                print(f"Processor model: {getattr(processor, 'dense_model_id', 'unknown')}")
            
            # Generate query embedding
            try:
                query_vector = processor.get_embedding(query)
                if self.verbose:
                    print(f"Generated query embedding with shape: {query_vector.shape}")
                    print(f"Vector norm: {np.linalg.norm(query_vector)}")
                    print(f"Vector sample (first 5 values): {query_vector[:5]}")
            except Exception as e:
                print(f"Error generating query embedding: {e}")
                print("Falling back to keyword search")
                return self.search_keyword(query, limit)
            
            # Always use "default" as the embedder name for consistency with our configuration
            embedder_name = "default"
            
            # Verify that embedder exists
            existing_embedders = self._get_existing_embedders()
            
            if self.verbose:
                print(f"Available embedders: {existing_embedders}")
                
            if embedder_name not in existing_embedders:
                if self.verbose:
                    print(f"Warning: Embedder '{embedder_name}' not found in index")
                    print(f"Available embedders: {list(existing_embedders.keys())}")
                    print(f"Will try to use any available embedder instead")
                
                # Fall back to first available embedder if possible
                if existing_embedders:
                    embedder_name = next(iter(existing_embedders.keys()))
                    if self.verbose:
                        print(f"Falling back to embedder: '{embedder_name}'")
                else:
                    # Fallback to keyword search if no embedders available
                    if self.verbose:
                        print(f"No embedders found in index, falling back to keyword search")
                    return self.search_keyword(query, limit)
            
            # Determine semantic ratio based on fusion type
            semantic_ratio = 0.5  # Default balanced ratio
            if fusion_type.lower() == 'vector':
                semantic_ratio = 0.9  # Mostly vector search
            elif fusion_type.lower() == 'keyword':
                semantic_ratio = 0.1  # Mostly keyword search
            
            if self.verbose:
                print(f"Using semantic ratio: {semantic_ratio}")
                print(f"Using embedder name: '{embedder_name}'")
            
            # Build search parameters based on available embedder
            # This matches Meilisearch v1.6+ search parameter format exactly
            search_params = {
                'limit': limit * 3,  # Get more results for better filtering
                'attributesToRetrieve': ['*'],
                'vector': query_vector.tolist(),
                'hybrid': {
                    'embedder': embedder_name,
                    'semanticRatio': semantic_ratio
                }
            }
            
            if score_threshold is not None:
                search_params['scoreThreshold'] = score_threshold
                
            if self.verbose:
                print(f"Search parameters: {json.dumps(search_params, indent=2)}")
            
            # Execute hybrid search with both query text and vector
            try:
                results = self.index.search(query, search_params)
                
                # Extract hits
                hits = []
                if hasattr(results, 'hits'):
                    hits = results.hits
                elif isinstance(results, dict) and 'hits' in results:
                    hits = results['hits']
                
                if self.verbose:
                    print(f"Hybrid search returned {len(hits)} raw results")
                    
                    # Examine top hits' content in detail for debugging
                    if hits:
                        print("\nTop 3 raw hits content:")
                        for i, hit in enumerate(hits[:3]):
                            print(f"\nHit #{i+1}:")
                            if isinstance(hit, dict):
                                if '_semanticScore' in hit:
                                    print(f"  Semantic score: {hit['_semanticScore']}")
                                print(f"  ID: {hit.get('id', 'unknown')}")
                                print(f"  Content preview: {hit.get('content', hit.get('text', ''))[:100]}...")
                                print(f"  Content length: {len(hit.get('content', hit.get('text', '')))}")
                                vector_info = hit.get('_vectors', {})
                                if vector_info:
                                    for vec_name, vec in vector_info.items():
                                        if isinstance(vec, list):
                                            print(f"  Vector '{vec_name}' dimensions: {len(vec)}")
                            else:
                                print(f"  Type: {type(hit)}")
                
                # Filter out results with too small content or headers only
                filtered_hits = []
                for hit in hits:
                    # Get text content from hit (may be in 'text' or 'content' field)
                    text = ""
                    if isinstance(hit, dict):
                        if 'content' in hit and hit['content']:
                            text = hit['content']
                        elif 'text' in hit and hit['text']:
                            text = hit['text']
                    else:
                        if hasattr(hit, 'content') and getattr(hit, 'content'):
                            text = getattr(hit, 'content')
                        elif hasattr(hit, 'text') and getattr(hit, 'text'):
                            text = getattr(hit, 'text')
                    
                    # Skip chunks that are likely headers or too short for meaningful content
                    if len(text) < 100:  # Skip very short chunks
                        if self.verbose:
                            print(f"Filtering out short chunk ({len(text)} chars): '{text[:50]}'")
                        continue
                        
                    # Skip chunks that are only page numbers or section headers
                    if re.match(r'^[\d]+\s*$', text.strip()) or re.match(r'^[A-Z\s]+$', text.strip()):
                        if self.verbose:
                            print(f"Filtering out header/page number: '{text.strip()}'")
                        continue
                    
                    # Check if query terms appear in the text - boost relevance if they do
                    query_lower = query.lower()
                    text_lower = text.lower()
                    
                    # Keep this result
                    filtered_hits.append(hit)
                
                if self.verbose:
                    print(f"\nFiltered to {len(filtered_hits)} results with meaningful content")
                
                # If no quality results from hybrid search, fall back to keyword search
                if not filtered_hits:
                    if self.verbose:
                        print(f"No quality results from hybrid search, falling back to keyword search")
                    return self.search_keyword(query, limit)
            
                # Format results
                formatted_results = []
                for hit in filtered_hits[:limit]:
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
                    
                    # Adjust score based on query term presence
                    text = result.get('content', result.get('text', ''))
                    query_lower = query.lower()
                    if query_lower in text.lower():
                        # Boost score if the query appears in the text
                        result['score'] = min(0.95, result['score'] + 0.15)
                    
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
                
            except Exception as search_e:
                print(f"Error during hybrid search operation: {search_e}")
                print("Falling back to keyword search")
                return self.search_keyword(query, limit)
                
        except Exception as e:
            if self.verbose:
                print(f"Error in hybrid search: {e}")
                import traceback
                traceback.print_exc()
                print("Falling back to keyword search")
            return self.search_keyword(query, limit)
    
    def search_dense(self, query: str, processor: Any, limit: int, score_threshold: float = None,
                    rerank: bool = False, reranker_type: str = None):
        """
        Perform vector search according to Meilisearch v1.6+ requirements with detailed logging.
        """
        if processor is None:
            return {"error": "Vector search requires an embedding model"}
        
        try:
            if self.verbose:
                print(f"\n===== VECTOR SEARCH DETAILS =====")
                print(f"Query: '{query}'")
                print(f"Processor model: {getattr(processor, 'dense_model_id', 'unknown')}")
            
            # Generate query embedding with error handling
            try:
                query_vector = processor.get_embedding(query)
                if self.verbose:
                    print(f"Generated query embedding with shape: {query_vector.shape}")
                    print(f"Vector norm: {np.linalg.norm(query_vector)}")
                    print(f"Vector sample (first 5 values): {query_vector[:5]}")
            except Exception as e:
                print(f"Error generating query embedding: {e}")
                print("Falling back to keyword search")
                return self.search_keyword(query, limit)
            
            # Always use "default" as the embedder name for consistency with our configuration
            embedder_name = "default"
            
            # Verify that embedder exists
            existing_embedders = self._get_existing_embedders()
            
            if self.verbose:
                print(f"Available embedders: {existing_embedders}")
                
            if embedder_name not in existing_embedders:
                if self.verbose:
                    print(f"Warning: Embedder '{embedder_name}' not found in index")
                    print(f"Available embedders: {list(existing_embedders.keys())}")
                    print(f"Will try to use any available embedder instead")
                
                # Fall back to first available embedder if possible
                if existing_embedders:
                    embedder_name = next(iter(existing_embedders.keys()))
                    if self.verbose:
                        print(f"Falling back to embedder: '{embedder_name}'")
                else:
                    # Fallback to keyword search if no embedders available
                    if self.verbose:
                        print(f"No embedders found in index, falling back to keyword search")
                    return self.search_keyword(query, limit)
            
            if self.verbose:
                print(f"Using vector search with embedder name: '{embedder_name}'")
            
            # Set up search parameters per Meilisearch v1.6+ (vector-only search)
            # Either use vector-only search or hybrid with semanticRatio=1.0
            try:
                # First try vector-only search (recommended for pure semantic search)
                search_params = {
                    'limit': limit * 3,  # Get more results than needed for filtering
                    'attributesToRetrieve': ['*'],
                    'vector': query_vector.tolist(),
                }
                
                if score_threshold is not None:
                    search_params['scoreThreshold'] = score_threshold
                    
                if self.verbose:
                    print(f"Using pure vector search parameters")
                
                # Try pure vector search first (empty query)
                results = self.index.search('', search_params)
                
            except Exception as e:
                if self.verbose:
                    print(f"Pure vector search failed: {e}")
                    print("Trying hybrid search with 100% semantic ratio")
                
                # Fall back to hybrid search with 100% semantic ratio
                search_params = {
                    'limit': limit * 3,  # Get more results than needed for filtering
                    'attributesToRetrieve': ['*'],
                    'vector': query_vector.tolist(),
                    'hybrid': {
                        'embedder': embedder_name,
                        'semanticRatio': 1.0  # Pure semantic search
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
                print(f"Vector search returned {len(hits)} raw results")
                
                # Examine top hits' content in detail for debugging
                if hits:
                    print("\nTop 3 raw hits content:")
                    for i, hit in enumerate(hits[:3]):
                        print(f"\nHit #{i+1}:")
                        if isinstance(hit, dict):
                            if '_semanticScore' in hit:
                                print(f"  Semantic score: {hit['_semanticScore']}")
                            print(f"  ID: {hit.get('id', 'unknown')}")
                            print(f"  Content preview: {hit.get('content', hit.get('text', ''))[:100]}...")
                            print(f"  Content length: {len(hit.get('content', hit.get('text', '')))}")
                            vector_info = hit.get('_vectors', {})
                            if vector_info:
                                for vec_name, vec in vector_info.items():
                                    if isinstance(vec, list):
                                        print(f"  Vector '{vec_name}' dimensions: {len(vec)}")
                        else:
                            print(f"  Type: {type(hit)}")
                
            # Filter out problematic results (too short, headers only, etc.)
            filtered_hits = []
            for hit in hits:
                # Get text content from hit (may be in 'text' or 'content' field)
                text = ""
                if isinstance(hit, dict):
                    if 'content' in hit and hit['content']:
                        text = hit['content']
                    elif 'text' in hit and hit['text']:
                        text = hit['text']
                else:
                    if hasattr(hit, 'content') and getattr(hit, 'content'):
                        text = getattr(hit, 'content')
                    elif hasattr(hit, 'text') and getattr(hit, 'text'):
                        text = getattr(hit, 'text')
                
                # Skip chunks that are likely headers or too short for meaningful content
                if len(text) < 100:  # Skip very short chunks
                    if self.verbose:
                        print(f"Filtering out short chunk ({len(text)} chars): '{text[:50]}'")
                    continue
                    
                # Skip chunks that are only page numbers or section headers
                if re.match(r'^[\d]+\s*$', text.strip()) or re.match(r'^[A-Z\s]+$', text.strip()):
                    if self.verbose:
                        print(f"Filtering out header/page number: '{text.strip()}'")
                    continue
                    
                # Keep this result
                filtered_hits.append(hit)
            
            if self.verbose:
                print(f"\nFiltered to {len(filtered_hits)} results with meaningful content")
                
            # If no quality results from vector search, fall back to keyword search
            if not filtered_hits:
                if self.verbose:
                    print(f"No quality results from vector search, falling back to keyword search")
                return self.search_keyword(query, limit)
            
            # Format results
            formatted_results = []
            for hit in filtered_hits[:limit]:
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
                
                # Adjust score based on query term presence
                text = result.get('content', result.get('text', ''))
                query_lower = query.lower()
                if query_lower in text.lower():
                    # Boost score if the query appears in the text
                    result['score'] = min(0.95, result['score'] + 0.15)
                
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
                import traceback
                traceback.print_exc()
                print("Falling back to keyword search")
            return self.search_keyword(query, limit)

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
                print("\n=== PREPARING INDEX FOR KEYWORD SEARCH ===")
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
                    
                print("=== END INDEX PREPARATION ===\n")
            
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
    
    def _check_embedder_configured(self, embedder_name):
        """Check if an embedder is configured in the index"""
        try:
            # Get existing embedders
            existing_embedders = self._get_existing_embedders()
            
            # Check if the embedder exists
            return embedder_name in existing_embedders
        except Exception as e:
            if self.verbose:
                print(f"Error checking embedder configuration: {e}")
            return False
        
    def insert_embeddings_with_sparse(self, embeddings_with_sparse: List[Tuple[np.ndarray, Dict[str, Any], Tuple[List[int], List[float]]]]) -> None:
        """
        Insert documents with both dense and sparse vector embeddings.
        Rewritten to ensure proper Meilisearch v1.6+ vector format compliance.
        
        Args:
            embeddings_with_sparse: List of (dense_embedding, payload, sparse_vector) tuples
        """
        if not embeddings_with_sparse:
            print("No embeddings provided to insert")
            return
        
        try:
            if not self.index:
                raise ValueError(f"Index '{self.collection_name}' not initialized")
            
            # CRITICAL: For Meilisearch v1.6+, embedder name must be "default" 
            # and must match exactly what's configured in the index
            embedder_name = "default"
            
            # Verify embedder exists
            existing_embedders = self._get_existing_embedders()
            
            if self.verbose:
                print(f"Current embedders configuration: {existing_embedders}")
                
            # Configure embedder if needed
            if embedder_name not in existing_embedders:
                if self.verbose:
                    print(f"Embedder '{embedder_name}' not configured. Setting it up now...")
                
                # Configure the embedder with proper dimensions
                embedder_settings = {
                    embedder_name: {
                        'source': 'userProvided',
                        'dimensions': self.vector_dim
                    }
                }
                
                try:
                    # Update embedders setting
                    self._update_embedders(embedder_settings)
                    existing_embedders = self._get_existing_embedders()
                    
                    if self.verbose:
                        print(f"Updated embedders configuration: {existing_embedders}")
                except Exception as e:
                    print(f"Failed to configure embedder: {e}")
                    print("Cannot proceed without a properly configured embedder")
                    return
            
            # Create a new list of validated documents
            validated_documents = []
            skipped_count = 0
            
            for i, (dense_embedding, payload, sparse_vector) in enumerate(embeddings_with_sparse):
                # STEP 1: Validate dense embedding
                if dense_embedding is None:
                    if self.verbose:
                        print(f"Skipping document #{i+1}: dense embedding is None")
                    skipped_count += 1
                    continue
                    
                if not isinstance(dense_embedding, np.ndarray):
                    if self.verbose:
                        print(f"Skipping document #{i+1}: dense embedding is not a numpy array (type: {type(dense_embedding)})")
                    skipped_count += 1
                    continue
                    
                if dense_embedding.size == 0:
                    if self.verbose:
                        print(f"Skipping document #{i+1}: dense embedding array is empty")
                    skipped_count += 1
                    continue
                    
                # STEP 2: Convert dense embedding to proper format
                try:
                    # Convert numpy array to regular list
                    vector_list = dense_embedding.tolist()
                    
                    # Validate the list
                    if not isinstance(vector_list, list):
                        if self.verbose:
                            print(f"Skipping document #{i+1}: dense embedding.tolist() did not return a list (type: {type(vector_list)})")
                        skipped_count += 1
                        continue
                        
                    if len(vector_list) == 0:
                        if self.verbose:
                            print(f"Skipping document #{i+1}: dense embedding list is empty after conversion")
                        skipped_count += 1
                        continue
                    
                    # Check dimensions - must match configured embedder
                    embedder_config = existing_embedders.get(embedder_name, {})
                    expected_dim = embedder_config.get('dimensions', self.vector_dim)
                    
                    if len(vector_list) != expected_dim:
                        if self.verbose:
                            print(f"Warning: Document #{i+1} has vector dimension {len(vector_list)}, " 
                                f"expected {expected_dim}. Will try to fix.")
                        
                        # Try to fix dimensions
                        if len(vector_list) > expected_dim:
                            # Truncate
                            vector_list = vector_list[:expected_dim]
                        else:
                            # Pad with zeros
                            vector_list.extend([0.0] * (expected_dim - len(vector_list)))
                except Exception as e:
                    if self.verbose:
                        print(f"Skipping document #{i+1}: error converting dense embedding: {e}")
                    skipped_count += 1
                    continue
                
                # STEP 3: Create document with proper vector format
                try:
                    # Generate document ID if not provided
                    if 'id' not in payload:
                        doc_id = str(uuid.uuid4())
                    else:
                        doc_id = payload['id']
                    
                    # Create a new clean document
                    document = {
                        'id': doc_id
                    }
                    
                    # Copy all payload fields except reserved ones
                    for key, value in payload.items():
                        if key not in ['_vectors', 'id']:
                            document[key] = value
                    
                    # CRITICAL: Format vectors according to Meilisearch v1.6+ requirements
                    # For userProvided source, must be exactly: '_vectors': {'default': [values]}
                    document['_vectors'] = {
                        embedder_name: vector_list
                    }
                    
                    # Process sparse vector for keyword search (not directly used by Meilisearch for vectors)
                    # Meilisearch doesn't support sparse vectors directly, but we can create a text field
                    # that simulates sparse vector search
                    if sparse_vector:
                        try:
                            sparse_indices, sparse_values = sparse_vector
                            
                            # Create sparse terms for better text matching
                            sparse_terms = []
                            for idx, val in zip(sparse_indices, sparse_values):
                                if val > 0.05:  # Only include significant terms
                                    sparse_terms.append(f"term_{idx}")
                            
                            # Add sparse terms field for text search
                            if sparse_terms:
                                document['sparse_terms'] = " ".join(sparse_terms)
                                
                        except Exception as e:
                            if self.verbose:
                                print(f"Warning: Could not process sparse vector for document #{i+1}: {e}")
                                print("Continuing with dense vector only")
                    
                    # Add to validated documents
                    validated_documents.append(document)
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Skipping document #{i+1}: error creating document: {e}")
                    skipped_count += 1
                    continue
            
            if skipped_count > 0:
                print(f"Warning: Skipped {skipped_count} documents due to invalid embeddings or errors")
                
            if not validated_documents:
                print("Error: No valid documents to insert after validation")
                return
                
            if self.verbose:
                print(f"Inserting {len(validated_documents)} validated documents")
                
                # Debug the first document
                first_doc = validated_documents[0]
                print(f"Sample document:")
                print(f"  ID: {first_doc['id']}")
                
                # Check content
                if 'content' in first_doc:
                    content = first_doc['content']
                    print(f"  Content: {content[:50]}..." if len(content) > 50 else content)
                elif 'text' in first_doc:
                    text = first_doc['text']
                    print(f"  Text: {text[:50]}..." if len(text) > 50 else text)
                    
                # Check vector format
                if '_vectors' in first_doc:
                    vectors = first_doc['_vectors']
                    print(f"  _vectors type: {type(vectors)}")
                    
                    if embedder_name in vectors:
                        vector_data = vectors[embedder_name]
                        print(f"  Vector data type: {type(vector_data)}")
                        print(f"  Vector dimensions: {len(vector_data)}")
                        print(f"  First 5 values: {vector_data[:5]}")
                    else:
                        print(f"  ERROR: embedder '{embedder_name}' not found in _vectors")
                else:
                    print(f"  ERROR: _vectors field missing")
                    
                # Check sparse terms
                if 'sparse_terms' in first_doc:
                    print(f"  Sparse terms count: {len(first_doc['sparse_terms'].split())}")
            
            # Insert documents in small batches
            batch_size = 10  # Smaller batches for better error isolation
            successful_count = 0
            
            for i in range(0, len(validated_documents), batch_size):
                batch = validated_documents[i:i+batch_size]
                
                if self.verbose:
                    print(f"Inserting batch {i//batch_size + 1}/{(len(validated_documents)-1)//batch_size + 1} ({len(batch)} documents)")
                
                try:
                    # Insert the batch
                    task = self.index.add_documents(batch)
                    self._wait_for_task(task)
                    successful_count += len(batch)
                    
                    if self.verbose:
                        print(f"Successfully inserted batch of {len(batch)} documents")
                        
                except Exception as e:
                    error_message = str(e)
                    print(f"Error inserting batch: {error_message}")
                    
                    if "no vectors provided" in error_message.lower():
                        print("Trying one document at a time to identify problematic ones...")
                        
                        # Try inserting documents one by one
                        for doc in batch:
                            try:
                                # Double-check vector format for this specific document
                                if '_vectors' not in doc:
                                    print(f"Skipping document {doc['id']}: missing _vectors field")
                                    continue
                                    
                                if not isinstance(doc['_vectors'], dict):
                                    print(f"Skipping document {doc['id']}: _vectors is not a dictionary (type: {type(doc['_vectors'])})")
                                    continue
                                    
                                if embedder_name not in doc['_vectors']:
                                    print(f"Skipping document {doc['id']}: missing embedder '{embedder_name}' in _vectors")
                                    continue
                                    
                                vector_data = doc['_vectors'][embedder_name]
                                if not isinstance(vector_data, list):
                                    print(f"Skipping document {doc['id']}: vector data is not a list (type: {type(vector_data)})")
                                    continue
                                    
                                if len(vector_data) == 0:
                                    print(f"Skipping document {doc['id']}: vector list is empty")
                                    continue
                                
                                # Try to insert just this one document
                                task = self.index.add_documents([doc])
                                self._wait_for_task(task)
                                successful_count += 1
                                
                                if self.verbose:
                                    print(f"Successfully inserted document {doc['id']}")
                                    
                            except Exception as doc_e:
                                print(f"Error inserting document {doc['id']}: {doc_e}")
                    else:
                        # Other type of error - might not be related to vector format
                        print(f"Unknown error type, cannot recover automatically.")
            
            print(f"Successfully inserted {successful_count}/{len(validated_documents)} documents")
            
        except Exception as e:
            print(f"Critical error in insert_embeddings_with_sparse: {e}")
            import traceback
            traceback.print_exc()

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
                    # Skip if result is in error format
                    if isinstance(points[i], dict) and "error" in points[i]:
                        continue
                        
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
                    elif hasattr(points[i], "score"):
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
            
    def search_sparse(self, query: str, processor: Any, limit: int, score_threshold: float = None,
                    rerank: bool = False, reranker_type: str = None):
        """
        Perform sparse vector search.
        
        Meilisearch doesn't natively support sparse vectors, so we simulate it using sparse_terms.
        """
        if processor is None:
            return {"error": "Sparse search requires an embedding model"}
        
        try:
            # Generate sparse vector for the query
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
                # Get the embedder model ID from the payload or use default
                embedder_model_id = payload.get("metadata", {}).get("embedder", self.dense_model_id)
                
                # Get the standardized embedder name for this model
                embedder_name = self.get_embedder_name(embedder_model_id)
                
                # Create document with updated data
                document = {
                    'id': doc_id,
                    **payload  # Include all payload fields
                }
                
                # Add embedding as _vectors field
                document['_vectors'] = {
                    embedder_name: embedding.tolist()
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

    def list_embedders(self) -> List[str]:
        """
        List all configured embedder names in the index.
        
        Returns:
            List of embedder names
        """
        try:
            # Get existing embedders
            existing_embedders = self._get_existing_embedders()
            
            # Return the embedder names
            return list(existing_embedders.keys())
        except Exception as e:
            if self.verbose:
                print(f"Error listing embedders: {e}")
            return []
    
    def verify_embedder_compatibility(self, processor: Any) -> bool:
        """
        Verify that the embedder from the processor is compatible with
        the index.
        
        Args:
            processor: Document processor with embedding capabilities
            
        Returns:
            True if compatible, False otherwise
        """
        try:
            # Get the model ID from the processor
            dense_model_id = getattr(processor, 'dense_model_id', self.dense_model_id)
            
            # Get the standardized embedder name for this model
            embedder_name = self.get_embedder_name(dense_model_id)
            
            # Verify that this embedder exists in the index
            existing_embedders = self._get_existing_embedders()
            
            # Check if embedder exists
            compatible = embedder_name in existing_embedders
            
            if self.verbose:
                if compatible:
                    print(f"Processor embedder '{embedder_name}' is compatible with the index")
                else:
                    print(f"Warning: Processor embedder '{embedder_name}' is not configured in the index")
                    print(f"Available embedders: {list(existing_embedders.keys())}")
                    if existing_embedders:
                        print(f"Will need to fall back to an available embedder during search")
                    else:
                        print(f"No embedders available, search will fall back to keyword search")
            
            return compatible
        except Exception as e:
            if self.verbose:
                print(f"Error verifying embedder compatibility: {e}")
            return False
    
    def add_embedder(self, model_id: str, vector_dim: int) -> bool:
        """
        Add a new embedder configuration for a model ID.
        
        Args:
            model_id: The model identifier
            vector_dim: The vector dimension
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedder name
            embedder_name = self.get_embedder_name(model_id)
            
            # Check if embedder already exists
            existing_embedders = self._get_existing_embedders()
            if embedder_name in existing_embedders:
                if self.verbose:
                    print(f"Embedder '{embedder_name}' already exists")
                return True
            
            # Create embedder configuration
            embedder_settings = {
                embedder_name: {
                    'source': 'userProvided',
                    'dimensions': vector_dim
                }
            }
            
            if self.verbose:
                print(f"Adding new embedder '{embedder_name}' with dimensions: {vector_dim}")
            
            # Update embedders
            self._update_embedders(embedder_settings)
            
            return True
        except Exception as e:
            if self.verbose:
                print(f"Error adding embedder: {e}")
            return False
            
    def debug_vector_storage(self, document_id: str = None) -> Dict[str, Any]:
        """
        Debug the vector storage for a particular document or general configuration.
        
        Args:
            document_id: Optional document ID to examine
            
        Returns:
            Debug information
        """
        debug_info = {
            "index_exists": False,
            "embedders": {},
            "document_count": 0
        }
        
        try:
            if not self.index:
                debug_info["error"] = "Index not initialized"
                return debug_info
                
            debug_info["index_exists"] = True
            
            # Get embedder configurations
            embedders = self._get_existing_embedders()
            debug_info["embedders"] = embedders
            
            # Get document count
            try:
                stats = self.index.get_stats()
                
                if hasattr(stats, "number_of_documents"):
                    debug_info["document_count"] = stats.number_of_documents
                elif isinstance(stats, dict) and "numberOfDocuments" in stats:
                    debug_info["document_count"] = stats["numberOfDocuments"]
            except Exception as e:
                debug_info["stats_error"] = str(e)
            
            # If document ID is provided, get that document
            if document_id:
                try:
                    document = self.index.get_document(document_id)
                    
                    # Create a safe version of the document for debug output
                    safe_doc = {}
                    
                    if isinstance(document, dict):
                        # For each field, either copy it or summarize it if it's large
                        for k, v in document.items():
                            if k == '_vectors':
                                # For vectors, just show which embedders are present
                                if isinstance(v, dict):
                                    safe_doc['_vectors'] = {embedder: f"[Vector with {len(vec)} elements]" 
                                                          for embedder, vec in v.items()}
                                else:
                                    safe_doc['_vectors'] = f"[Unexpected format: {type(v)}]"
                            elif k == 'content' or k == 'text':
                                # For content/text, show a preview
                                if isinstance(v, str):
                                    safe_doc[k] = v[:100] + "..." if len(v) > 100 else v
                                else:
                                    safe_doc[k] = f"[Unexpected format: {type(v)}]"
                            else:
                                # For other fields, copy directly
                                safe_doc[k] = v
                    else:
                        safe_doc["error"] = f"Document is not a dictionary: {type(document)}"
                    
                    debug_info["document"] = safe_doc
                except Exception as e:
                    debug_info["document_error"] = str(e)
            
            # Get embedder name mapping
            debug_info["embedder_name_map"] = self.embedder_name_map
            
            return debug_info
            
        except Exception as e:
            debug_info["error"] = str(e)
            return debug_info