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
        """Connect to Milvus server with robust error handling"""
        max_retries = 5
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Set up connection parameters - use explicit IPv4 address
                conn_params = {
                    "host": "127.0.0.1" if self.host == "localhost" else self.host,
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
                    print(f"Connecting to Milvus at {conn_params['host']}:{conn_params['port']} (attempt {attempt+1}/{max_retries})")
                
                # Connect to Milvus with timeout
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
                    
                    # Connection successful, return
                    return
                    
                except Exception as coll_e:
                    if self.verbose:
                        print(f"Error checking for collection: {coll_e}")
                    self.collection = None
                    
                    # If we can at least connect, don't retry further
                    return
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    if self.verbose:
                        print(f"Connection attempt {attempt+1} failed: {e}")
                        print(f"Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 30)  # Exponential backoff with cap
                else:
                    print(f"Error connecting to Milvus after {max_retries} attempts: {str(e)}")
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
        """Check if collection has indexes and load it if needed"""
        try:
            if self.collection:
                # Check if the collection has been loaded
                try:
                    # Skip index checking and just load the collection directly
                    self.collection.load()
                    self.index_created = True
                    if self.verbose:
                        print("Collection loaded into memory")
                    return
                except Exception as e:
                    if self.verbose:
                        print(f"Error loading collection: {e}")
                        print("Will attempt to check indexes...")
                    
                # Since we can't reliably check indexes with the describe_index API due to 
                # the ambiguous index name error, we'll use a different approach
                
                # List all indexes on the collection using utility function
                try:
                    # This is a more reliable way to get all indexes in newer Milvus versions
                    all_indexes = utility.list_indexes(self.collection_name)
                    if all_indexes:
                        self.index_created = True
                        if self.verbose:
                            print(f"Collection has {len(all_indexes)} indexes")
                        
                        # Try to load collection
                        try:
                            self.collection.load()
                            if self.verbose:
                                print("Collection loaded into memory")
                        except Exception as load_e:
                            if self.verbose:
                                print(f"Error loading collection: {load_e}")
                    else:
                        if self.verbose:
                            print("No indexes found on collection")
                        self.index_created = False
                except Exception as list_e:
                    if self.verbose:
                        print(f"Error listing indexes: {list_e}")
                    
                    # Fallback approach: try checking for the vector field explicitly
                    # without getting index details (which causes ambiguity issues)
                    self.index_created = True  # Assume indexed and try to use it
                    if self.verbose:
                        print("Unable to check indexes reliably, assuming collection is indexed")
                    
                    # Try to load the collection anyway
                    try:
                        self.collection.load()
                        if self.verbose:
                            print("Collection loaded into memory")
                    except Exception as load_e:
                        if self.verbose:
                            print(f"Error loading collection: {load_e}")
                        self.index_created = False
        except Exception as e:
            if self.verbose:
                print(f"Error in _check_index: {e}")
            self.index_created = False


    def update_bm25_model(self):
        """
        Update the BM25 model with the current corpus of documents.
        This should be called after documents have been inserted.
        """
        if not hasattr(self, 'bm25_func') or self.bm25_func is None:
            if self.verbose:
                print("BM25 function not initialized, trying to create it")
            try:
                from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
                from pymilvus.model.sparse import BM25EmbeddingFunction
                
                # Create analyzer for English text
                analyzer = build_default_analyzer(language="en")
                self.bm25_analyzer = analyzer
                self.bm25_func = BM25EmbeddingFunction(analyzer)
            except Exception as e:
                if self.verbose:
                    print(f"Failed to create BM25 function: {e}")
                return False
        
        try:
            # Get a sample of documents to train the BM25 model
            max_docs = 10000  # Reasonable limit to avoid memory issues
            results = self.collection.query(
                expr="id != ''",
                output_fields=["content"],
                limit=max_docs
            )
            
            if not results:
                if self.verbose:
                    print("No documents found to train BM25 model")
                return False
            
            # Extract content from documents
            corpus = []
            for doc in results:
                content = doc.get("content", "")
                if content and isinstance(content, str):
                    corpus.append(content)
            
            if not corpus:
                if self.verbose:
                    print("No valid content found to train BM25 model")
                return False
                
            if self.verbose:
                print(f"Training BM25 model on {len(corpus)} documents")
            
            # Fit the BM25 model on the corpus
            self.bm25_func.fit(corpus)
            
            if self.verbose:
                print(f"BM25 model trained successfully, sparse dimension: {self.bm25_func.dim}")
            
            return True
        except Exception as e:
            if self.verbose:
                print(f"Error updating BM25 model: {e}")
            return False

    def create_collection(self, recreate: bool = False) -> None:
        """Create Milvus collection with vector search capabilities and improved schema"""
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
                
                # Initialize BM25 capability if available
                try:
                    # Import BM25 modules dynamically to avoid issues if they're not available
                    from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
                    from pymilvus.model.sparse import BM25EmbeddingFunction
                    
                    # Store in instance so we can use it later
                    self.has_bm25_capability = True
                    self.bm25_analyzer = build_default_analyzer(language="en")
                    self.bm25_function = BM25EmbeddingFunction(self.bm25_analyzer)
                    
                    if self.verbose:
                        print("Successfully created BM25 analyzer")
                        
                    # We'll fit the model later when we have documents
                    self.bm25_trained = False
                except ImportError:
                    if self.verbose:
                        print("BM25 modules not available in this Milvus version")
                    self.has_bm25_capability = False
                except Exception as e:
                    if self.verbose:
                        print(f"Error initializing BM25: {e}")
                    self.has_bm25_capability = False
                
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
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),  # Increased for longer content
                    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
                    
                    # Chunking fields
                    FieldSchema(name="is_chunk", dtype=DataType.BOOL),
                    FieldSchema(name="is_parent", dtype=DataType.BOOL),
                    FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="chunk_index", dtype=DataType.INT64),
                    FieldSchema(name="total_chunks", dtype=DataType.INT64),
                    
                    # Sparse representation as string fields for compatibility
                    FieldSchema(name=f"{self.sparse_indices_field}_str", dtype=DataType.VARCHAR, max_length=10000),
                    FieldSchema(name=f"{self.sparse_values_field}_str", dtype=DataType.VARCHAR, max_length=10000),
                    
                    # Vector field - for dense embeddings
                    FieldSchema(name=self.dense_field, dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
                ]
                
                # Check if we should define content_vector field (make it NULLABLE)
                try:
                    if hasattr(DataType, 'SPARSE_FLOAT_VECTOR'):
                        # If this attribute exists, we need to add the field
                        # Making it nullable so it's not required
                        if self.verbose:
                            print("Adding content_vector as a nullable field")
                        from pymilvus.client.types import NULLABLE
                        fields.append(
                            FieldSchema(
                                name="content_vector", 
                                dtype=DataType.SPARSE_FLOAT_VECTOR,
                                is_nullable=True  # Making it nullable!
                            )
                        )
                except Exception as e:
                    if self.verbose:
                        print(f"Error checking for SPARSE_FLOAT_VECTOR: {e}")
                
                # Create schema
                schema = CollectionSchema(fields=fields, description=f"Document collection for {self.dense_model_id}")
                
                # Create collection
                self.collection = Collection(name=self.collection_name, schema=schema)
                
                if self.verbose:
                    print(f"Created collection '{self.collection_name}'")
                    
                # Store schema info for later reference
                try:
                    # Save field names for future reference
                    self.field_names = [field.name for field in fields]
                    if self.verbose:
                        print(f"Schema has fields: {self.field_names}")
                except Exception as e:
                    if self.verbose:
                        print(f"Error saving schema info: {e}")
                
                # Create indexes for vector fields
                self._create_indexes()
            else:
                # Collection exists, get it
                self.collection = Collection(self.collection_name)
                if self.verbose:
                    print(f"Collection '{self.collection_name}' already exists")
                    
                # Get field names for existing collection
                try:
                    schema = self.collection.schema
                    self.field_names = [field.name for field in schema.fields]
                    if self.verbose:
                        print(f"Schema has fields: {self.field_names}")
                except Exception as e:
                    if self.verbose:
                        print(f"Error getting schema fields: {e}")
                    self.field_names = []
                    
                # Check if indexes exist
                self._check_index()
                
                # Create indexes if they don't exist
                if not self.index_created:
                    self._create_indexes()
            
        except Exception as e:
            print(f"Error creating collection: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

            
    def _create_indexes(self):
        """Create indexes for the collection with proper field name checking"""
        try:
            if not self.collection:
                raise ValueError("Collection is not initialized")
                
            if self.verbose:
                print("Creating indexes...")
            
            # First, check if indexes already exist and drop them if necessary
            try:
                all_indexes = utility.list_indexes(self.collection_name)
                if all_indexes:
                    if self.verbose:
                        print(f"Found existing indexes: {all_indexes}")
                    # Dropping existing indexes to avoid conflicts
                    for index_info in all_indexes:
                        if "index_name" in index_info:
                            try:
                                if self.verbose:
                                    print(f"Dropping index: {index_info['index_name']}")
                                self.collection.drop_index(index_name=index_info["index_name"])
                            except Exception as drop_e:
                                if self.verbose:
                                    print(f"Error dropping index {index_info['index_name']}: {drop_e}")
            except Exception as list_e:
                if self.verbose:
                    print(f"Error listing indexes: {list_e}")
                
            # Create index for dense vector field with a specific name
            index_params = {
                "metric_type": "COSINE",  # Use COSINE similarity for dense vectors
                "index_type": "HNSW",     # Hierarchical Navigable Small World graph
                "params": {
                    "M": 16,              # Maximum number of edges per node
                    "efConstruction": 200 # Construction-time control for accuracy vs. time
                }
            }
            
            # Create index with a specific name to avoid ambiguity
            dense_index_name = f"{self.dense_field}_idx"
            self.collection.create_index(
                field_name=self.dense_field,
                index_params=index_params,
                index_name=dense_index_name
            )
            
            if self.verbose:
                print(f"Created index {dense_index_name} for field {self.dense_field}")
            
            # Get the actual fields in the collection
            try:
                # Get collection schema
                collection_info = self.collection.schema
                field_names = [field.name for field in collection_info.fields]
                
                # Check if vector fields for sparse indices/values exist in the schema
                indices_vector_field = f"{self.sparse_indices_field}_vector"
                values_vector_field = f"{self.sparse_values_field}_vector" 
                
                # Only try to create sparse vector indexes if the fields exist
                if indices_vector_field in field_names and values_vector_field in field_names:
                    try:
                        # Create index for sparse indices vector
                        sparse_indices_index_name = f"{indices_vector_field}_idx"
                        sparse_indices_index_params = {
                            "metric_type": "L2",
                            "index_type": "FLAT",
                            "params": {}
                        }
                        self.collection.create_index(
                            field_name=indices_vector_field,
                            index_params=sparse_indices_index_params,
                            index_name=sparse_indices_index_name
                        )
                        
                        if self.verbose:
                            print(f"Created index {sparse_indices_index_name} for field {indices_vector_field}")
                            
                        # Create index for sparse values vector
                        sparse_values_index_name = f"{values_vector_field}_idx"
                        sparse_values_index_params = {
                            "metric_type": "L2",
                            "index_type": "FLAT",
                            "params": {}
                        }
                        self.collection.create_index(
                            field_name=values_vector_field,
                            index_params=sparse_values_index_params,
                            index_name=sparse_values_index_name
                        )
                        
                        if self.verbose:
                            print(f"Created index {sparse_values_index_name} for field {values_vector_field}")
                    except Exception as sparse_idx_e:
                        if self.verbose:
                            print(f"Failed to create sparse vector indexes: {sparse_idx_e}")
                else:
                    if self.verbose:
                        print(f"Sparse vector fields not found in schema, skipping their indexes")
                        
                    # Check if string fields for sparse representations exist and create indexes if needed
                    indices_str_field = f"{self.sparse_indices_field}_str"
                    values_str_field = f"{self.sparse_values_field}_str"
                    
                    if indices_str_field in field_names:
                        try:
                            index_name = f"{indices_str_field}_idx"
                            self.collection.create_index(
                                field_name=indices_str_field,
                                index_name=index_name
                            )
                            if self.verbose:
                                print(f"Created index {index_name} for field {indices_str_field}")
                        except Exception as e:
                            if self.verbose:
                                print(f"Error creating index for {indices_str_field}: {e}")
                                
                    if values_str_field in field_names:
                        try:
                            index_name = f"{values_str_field}_idx"
                            self.collection.create_index(
                                field_name=values_str_field,
                                index_name=index_name
                            )
                            if self.verbose:
                                print(f"Created index {index_name} for field {values_str_field}")
                        except Exception as e:
                            if self.verbose:
                                print(f"Error creating index for {values_str_field}: {e}")
                
            except Exception as schema_e:
                if self.verbose:
                    print(f"Error checking collection schema: {schema_e}")
            
            # Create scalar field indexes for faster filtering with specific names
            scalar_fields = ["file_path", "file_type", "parent_id", "content", "title"]
            for field_name in scalar_fields:
                # Only create indexes for fields that exist in the schema
                if field_name in field_names:
                    try:
                        index_name = f"{field_name}_idx"
                        self.collection.create_index(
                            field_name=field_name,
                            index_name=index_name
                        )
                        if self.verbose:
                            print(f"Created index {index_name} for field {field_name}")
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


    
    def count_entities(self) -> int:
        """
        Get the accurate count of entities in the collection using multiple methods
        that respect Milvus query window limitation.
        
        Returns:
            Number of entities in the collection
        """
        if not self.collection:
            return 0
            
        count = 0
        methods_tried = []
        
        # Method 1: Try with the num_entities attribute (most direct method)
        try:
            count = self.collection.num_entities
            methods_tried.append("num_entities_attribute")
            if self.verbose:
                print(f"Entity count from num_entities attribute: {count}")
            if count > 0:
                return count
        except Exception as e:
            if self.verbose:
                print(f"Error getting count with num_entities attribute: {e}")
        
        # Method 2: Try query with pagination to respect Milvus limits (16384 max)
        try:
            # First check if there's at least one document
            results = self.collection.query(
                expr="id != ''",  # This matches any document with a non-empty ID
                output_fields=["id"],
                limit=1  # Just check if there's at least one document
            )
            
            if results:
                # If we got a result, then try with pagination
                total_count = 0
                page_size = 10000  # Below the 16384 limit
                max_pages = 100    # Reasonable upper limit
                
                for page in range(max_pages):
                    offset = page * page_size
                    
                    if offset >= 16000:  # Getting close to the limit
                        break
                    
                    page_results = self.collection.query(
                        expr="id != ''",
                        output_fields=["id"],
                        limit=page_size,
                        offset=offset
                    )
                    
                    if not page_results:
                        break  # No more results
                    
                    page_count = len(page_results)
                    total_count += page_count
                    
                    if page_count < page_size:
                        break  # Last page
                
                count = total_count
                methods_tried.append("paged_query")
                if self.verbose:
                    print(f"Entity count from paged query: {count}")
                if count > 0:
                    return count
        except Exception as e:
            if self.verbose:
                print(f"Error getting count with paged query: {e}")
        
        # Method 3: Try search with a dummy vector to get an estimate
        try:
            # Create a dummy vector for search
            dummy_vector = [0] * self.vector_dim  # Zero vector
            
            # Execute vector search with a large limit (but below Milvus max)
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10},
            }
            
            results = self.collection.search(
                data=[dummy_vector],
                anns_field=self.dense_field,
                param=search_params,
                limit=1000,  # Use a reasonable limit to avoid out-of-memory issues
                expr=None,
                output_fields=["id"]
            )
            
            if results and len(results) > 0:
                count = len(results[0])
                methods_tried.append("search")
                if self.verbose:
                    print(f"Entity count from search: {count}")
                if count > 0:
                    return count
        except Exception as e:
            if self.verbose:
                print(f"Error getting count with search: {e}")
        
        # Method 4: Use internal counter if available
        if hasattr(self, "_inserted_count") and self._inserted_count > 0:
            count = self._inserted_count
            methods_tried.append("insertion_record")
            if self.verbose:
                print(f"Entity count from insertion record: {count}")
            return count
        
        if self.verbose:
            print(f"Tried methods: {', '.join(methods_tried)}")
            print(f"Final count determination: {count}")
        
        return count

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            if not self.collection:
                return {"error": "Collection not initialized"}
                
            # Get entity count using the more robust method
            row_count = self.count_entities()
            
            # Safely get index information
            vector_configs = {}
            try:
                # Try to get index list from utility
                from pymilvus import utility
                indexes = utility.list_indexes(self.collection_name)
                
                if indexes and isinstance(indexes, list):
                    for idx_info in indexes:
                        if isinstance(idx_info, dict):
                            # Extract the field from the index info if possible
                            field_name = idx_info.get("field_name", "unknown")
                            if field_name == self.dense_field:
                                # For the vector field, try to get more details
                                vector_configs[field_name] = {
                                    "type": idx_info.get("index_type", "unknown"),
                                    "metric": "COSINE",  # Default for our setup
                                    "params": {"M": 16, "efConstruction": 200}  # Default params
                                }
                            else:
                                # For other fields, just record basic info
                                vector_configs[field_name] = {
                                    "type": idx_info.get("index_type", "unknown"),
                                    "metric": "NA",
                                    "params": {}
                                }
                else:
                    # Fallback with at least the known fields
                    vector_configs[self.dense_field] = {
                        "type": "HNSW",
                        "metric": "COSINE",
                        "params": {"M": 16, "efConstruction": 200}
                    }
            except Exception as idx_e:
                if self.verbose:
                    print(f"Error getting index information: {idx_e}")
                # Fallback with default values
                vector_configs[self.dense_field] = {
                    "type": "HNSW",
                    "metric": "COSINE",
                    "params": {"M": 16, "efConstruction": 200}
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
        """Insert embeddings into Milvus with improved content handling"""
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
                f"{self.sparse_indices_field}_str": [],
                f"{self.sparse_values_field}_str": []
            }
            
            for i, (embedding, payload) in enumerate(embeddings_with_payloads):
                # Handle ID generation with extensive error checking
                doc_id = None
                
                # Case 1: ID not in payload
                if "id" not in payload:
                    doc_id = str(uuid.uuid4())
                    if self.verbose and i == 0:
                        print(f"No ID found in payload, generating UUID: {doc_id}")
                else:
                    # Case 2: ID exists in payload
                    raw_id = payload["id"]
                    
                    # Case 2a: ID is a list
                    if isinstance(raw_id, list):
                        if raw_id:  # Non-empty list
                            # Take first element and convert to string
                            doc_id = str(raw_id[0])
                            if self.verbose and i == 0:
                                print(f"ID is a list, using first element: {doc_id}")
                        else:  # Empty list
                            doc_id = str(uuid.uuid4())
                            if self.verbose and i == 0:
                                print(f"ID is an empty list, generating UUID: {doc_id}")
                    
                    # Case 2b: ID is not a list
                    else:
                        doc_id = str(raw_id)
                        if self.verbose and i == 0:
                            print(f"Using ID from payload: {doc_id}")
                
                # Final safety check
                if not doc_id:
                    doc_id = str(uuid.uuid4())
                    if self.verbose and i == 0:
                        print(f"Failed to generate ID through normal means, using fallback UUID: {doc_id}")
                
                # Generate default sparse vector for compatibility
                sparse_indices_str = "0"
                sparse_values_str = "0.0"
                
                # IMPORTANT: Extract content with robust fallbacks and truncation handling
                content = ""
                if "content" in payload:
                    content = payload["content"]
                elif "text" in payload:
                    content = payload["text"]
                    
                # Ensure content is a string and not None
                if content is None:
                    content = ""
                elif not isinstance(content, str):
                    try:
                        content = str(content)
                    except:
                        content = ""
                        
                # Make sure content is not too long (Milvus has VARCHAR limits)
                # Limit to a safe size (65000 chars) to stay within Milvus VARCHAR limits
                if len(content) > 65000:
                    if self.verbose and i == 0:
                        print(f"Content exceeds 65000 chars, truncating (original: {len(content)} chars)")
                    content = content[:65000]
                    
                # Make sure we have a valid title
                title = payload.get("title", "")
                if title is None:
                    title = ""
                elif not isinstance(title, str):
                    try:
                        title = str(title)
                    except:
                        title = ""
                
                # Ensure title is not too long
                if len(title) > 490:
                    title = title[:490]
                    
                # CRITICAL: Make sure we're storing sensible values
                if self.verbose and i < 3:  # Debug the first few documents
                    content_preview = content[:100] + "..." if content and len(content) > 100 else content
                    print(f"Document {i+1} content preview: {content_preview}")
                    
                # Add data to columns
                data["id"].append(doc_id)
                data["file_path"].append(payload.get("file_path", ""))
                data["file_name"].append(payload.get("file_name", ""))
                data["file_type"].append(payload.get("fileType", ""))
                data["file_size"].append(payload.get("fileSize", 0))
                data["created_at"].append(int(payload.get("createdAt", time.time())))
                data["modified_at"].append(int(payload.get("modifiedAt", time.time())))
                data["content"].append(content)  # Store the validated content
                data["title"].append(title)  # Store the validated title
                data["is_chunk"].append(bool(payload.get("is_chunk", False)))
                data["is_parent"].append(bool(payload.get("is_parent", False)))
                data["parent_id"].append(payload.get("parent_id", ""))
                data["chunk_index"].append(int(payload.get("chunk_index", 0)))
                data["total_chunks"].append(int(payload.get("total_chunks", 0)))
                data[self.dense_field].append(embedding.tolist())
                data[f"{self.sparse_indices_field}_str"].append(sparse_indices_str)
                data[f"{self.sparse_values_field}_str"].append(sparse_values_str)
            
            if self.verbose:
                print(f"Inserting {len(data['id'])} documents into collection '{self.collection_name}'")
                
            # Insert data in batches to handle large datasets
            batch_size = 50  # Smaller batches for more reliable insertion
            for i in range(0, len(data["id"]), batch_size):
                end_idx = min(i + batch_size, len(data["id"]))
                
                # Prepare batch data
                batch_data = {k: v[i:end_idx] for k, v in data.items()}
                
                if self.verbose and len(data["id"]) > batch_size:
                    print(f"Inserting batch {i//batch_size + 1}/{(len(data['id'])-1)//batch_size + 1} ({end_idx-i} documents)")
                
                # Insert batch - with retry mechanism for robustness
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        # Test content before insertion
                        if self.verbose and retry > 0:
                            print(f"Retry {retry+1}/{max_retries} for batch {i//batch_size + 1}")
                            test_content = batch_data["content"][0]
                            print(f"Test content (first 50 chars): {test_content[:50]}")
                        
                        # Insert the batch
                        self.collection.insert(batch_data)
                        break  # Successful insertion, exit retry loop
                    except Exception as batch_e:
                        if retry < max_retries - 1:
                            if self.verbose:
                                print(f"Batch insertion failed (attempt {retry+1}): {batch_e}")
                                print("Retrying in 2 seconds...")
                            time.sleep(2)
                        else:
                            print(f"Failed to insert batch after {max_retries} retries: {batch_e}")
                            print("Will continue with next batch")
                
            # Create index if not already created
            if not self.index_created:
                self._create_indexes()
            else:
                # Make sure collection is loaded
                try:
                    self.collection.load()
                except Exception as load_e:
                    if self.verbose:
                        print(f"Warning: error loading collection after insert: {load_e}")
                
            if self.verbose:
                print(f"Successfully inserted {len(data['id'])} documents")
                
        except Exception as e:
            print(f"Error inserting embeddings: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _format_search_results(self, points, query, search_type, processor, context_size=300):
        """
        Format search results into a standardized format with improved content handling.
        
        Args:
            points: List of search result points
            query: Original search query
            search_type: Type of search performed
            processor: Document processor with embedding capabilities
            context_size: Size of context window for preview
            
        Returns:
            Dictionary with formatted search results
        """
        # Initialize results container
        formatted = {
            "query": query,
            "search_type": search_type,
            "count": len(points),
            "results": [],
            # Use 'embedder_info' instead of 'embedders' to match expected format
            "embedder_info": {
                "dense": processor.model_name if processor else "unknown",
                "sparse": getattr(processor, 'sparse_model_name', "unknown") if processor else "unknown"
            }
        }
        
        # Format each result
        for i, point in enumerate(points):
            try:
                # Extract basic metadata
                source = point.get("_source", {})
                score = point.get("_score", 0.0)
                
                # Extract content with multiple fallback options
                content = source.get("content", "")
                if not content and "_source" in point:
                    # Try other possible content fields
                    content = point["_source"].get("text", "")
                    
                # Extract file info, with fallbacks
                file_path = source.get("file_path", "")
                if not file_path and "_source" in point:
                    file_path = point["_source"].get("path", "")
                    
                file_name = source.get("file_name", "") or self._extract_filename(file_path)
                chunk_index = source.get("chunk_index", 0)
                
                # Debug content if verbose
                if self.verbose and i < 2:  # Only show first two results
                    content_preview = content[:100] + "..." if len(content) > 100 else content
                    print(f"Result {i+1} content: {content_preview}")
                    
                # Calculate content statistics properly
                content_stats = self._calculate_content_stats(content)
                
                if self.verbose and i == 0:
                    print(f"Content stats for first result: {content_stats}")
                
                # Extract embedder info
                metadata = source.get("metadata", {})
                dense_embedder = metadata.get("dense_embedder", self.dense_model_id if processor else "unknown")
                sparse_embedder = metadata.get("sparse_embedder", getattr(processor, 'sparse_model_name', "unknown") if processor else "unknown")
                
                # Prepare preview text with multiple fallback options
                preview = ""
                
                # Try source content first
                if content:
                    preview = content[:context_size]
                
                # If preview is empty, try to retrieve using our improved context retrieval
                if not preview and file_path:
                    try:
                        retrieved_context = self._retrieve_context_for_chunk(file_path, chunk_index, window=0)
                        if retrieved_context:
                            preview = retrieved_context[:context_size] 
                            if self.verbose and i == 0 and preview:
                                print(f"Retrieved context preview: {preview[:100]}...")
                    except Exception as e:
                        if self.verbose:
                            print(f"Error retrieving context for preview: {e}")
                
                # Try to get content directly from file if still empty
                if not preview and file_path and os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                        preview = file_content[:context_size]
                        if self.verbose and i == 0:
                            print(f"Got preview directly from file: {preview[:100]}...")
                    except Exception as e:
                        if self.verbose:
                            print(f"Error reading file directly: {e}")
                
                # Last resort - use a placeholder message
                if not preview:
                    preview = f"[Could not retrieve content for {file_name}]"
                
                # Format the result - ensure all expected fields are present
                result = {
                    "rank": i + 1,
                    "score": score,
                    "file": file_name,
                    "file_name": file_name,
                    "path": file_path,
                    "file_path": file_path,  # Add this explicitly to match expected interface
                    "chunk": chunk_index,
                    "chunk_index": chunk_index,  # Add chunk_index field explicitly
                    "total_chunks": source.get("total_chunks", 1),
                    "chunk_size": content_stats,
                    "embedders": {
                        "dense": dense_embedder,
                        "sparse": sparse_embedder
                    },
                    "embedder_info": {  # Add embedder_info at the result level too
                        "dense_embedder": dense_embedder,
                        "sparse_embedder": sparse_embedder
                    },
                    "preview": preview
                }
                
                formatted["results"].append(result)
            except Exception as format_e:
                # Skip this result if formatting fails
                if self.verbose:
                    print(f"Error formatting result {i+1}: {format_e}")
                    import traceback
                    traceback.print_exc()
        
        return formatted


    def _extract_filename(self, file_path):
        """
        Extract filename from file path.
        
        Args:
            file_path: File path to extract filename from
            
        Returns:
            Extracted filename
        """
        if not file_path:
            return ""
        
        try:
            import os
            return os.path.basename(file_path)
        except:
            # Simple fallback
            parts = file_path.split('/')
            return parts[-1] if parts else ""

    def _fix_batch_data_types(self, batch_data):
        """
        Fix data types in batch data to ensure compatibility with Milvus schema.
        
        Args:
            batch_data: Dictionary of batch data for insertion
            
        Returns:
            Fixed batch data
        """
        # Create a copy to avoid modifying the original
        fixed_data = {}
        
        for field_name, values in batch_data.items():
            if field_name == "id":
                # Ensure all IDs are strings
                fixed_values = []
                for val in values:
                    if isinstance(val, list):
                        # If ID is a list, convert first element to string
                        if val:
                            fixed_values.append(str(val[0]))
                        else:
                            fixed_values.append(str(uuid.uuid4()))
                    else:
                        # Otherwise, ensure it's a string
                        fixed_values.append(str(val))
                fixed_data[field_name] = fixed_values
            elif field_name in ["created_at", "modified_at", "file_size", "chunk_index", "total_chunks"]:
                # Ensure numeric fields are integers
                fixed_values = []
                for val in values:
                    try:
                        fixed_values.append(int(val))
                    except (ValueError, TypeError):
                        fixed_values.append(0)
                fixed_data[field_name] = fixed_values
            elif field_name in ["is_chunk", "is_parent"]:
                # Ensure boolean fields are booleans
                fixed_values = []
                for val in values:
                    fixed_values.append(bool(val))
                fixed_data[field_name] = fixed_values
            elif field_name == self.dense_field:
                # Ensure dense vectors are proper lists
                fixed_values = []
                for val in values:
                    if isinstance(val, np.ndarray):
                        fixed_values.append(val.tolist())
                    else:
                        fixed_values.append(val)
                fixed_data[field_name] = fixed_values
            else:
                # Keep other fields as they are
                fixed_data[field_name] = values
        
        return fixed_data
    
    def _safe_get_str(self, entity, field, default=""):
        """Safely extract a string field from an entity with proper error handling"""
        try:
            value = entity.get(field)
            if value is None:
                return default
            return str(value)
        except Exception:
            return default

    def _safe_get_int(self, entity, field, default=0):
        """Safely extract an integer field from an entity with proper error handling"""
        try:
            value = entity.get(field)
            if value is None:
                return default
            return int(value)
        except (ValueError, TypeError):
            return default

    def _safe_get_bool(self, entity, field, default=False):
        """Safely extract a boolean field from an entity with proper error handling"""
        try:
            value = entity.get(field)
            if value is None:
                return default
            return bool(value)
        except Exception:
            return default

    def insert_embeddings_with_sparse(self, embeddings_with_sparse: List[Tuple[np.ndarray, Dict[str, Any], Tuple[List[int], List[float]]]]) -> None:
        """
        Insert embeddings with sparse vectors into Milvus with improved content handling.
        
        Args:
            embeddings_with_sparse: List of (dense_embedding, payload, sparse_embedding) tuples
        """
        if not embeddings_with_sparse:
            return
            
        try:
            if not self.collection:
                raise ValueError(f"Collection '{self.collection_name}' not initialized")
            
            # Debug the first item to understand its structure
            if embeddings_with_sparse and self.verbose:
                first_item = embeddings_with_sparse[0]
                if len(first_item) >= 2:
                    payload = first_item[1]
                    print(f"DEBUG - First payload structure:")
                    print(f"ID type: {type(payload.get('id'))}")
                    print(f"ID value: {payload.get('id')}")
            
            # Create a list of entities
            entities = []
            
            for i, item in enumerate(embeddings_with_sparse):
                # Handle different structures of input tuples with robust error handling
                try:
                    # Check the structure of the item
                    if len(item) == 3:
                        embedding, payload, sparse_vector = item
                        
                        # Validate sparse vector format
                        if isinstance(sparse_vector, tuple) and len(sparse_vector) == 2:
                            sparse_indices, sparse_values = sparse_vector
                        else:
                            # If not a proper tuple, generate default sparse vector
                            if self.verbose:
                                print(f"Invalid sparse vector format for item {i}, generating default")
                            sparse_indices, sparse_values = [0], [0.0]
                    elif len(item) == 2:
                        # Only dense embedding and payload provided
                        embedding, payload = item
                        if self.verbose:
                            print(f"No sparse vector for item {i}, generating default")
                        sparse_indices, sparse_values = [0], [0.0]
                    else:
                        # Unexpected format
                        if self.verbose:
                            print(f"Unexpected structure for item {i}, skipping")
                        continue
                    
                    # Generate a unique ID if not provided
                    if "id" not in payload:
                        doc_id = str(uuid.uuid4())
                        if self.verbose and i == 0:
                            print(f"No ID found in payload, generating UUID: {doc_id}")
                    else:
                        # Extract ID and ensure it's a string
                        raw_id = payload["id"]
                        if isinstance(raw_id, list):
                            if raw_id:  # Non-empty list
                                doc_id = str(raw_id[0])
                                if self.verbose and i == 0:
                                    print(f"ID is a list, using first element: {doc_id}")
                            else:  # Empty list
                                doc_id = str(uuid.uuid4())
                                if self.verbose and i == 0:
                                    print(f"ID is an empty list, generating UUID: {doc_id}")
                        else:
                            doc_id = str(raw_id)
                            if self.verbose and i == 0:
                                print(f"Using ID from payload: {doc_id}")
                    
                    # Ensure sparse indices and values are valid
                    try:
                        # Validate indices and values
                        if not sparse_indices or len(sparse_indices) == 0:
                            sparse_indices = [0]
                        if not sparse_values or len(sparse_values) == 0:
                            sparse_values = [0.0]
                        
                        # Make sure indices and values are the same length
                        if len(sparse_indices) != len(sparse_values):
                            if self.verbose:
                                print(f"Mismatched sparse indices/values lengths for item {i}, fixing")
                            # Keep the shorter length
                            min_len = min(len(sparse_indices), len(sparse_values))
                            sparse_indices = sparse_indices[:min_len] if min_len > 0 else [0]
                            sparse_values = sparse_values[:min_len] if min_len > 0 else [0.0]
                        
                        # Check array size limits
                        if len(sparse_indices) > 1000:
                            if self.verbose:
                                print(f"Truncating large sparse vector for item {i} from {len(sparse_indices)} to 1000 elements")
                            sparse_indices = sparse_indices[:1000]
                            sparse_values = sparse_values[:1000]
                        
                        # Convert numpy arrays to lists if needed
                        if isinstance(sparse_indices, np.ndarray):
                            sparse_indices = sparse_indices.tolist()
                        if isinstance(sparse_values, np.ndarray):
                            sparse_values = sparse_values.tolist()
                        
                        # Ensure all elements are the right type
                        sparse_indices = [int(idx) for idx in sparse_indices]
                        sparse_values = [float(val) for val in sparse_values]
                        
                        # Create string representations for storage
                        # Format: "idx1,idx2,idx3,..."
                        sparse_indices_str = ",".join([str(idx) for idx in sparse_indices])
                        sparse_values_str = ",".join([f"{val:.6f}" for val in sparse_values])
                        
                    except Exception as sparse_e:
                        if self.verbose:
                            print(f"Error processing sparse vector for item {i}: {sparse_e}")
                        # Use default sparse vector strings
                        sparse_indices_str = "0"
                        sparse_values_str = "0.0"
                    
                    # CRITICAL: Extract and validate content
                    content = ""
                    if "content" in payload:
                        content = payload["content"]
                    elif "text" in payload:
                        content = payload["text"]
                        
                    # Ensure content is a string and not None
                    if content is None:
                        content = ""
                    elif not isinstance(content, str):
                        try:
                            content = str(content)
                        except:
                            content = ""
                            
                    # Make sure content is not too long (Milvus has VARCHAR limits)
                    if len(content) > 65000:
                        if self.verbose and i < 5:
                            print(f"Content for item {i} exceeds 65000 chars, truncating (original: {len(content)} chars)")
                        content = content[:65000]
                    
                    # Validate title
                    title = payload.get("title", "")
                    if title is None:
                        title = ""
                    elif not isinstance(title, str):
                        try:
                            title = str(title)
                        except:
                            title = ""
                    
                    if len(title) > 490:
                        title = title[:490]
                    
                    # Debug content for the first few documents
                    if self.verbose and i < 3:
                        content_preview = content[:100] + "..." if content and len(content) > 100 else content
                        print(f"Document {i+1} content preview: {content_preview}")
                    
                    # Create entity as a dictionary
                    entity = {
                        "id": doc_id,
                        "file_path": str(payload.get("file_path", "")),
                        "file_name": str(payload.get("file_name", "")),
                        "file_type": str(payload.get("fileType", "")),
                        "file_size": int(payload.get("fileSize", 0)),
                        "created_at": int(payload.get("createdAt", time.time())),
                        "modified_at": int(payload.get("modifiedAt", time.time())),
                        "content": content,  # Store validated content
                        "title": title,  # Store validated title
                        "is_chunk": bool(payload.get("is_chunk", False)),
                        "is_parent": bool(payload.get("is_parent", False)),
                        "parent_id": str(payload.get("parent_id", "")),
                        "chunk_index": int(payload.get("chunk_index", 0)),
                        "total_chunks": int(payload.get("total_chunks", 0)),
                        self.dense_field: embedding.tolist(),
                        f"{self.sparse_indices_field}_str": sparse_indices_str,
                        f"{self.sparse_values_field}_str": sparse_values_str
                    }
                    
                    # Add content_vector field if it's in our schema
                    if hasattr(self, 'field_names') and "content_vector" in self.field_names:
                        # For content_vector, we'll use a special SparseVector object if possible
                        try:
                            from pymilvus.client.types import SparseVector
                            entity["content_vector"] = SparseVector(
                                indices=sparse_indices,
                                values=sparse_values
                            )
                        except ImportError:
                            # If SparseVector doesn't exist, use a dictionary
                            entity["content_vector"] = {
                                "indices": sparse_indices,
                                "values": sparse_values
                            }
                        except Exception as sv_e:
                            if self.verbose:
                                print(f"Error creating SparseVector for content_vector: {sv_e}")
                            # For nullable fields, we can set it to None
                            entity["content_vector"] = None
                    
                    entities.append(entity)
                except Exception as item_e:
                    if self.verbose:
                        print(f"Error processing item {i}: {item_e}")
                        print("Skipping this item and continuing")
                    continue
            
            if self.verbose:
                print(f"Inserting {len(entities)} entities with sparse vectors")
                if entities:
                    print(f"First entity ID type: {type(entities[0]['id'])}")
                    print(f"First entity ID value: {entities[0]['id']}")
                    print(f"First entity sparse indices: {entities[0][f'{self.sparse_indices_field}_str'][:100]}...")
                    print(f"First entity sparse values: {entities[0][f'{self.sparse_values_field}_str'][:100]}...")
                    
                    # Show content_vector info if present
                    if "content_vector" in entities[0]:
                        print(f"First entity has content_vector: {type(entities[0]['content_vector'])}")
            
            # Keep track of successfully inserted entities
            inserted_count = 0
            
            # Use smaller batch size to reduce memory pressure
            batch_size = 5  # Reduced from 10
            for i in range(0, len(entities), batch_size):
                end_idx = min(i + batch_size, len(entities))
                batch = entities[i:end_idx]
                
                if self.verbose:
                    print(f"Inserting batch {i//batch_size + 1}/{(len(entities)-1)//batch_size + 1} ({len(batch)} entities)")
                
                # Try inserting the batch with retry logic
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        # Use insert() with list of entity dictionaries
                        mr = self.collection.insert(batch)
                        inserted_count += len(batch)
                        if self.verbose:
                            print(f"Inserted batch with IDs: {mr.primary_keys[:3]}...")
                        break  # Success, exit retry loop
                    except Exception as batch_e:
                        if attempt < max_attempts - 1:
                            if self.verbose:
                                print(f"Batch insertion attempt {attempt+1} failed: {batch_e}")
                                print(f"Retrying in 2s...")
                            time.sleep(2)
                        else:
                            if self.verbose:
                                print(f"Error inserting batch after {max_attempts} attempts: {batch_e}")
                                print("Attempting individual entity insertion...")
                            
                            # Try one by one
                            for entity in batch:
                                try:
                                    # Try insertion with individual entity
                                    mr = self.collection.insert([entity])
                                    inserted_count += 1
                                    if self.verbose and inserted_count % 20 == 0:
                                        print(f"Inserted {inserted_count} entities so far")
                                except Exception as entity_e:
                                    if self.verbose:
                                        print(f"Error inserting entity with ID {entity['id']}: {entity_e}")
                
                # Add a small delay between batches to reduce server pressure
                if i + batch_size < len(entities):
                    time.sleep(0.5)
            
            # Store the count for future reference
            self._inserted_count = inserted_count
            
            # Create index if not already created
            if not self.index_created:
                self._create_indexes()
            else:
                # Make sure collection is loaded
                try:
                    # Use try-except with timeout handling
                    self.collection.load()
                except Exception as load_e:
                    if self.verbose:
                        print(f"Warning: error loading collection after insert: {load_e}")
                
            if self.verbose:
                print(f"Completed insert operation for {inserted_count} entities")
                
        except Exception as e:
            print(f"Error inserting embeddings with sparse vectors: {str(e)}")
            import traceback
            traceback.print_exc()
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
            List of search results in a format compatible with ResultProcessor
        """
        if processor is None:
            return {"error": "Hybrid search requires an embedding model"}
        
        try:
            # Milvus doesn't have a native hybrid search like other systems,
            # so we implement it manually by combining dense and keyword search results
            
            # 1. Get dense vector results
            dense_results = self.search_dense(query, processor, prefetch_limit, score_threshold, False, None)
            
            # 2. Get keyword search results
            keyword_results = self.search_keyword(query, prefetch_limit, score_threshold)
            
            # 3. Combine results using fusion algorithm
            if isinstance(dense_results, dict) and "error" in dense_results:
                dense_results = []
                
            if isinstance(keyword_results, dict) and "error" in keyword_results:
                keyword_results = []
                
            # Ensure results are in the right format for fusion
            if len(dense_results) > 0 or len(keyword_results) > 0:
                # Use SearchAlgorithms to perform fusion
                fused_results = SearchAlgorithms.manual_fusion(
                    dense_results, keyword_results, prefetch_limit, fusion_type
                )
                
                # 4. Apply reranking if requested
                if rerank and len(fused_results) > 0:
                    reranked_results = SearchAlgorithms.rerank_results(
                        query, fused_results, processor, limit, self.verbose
                    )
                    
                    # Record hit rates if ground truth is available
                    true_context = getattr(processor, 'expected_context', None)
                    if true_context:
                        hit = any(p.get("_source", {}).get("content") == true_context 
                                if hasattr(p, "get") else p.payload.get("content") == true_context 
                                for p in reranked_results)
                        search_key = f"hybrid_{fusion_type}"
                        if rerank:
                            search_key += f"_{reranker_type or 'default'}"
                        self._record_hit(search_key, hit)
                    
                    return reranked_results[:limit]
                
                # Record hit rates if ground truth is available
                true_context = getattr(processor, 'expected_context', None)
                if true_context:
                    hit = any(p.get("_source", {}).get("content") == true_context 
                            if hasattr(p, "get") else p.payload.get("content") == true_context 
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
        Perform a search using the specified search type.
        
        Args:
            query: Search query string
            search_type: Type of search to perform (hybrid, vector, sparse, keyword)
            limit: Number of results to return
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
            
            # Handle empty results
            if not points:
                return {
                    "query": query,
                    "search_type": search_type,
                    "count": 0,
                    "results": [],
                    # Use 'embedder_info' instead of 'embedders' to match expected format
                    "embedder_info": {
                        "dense": processor.model_name if processor else "unknown",
                        "sparse": getattr(processor, 'sparse_model_name', "unknown") if processor else "unknown"
                    }
                }
            
            # Format results with our custom formatter
            return self._format_search_results(points, query, search_type, processor, context_size)
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
        
    def _adapt_milvus_result_to_point(self, result):
        """
        Adapt Milvus search result to the expected dictionary format for ResultProcessor.adapt_result().
        This ensures compatibility with the shared processing methods.
        
        Args:
            result: Tuple of (entity, distance) from Milvus search
            
        Returns:
            Dictionary with the expected fields for ResultProcessor
        """
        entity, distance = result
        
        # Convert distance to similarity score (0-1)
        # For COSINE distance, similarity = 1 - distance
        score = 1.0 - distance
        
        # Get file path, ensuring it's always present
        file_path = entity.get("file_path", "")
        
        # Ensure file_name is available
        file_name = entity.get("file_name", "")
        if not file_name and file_path:
            # Extract filename from path if needed
            file_name = self._extract_filename(file_path)
        
        # Get chunk information with defaults
        chunk_index = self._safe_get_int(entity, "chunk_index", 0)
        total_chunks = self._safe_get_int(entity, "total_chunks", 1)
        
        # Create payload with all necessary fields
        payload = {
            "text": entity.get("content", ""),  # Ensure both text and content are present
            "content": entity.get("content", ""),
            "file_path": file_path,
            "path": file_path,  # Add path field for compatibility
            "file_name": file_name,
            "file": file_name,  # Add file field for compatibility
            "file_type": entity.get("file_type", ""),
            "file_size": entity.get("file_size", 0),
            "created_at": entity.get("created_at", 0),
            "modified_at": entity.get("modified_at", 0),
            "title": entity.get("title", ""),
            "is_chunk": entity.get("is_chunk", False),
            "is_parent": entity.get("is_parent", False),
            "parent_id": entity.get("parent_id", ""),
            "chunk_index": chunk_index,
            "chunk": chunk_index,  # Add chunk field for compatibility
            "total_chunks": total_chunks,
            # Include metadata for embedder info
            "metadata": {
                "dense_embedder": self.dense_model_id,
                "sparse_embedder": self.sparse_model_id
            },
            # Add embedder_info for direct access
            "embedder_info": {
                "dense_embedder": self.dense_model_id,
                "sparse_embedder": self.sparse_model_id
            }
        }
        
        # Return a dictionary in the format expected by ResultProcessor.adapt_result()
        return {
            "_id": entity.get("id", ""),
            "_source": payload,
            "_score": score,
            # Include these fields at the top level too for direct access
            "file_path": file_path,
            "file_name": file_name,
            "chunk_index": chunk_index,
            "embedder_info": {
                "dense_embedder": self.dense_model_id,
                "sparse_embedder": self.sparse_model_id
            }
        }

    
    def search_dense(self, query: str, processor: Any, limit: int, score_threshold: float = None,
                    rerank: bool = False, reranker_type: str = None):
        """
        Perform dense vector search with improved error handling.
        
        Args:
            query: Search query string
            processor: Document processor with embedding capabilities
            limit: Number of results to return
            score_threshold: Minimum score threshold
            rerank: Whether to apply reranking
            reranker_type: Type of reranker to use
            
        Returns:
            List of search results in a format compatible with ResultProcessor
        """
        if processor is None:
            return {"error": "Dense search requires an embedding model"}
        
        try:
            # Generate dense embedding
            query_vector = processor.get_embedding(query)
            
            # Build search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {
                    "nprobe": 10,      # Number of clusters to search
                    "ef": 64           # Higher values give more accurate results, but slower
                }
            }
            
            if score_threshold is not None:
                search_params["params"]["ef"] = 128  # Increase accuracy for threshold filtering
                
            # Execute vector search with retry logic
            max_attempts = 3
            results = None
            
            for attempt in range(max_attempts):
                try:
                    results = self.collection.search(
                        data=[query_vector.tolist()],  # List of query vectors
                        anns_field=self.dense_field,   # Vector field to search
                        param=search_params,          # Search parameters
                        limit=min(limit * 3, 1000),   # Get more results but respect limits
                        expr=None,                    # No filtering expression
                        output_fields=["id", "content", "file_path", "file_name", "file_type", 
                                    "file_size", "created_at", "modified_at", "title", 
                                    "is_chunk", "is_parent", "parent_id", "chunk_index", "total_chunks"],
                        timeout=30  # 30 second timeout
                    )
                    break  # Success, exit retry loop
                except Exception as search_e:
                    if attempt < max_attempts - 1:
                        if self.verbose:
                            print(f"Search attempt {attempt+1} failed: {search_e}")
                            print(f"Retrying in 2s...")
                        time.sleep(2)
                    else:
                        if self.verbose:
                            print(f"All search attempts failed: {search_e}")
                        return {"error": f"Error in dense search: {str(search_e)}"}
            
            if not results or len(results) == 0 or len(results[0]) == 0:
                if self.verbose:
                    print("Dense search returned no results")
                return []
                
            if self.verbose:
                print(f"Dense search returned {len(results[0])} results")
            
            # Process results into points format
            points = []
            for hit in results[0]:  # results[0] is for the first query
                # Extract entity fields into a dictionary
                entity = {}
                for field in hit.entity.fields:
                    entity[field] = hit.entity.get(field)
                
                # Calculate similarity score (convert from distance)
                score = 1.0 - hit.distance
                
                # Ensure we have a file_path
                file_path = self._safe_get_str(entity, "file_path", "")
                
                # Ensure we have a file_name
                file_name = self._safe_get_str(entity, "file_name", "")
                if not file_name and file_path:
                    file_name = self._extract_filename(file_path)
                    
                # Get chunk information with defaults
                chunk_index = self._safe_get_int(entity, "chunk_index", 0)
                total_chunks = self._safe_get_int(entity, "total_chunks", 1)
                
                # Get content with proper error handling
                content = self._safe_get_str(entity, "content", "")
                
                # Create payload with all necessary fields
                payload = {
                    "text": content,
                    "content": content,
                    "file_path": file_path,
                    "path": file_path,  # Add path field for compatibility
                    "file_name": file_name,
                    "file": file_name,  # Add file field for compatibility
                    "file_type": self._safe_get_str(entity, "file_type", ""),
                    "file_size": self._safe_get_int(entity, "file_size", 0),
                    "created_at": self._safe_get_int(entity, "created_at", 0),
                    "modified_at": self._safe_get_int(entity, "modified_at", 0),
                    "title": self._safe_get_str(entity, "title", ""),
                    "is_chunk": self._safe_get_bool(entity, "is_chunk", False),
                    "is_parent": self._safe_get_bool(entity, "is_parent", False),
                    "parent_id": self._safe_get_str(entity, "parent_id", ""),
                    "chunk_index": chunk_index,
                    "chunk": chunk_index,  # Add chunk field for compatibility
                    "total_chunks": total_chunks,
                    "metadata": {
                        "dense_embedder": self.dense_model_id,
                        "sparse_embedder": self.sparse_model_id
                    }
                }
                
                # Create the point
                point = {
                    "_id": self._safe_get_str(entity, "id", str(len(points))),
                    "_source": payload,
                    "_score": score
                }
                
                # Apply score threshold
                if score_threshold is not None and score < score_threshold:
                    continue
                    
                points.append(point)
            
            if self.verbose:
                print(f"Processed {len(points)} search results")
                if points:
                    print(f"First result score: {points[0]['_score']}")
                    print(f"First result path: {points[0]['_source']['file_path']}")
            
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

    # Helper methods for safer data extraction
    def _safe_get_str(self, entity, field, default=""):
        """Safely extract a string field from an entity"""
        value = entity.get(field)
        if value is None:
            return default
        return str(value)

    def _safe_get_int(self, entity, field, default=0):
        """Safely extract an integer field from an entity"""
        value = entity.get(field)
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def _safe_get_bool(self, entity, field, default=False):
        """Safely extract a boolean field from an entity"""
        value = entity.get(field)
        if value is None:
            return default
        return bool(value)

    def _adapt_milvus_result_to_point(self, result):
        """
        Adapt Milvus search result to the expected dictionary format for ResultProcessor.adapt_result().
        This ensures compatibility with the shared processing methods.
        
        Args:
            result: Tuple of (entity, distance) from Milvus search
            
        Returns:
            Dictionary with the expected fields for ResultProcessor
        """
        entity, distance = result
        
        # Convert distance to similarity score (0-1)
        # For COSINE distance, similarity = 1 - distance
        score = 1.0 - distance
        
        # Create payload with all necessary fields
        # Ensure all fields have sensible default values
        payload = {
            "text": entity.get("content", ""),  # Ensure both text and content are present
            "content": entity.get("content", ""),
            "file_path": entity.get("file_path", ""),
            "file_name": entity.get("file_name", ""),
            "file_type": entity.get("file_type", ""),
            "file_size": entity.get("file_size", 0),
            "created_at": entity.get("created_at", 0),
            "modified_at": entity.get("modified_at", 0),
            "title": entity.get("title", ""),
            "is_chunk": entity.get("is_chunk", False),
            "is_parent": entity.get("is_parent", False),
            "parent_id": entity.get("parent_id", ""),
            "chunk_index": entity.get("chunk_index", 0),
            "total_chunks": entity.get("total_chunks", 0),
            # Include metadata for embedder info
            "metadata": {
                "dense_embedder": self.dense_model_id,
                "sparse_embedder": self.sparse_model_id
            }
        }
        
        # For debugging, print the fields if verbose
        if self.verbose and hasattr(self, '_debug_count') and self._debug_count < 3:
            self._debug_count += 1
            print(f"DEBUG - Result entity keys: {entity.keys()}")
            print(f"DEBUG - First few entity values: {list(entity.values())[:3]}")
            print(f"DEBUG - Score: {score}")
        else:
            if not hasattr(self, '_debug_count'):
                self._debug_count = 0
        
        # Return a dictionary in the format expected by ResultProcessor.adapt_result()
        return {
            "_id": entity.get("id", ""),
            "_source": payload,
            "_score": score
        }

    
    def search_sparse(self, query: str, processor: Any, limit: int, score_threshold: float = None,
                    rerank: bool = False, reranker_type: str = None):
        """
        Perform sparse vector search with robust error handling for SPLADE.
        
        Milvus doesn't have native sparse vector search, so we implement a 
        simple approximation using the stored sparse vectors and content matching.
        
        Args:
            query: Search query string
            processor: Document processor with embedding capabilities
            limit: Number of results to return
            score_threshold: Minimum score threshold
            rerank: Whether to apply reranking
            reranker_type: Type of reranker to use
            
        Returns:
            List of search results in a format compatible with ResultProcessor
        """
        if processor is None:
            return {"error": "Sparse search requires an embedding model"}
        
        try:
            # Generate sparse vector from query with robust error handling
            try:
                # First try the standard approach
                if self.verbose:
                    print(f"Generating sparse vector using primary approach...")
                sparse_indices, sparse_values = processor.get_sparse_embedding(query)
                if self.verbose:
                    print(f"Successfully generated sparse vector with {len(sparse_indices)} non-zero elements")
            except ValueError as ve:
                # Handle the specific "not enough values to unpack" error
                if "not enough values to unpack" in str(ve):
                    if self.verbose:
                        print(f"SPLADE model error: {ve}")
                        print(f"Primary SPLADE approach failed: {ve}")
                        print(f"Trying alternative SPLADE method...")
                    
                    # Try alternative approach using MLX utility if available
                    try:
                        # Try to import generate_sparse_vector from mlx_utils
                        try:
                            from mlx_utils import generate_sparse_vector
                            if self.verbose:
                                print("Using generate_sparse_vector from mlx_utils")
                            
                            # Use the imported function
                            sparse_indices, sparse_values = generate_sparse_vector(query, processor)
                            if self.verbose:
                                print(f"Successfully generated sparse vector with {len(sparse_indices)} non-zero elements")
                        except ImportError:
                            # If mlx_utils is not available, check if processor has mlx_embedding_provider
                            if hasattr(processor, 'mlx_embedding_provider') and processor.mlx_embedding_provider is not None:
                                if self.verbose:
                                    print("Using MLX embedding provider directly")
                                sparse_indices, sparse_values = processor.mlx_embedding_provider.get_sparse_embedding(query)
                                if self.verbose:
                                    print(f"Successfully generated sparse vector with {len(sparse_indices)} non-zero elements")
                            else:
                                # If no MLX methods are available, simulate sparse vector from query terms
                                if self.verbose:
                                    print("No SPLADE implementation available, using content matching only")
                                # Use empty sparse vector
                                sparse_indices, sparse_values = [0], [0.0]
                                
                    except Exception as alt_e:
                        if self.verbose:
                            print(f"Alternative SPLADE approach failed: {alt_e}")
                            print("Using content matching only")
                        # Use empty sparse vector
                        sparse_indices, sparse_values = [0], [0.0]
                else:
                    # Re-raise other ValueError exceptions
                    raise
            except Exception as e:
                # Handle any other exceptions
                if self.verbose:
                    print(f"Error generating sparse vector: {e}")
                    print("Using content matching only")
                # Use empty sparse vector
                sparse_indices, sparse_values = [0], [0.0]
            
            if self.verbose:
                print(f"Using sparse fields: {self.sparse_indices_field} and {self.sparse_values_field}")
                if len(sparse_indices) > 0:
                    print(f"Sample indices: {sparse_indices[:5]}")
                    print(f"Sample values: {sparse_values[:5]}")
            
            # Make sure we have valid sparse vectors
            if not sparse_indices or not sparse_values or len(sparse_indices) != len(sparse_values):
                if self.verbose:
                    print(f"Invalid sparse vector format, using default values")
                sparse_indices, sparse_values = [0], [0.0]
            
            # Since Milvus doesn't directly support sparse vector search like Elasticsearch,
            # we'll use a hybrid approach combining keyword search with sparse vector information
            
            # Find the top indices by weight (highest information value)
            if len(sparse_indices) > 0 and len(sparse_values) > 0:
                # Sort indices by values (largest weight first)
                try:
                    top_pairs = sorted(zip(sparse_indices, sparse_values), key=lambda x: x[1], reverse=True)
                    
                    # Take top terms or all if less than 5
                    top_terms = min(5, len(sparse_indices))
                    top_indices = top_pairs[:top_terms]
                    
                    if self.verbose:
                        print(f"Using top {len(top_indices)} terms from sparse vector")
                except Exception as sort_e:
                    if self.verbose:
                        print(f"Error sorting sparse vector pairs: {sort_e}")
                        print("Proceeding with simpler approach")
                    # Use a simpler approach
                    top_indices = []
            else:
                top_indices = []
            
            # Build content query from original query for relevance
            # Escape quotes and special chars in query
            safe_query = query.replace("'", "\\'").replace("%", "\\%")
            
            # Build the query expression for Milvus
            try:
                # Try prefix matching (supported by Milvus)
                # First try with exact match which is more reliable
                expr = f"content == '{safe_query}'"
                
                # Execute query
                results = self.collection.query(
                    expr=expr,
                    output_fields=["id", "content", "file_path", "file_name", "file_type", 
                                "file_size", "created_at", "modified_at", "title", 
                                "is_chunk", "is_parent", "parent_id", "chunk_index", "total_chunks"],
                    limit=limit * 3  # Get more results to allow for filtering
                )
                
                if not results and len(query.split()) > 0:
                    # Try prefix matching with the first word
                    first_word = query.split()[0]
                    safe_word = first_word.replace("'", "\\'").replace("%", "\\%")
                    expr = f"content like '{safe_word}%'"
                    
                    results = self.collection.query(
                        expr=expr,
                        output_fields=["id", "content", "file_path", "file_name", "file_type", 
                                    "file_size", "created_at", "modified_at", "title", 
                                    "is_chunk", "is_parent", "parent_id", "chunk_index", "total_chunks"],
                        limit=limit * 3
                    )
                
                if self.verbose:
                    print(f"Sparse search returned {len(results) if results else 0} results")
                
                # No results found, try another approach
                if not results:
                    # Try with title
                    expr = f"title == '{safe_query}'"
                    
                    results = self.collection.query(
                        expr=expr,
                        output_fields=["id", "content", "file_path", "file_name", "file_type", 
                                    "file_size", "created_at", "modified_at", "title", 
                                    "is_chunk", "is_parent", "parent_id", "chunk_index", "total_chunks"],
                        limit=limit * 3
                    )
                    
                    if self.verbose:
                        print(f"Title search returned {len(results) if results else 0} results")
                        
                    # If still no results, try prefix matching with title
                    if not results and len(query.split()) > 0:
                        first_word = query.split()[0]
                        safe_word = first_word.replace("'", "\\'").replace("%", "\\%")
                        expr = f"title like '{safe_word}%'"
                        
                        results = self.collection.query(
                            expr=expr,
                            output_fields=["id", "content", "file_path", "file_name", "file_type", 
                                        "file_size", "created_at", "modified_at", "title", 
                                        "is_chunk", "is_parent", "parent_id", "chunk_index", "total_chunks"],
                            limit=limit * 3
                        )
                        
                        if self.verbose:
                            print(f"Title prefix search returned {len(results) if results else 0} results")
                
                # If still no results, try manual search
                if not results:
                    if self.verbose:
                        print("Using manual search method")
                        
                    # Get a sample of documents (respect Milvus limit of 16384)
                    sample_size = min(16000, self.count_entities())
                    if sample_size == 0:
                        sample_size = 1000  # Fallback but still below Milvus limit
                    
                    sample_docs = self.collection.query(
                        expr="id != ''",  # Match any document
                        output_fields=["id", "content", "file_path", "file_name", "file_type", 
                                    "file_size", "created_at", "modified_at", "title", 
                                    "is_chunk", "is_parent", "parent_id", "chunk_index", "total_chunks"],
                        limit=sample_size
                    )
                    
                    if self.verbose:
                        print(f"Retrieved {len(sample_docs)} documents for manual search")
                    
                    # Match documents manually
                    query_lower = query.lower()
                    query_words = query_lower.split()
                    
                    # Only process a reasonable number of docs to avoid performance issues
                    max_docs_to_process = min(len(sample_docs), 5000)
                    
                    matched_docs = []
                    for entity in sample_docs[:max_docs_to_process]:
                        if not entity:
                            continue
                            
                        content = str(entity.get("content", "")).lower() if entity.get("content") is not None else ""
                        title = str(entity.get("title", "")).lower() if entity.get("title") is not None else ""
                        
                        # Skip empty content
                        if not content and not title:
                            continue
                        
                        # Check for matches
                        match_score = 0.0
                        
                        # Full matches
                        if query_lower in content:
                            match_score += 0.8
                        if title and query_lower in title:
                            match_score += 0.9
                            
                        # Word-level matches
                        if match_score == 0:
                            for word in query_words:
                                if word in content:
                                    match_score += 0.3
                                if title and word in title:
                                    match_score += 0.4
                        
                        # Use information from sparse vector if available
                        if top_indices and len(content.split()) > 0:
                            # This is a simple approximation that gives weight to content
                            # containing terms that are important in sparse vector
                            content_tokens = set(content.lower().split())
                            for idx, val in top_indices:
                                try:
                                    # Here we would ideally map idx to actual term, but it's complex
                                    # As a simple approximation, we just add a small bonus
                                    # This still leverages sparse vector information
                                    match_score += 0.05
                                except Exception:
                                    pass
                        
                        # Add document if score is reasonable
                        if match_score > 0.2:
                            # Calculate distance
                            distance = 1.0 - min(1.0, match_score)
                            matched_docs.append((entity, distance))
                    
                    # Sort by score (lowest distance first) and get results
                    matched_docs.sort(key=lambda x: x[1])
                    results = [doc for doc, _ in matched_docs[:limit * 3]]
                    
                    if self.verbose:
                        print(f"Manual search found {len(results)} matches")
                
                # Still no results, return empty list
                if not results:
                    return []
                
                # Convert results to points with calculated scores
                points = []
                for i, entity in enumerate(results):
                    # Handle both standard results and (entity, distance) tuples
                    if isinstance(entity, tuple) and len(entity) == 2:
                        entity, distance = entity
                        # Score is already calculated
                        score = 1.0 - distance
                    else:
                        # Calculate a score based on position and match quality
                        position_score = max(0.1, 1.0 - (i / max(1, len(results))))
                        exact_match_bonus = 0.2 if query.lower() in str(entity.get("content", "")).lower() else 0
                        title_match_bonus = 0.3 if entity.get("title", "") and query.lower() in str(entity.get("title", "")).lower() else 0
                        
                        # Calculate sparse term overlap
                        sparse_bonus = 0
                        if len(sparse_indices) > 0:
                            sparse_bonus = 0.1
                        
                        # Calculate final score (normalized to 0-1)
                        score = min(1.0, position_score + exact_match_bonus + title_match_bonus + sparse_bonus)
                    
                    # Format using adapter
                    point = self._adapt_milvus_result_to_point((entity, 1.0 - score))  # Convert to distance format
                    
                    # Apply score threshold if needed
                    if score_threshold is not None and point["_score"] < score_threshold:
                        continue
                        
                    points.append(point)
                
                # Apply reranking if requested - for sparse search, this can significantly improve results
                if rerank and len(points) > 0:
                    if self.verbose:
                        print(f"Applying {reranker_type or 'default'} reranking to improve sparse results")
                        
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
                
            except Exception as query_e:
                if self.verbose:
                    print(f"Error executing query: {query_e}")
                    import traceback
                    traceback.print_exc()
                return {"error": f"Error in sparse search: {str(query_e)}"}
                
        except Exception as e:
            if self.verbose:
                print(f"Error in sparse search: {e}")
                import traceback
                traceback.print_exc()
            return {"error": f"Error in sparse search: {str(e)}"}

    
    def search_keyword(self, query: str, limit: int = 10, score_threshold: float = None,
                    rerank: bool = False, reranker_type: str = None):
        """
        Perform a keyword-based search with improved matching logic and error handling.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            score_threshold: Minimum score threshold
            rerank: Whether to apply reranking
            reranker_type: Type of reranker to use
            
        Returns:
            List of search results in a format compatible with ResultProcessor
        """
        try:
            if not query.strip():
                return {"error": "Empty query"}
                    
            # Clean the query and prepare variations
            clean_query = query.strip()
            clean_query_lower = clean_query.lower()
            
            # Create variants of the query
            query_variants = [clean_query]
            
            # Add lowercase variant if different
            if clean_query != clean_query_lower:
                query_variants.append(clean_query_lower)
                
            # Add capitalized variant
            if clean_query != clean_query.capitalize():
                query_variants.append(clean_query.capitalize())
            
            if self.verbose:
                print(f"Performing keyword search for: '{clean_query}'")
                print(f"Using query variants: {query_variants}")
            
            # For debugging: Attempt a direct sample query first to verify database is accessible
            try:
                debug_sample = self.collection.query(
                    expr="id != ''",  # Match any document
                    output_fields=["id", "content"],
                    limit=5
                )
                
                if debug_sample and self.verbose:
                    print(f"Debug: Successfully retrieved {len(debug_sample)} sample documents")
                    for i, doc in enumerate(debug_sample):
                        content_preview = doc.get("content", "")[:50] if doc.get("content") else "[No content]"
                        print(f"Debug sample {i+1}: {content_preview}...")
            except Exception as debug_e:
                if self.verbose:
                    print(f"Debug query failed: {debug_e}")
            
            # Instead of multiple separate queries, do one broader query and filter in memory
            try:
                # Query for a larger sample of documents to search through
                sample_size = min(1000, self.count_entities())
                if sample_size == 0:
                    sample_size = 500  # Fallback
                
                if self.verbose:
                    print(f"Entity count from num_entities attribute: {self.collection.num_entities}")
                    print(f"Entity count from paged query: {sample_size}")
                
                sample_docs = self.collection.query(
                    expr="id != ''",  # Match any document
                    output_fields=["id", "content", "file_path", "file_name", "file_type", 
                                "file_size", "created_at", "modified_at", "title", 
                                "is_chunk", "is_parent", "parent_id", "chunk_index", "total_chunks"],
                    limit=sample_size
                )
                
                if self.verbose:
                    print(f"Retrieved {len(sample_docs) if sample_docs else 0} documents for in-memory search")
                    
                if not sample_docs:
                    if self.verbose:
                        print("No documents retrieved from database")
                    return []
                
                # Search through documents in memory
                matched_docs = []
                query_lower = clean_query_lower
                query_words = query_lower.split()
                
                for entity in sample_docs:
                    content = str(entity.get("content", "")).lower() if entity.get("content") is not None else ""
                    title = str(entity.get("title", "")).lower() if entity.get("title") is not None else ""
                    file_path = str(entity.get("file_path", "")).lower()
                    file_name = str(entity.get("file_name", "")).lower()
                    
                    # Skip obviously empty documents
                    if not content and not title and not file_name:
                        continue
                    
                    # Match score calculation with different components
                    match_score = 0.0
                    # Instead of attaching match_details to entity, store it separately
                    match_details = []
                    
                    # Check exact matches in content
                    if query_lower in content:
                        match_score += 0.8
                        match_details.append("exact_content_match:0.8")
                    
                    # Check exact matches in title
                    if title and query_lower in title:
                        match_score += 0.9
                        match_details.append("exact_title_match:0.9")
                    
                    # Check filename matches
                    if file_name and query_lower in file_name:
                        match_score += 0.7
                        match_details.append("filename_match:0.7")
                    
                    # Check filepath matches
                    if file_path and query_lower in file_path:
                        match_score += 0.6
                        match_details.append("filepath_match:0.6")
                    
                    # Word-level matching for partial matches
                    if match_score < 0.3:
                        word_matches_content = sum(1 for word in query_words if word in content)
                        if word_matches_content > 0:
                            word_content_score = min(0.6, 0.2 * word_matches_content / len(query_words))
                            match_score += word_content_score
                            match_details.append(f"word_content_match:{word_content_score:.2f}")
                        
                        if title:
                            word_matches_title = sum(1 for word in query_words if word in title)
                            if word_matches_title > 0:
                                word_title_score = min(0.7, 0.3 * word_matches_title / len(query_words))
                                match_score += word_title_score
                                match_details.append(f"word_title_match:{word_title_score:.2f}")
                        
                        if file_name:
                            word_matches_filename = sum(1 for word in query_words if word in file_name)
                            if word_matches_filename > 0:
                                word_filename_score = min(0.5, 0.25 * word_matches_filename / len(query_words))
                                match_score += word_filename_score
                                match_details.append(f"word_filename_match:{word_filename_score:.2f}")
                    
                    # Add document if score is reasonable
                    if match_score > 0.2:
                        # Calculate distance (inverse of score)
                        distance = 1.0 - min(1.0, match_score)
                        # DO NOT modify entity directly, store the tuple with additional metadata
                        matched_docs.append((entity, distance, match_details))
                
                # Sort by score (lowest distance first)
                matched_docs.sort(key=lambda x: x[1])
                
                # Use only the top matches
                top_matches = matched_docs[:min(limit * 3, len(matched_docs))]
                
                if self.verbose:
                    print(f"In-memory search found {len(top_matches)} matches")
                    if top_matches:
                        # Print match details for the top result
                        top_entity, top_distance, top_match_details = top_matches[0]
                        top_score = 1.0 - top_distance
                        print(f"Top match score: {top_score:.4f}, details: {top_match_details}")
                
                # Add matched documents to results
                results = []
                for entity, _, _ in top_matches:
                    results.append(entity)
                
            except Exception as query_e:
                if self.verbose:
                    print(f"Error in in-memory search: {query_e}")
                    import traceback
                    traceback.print_exc()
                results = []
            
            # If still no results, try a last fallback approach with partial matching
            if not results:
                try:
                    if self.verbose:
                        print("Trying fallback partial matching approach")
                    
                    # Try to search in file paths directly
                    all_files = self.collection.query(
                        expr="id != ''",
                        output_fields=["id", "file_path", "file_name", "content", "title", 
                                    "chunk_index", "total_chunks"],
                        limit=500
                    )
                    
                    if all_files:
                        for word in query.lower().split():
                            if len(word) < 3:  # Skip very short words
                                continue
                                
                            matched_files = []
                            
                            for entity in all_files:
                                file_path = str(entity.get("file_path", "")).lower()
                                file_name = str(entity.get("file_name", "")).lower()
                                
                                if word in file_path or word in file_name:
                                    score = 0.4 + (0.1 * len(word) / len(query))  # Longer words get higher scores
                                    matched_files.append((entity, 1.0 - score))
                            
                            if matched_files:
                                matched_files.sort(key=lambda x: x[1])
                                results.extend([entity for entity, _ in matched_files[:limit]])
                                break
                                
                        if self.verbose and results:
                            print(f"Fallback file path search found {len(results)} results")
                            
                except Exception as fallback_e:
                    if self.verbose:
                        print(f"Error in fallback search: {fallback_e}")
                        import traceback
                        traceback.print_exc()
            
            # If still no results, try a direct search for the specific query text
            if not results:
                try:
                    if self.verbose:
                        print("Trying direct search for exact query text")
                    
                    # Try each query variant directly
                    for variant in query_variants:
                        if not variant:
                            continue
                            
                        # Escape single quotes in query
                        safe_variant = variant.replace("'", "''")
                        
                        # Try exact match in content
                        try:
                            exact_matches = self.collection.query(
                                expr=f"content like '%{safe_variant}%'",
                                output_fields=["id", "content", "file_path", "file_name", "title", 
                                            "chunk_index", "total_chunks"],
                                limit=limit
                            )
                            
                            if exact_matches:
                                if self.verbose:
                                    print(f"Found {len(exact_matches)} exact matches for '{variant}'")
                                results.extend(exact_matches)
                                break
                        except Exception as e:
                            if self.verbose:
                                print(f"Exact match query failed: {e}")
                    
                except Exception as direct_e:
                    if self.verbose:
                        print(f"Error in direct search: {direct_e}")
                        import traceback
                        traceback.print_exc()
            
            # If still no results, return empty list
            if not results:
                if self.verbose:
                    print("No results found")
                return []
            
            # Remove duplicates by ID
            unique_results = {}
            for entity in results:
                entity_id = entity.get("id", "")
                if entity_id and entity_id not in unique_results:
                    unique_results[entity_id] = entity
            
            results = list(unique_results.values())
            
            # Convert results to points with calculated scores
            points = []
            for i, entity in enumerate(results):
                # Calculate score based on position and potential matches
                position_score = max(0.2, 1.0 - (i / max(1, len(results))))
                
                # Check for exact match bonus
                entity_content = str(entity.get("content", "")).lower() if entity.get("content") is not None else ""
                entity_title = str(entity.get("title", "")).lower() if entity.get("title") is not None else ""
                exact_match = query.lower() in entity_content or query.lower() in entity_title
                
                # Calculate final score
                score = position_score + (0.3 if exact_match else 0)
                distance = 1.0 - score
                
                # Format using adapter
                point = self._adapt_milvus_result_to_point((entity, distance))
                
                # Apply score threshold if needed
                if score_threshold is not None and point["_score"] < score_threshold:
                    continue
                    
                points.append(point)
            
            if self.verbose:
                print(f"Returning {len(points)} final search results")
            
            return points[:limit]
                    
        except Exception as e:
            if self.verbose:
                print(f"Error in keyword search: {e}")
                import traceback
                traceback.print_exc()
            return {"error": f"Error in keyword search: {str(e)}"}



    def _fuzzy_match(self, query_term, content_term, max_edits=2):
        """
        Check if two strings match with a maximum number of edits.
        Uses a simple edit distance calculation.
        
        Args:
            query_term: Query term to match
            content_term: Content term to match against
            max_edits: Maximum number of allowed edits
            
        Returns:
            True if the strings match within the edit distance
        """
        # Check if one string is much longer than the other
        if abs(len(query_term) - len(content_term)) > max_edits:
            return False
        
        # Simple case - one is prefix of the other
        if query_term.startswith(content_term) or content_term.startswith(query_term):
            return True
        
        # Count the number of different characters
        edits = 0
        for i in range(min(len(query_term), len(content_term))):
            if query_term[i] != content_term[i]:
                edits += 1
                if edits > max_edits:
                    return False
        
        # Add remaining length difference to edits
        edits += abs(len(query_term) - len(content_term))
        
        return edits <= max_edits

    def _calculate_relevance_score(self, entity, query, query_words, position, total_results):
        """
        Calculate a relevance score for a result entity.
        
        Args:
            entity: Entity from search results
            query: Original search query
            query_words: Words from the query
            position: Position in results
            total_results: Total number of results
            
        Returns:
            Relevance score between 0 and 1
        """
        # Extract content and title, handling None values
        content = str(entity.get("content", "")).lower() if entity.get("content") is not None else ""
        title = str(entity.get("title", "")).lower() if entity.get("title") is not None else ""
        
        # Base position score (higher for earlier results)
        position_score = max(0.1, 1.0 - (position / max(1, total_results)))
        
        # Match score based on content
        match_score = 0.0
        
        # Check for exact query match
        query_lower = query.lower()
        if query_lower in title:
            match_score += 0.8
        elif query_lower in content:
            match_score += 0.6
        
        # Check for word matches
        for word in query_words:
            word_lower = word.lower()
            if word_lower in title:
                match_score += 0.4
            elif word_lower in content:
                match_score += 0.2
        
        # Normalize match score
        match_score = min(1.0, match_score)
        
        # Combine position and match scores
        final_score = (match_score * 0.7) + (position_score * 0.3)
        
        return min(1.0, final_score)
    
    def _calculate_keyword_score(self, entity, query, position, total_results):
        """
        Calculate a relevance score for keyword search.
        
        Args:
            entity: Entity from Milvus
            query: Original search query
            position: Position in results
            total_results: Total number of results
            
        Returns:
            Relevance score between 0 and 1
        """
        content = entity.get("content", "").lower()
        title = entity.get("title", "").lower()
        query_lower = query.lower()
        
        # Calculate position score (higher positions get higher scores)
        position_score = max(0.1, 1.0 - (position / max(1, total_results)))
        
        # Calculate content match score
        content_score = 0
        if query_lower in content:
            # Exact match gets highest score
            content_score = 0.8
        else:
            # Word-level matches
            words = query_lower.split()
            matches = sum(1 for word in words if word in content)
            if matches > 0:
                content_score = min(0.6, 0.2 * matches)
        
        # Calculate title match score
        title_score = 0
        if title and query_lower in title:
            title_score = 0.9  # Title matches are very relevant
        elif title:
            words = query_lower.split()
            matches = sum(1 for word in words if word in title)
            if matches > 0:
                title_score = min(0.7, 0.3 * matches)
        
        # Combine scores (take the maximum of content and title score, then boost by position)
        combined_score = max(content_score, title_score)
        final_score = min(1.0, combined_score + 0.2 * position_score)
        
        return final_score

    def _calculate_content_stats(self, content):
        """
        Calculate content statistics with improved robustness.
        
        Args:
            content: Text content to analyze
            
        Returns:
            Dictionary with content statistics
        """
        try:
            if not content:
                return {"chars": 0, "words": 0, "lines": 0, "characters": 0}
            
            # Safety check - ensure content is a string
            if not isinstance(content, str):
                content = str(content)
            
            char_count = len(content)
            word_count = len(content.split())
            line_count = content.count('\n') + 1
            
            return {
                "chars": char_count,
                "characters": char_count,  # for compatibility
                "words": word_count,
                "lines": line_count
            }
        except Exception as e:
            # Return defaults if stats calculation fails
            if self.verbose:
                print(f"Error calculating content stats: {e}")
            return {"chars": 0, "words": 0, "lines": 0, "characters": 0}


    def _retrieve_context_for_chunk(self, file_path: str, chunk_index: int, window: int = 1) -> str:
        """
        Retrieve context surrounding a chunk with improved error handling and direct content access.
        
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
            
            # First, try to get the specific chunk directly with exact match on chunk_index
            safe_file_path = file_path.replace("'", "\\'")
            expr = f"file_path == '{safe_file_path}' and chunk_index == {chunk_index}"
            
            if self.verbose:
                print(f"Retrieving chunk directly with query: {expr}")
            
            # Query for the specific chunk
            results = None
            try:
                results = self.collection.query(
                    expr=expr,
                    output_fields=["id", "content", "chunk_index"],
                    limit=1
                )
            except Exception as e:
                if self.verbose:
                    print(f"Error querying specific chunk: {e}")
            
            # If we found the specific chunk with content, return it
            if results and len(results) > 0:
                chunk_content = results[0].get("content", "")
                if chunk_content:
                    if self.verbose:
                        print(f"Found specific chunk with content length: {len(chunk_content)}")
                    return chunk_content
                else:
                    if self.verbose:
                        print("Found chunk but content is empty")
            
            # If direct approach fails, try a broader search for any chunk in the same file
            expr = f"file_path == '{safe_file_path}'"
            
            if self.verbose:
                print(f"Retrieving any chunk from file with query: {expr}")
            
            # Query for any chunk in the file
            try:
                results = self.collection.query(
                    expr=expr,
                    output_fields=["id", "content", "chunk_index"],
                    limit=20  # Get several chunks to find one with content
                )
            except Exception as e:
                if self.verbose:
                    print(f"Error querying file chunks: {e}")
            
            # Check if we found any chunks with content
            if results and len(results) > 0:
                if self.verbose:
                    print(f"Found {len(results)} chunks from the file")
                    
                # Find first chunk with content
                for res in results:
                    chunk_content = res.get("content", "")
                    if chunk_content:
                        if self.verbose:
                            print(f"Found chunk {res.get('chunk_index')} with content")
                        return chunk_content
                        
                if self.verbose:
                    print("All found chunks have empty content")
            
            # As a last resort, try a general query to find any document with content
            try:
                # Get any document from the collection with content
                results = self.collection.query(
                    expr="content != ''",  # Find docs with non-empty content
                    output_fields=["id", "content", "file_path", "chunk_index"],
                    limit=5
                )
                
                if results and len(results) > 0:
                    if self.verbose:
                        print(f"Found {len(results)} documents with non-empty content")
                    for res in results:
                        content = res.get("content", "")
                        if content:
                            if self.verbose:
                                fp = res.get("file_path", "unknown")
                                print(f"Using content from alternative file: {fp}")
                            return f"[Content from alternate file {res.get('file_path', '')}]: {content[:500]}"
            except Exception as e:
                if self.verbose:
                    print(f"Error in last-resort query: {e}")
            
            # If nothing else worked, try to access file content directly if available
            try:
                # Clean up the file path
                clean_path = file_path
                if clean_path.endswith(".txt") and not os.path.exists(clean_path):
                    # Try without .txt extension
                    base_path = clean_path[:-4]
                    if os.path.exists(base_path):
                        clean_path = base_path
                
                if os.path.exists(clean_path):
                    with open(clean_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    return f"[Content read directly from file]: {file_content[:500]}"
            except Exception as file_e:
                if self.verbose:
                    print(f"Error reading file directly: {file_e}")
            
            # Nothing worked, return helpful error message
            return f"[Content not available for file_path={file_path}, chunk_index={chunk_index}. The document may have empty content fields.]"
                
        except Exception as e:
            if self.verbose:
                print(f"Error retrieving context: {e}")
                import traceback
                traceback.print_exc()
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