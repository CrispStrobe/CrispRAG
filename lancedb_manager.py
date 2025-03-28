# lancedb_manager.py:
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
from utils import SearchAlgorithms, TextProcessor, ResultProcessor

# Try to import LanceDB client with the correct rerankers
try:
    import lancedb
    import pyarrow as pa
    from lancedb.rerankers import (
        RRFReranker,  # Default reranker for hybrid search
        CrossEncoderReranker,
        ColbertReranker,  # Recommended reranker
        CohereReranker,
        JinaReranker
        # Note: LinearCombinationReranker is deprecated according to docs
    )
    lancedb_available = True
except Exception as e:
    print(f"Warning: LanceDB or PyArrow not available: {e}")
    lancedb_available = False

class LanceDBManager(VectorDBInterface):
    """Manager for LanceDB vector database operations with enhanced retrieval performance"""
    
    def __init__(self, 
                uri: str = None, 
                collection_name: str = "documents",
                vector_size: int = 384,
                storage_path: str = None,
                verbose: bool = False,
                dense_model_id: str = "bge-small",
                sparse_model_id: str = "distilbert-splade"):
        """
        Initialize LanceDBManager with model-specific vector configuration.
        
        Args:
            uri: LanceDB URI (if None, will use local storage)
            collection_name: Collection name
            vector_size: Vector dimension
            storage_path: Path to storage directory (for local mode)
            verbose: Verbose output
            dense_model_id: ID of dense model for naming vectors
            sparse_model_id: ID of sparse model for naming vectors
        """
        if not lancedb_available:
            raise ImportError("LanceDB client not available. Install with: pip install lancedb pyarrow")
            
        # If no URI is provided, create a local database at storage_path
        # If neither URI nor storage_path is provided, create a local database in the current working directory
        self.uri = uri
        self.collection_name = collection_name
        self.vector_dim = vector_size
        
        # If storage_path is not provided but we don't have a URI, use a default path
        if storage_path is None and uri is None:
            storage_path = os.path.join(os.getcwd(), "lancedb_storage")
            if verbose:
                print(f"No URI or storage_path provided. Using default path: {storage_path}")
        
        self.storage_path = storage_path
        self.verbose = verbose
        self.db = None
        self.table = None
        
        # Store the original model IDs
        self.dense_model_id = dense_model_id
        self.sparse_model_id = sparse_model_id
        
        # Default reranker for hybrid search - Reciprocal Rank Fusion
        self.reranker = RRFReranker()
        
        # Keep track of whether we have indexes
        self.has_vector_index = False
        self.has_fts_index = False
        self.has_scalar_indexes = False
        
        # Performance tracking
        self._hit_rates = {}
        
        self.connect()

    def connect(self) -> None:
        """Establish connection to LanceDB with improved error handling"""
        try:
            # Connect to LanceDB using URI if provided, otherwise use local storage
            conn_uri = self.uri if self.uri else self.storage_path
            
            # Make sure the directory exists for local storage
            if not self.uri and self.storage_path:
                os.makedirs(self.storage_path, exist_ok=True)
                
            if self.verbose:
                print(f"Connecting to LanceDB at {conn_uri}")
                
            self.db = lancedb.connect(conn_uri)
            
            # Try to open existing collection if it exists
            try:
                self.table = self.db.open_table(self.collection_name)
                if self.verbose:
                    print(f"Opened existing collection '{self.collection_name}'")
                    
                # Check if table has indexes
                self._check_existing_indexes()
            except Exception as e:
                if self.verbose:
                    print(f"Collection '{self.collection_name}' does not exist yet: {e}")
                self.table = None
                
        except Exception as e:
            print(f"Error connecting to LanceDB: {str(e)}")
            raise
            
    def _check_existing_indexes(self):
        """Check if table has existing indexes"""
        try:
            indices = self.table.list_indices()
            
            for index in indices:
                index_name = index.get("name", "")
                index_type = index.get("type", "")
                index_column = index.get("column", "")
                
                if index_type in ["IVF_PQ", "HNSW", "IVF_FLAT"]:
                    self.has_vector_index = True
                    if self.verbose:
                        print(f"Found existing vector index: {index_name} on column {index_column}")
                        
                elif index_type == "FULLTEXT":
                    self.has_fts_index = True
                    if self.verbose:
                        print(f"Found existing full-text search index: {index_name} on column {index_column}")
                        
                elif index_type == "SCALAR":
                    if not self.has_scalar_indexes:
                        # Only mark as true if we find multiple scalar indexes
                        scalar_indices = [idx for idx in indices if idx.get("type") == "SCALAR"]
                        if len(scalar_indices) >= 2:  # We need at least file_path and chunk_index
                            self.has_scalar_indexes = True
                    
                    if self.verbose:
                        print(f"Found existing scalar index: {index_name} on column {index_column}")
        except Exception as e:
            if self.verbose:
                print(f"Could not check existing indexes: {e}")
        
    def create_collection(self, recreate: bool = False) -> None:
        """Create or recreate a LanceDB collection with improved reliability"""
        try:
            # Ensure connection is active
            if self.db is None:
                self.connect()
                
            # Check if collection exists
            collection_exists = False
            try:
                self.table = self.db.open_table(self.collection_name)
                collection_exists = True
            except Exception as e:
                if self.verbose:
                    print(f"Collection does not exist: {e}")
                pass

            if collection_exists:
                if recreate:
                    if self.verbose:
                        print(f"Recreating collection '{self.collection_name}'...")
                    try:
                        self.db.drop_table(self.collection_name)
                        self.table = None
                    except Exception as e:
                        print(f"Warning: Error dropping table: {e}")
                        # Continue anyway as we'll overwrite it
                else:
                    if self.verbose:
                        print(f"Collection '{self.collection_name}' already exists.")
                    return

            if self.verbose:
                print(f"Creating collection '{self.collection_name}' with vector size {self.vector_dim}")

            # Define schema for the collection with both dense and sparse vectors
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.vector_dim)),  # Dense vector
                pa.field("sparse_indices", pa.list_(pa.int32())),  # Sparse vector indices
                pa.field("sparse_values", pa.list_(pa.float32())),  # Sparse vector values
                pa.field("text", pa.string()),
                pa.field("file_path", pa.string()),
                pa.field("file_name", pa.string()),
                pa.field("chunk_index", pa.int32()),
                pa.field("total_chunks", pa.int32()),
                pa.field("token_count", pa.int32()),
                pa.field("metadata", pa.string())  # JSON string for metadata
            ])

            # Create the table with proper error handling
            try:
                self.table = self.db.create_table(
                    self.collection_name,
                    schema=schema,
                    mode="overwrite" if recreate else "create"
                )
                
                # Always double-check the table was created and accessible
                if self.table is None:
                    raise ValueError(f"Table creation failed - returned None")
                    
                # Verify we can access the table
                self.table.schema
                
                if self.verbose:
                    print(f"Successfully created collection '{self.collection_name}'")
                    
            except Exception as e:
                print(f"Error during table creation: {e}")
                # Try to recover by opening the table if it might have been created
                try:
                    self.table = self.db.open_table(self.collection_name)
                    if self.verbose:
                        print(f"Recovered by opening existing table")
                except Exception as e2:
                    print(f"Failed to recover: {e2}")
                    raise ValueError(f"Could not create or open table: {e}")

            # For empty tables, we'll defer index creation until data is inserted
            # Set flags to indicate indexes haven't been created yet
            self.has_vector_index = False
            self.has_fts_index = False
            self.has_scalar_indexes = False
            
            if self.verbose:
                print(f"✅ Collection '{self.collection_name}' created successfully")
                print("Note: Indexes will be created after data insertion")

        except Exception as e:
            print(f"❌ Error creating collection '{self.collection_name}': {str(e)}")
            raise
        
            
    def _wait_for_indexes(self, timeout=60, poll_interval=5):
        """
        Wait for indexes to be ready with improved feedback
        
        Args:
            timeout: Maximum time to wait in seconds
            poll_interval: How often to check status in seconds
        """
        if not self.verbose:
            return
            
        print("Checking index status...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                indices = self.table.list_indices()
                
                if not indices:
                    print("⏳ No indexes found yet, waiting...")
                    time.sleep(poll_interval)
                    continue
                    
                # Get status of all indexes
                incomplete_indexes = []
                complete_indexes = []
                
                for idx in indices:
                    idx_name = idx.name
                    try:
                        stats = self.table.index_stats(idx_name)
                        if hasattr(stats, 'num_unindexed_rows') and stats.num_unindexed_rows > 0:
                            # This index is still processing
                            incomplete_indexes.append((idx_name, stats.num_unindexed_rows))
                        else:
                            # This index is done
                            complete_indexes.append(idx_name)
                    except:
                        # Can't get stats for this index yet
                        incomplete_indexes.append((idx_name, "unknown"))
                
                if incomplete_indexes:
                    # Some indexes still building
                    status_str = ", ".join([f"{name}({rows} rows left)" for name, rows in incomplete_indexes])
                    print(f"⏳ Still building indexes: {status_str}")
                else:
                    # All indexes complete
                    print(f"✅ All {len(complete_indexes)} indexes are ready!")
                    return
                    
            except Exception as e:
                print(f"Error checking index status: {e}")
            
            time.sleep(poll_interval)
            
        print("⚠️ Timeout waiting for indexes to be ready. Continuing anyway.")
        print("Indexes will continue building in the background.")

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection with compatibility fixes for API changes"""
        try:
            if not self.db or not self.table:
                return {"error": "Not connected to LanceDB or collection not found"}
                
            # Get statistics about the table
            row_count = 0
            try:
                # Try the stats() method first (may not exist in newer versions)
                try:
                    stats = self.table.stats()
                    row_count = stats.get("num_rows", 0)
                except AttributeError:
                    # stats() doesn't exist, try counting with SQL
                    count_result = self.table.sql("SELECT COUNT(*) as count FROM data").to_arrow()
                    if hasattr(count_result, 'to_pandas'):
                        row_count = count_result.to_pandas()["count"].iloc[0]
                    else:
                        row_count = count_result["count"][0]
            except:
                # Last resort: try to count by retrieving all rows
                try:
                    # Try with to_arrow() first
                    data = self.table.to_arrow()
                    row_count = len(data)
                except:
                    # Fall back to loading a small sample
                    try:
                        # Different versions have different APIs for to_pandas
                        try:
                            df = self.table.to_pandas(limit=10)
                        except TypeError:
                            # If limit is not supported, try without it
                            # Note: this could be slow for large tables
                            df = self.table.to_pandas()
                            
                        # Just note we have some data but not the exact count
                        if len(df) > 0:
                            row_count = "some data available (count unknown)"
                        else:
                            row_count = 0
                    except:
                        row_count = "unknown"
            
            # Get disk usage if possible
            disk_usage = None
            if not self.uri:  # Only for local mode
                try:
                    table_path = Path(f"{self.storage_path}/{self.collection_name}")
                    if table_path.exists():
                        disk_usage = sum(f.stat().st_size for f in table_path.glob('**/*') if f.is_file())
                except Exception as e:
                    if self.verbose:
                        print(f"Could not calculate disk usage: {str(e)}")
            
            # Get table schema - FIX: access schema as a property, not a method
            try:
                schema = self.table.schema
                schema_str = str(schema)
            except Exception as schema_err:
                if self.verbose:
                    print(f"Error accessing schema: {schema_err}")
                schema_str = "schema unavailable"
            
            # Get index information
            indexes = {}
            try:
                index_info = self.table.list_indices()
                for idx in index_info:
                    idx_name = idx.name
                    idx_type = getattr(idx, 'type', 'unknown')
                    idx_column = getattr(idx, 'column', 'unknown')
                    indexes[idx_name] = {"type": idx_type, "column": idx_column}
                        
                    # Get additional index stats if available
                    try:
                        idx_stats = self.table.index_stats(idx_name)
                        indexes[idx_name]["stats"] = {
                            "indexed_rows": getattr(idx_stats, 'num_indexed_rows', 'unknown'),
                            "unindexed_rows": getattr(idx_stats, 'num_unindexed_rows', 'unknown'),
                            "distance_type": getattr(idx_stats, 'distance_type', None)
                        }
                    except:
                        pass
            except Exception as e:
                if self.verbose:
                    print(f"Could not fetch index information: {str(e)}")
            
            # Include performance data if available
            performance = {}
            if self._hit_rates:
                performance["hit_rates"] = self._hit_rates
            
            return {
                "name": self.collection_name,
                "points_count": row_count,
                "disk_usage": disk_usage,
                "schema": schema_str,
                "indexes": indexes,
                "vector_dim": self.vector_dim,
                "dense_model": self.dense_model_id,
                "sparse_model": self.sparse_model_id,
                "performance": performance
            }
        except Exception as e:
            if self.verbose:
                print(f"Error getting collection info: {str(e)}")
                import traceback
                traceback.print_exc()
            return {"error": str(e)}
    
    def insert_embeddings(self, embeddings_with_payloads: List[Tuple[np.ndarray, Dict[str, Any]]]) -> None:
        """Insert dense embeddings into LanceDB"""
        if not embeddings_with_payloads:
            return
            
        try:
            if not self.table:
                raise ValueError(f"Collection '{self.collection_name}' not found or not initialized")
                
            # Prepare data for insertion
            rows = []
            for embedding, payload in embeddings_with_payloads:
                # Generate a UUID for the point
                point_id = str(uuid.uuid4())
                
                # Generate sparse vector for the text (as fallback)
                text = payload.get("text", "")
                sparse_indices, sparse_values = self._generate_sparse_vector(text)
                
                # Format metadata as JSON string
                metadata = payload.get("metadata", {})
                if not isinstance(metadata, str):
                    metadata = json.dumps(metadata)
                
                # Create row for insertion
                row = {
                    "id": point_id,
                    "vector": embedding.tolist(),
                    "sparse_indices": sparse_indices,
                    "sparse_values": sparse_values,
                    "text": text,
                    "file_path": payload.get("file_path", ""),
                    "file_name": payload.get("file_name", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "total_chunks": payload.get("total_chunks", 0),
                    "token_count": payload.get("token_count", 0),
                    "metadata": metadata
                }
                rows.append(row)
            
            if self.verbose:
                print(f"Inserting {len(rows)} points into collection '{self.collection_name}'")
                
            # Insert data in batches for better performance
            BATCH_SIZE = 100
            for i in range(0, len(rows), BATCH_SIZE):
                batch = rows[i:i+BATCH_SIZE]
                if self.verbose and len(rows) > BATCH_SIZE:
                    print(f"Inserting batch {i//BATCH_SIZE + 1}/{(len(rows)-1)//BATCH_SIZE + 1} ({len(batch)} points)")
                
                self.table.add(batch)
                
            if self.verbose:
                print(f"Successfully inserted {len(rows)} points")
                
            # Now that we have data, create indexes if needed
            self._ensure_indexes()
        except Exception as e:
            print(f"Error inserting embeddings: {str(e)}")
            raise

    def insert_embeddings_with_sparse(self, embeddings_with_sparse: List[Tuple[np.ndarray, Dict[str, Any], Tuple[List[int], List[float]]]]) -> None:
        """Insert embeddings with sparse vectors into LanceDB with improved error handling"""
        if not embeddings_with_sparse:
            return
            
        try:
            # Check if table is accessible and reconnect if needed
            if self.table is None:
                if self.verbose:
                    print(f"Table not initialized, attempting to reconnect and open")
                try:
                    self.connect()
                    try:
                        self.table = self.db.open_table(self.collection_name)
                    except Exception as open_err:
                        if self.verbose:
                            print(f"Could not open table: {open_err}")
                        # If we can't open it, try to create it
                        self.create_collection(recreate=False)
                except Exception as conn_err:
                    print(f"Failed to reconnect: {conn_err}")
                    raise ValueError(f"Collection '{self.collection_name}' not found or not initialized")
            
            # Verify table is accessible by checking schema
            try:
                self.table.schema
            except Exception as schema_err:
                if self.verbose:
                    print(f"Table connection lost, attempting to reconnect: {schema_err}")
                self.connect()
                self.table = self.db.open_table(self.collection_name)
            
            # Prepare data for insertion
            rows = []
            for embedding, payload, sparse_vector in embeddings_with_sparse:
                # Generate a UUID for the point
                point_id = str(uuid.uuid4())
                
                # Extract sparse indices and values
                sparse_indices, sparse_values = sparse_vector
                
                # Format metadata as JSON string
                metadata = payload.get("metadata", {})
                if not isinstance(metadata, str):
                    metadata = json.dumps(metadata)
                
                # Create row for insertion
                row = {
                    "id": point_id,
                    "vector": embedding.tolist(),
                    "sparse_indices": sparse_indices,
                    "sparse_values": sparse_values,
                    "text": payload.get("text", ""),
                    "file_path": payload.get("file_path", ""),
                    "file_name": payload.get("file_name", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "total_chunks": payload.get("total_chunks", 0),
                    "token_count": payload.get("token_count", 0),
                    "metadata": metadata
                }
                rows.append(row)
            
            if self.verbose:
                print(f"Inserting {len(rows)} points with sparse vectors into collection '{self.collection_name}'")
                
            # Insert data in batches for better performance
            BATCH_SIZE = 100
            for i in range(0, len(rows), BATCH_SIZE):
                batch = rows[i:i+BATCH_SIZE]
                if self.verbose and len(rows) > BATCH_SIZE:
                    print(f"Inserting batch {i//BATCH_SIZE + 1}/{(len(rows)-1)//BATCH_SIZE + 1} ({len(batch)} points)")
                
                try:
                    self.table.add(batch)
                except Exception as add_err:
                    print(f"Error adding batch: {add_err}")
                    # Try to reconnect and add again
                    self.connect()
                    self.table = self.db.open_table(self.collection_name)
                    self.table.add(batch)
                
            if self.verbose:
                print(f"Successfully inserted {len(rows)} points with sparse vectors")
                
            # Now that we have data, create indexes if needed
            self._ensure_indexes()
        except Exception as e:
            print(f"Error inserting embeddings with sparse vectors: {str(e)}")
            raise
            

    def _ensure_indexes(self):
        """Create indexes if they don't exist and we have data in the table with improved API compatibility"""
        try:
            # First check if we have any data in the table and how much
            data_count = 0
            try:
                # Try to get data count with SQL
                try:
                    # Use SQL to count rows efficiently
                    count_result = self.table.sql("SELECT COUNT(*) as count FROM data").to_arrow()
                    if hasattr(count_result, 'to_pandas'):
                        data_count = int(count_result.to_pandas()["count"].iloc[0])
                    else:
                        data_count = int(count_result["count"][0])
                        
                    if self.verbose:
                        print(f"Table has {data_count} rows")
                except:
                    # Fallback method - count by sampling
                    try:
                        # Use to_arrow with limit parameter (newer API)
                        sample_data = self.table.to_arrow(limit=1000)  # Sample up to 1000 rows
                        sample_count = len(sample_data)
                        
                        if sample_count == 1000:
                            # We hit the limit, so there are at least 1000 rows
                            data_count = 1000
                            if self.verbose:
                                print(f"Table has at least {data_count} rows")
                        else:
                            # We got all rows
                            data_count = sample_count
                            if self.verbose:
                                print(f"Table has {data_count} rows")
                    except TypeError:
                        # Fallback for older versions - try without limit
                        try:
                            all_data = self.table.to_arrow()
                            data_count = len(all_data)
                            if self.verbose:
                                print(f"Table has {data_count} rows")
                        except:
                            # Can't determine count
                            if self.verbose:
                                print("Table has data (count unknown)")
                            data_count = 100  # Conservative default
            except Exception as e:
                if self.verbose:
                    print(f"Error checking table data: {e}")
                data_count = 0
                
            if data_count == 0:
                if self.verbose:
                    print("Table is empty, skipping index creation")
                return
                
            # Check for vector index
            if not self.has_vector_index:
                if self.verbose:
                    print("Creating vector index...")
                    
                try:
                    # Try to determine GPU acceleration support
                    gpu_available = False
                    gpu_type = None
                    
                    # Check if PyTorch with CUDA is available
                    try:
                        import torch
                        if torch.cuda.is_available():
                            gpu_available = True
                            gpu_type = "cuda"
                            if self.verbose:
                                print(f"CUDA is available with {torch.cuda.device_count()} device(s)")
                                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                    except:
                        pass
                        
                    # Check for MPS (Apple Silicon)
                    if not gpu_available:
                        try:
                            import torch
                            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                                gpu_available = True
                                gpu_type = "mps"
                                if self.verbose:
                                    print(f"Apple MPS (Metal) acceleration is available")
                        except:
                            pass
                            
                    # Calculate good parameters based on data size and vector dimension
                    vector_dim = self.vector_dim  # Usually 384 for BGE models
                    
                    # For very small datasets, IVF_PQ doesn't work well
                    # We need to use IVF_FLAT which has fewer constraints but still 
                    # provides better performance than brute force
                    if data_count < 1000:
                        # For small datasets, use IVF_FLAT
                        # Calculate optimal number of partitions (clusters)
                        # For very small datasets, use minimal partitions
                        num_partitions = max(2, min(data_count // 4, 16))
                        
                        if self.verbose:
                            print(f"Using IVF_FLAT index for small dataset ({data_count} vectors)")
                            print(f"  - num_partitions: {num_partitions}")
                            print(f"  - vectors per partition: ~{data_count/num_partitions:.1f}")
                        
                        # Create index with appropriate parameters for IVF_FLAT
                        index_params = {
                            "vector_column_name": "vector", 
                            "metric": "cosine",
                            "replace": True,
                            "num_partitions": num_partitions,
                            "index_type": "IVF_FLAT"  # Use FLAT for small datasets
                        }
                    else:
                        # For larger datasets, use IVF_PQ with optimized parameters
                        # Calculate optimal number of partitions (clusters)
                        if data_count < 10000:
                            # For small-medium datasets
                            num_partitions = int(data_count ** 0.4)  # Slightly less than sqrt
                        else:
                            # For larger datasets
                            # Aim for ~3K vectors per partition
                            num_partitions = max(10, min(256, data_count // 3000))
                        
                        # Calculate optimal number of sub-vectors
                        # Ensure dimension / num_sub_vectors is a multiple of 8 for SIMD efficiency
                        # For 384-dim: good values are 48 (384/48=8) or 64 (384/64=6)
                        # Adjust for different dimensions
                        if vector_dim % 64 == 0:
                            num_sub_vectors = 64  # Good for 384, 512, 768, 1024
                        elif vector_dim % 48 == 0:
                            num_sub_vectors = 48  # Good for 384, 768
                        else:
                            # Find largest divisor that results in multiple of 8
                            for divisor in [32, 24, 16, 8]:
                                if vector_dim % divisor == 0:
                                    num_sub_vectors = divisor
                                    break
                            else:
                                # Fallback: use dimension / 8
                                num_sub_vectors = max(1, vector_dim // 8)
                        
                        if self.verbose:
                            print(f"Using IVF_PQ index for dataset with {data_count} vectors with {vector_dim} dimensions:")
                            print(f"  - num_partitions: {num_partitions}")
                            print(f"  - num_sub_vectors: {num_sub_vectors}")
                            print(f"  - vectors per partition: ~{data_count/num_partitions:.1f}")
                            print(f"  - dimension / num_sub_vectors = {vector_dim/num_sub_vectors:.1f}")
                        
                        # Create index with appropriate parameters for IVF_PQ
                        index_params = {
                            "vector_column_name": "vector", 
                            "metric": "cosine",
                            "replace": True,
                            "num_partitions": num_partitions,
                            "num_sub_vectors": num_sub_vectors,
                            "num_bits": 8,  # Explicit: only 4 or 8 supported, 8 is better quality
                            "index_type": "IVF_PQ"  # Explicitly set index type
                        }
                    
                    # Add accelerator if GPU is available
                    if gpu_available:
                        if self.verbose:
                            print(f"Creating vector index with {gpu_type} acceleration")
                        index_params["accelerator"] = gpu_type
                    else:
                        if self.verbose:
                            print("Creating vector index with CPU (no GPU acceleration available)")
                    
                    # Create the index with all parameters
                    self.table.create_index(**index_params)
                        
                    self.has_vector_index = True
                    
                    if self.verbose:
                        print("Vector index creation initiated")
                        print("Note: Index building will continue in the background")
                        
                except Exception as e:
                    if self.verbose:
                        print(f"Error creating vector index: {e}")
                        print("Will continue without vector index")
            
            # Check for FTS index
            if not self.has_fts_index:
                if self.verbose:
                    print("Creating full-text search index...")
                try:
                    self.table.create_fts_index(
                        "text", 
                        replace=True,
                        with_position=True,
                        lower_case=True,
                        stem=True,
                        remove_stop_words=True
                    )
                    self.has_fts_index = True
                    if self.verbose:
                        print("FTS index creation initiated")
                except Exception as e:
                    if self.verbose:
                        print(f"Error creating FTS index: {e}")
                        print("Will continue without FTS index")
            
            # Check for scalar indexes
            if not self.has_scalar_indexes:
                if self.verbose:
                    print("Creating scalar indexes...")
                
                # Try to create file_path index (using BTREE which is good for many unique values)
                try:
                    self.table.create_scalar_index("file_path", index_type="BTREE")
                    if self.verbose:
                        print("Created scalar index on file_path")
                except Exception as e:
                    if self.verbose:
                        print(f"Error creating file_path index: {e}")
                
                # Try to create chunk_index (using BTREE for numeric values)
                try:
                    self.table.create_scalar_index("chunk_index", index_type="BTREE")
                    if self.verbose:
                        print("Created scalar index on chunk_index")
                except Exception as e:
                    if self.verbose:
                        print(f"Error creating chunk_index index: {e}")
                        
                self.has_scalar_indexes = True
                if self.verbose:
                    print("Scalar indexes creation initiated")
                    
        except Exception as e:
            if self.verbose:
                print(f"Error ensuring indexes: {e}")
                print("Will continue without additional indexes")



    def _get_reranker(self, reranker_type: str = "rrf", verbose: bool = False):
        """
        Get a reranker instance based on type
        
        Args:
            reranker_type: Type of reranker to use ('rrf', 'colbert', 'cohere', 'jina', 'cross')
            verbose: Whether to show verbose output
            
        Returns:
            Reranker instance
        """
        reranker_type = reranker_type.lower()
        
        try:
            if reranker_type == "rrf":
                # RRF is the default and recommended reranker for hybrid search
                # k=60 is the default and near-optimal according to the docs
                return RRFReranker(k=60, return_score="relevance")
                
            elif reranker_type == "colbert":
                # ColBERT reranker - good for all search types (hybrid, vector, FTS)
                return ColbertReranker(
                    model_name="colbert-ir/colbertv2.0",
                    column="text",
                    return_score="relevance"
                )
                
            elif reranker_type == "dbsf" or reranker_type == "fusion":
                # For DBSF fusion, use RRFReranker as it's more suitable than LinearCombinationReranker
                if verbose:
                    print(f"Using RRFReranker for fusion type '{reranker_type}'")
                return RRFReranker(k=60, return_score="all")  # Return all scores for better fusion
                
            elif reranker_type == "cohere":
                # Cohere requires an API key which we don't provide here
                if verbose:
                    print("Warning: Cohere reranker requires an API key")
                return CohereReranker()
                
            elif reranker_type == "jina":
                return JinaReranker()
                
            elif reranker_type == "cross":
                return CrossEncoderReranker()
                
            else:
                if verbose:
                    print(f"Unknown reranker type: {reranker_type}, using RRF")
                return RRFReranker()
                
        except Exception as e:
            if verbose:
                print(f"Error creating reranker '{reranker_type}': {e}")
                print("Falling back to RRF reranker")
            return RRFReranker()

    def search_hybrid(self, query: str, processor: Any, limit: int, 
                    prefetch_limit: int = 50, fusion_type: str = "rrf",
                    score_threshold: float = None, rerank: bool = False,
                    reranker_type: str = None):
        """
        Perform a hybrid search combining both dense and sparse vectors with optional reranking
        
        This uses LanceDB's native hybrid search capabilities with appropriate rerankers
        
        Args:
            query: Search query string
            processor: Document processor with embedding capabilities
            limit: Number of results to return
            prefetch_limit: Number of results to prefetch for fusion
            fusion_type: Type of fusion to use (rrf or dbsf, with rrf recommended)
            score_threshold: Minimum score threshold
            rerank: Whether to apply additional reranking after fusion
            reranker_type: Type of reranker to use for final reranking (colbert recommended, or cohere, jina, cross)
        """
        if processor is None:
            return {"error": "Hybrid search requires an embedding model"}
        
        try:
            # Ensure we have a working table connection
            if self.table is None:
                try:
                    self.connect()
                    self.table = self.db.open_table(self.collection_name)
                except Exception as e:
                    return {"error": f"Could not connect to table: {e}"}
            
            # Generate the query embedding
            query_vector = processor.get_embedding(query)
            
            # Ensure we have both vector and FTS indexes
            self._ensure_indexes()
            
            # First select the fusion reranker - this is for combining vector and FTS results
            fusion_reranker = self._get_reranker(fusion_type, self.verbose)
            
            # If additional reranking is requested, use a different reranker after fusion
            reranker = None
            if rerank:
                reranker_type = reranker_type or "cross"  # Default to cross-encoder for reranking
                reranker = self._get_reranker(reranker_type, self.verbose)
            
            # Perform hybrid search
            if self.verbose:
                print(f"Performing hybrid search for: '{query}'")
                print(f"Using {fusion_type} fusion and {reranker_type if rerank else 'no'} reranking")

            try:
                # First perform hybrid search with fusion
                search_results = (
                    self.table.search(
                        query,
                        query_type="hybrid",
                        # Add fast_search parameter to use available indices
                        fast_search=True
                    )
                    .rerank(reranker=fusion_reranker)  # Apply fusion reranker
                    .limit(prefetch_limit)              # Get more results than needed
                )
                
                # Apply additional reranking if requested
                if rerank and reranker:
                    search_results = search_results.rerank(reranker=reranker)
                
                # Apply score threshold if provided
                if score_threshold is not None:
                    search_results = search_results.where(f"_distance >= {score_threshold}", prefilter=False)
                
                # Limit to final number of results
                search_results = search_results.limit(limit)
                
                # Get the results as pandas DataFrame, handling API differences
                try:
                    # First try with to_pandas() (newer versions)
                    results_df = search_results.to_pandas()
                except Exception as e1:
                    try:
                        # Try with to_arrow().to_pandas() (older versions)
                        results_arr = search_results.to_arrow()
                        if hasattr(results_arr, 'to_pandas'):
                            results_df = results_arr.to_pandas()
                        else:
                            # Manual conversion from Arrow to pandas
                            import pandas as pd
                            results_df = pd.DataFrame({
                                col: results_arr[col].to_numpy() 
                                for col in results_arr.column_names
                            })
                    except Exception as e2:
                        if self.verbose:
                            print(f"Error converting results: {e1}; then {e2}")
                        return {"error": "Could not process search results"}
                
                # Record hit rate metrics if available
                true_context = getattr(processor, 'expected_context', None)
                if true_context:
                    hit = any(row["text"] == true_context for _, row in results_df.iterrows())
                    search_key = f"hybrid_{fusion_type}"
                    if rerank:
                        search_key += f"_{reranker_type}"
                    self._record_hit(search_key, hit)
                
                # Convert results to the expected format
                results = []
                for _, row in results_df.iterrows():
                    # Create a result object (compatible with other backends)
                    class ResultItem:
                        def __init__(self, id, payload, score):
                            self.id = id
                            self.payload = payload
                            self.score = score
                    
                    # Parse metadata JSON
                    metadata = row.get("metadata", "{}")
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except:
                            metadata = {}
                    
                    # Create payload dictionary
                    payload = {
                        "text": row["text"],
                        "file_path": row["file_path"],
                        "file_name": row["file_name"],
                        "chunk_index": row["chunk_index"],
                        "total_chunks": row.get("total_chunks", 0),
                        "metadata": metadata
                    }
                    
                    # Create result object with the score from the appropriate column
                    # Different versions use different column names
                    score = 0.0
                    for score_col in ["_distance", "_relevance_score", "_score", "score"]:
                        if score_col in row:
                            score = row[score_col]
                            break
                            
                    result = ResultItem(
                        id=row["id"],
                        payload=payload,
                        score=score
                    )
                    
                    results.append(result)
                
                return results
            except Exception as search_err:
                if self.verbose:
                    print(f"Hybrid search error: {search_err}")
                    
                # If hybrid search fails (possibly missing indexes), fall back to vector-only search
                if self.verbose:
                    print("Falling back to vector-only search")
                return self.search_dense(query, processor, limit, score_threshold, rerank, reranker_type)
                
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
        Search with various options
        
        Args:
            query: Search query string
            search_type: Type of search to perform (hybrid, vector, sparse, keyword)
            limit: Maximum number of results to return
            processor: Document processor with embedding capabilities
            prefetch_limit: Number of results to prefetch for fusion
            fusion_type: Type of fusion to use (rrf, linear, dbsf)
            relevance_tuning: Whether to apply relevance tuning
            context_size: Size of context window for preview
            score_threshold: Minimum score threshold
            rerank: Whether to apply reranking
            reranker_type: Type of reranker to use (colbert, cohere, jina, cross)
        """
        try:
            query = query.strip()
            if not query:
                return {"error": "Empty query"}
            
            # Check if collection exists
            if self.table is None:
                return {"error": f"Collection {self.collection_name} does not exist"}
            
            # Determine correct search method based on type
            if search_type.lower() in ["vector", "dense"]:
                points = self.search_dense(query, processor, limit, score_threshold, rerank, reranker_type)
            elif search_type.lower() == "sparse":
                points = self.search_sparse(query, processor, limit, score_threshold, rerank, reranker_type)
            elif search_type.lower() in ["keyword", "fts"]:
                points = self.search_keyword(query, limit, score_threshold, rerank, reranker_type)
            else:  # Default to hybrid
                points = self.search_hybrid(query, processor, limit, prefetch_limit, 
                                        fusion_type, score_threshold, rerank, reranker_type)
            
            # Handle errors
            if isinstance(points, dict) and "error" in points:
                return points
            
            # Create a retriever function to pass to the formatter
            def context_retriever(file_path, chunk_index, window=1):
                return self._retrieve_context_for_chunk(file_path, chunk_index, window)
            
            # Format results with TextProcessor
            return TextProcessor.format_search_results(
                points, query, search_type, processor, context_size,
                retriever=context_retriever,
                db_type="lancedb"
            )
        except Exception as e:
            if self.verbose:
                print(f"Error during search: {e}")
                import traceback
                traceback.print_exc()
            return {"error": str(e)}
        

    def search_dense(self, query: str, processor: Any, limit: int, score_threshold: float = None,
                    rerank: bool = False, reranker_type: str = None):
        """Perform dense vector search with consistent handling and optimized parameters"""
        if processor is None:
            return {"error": "Dense search requires an embedding model"}
        
        try:
            # Ensure we have a working table connection
            if self.table is None:
                try:
                    self.connect()
                    self.table = self.db.open_table(self.collection_name)
                except Exception as e:
                    return {"error": f"Could not connect to table: {e}"}
                    
            # Generate dense embedding
            query_vector = processor.get_embedding(query)
            
            # Calculate appropriate nprobes based on collection size
            # Estimate table size if possible
            try:
                # Try to count rows for probes calculation
                count_result = self.table.sql("SELECT COUNT(*) as count FROM data").to_arrow()
                if hasattr(count_result, 'to_pandas'):
                    row_count = int(count_result.to_pandas()["count"].iloc[0])
                else:
                    row_count = int(count_result["count"][0])
            except:
                # Default assumption
                row_count = 1000  # Conservative default
                
            # Calculate appropriate nprobes (5-15% of partitions)
            # First get index information to see how many partitions we have
            index_info = None
            try:
                indices = self.table.list_indices()
                for idx in indices:
                    if hasattr(idx, 'type') and idx.type == 'IVF_PQ' and hasattr(idx, 'vector_column_name') and idx.vector_column_name == 'vector':
                        index_info = idx
                        break
            except:
                pass
                
            # Set nprobes based on partitions or data size
            if index_info and hasattr(index_info, 'num_partitions'):
                num_partitions = index_info.num_partitions
                nprobes = max(2, min(256, int(num_partitions * 0.1)))  # 10% of partitions
            else:
                # Estimate based on data size (square root is often used for num_partitions)
                estimated_partitions = max(2, min(256, int(row_count ** 0.5)))
                nprobes = max(2, min(64, int(estimated_partitions * 0.1)))  # 10% of estimated partitions
                
            # Set refine_factor based on data size
            # For smaller datasets, higher refine_factor is feasible
            if row_count < 1000:
                refine_factor = 10  # Higher refine for small datasets
            elif row_count < 10000:
                refine_factor = 5
            else:
                refine_factor = 3  # Lower refine for large datasets
                
            if self.verbose:
                print(f"Vector search parameters: nprobes={nprobes}, refine_factor={refine_factor}")
            
            # Get initial search results with optimized parameters
            search_result = None
            try:
                # Try with newer API first
                search_result = (
                    self.table.search(
                        query_vector.tolist(),
                        vector_column_name="vector",
                        metric="cosine"
                    )
                    .nprobes(nprobes)          # Parameter for index efficiency
                    .refine_factor(refine_factor)  # Parameter for result quality
                    .limit(limit * 3)          # Get more results for filtering
                )
                
                # Apply score threshold if provided
                if score_threshold is not None:
                    search_result = search_result.where(f"_distance >= {score_threshold}")
                    
                # Get results
                search_result = search_result.to_pandas()
            except Exception as e:
                if self.verbose:
                    print(f"Error with optimized search: {e}")
                    print("Trying alternative search method...")
                    
                # Try alternative methods if the first approach fails
                try:
                    search_result = self.table.search(
                        query_vector.tolist(),
                        vector_column_name="vector",
                        metric="cosine",
                        limit=limit * 3
                    )
                    
                    # Convert to pandas
                    if hasattr(search_result, 'to_pandas'):
                        search_result = search_result.to_pandas()
                    else:
                        # Manual conversion if needed
                        import pandas as pd
                        arrow_result = search_result.to_arrow()
                        search_result = pd.DataFrame({
                            col: arrow_result[col].to_numpy() 
                            for col in arrow_result.column_names
                        })
                except Exception as e2:
                    if self.verbose:
                        print(f"All vector search methods failed: {e}, {e2}")
                    return {"error": f"Vector search failed: {e2}"}
            
            # Apply reranking if requested and we have results
            if rerank and search_result is not None and len(search_result) > 0:
                try:
                    # Extract text data for reranking
                    texts = search_result["text"].tolist()
                    
                    # Get reranker based on type
                    reranker = self._get_reranker(reranker_type or "rrf", self.verbose)
                    
                    # Different reranking approaches based on availability
                    if hasattr(processor, 'fastembed_provider') and processor.fastembed_provider is not None:
                        if self.verbose:
                            print(f"Reranking with FastEmbed")
                        scores = processor.fastembed_provider.rerank_with_fastembed(query, texts)
                    elif hasattr(processor, 'mlx_embedding_provider') and processor.mlx_embedding_provider is not None:
                        if self.verbose:
                            print(f"Reranking with MLX")
                        scores = processor.mlx_embedding_provider.get_dense_embedding(query)
                    elif hasattr(processor, 'ollama_embedding_provider') and processor.ollama_embedding_provider is not None:
                        if self.verbose:
                            print(f"Reranking with Ollama")
                        scores = processor.ollama_embedding_provider.rerank_with_ollama(query, texts)
                    else:
                        # Simple cosine reranking
                        if self.verbose:
                            print(f"Reranking with simple cosine similarity")
                        query_emb = processor.get_embedding(query)
                        scores = []
                        for text in texts:
                            text_emb = processor.get_embedding(text)
                            score = np.dot(query_emb, text_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(text_emb))
                            scores.append(float(score))
                    
                    # Add scores to dataframe
                    search_result["rerank_score"] = scores
                    
                    # Sort by reranker scores
                    search_result = search_result.sort_values("rerank_score", ascending=False)
                except Exception as e:
                    if self.verbose:
                        print(f"Reranking failed: {e}")
                    # Continue with original results
            
            # Limit to requested number
            search_result = search_result.head(limit)
            
            # Format as expected output
            results = []
            for _, row in search_result.iterrows():
                # Create a result object (compatible with other backends)
                class ResultItem:
                    def __init__(self, id, payload, score):
                        self.id = id
                        self.payload = payload
                        self.score = score
                
                # Parse metadata JSON
                metadata = row.get("metadata", "{}")
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                
                # Create payload dictionary
                payload = {
                    "text": row["text"],
                    "file_path": row["file_path"],
                    "file_name": row["file_name"],
                    "chunk_index": row["chunk_index"],
                    "total_chunks": row.get("total_chunks", 0),
                    "metadata": metadata
                }
                
                # Get score from appropriate column
                if rerank and "rerank_score" in row:
                    score = row["rerank_score"]
                else:
                    score = row.get("_distance", 0.0)
                
                # Create result object
                result = ResultItem(
                    id=row["id"],
                    payload=payload,
                    score=score
                )
                
                results.append(result)
            
            # Record metrics if ground truth is available
            true_context = getattr(processor, 'expected_context', None)
            if true_context:
                hit = any(hasattr(p, "payload") and p.payload and p.payload.get("text") == true_context for p in results)
                search_key = "vector"
                if rerank:
                    search_key += f"_{reranker_type or 'default'}"
                self._record_hit(search_key, hit)
            
            return results
                
        except Exception as e:
            if self.verbose:
                print(f"Error in dense search: {str(e)}")
                import traceback
                traceback.print_exc()
            return {"error": f"Error in dense search: {str(e)}"}

    def search_sparse(self, query: str, processor: Any, limit: int, score_threshold: float = None,
                    rerank: bool = False, reranker_type: str = None):
        """
        Perform sparse vector search
        
        Since LanceDB doesn't natively support sparse vector search yet, we use full-text search
        as the closest approximation, and apply reranking if needed.
        
        Args:
            query: Search query string
            processor: Document processor with embedding capabilities
            limit: Number of results to return
            score_threshold: Minimum score threshold
            rerank: Whether to apply reranking
            reranker_type: Type of reranker to use (colbert, cohere, jina, cross)
        """
        if processor is None:
            return {"error": "Sparse search requires an embedding model"}
        
        try:
            # Make sure FTS index exists
            if not self.has_fts_index:
                self._ensure_indexes()
            
            # If reranking is enabled, fetch more results for reranker to choose from
            overfetch_factor = 3 if rerank else 1
            fetch_limit = limit * overfetch_factor
            
            # Perform keyword-based search
            if self.verbose:
                print(f"Performing full-text search (as sparse approximation) for: '{query}'")
                if rerank:
                    print(f"Will oversample by factor {overfetch_factor} for reranking")
            
            # Get FTS results
            search_results = self.table.search(
                query, 
                query_type="fts"
            )
            
            # Apply score threshold if provided
            if score_threshold is not None:
                search_results = search_results.where(f"_bm25 >= {score_threshold}", prefilter=False)
            
            # Limit results - get more if we're going to rerank
            search_results = search_results.limit(fetch_limit)
            
            # Apply reranking if requested
            if rerank:
                # Get reranker
                reranker = self._get_reranker(reranker_type or "cross", self.verbose)
                
                if self.verbose:
                    print(f"Applying {reranker_type or 'cross'} reranker to FTS search results")
                
                # Apply reranking
                search_results = search_results.rerank(reranker=reranker)
            
            # Get the final result set
            results_df = search_results.limit(limit).to_pandas()
            
            # Record hit rate metrics if available
            true_context = getattr(processor, 'expected_context', None)
            if true_context:
                hit = any(row["text"] == true_context for _, row in results_df.iterrows())
                search_key = "fts"
                if rerank:
                    search_key += f"_{reranker_type or 'cross'}"
                self._record_hit(search_key, hit)
            
            # Convert results to the expected format
            results = []
            for _, row in results_df.iterrows():
                # Parse metadata JSON
                metadata = row.get("metadata", "{}")
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                
                # Create a result object (compatible with other backends)
                class ResultItem:
                    def __init__(self, id, payload, score):
                        self.id = id
                        self.payload = payload
                        self.score = score
                
                payload = {
                    "text": row["text"],
                    "file_path": row["file_path"],
                    "file_name": row["file_name"],
                    "chunk_index": row["chunk_index"],
                    "total_chunks": row.get("total_chunks", 0),
                    "metadata": metadata
                }
                
                result = ResultItem(
                    id=row["id"],
                    payload=payload,
                    score=row.get("_bm25", 0.0)  # Use BM25 score for keyword-based search
                )
                
                results.append(result)
            
            return results
            
        except Exception as e:
            if self.verbose:
                print(f"Error in sparse/keyword search: {e}")
            return {"error": f"Error in sparse search: {str(e)}"}


    def search_keyword(self, query: str, limit: int = 10, score_threshold: float = None,
                    rerank: bool = False, reranker_type: str = None):
        """
        Perform a keyword-based search using full-text search index
        
        Updated to handle API changes in newer LanceDB versions
        
        Args:
            query: Search query string
            limit: Number of results to return
            score_threshold: Minimum score threshold
            rerank: Whether to apply reranking
            reranker_type: Type of reranker to use
            
        Returns:
            List of search results
        """
        try:
            if not query.strip():
                return {"error": "Empty query"}
                
            # Ensure we have a working table connection
            if self.table is None:
                try:
                    self.connect()
                    self.table = self.db.open_table(self.collection_name)
                except Exception as e:
                    return {"error": f"Could not connect to table: {e}"}
                    
            # Create FTS index if needed
            if not self.has_fts_index:
                if self.verbose:
                    print("Creating FTS index for text search...")
                self._ensure_indexes()
                    
            if self.verbose:
                print(f"Performing keyword search for: '{query}'")
            
            try:
                # Try newest API first - FTS search with limit
                search_results = (
                    self.table.search(
                        query,
                        query_type="fts"
                    )
                    .limit(limit * 3)  # Get more results for possible filtering
                )
                
                # Apply score threshold if provided
                # Note: Different LanceDB versions use different column names for scores
                if score_threshold is not None:
                    try:
                        # Try with _bm25 score filter
                        search_results = search_results.where(f"_bm25 >= {score_threshold}", prefilter=False)
                    except:
                        try:
                            # Try with _score filter
                            search_results = search_results.where(f"_score >= {score_threshold}", prefilter=False)
                        except:
                            if self.verbose:
                                print(f"Could not apply score threshold - filter not supported")
                
                # Get the results as pandas DataFrame
                try:
                    # First try to_pandas()
                    results_df = search_results.to_pandas()
                except Exception as e1:
                    try:
                        # Try with to_arrow().to_pandas()
                        results_arr = search_results.to_arrow()
                        if hasattr(results_arr, 'to_pandas'):
                            results_df = results_arr.to_pandas()
                        else:
                            # Manual conversion if needed
                            import pandas as pd
                            results_df = pd.DataFrame({
                                col: results_arr[col].to_numpy() 
                                for col in results_arr.column_names
                            })
                    except Exception as e2:
                        return {"error": f"Error converting search results: {e1}; then {e2}"}
                        
                # Sort by score if available (different column names in different versions)
                for score_col in ['_bm25', '_score', 'score']:
                    if score_col in results_df.columns:
                        results_df = results_df.sort_values(by=score_col, ascending=False)
                        break
                
                # Limit to requested number after sorting
                results_df = results_df.head(limit)
                
                # Format results in the expected format
                results = []
                for _, row in results_df.iterrows():
                    # Create a result object (compatible with other backends)
                    class ResultItem:
                        def __init__(self, id, payload, score):
                            self.id = id
                            self.payload = payload
                            self.score = score
                    
                    # Parse metadata JSON
                    metadata = row.get("metadata", "{}")
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except:
                            metadata = {}
                    
                    # Create payload dictionary
                    payload = {
                        "text": row["text"],
                        "file_path": row["file_path"],
                        "file_name": row["file_name"],
                        "chunk_index": row["chunk_index"],
                        "total_chunks": row.get("total_chunks", 0),
                        "metadata": metadata
                    }
                    
                    # Extract score from appropriate column
                    score = 0.0
                    for score_col in ['_bm25', '_score', 'score']:
                        if score_col in row:
                            score = row[score_col]
                            break
                    
                    # Create result object
                    result = ResultItem(
                        id=row["id"],
                        payload=payload,
                        score=score
                    )
                    
                    results.append(result)
                    
                return results
                    
            except Exception as e:
                if self.verbose:
                    print(f"Error in keyword search: {e}")
                    import traceback
                    traceback.print_exc()
                    
                # Try alternate approach using SQL
                try:
                    if self.verbose:
                        print("Trying SQL-based keyword search...")
                        
                    # Escape single quotes in query
                    sql_query = query.replace("'", "''")
                    
                    # Prepare SQL query
                    sql = f"""
                    SELECT *,
                        SCORE() AS _score
                    FROM data
                    WHERE MATCH(text) AGAINST('{sql_query}')
                    ORDER BY _score DESC
                    LIMIT {limit * 3}
                    """
                    
                    # Execute SQL query
                    results = self.table.sql(sql)
                    
                    # Convert to pandas DataFrame
                    try:
                        results_df = results.to_pandas()
                    except:
                        # Try with to_arrow first
                        results_arr = results.to_arrow()
                        if hasattr(results_arr, 'to_pandas'):
                            results_df = results_arr.to_pandas()
                        else:
                            # Manual conversion
                            import pandas as pd
                            results_df = pd.DataFrame({
                                col: results_arr[col].to_numpy() 
                                for col in results_arr.column_names
                            })
                    
                    # Limit to requested number
                    results_df = results_df.head(limit)
                    
                    # Format results in the expected format
                    results = []
                    for _, row in results_df.iterrows():
                        # Create a result object (compatible with other backends)
                        class ResultItem:
                            def __init__(self, id, payload, score):
                                self.id = id
                                self.payload = payload
                                self.score = score
                        
                        # Parse metadata JSON
                        metadata = row.get("metadata", "{}")
                        if isinstance(metadata, str):
                            try:
                                metadata = json.loads(metadata)
                            except:
                                metadata = {}
                        
                        # Create payload dictionary
                        payload = {
                            "text": row["text"],
                            "file_path": row.get("file_path", ""),
                            "file_name": row.get("file_name", ""),
                            "chunk_index": row.get("chunk_index", 0),
                            "total_chunks": row.get("total_chunks", 0),
                            "metadata": metadata
                        }
                        
                        # Extract score from appropriate column
                        score = row.get("_score", 0.0)
                        
                        # Create result object
                        result = ResultItem(
                            id=row.get("id", f"result_{_}"),
                            payload=payload,
                            score=score
                        )
                        
                        results.append(result)
                        
                    return results
                except Exception as e2:
                    if self.verbose:
                        print(f"SQL-based search also failed: {e2}")
                    return {"error": f"Error in keyword search: {str(e)}; SQL fallback: {str(e2)}"}
        
        except Exception as e:
            if self.verbose:
                print(f"Error in keyword search: {str(e)}")
                import traceback
                traceback.print_exc()
            return {"error": f"Error in keyword search: {str(e)}"}

    def _retrieve_context_for_chunk(self, file_path: str, chunk_index: int, window: int = 1) -> str:
        """Retrieve context surrounding a chunk"""
        try:
            # Define minimum and maximum chunk indices to retrieve
            min_idx = max(0, chunk_index - window)
            max_idx = chunk_index + window
            
            # Use scalar indexes for efficient filtering if available
            if self.has_scalar_indexes:
                # Fetch chunks from the same file within the window
                query = self.table.search().where(
                    f"file_path = '{file_path}' AND chunk_index >= {min_idx} AND chunk_index <= {max_idx}"
                )
                
                # Order by chunk index
                query = query.order_by("chunk_index")
                
                # Execute query and get results
                context_chunks = query.to_pandas()
            else:
                # Fall back to full table scan if no scalar indexes
                if self.verbose:
                    print("Warning: Scalar indexes not available. Using full table scan for context retrieval.")
                all_docs = self.table.to_pandas()
                context_chunks = all_docs[
                    (all_docs["file_path"] == file_path) & 
                    (all_docs["chunk_index"] >= min_idx) & 
                    (all_docs["chunk_index"] <= max_idx)
                ].sort_values(by="chunk_index")
            
            # Combine the texts
            if len(context_chunks) > 0:
                combined_text = "\n".join(context_chunks["text"].tolist())
            else:
                combined_text = ""
            
            return combined_text
        except Exception as e:
            if self.verbose:
                print(f"Error retrieving context: {e}")
            return ""

    def _generate_sparse_vector(self, text: str) -> Tuple[List[int], List[float]]:
        """Generate a simple sparse vector using term frequencies"""
        from collections import Counter
        import re
        
        # Simple tokenization
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Remove very common words (simple stopwords)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'of', 'to', 'in', 'that', 'it', 'with', 'for'}
        tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
        
        # Count terms
        counter = Counter(tokens)
        
        # Convert to sparse vector format (for simplicity, use term hashes as indices)
        indices = []
        values = []
        
        for term, count in counter.items():
            # Simple hash function for terms - use a better one in production
            # Limit to 100000 dimensions
            term_index = hash(term) % 100000
            term_value = count / max(1, len(tokens))  # Normalize by document length, avoid division by zero
            indices.append(term_index)
            values.append(term_value)
        
        # If empty, return a single default dimension
        if not indices:
            return [0], [0.0]
                
        return indices, values
        
    def update_embeddings(self, id_embedding_payload_tuples: List[Tuple[str, np.ndarray, Dict[str, Any]]]) -> None:
        """
        Update existing embeddings with new values
        
        Args:
            id_embedding_payload_tuples: List of (id, embedding, payload) tuples to update
        """
        if not id_embedding_payload_tuples:
            return
            
        try:
            for id_str, embedding, payload in id_embedding_payload_tuples:
                # Generate sparse vector for the text (as fallback)
                text = payload.get("text", "")
                sparse_indices, sparse_values = self._generate_sparse_vector(text)
                
                # Format metadata as JSON string
                metadata = payload.get("metadata", {})
                if not isinstance(metadata, str):
                    metadata = json.dumps(metadata)
                
                # Create update values
                values = {
                    "vector": embedding.tolist(),
                    "sparse_indices": sparse_indices,
                    "sparse_values": sparse_values,
                    "text": text,
                    "file_path": payload.get("file_path", ""),
                    "file_name": payload.get("file_name", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "total_chunks": payload.get("total_chunks", 0),
                    "token_count": payload.get("token_count", 0),
                    "metadata": metadata
                }
                
                # Update the row
                self.table.update(where=f"id = '{id_str}'", values=values)
                
            if self.verbose:
                print(f"Updated {len(id_embedding_payload_tuples)} embeddings")
        except Exception as e:
            print(f"Error updating embeddings: {str(e)}")
            raise
            
    def delete_by_filter(self, filter_condition: str) -> int:
        """
        Delete entries matching a filter condition
        
        Args:
            filter_condition: SQL-like filter condition (e.g., "file_path = '/path/to/file'")
            
        Returns:
            Number of deleted entries
        """
        try:
            # Count rows that match the filter
            count_query = self.table.search().where(filter_condition)
            matching_count = len(count_query.to_pandas())
            
            # Execute deletion
            self.table.delete(filter_condition)
            
            if self.verbose:
                print(f"Deleted {matching_count} entries matching filter: {filter_condition}")
                
            return matching_count
        except Exception as e:
            print(f"Error deleting entries: {str(e)}")
            return 0
    
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

    def cleanup(self, remove_storage: bool = False) -> None:
        """Clean up resources"""
        # Close the database connection
        self.db = None
        self.table = None
        
        # Remove storage if requested
        if remove_storage and not self.uri:
            try:
                if self.verbose:
                    print(f"Removing storage directory: {self.storage_path}")
                shutil.rmtree(self.storage_path, ignore_errors=True)
            except Exception as e:
                print(f"Error cleaning up storage: {e}")