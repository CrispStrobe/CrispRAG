import os
import time
import json
import shutil
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set, Iterator, Union, Literal
import numpy as np

from .vector_db_interface import VectorDBInterface

# Try to import LanceDB client
try:
    import lancedb
    import pyarrow as pa
    lancedb_available = True
except ImportError:
    lancedb_available = False


class LanceDBManager(VectorDBInterface):
    """Manager for LanceDB vector database operations"""
    
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
            
        self.uri = uri
        self.collection_name = collection_name
        self.vector_dim = vector_size
        self.storage_path = storage_path if storage_path else os.path.join(os.getcwd(), "lancedb_storage")
        self.verbose = verbose
        self.db = None
        self.table = None
        
        # Store the original model IDs
        self.dense_model_id = dense_model_id
        self.sparse_model_id = sparse_model_id
        
        self.connect()

    def connect(self) -> None:
        """Establish connection to LanceDB"""
        try:
            # Connect to LanceDB using URI if provided, otherwise use local storage
            uri = self.uri if self.uri else self.storage_path
            
            # Make sure the directory exists for local storage
            if not self.uri:
                os.makedirs(self.storage_path, exist_ok=True)
                
            if self.verbose:
                print(f"Connecting to LanceDB at {uri}")
                
            self.db = lancedb.connect(uri)
            
            # Try to open existing collection if it exists
            try:
                self.table = self.db.open_table(self.collection_name)
                if self.verbose:
                    print(f"Opened existing collection '{self.collection_name}'")
            except:
                if self.verbose:
                    print(f"Collection '{self.collection_name}' does not exist yet")
                self.table = None
                
        except Exception as e:
            print(f"Error connecting to LanceDB: {str(e)}")
            raise
        
    def create_collection(self, recreate: bool = False) -> None:
        """Create or recreate a LanceDB collection"""
        try:
            # Check if collection exists
            collection_exists = False
            try:
                self.table = self.db.open_table(self.collection_name)
                collection_exists = True
            except:
                pass

            if collection_exists:
                if recreate:
                    if self.verbose:
                        print(f"Recreating collection '{self.collection_name}'...")
                    # LanceDB doesn't have a direct drop_table, we need to recreate the table
                    self.db.drop_table(self.collection_name)
                    self.table = None
                else:
                    if self.verbose:
                        print(f"Collection '{self.collection_name}' already exists.")
                    return

            if self.verbose:
                print(f"Creating collection '{self.collection_name}' with vector size {self.vector_dim}")

            # Define schema for the collection with both dense and sparse vectors
            # LanceDB doesn't support sparse vectors directly, so we'll store them as arrays
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
                pa.field("metadata", pa.string())  # JSON string for metadata
            ])

            # Create the table
            self.table = self.db.create_table(
                self.collection_name,
                schema=schema,
                mode="overwrite" if recreate else "create"
            )

            # Create index on the vector field for similarity search
            self.table.create_index(
                ["vector"], 
                index_type="IVF_PQ", 
                replace=True
            )
            
            if self.verbose:
                print(f"✅ Collection '{self.collection_name}' created successfully with index")

        except Exception as e:
            print(f"❌ Error creating collection '{self.collection_name}': {str(e)}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            if not self.db or not self.table:
                return {"error": "Not connected to LanceDB or collection not found"}
                
            # Get statistics about the table
            try:
                stats = self.table.stats()
                row_count = stats.get("num_rows", 0)
            except:
                # Fall back to counting rows manually
                df = self.table.to_pandas(columns=["id"])
                row_count = len(df)
            
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
            
            # Get schema information
            schema = self.table.schema()
            
            return {
                "name": self.collection_name,
                "points_count": row_count,
                "disk_usage": disk_usage,
                "schema": str(schema),
                "vector_dim": self.vector_dim
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
                    "metadata": json.dumps(payload.get("metadata", {}))
                }
                rows.append(row)
            
            if self.verbose:
                print(f"Inserting {len(rows)} points into collection '{self.collection_name}'")
                
            # Insert data in batches
            BATCH_SIZE = 100
            for i in range(0, len(rows), BATCH_SIZE):
                batch = rows[i:i+BATCH_SIZE]
                if self.verbose and len(rows) > BATCH_SIZE:
                    print(f"Inserting batch {i//BATCH_SIZE + 1}/{(len(rows)-1)//BATCH_SIZE + 1} ({len(batch)} points)")
                
                self.table.add(batch)
                
            if self.verbose:
                print(f"Successfully inserted {len(rows)} points")
        except Exception as e:
            print(f"Error inserting embeddings: {str(e)}")
            raise

    def insert_embeddings_with_sparse(self, embeddings_with_sparse: List[Tuple[np.ndarray, Dict[str, Any], Tuple[List[int], List[float]]]]) -> None:
        """Insert embeddings with sparse vectors into LanceDB"""
        if not embeddings_with_sparse:
            return
            
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
            db_type="lancedb"  # Pass the db_type
        )