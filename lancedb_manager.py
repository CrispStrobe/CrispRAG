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

# Try to import LanceDB client
try:
    import lancedb
    import pyarrow as pa
    from lancedb.rerankers import (
        RRFReranker, 
        LinearCombinationReranker,
        CombineReranker,
        CrossEncoderReranker,
        ColbertReranker,
        CohereReranker,
        JinaReranker
    )
    lancedb_available = True
except ImportError:
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
        """Create or recreate a LanceDB collection with vector and text indexes"""
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
                    self.db.drop_table(self.collection_name)
                    self.table = None
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

            # Create the table
            self.table = self.db.create_table(
                self.collection_name,
                schema=schema,
                mode="overwrite" if recreate else "create"
            )

            # Create vector index for similarity search
            if self.verbose:
                print("Creating vector index...")
                
            self.table.create_index(
                ["vector"], 
                index_type="IVF_PQ", 
                metric="cosine",
                replace=True
            )
            self.has_vector_index = True
            
            # Create full-text search index
            if self.verbose:
                print("Creating full-text search index...")
                
            # Use more advanced FTS configuration
            self.table.create_fts_index(
                "text", 
                replace=True,
                with_position=True,  # Enable phrase queries
                lower_case=True,     # Case-insensitive search
                stem=True,           # Enable stemming
                remove_stop_words=True  # Remove common stop words
            )
            self.has_fts_index = True
            
            # Create scalar indices for efficient filtering
            if self.verbose:
                print("Creating scalar indices for efficient filtering...")
                
            self.table.create_scalar_index("file_path")
            self.table.create_scalar_index("chunk_index")
            self.has_scalar_indexes = True
            
            if self.verbose:
                print(f"✅ Collection '{self.collection_name}' created successfully with indexes")
                
            # Wait for indexes to be ready
            self._wait_for_indexes()

        except Exception as e:
            print(f"❌ Error creating collection '{self.collection_name}': {str(e)}")
            raise
            
    def _wait_for_indexes(self, timeout=300, poll_interval=10):
        """Wait for indexes to be ready"""
        if not self.verbose:
            return
            
        print("Waiting for indexes to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                indices = self.table.list_indices()
                if not indices:
                    time.sleep(poll_interval)
                    continue
                    
                # Check if we have vector and FTS indexes
                has_vector = any(idx.get("type", "") in ["IVF_PQ", "HNSW", "IVF_FLAT"] for idx in indices)
                has_fts = any(idx.get("type", "") == "FULLTEXT" for idx in indices)
                
                if has_vector and has_fts:
                    # Check if indices are fully built
                    all_ready = True
                    for idx in indices:
                        idx_name = idx.get("name", "")
                        try:
                            stats = self.table.index_stats(idx_name)
                            if stats.num_unindexed_rows > 0:
                                all_ready = False
                                break
                        except:
                            all_ready = False
                            break
                    
                    if all_ready:
                        print("All indexes are ready.")
                        return
            except Exception as e:
                print(f"Error checking index status: {e}")
            
            print(f"⏳ Waiting for indexes to be ready... ({int(time.time() - start_time)}s)")
            time.sleep(poll_interval)
            
        print("⚠️ Timeout waiting for indexes to be ready. Continuing anyway.")

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
            
            # Get table schema and index information
            schema = self.table.schema()
            
            # Get index information
            indexes = {}
            try:
                index_info = self.table.list_indices()
                for idx in index_info:
                    idx_name = idx.get("name", "unknown")
                    idx_type = idx.get("type", "unknown")
                    idx_column = idx.get("column", "unknown")
                    indexes[idx_name] = {"type": idx_type, "column": idx_column}
                    
                    # Get additional index stats if available
                    try:
                        idx_stats = self.table.index_stats(idx_name)
                        indexes[idx_name]["stats"] = {
                            "indexed_rows": idx_stats.num_indexed_rows,
                            "unindexed_rows": idx_stats.num_unindexed_rows,
                            "distance_type": getattr(idx_stats, "distance_type", None)
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
                "schema": str(schema),
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
                
            # Make sure we have appropriate indexes
            self._ensure_indexes()
        except Exception as e:
            print(f"Error inserting embeddings: {str(e)}")
            raise

    def insert_embeddings_with_sparse(self, embeddings_with_sparse: List[Tuple[np.ndarray, Dict[str, Any], Tuple[List[int], List[float]]]]) -> None:
        """Insert embeddings with sparse vectors into LanceDB"""
        if not embeddings_with_sparse:
            return
            
        try:
            if not self.table:
                raise ValueError(f"Collection '{self.collection_name}' not found or not initialized")
                
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
                
                self.table.add(batch)
                
            if self.verbose:
                print(f"Successfully inserted {len(rows)} points with sparse vectors")
                
            # Make sure we have appropriate indexes
            self._ensure_indexes()
        except Exception as e:
            print(f"Error inserting embeddings with sparse vectors: {str(e)}")
            raise
            
    def _ensure_indexes(self):
        """Ensure that necessary indexes exist"""
        try:
            # Check if we have necessary indexes
            indices = self.table.list_indices()
            
            # Check for vector index
            vector_index = any(idx.get("type", "") in ["IVF_PQ", "HNSW", "IVF_FLAT"] for idx in indices)
            if not vector_index and not self.has_vector_index:
                if self.verbose:
                    print("Creating vector index...")
                self.table.create_index(
                    ["vector"], 
                    index_type="IVF_PQ", 
                    metric="cosine",
                    replace=True
                )
                self.has_vector_index = True
            
            # Check for FTS index
            fts_index = any(idx.get("type", "") == "FULLTEXT" for idx in indices)
            if not fts_index and not self.has_fts_index:
                if self.verbose:
                    print("Creating full-text search index...")
                self.table.create_fts_index(
                    "text", 
                    replace=True,
                    with_position=True,
                    lower_case=True,
                    stem=True,
                    remove_stop_words=True
                )
                self.has_fts_index = True
            
            # Check for scalar indexes
            file_path_index = any(idx.get("column", "") == "file_path" and idx.get("type", "") == "SCALAR" for idx in indices)
            chunk_index_index = any(idx.get("column", "") == "chunk_index" and idx.get("type", "") == "SCALAR" for idx in indices)
            
            if not file_path_index or not chunk_index_index:
                self.has_scalar_indexes = False
                
            if not self.has_scalar_indexes:
                if self.verbose:
                    print("Creating scalar indexes...")
                    
                if not file_path_index:
                    self.table.create_scalar_index("file_path")
                    
                if not chunk_index_index:
                    self.table.create_scalar_index("chunk_index")
                    
                self.has_scalar_indexes = True
        
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not ensure indexes: {e}")

    def _get_reranker(self, reranker_type: str = "rrf", verbose: bool = False):
        """
        Get a reranker instance based on type
        
        Args:
            reranker_type: Type of reranker to use ('rrf', 'linear', 'dbsf', 'colbert', 'cohere', 'jina', 'cross')
            verbose: Whether to show verbose output
            
        Returns:
            Reranker instance
        """
        reranker_type = reranker_type.lower()
        
        try:
            if reranker_type == "rrf":
                return RRFReranker()
            elif reranker_type == "linear":
                return LinearCombinationReranker(weight=0.7)  # 0.7 for vector, 0.3 for FTS
            elif reranker_type == "dbsf":
                return CombineReranker(normalize="score")
            elif reranker_type == "colbert":
                return ColbertReranker()
            elif reranker_type == "cohere":
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
            fusion_type: Type of fusion to use (rrf, linear, dbsf)
            score_threshold: Minimum score threshold
            rerank: Whether to apply additional reranking after fusion
            reranker_type: Type of reranker to use for final reranking (colbert, cohere, jina, cross)
        """
        if processor is None:
            return {"error": "Hybrid search requires an embedding model"}
        
        try:
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

            # First perform hybrid search with fusion
            search_results = (
                self.table.search(
                    query,
                    query_type="hybrid"
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
            
            # Get the results as pandas DataFrame
            results_df = search_results.to_pandas()
            
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
                
                # Create result object
                result = ResultItem(
                    id=row["id"],
                    payload=payload,
                    score=row.get("_distance", 0.0)
                )
                
                results.append(result)
            
            return results
            
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
        """
        Perform dense vector search with optional reranking
        
        Args:
            query: Search query string
            processor: Document processor with embedding capabilities
            limit: Number of results to return
            score_threshold: Minimum score threshold
            rerank: Whether to apply reranking
            reranker_type: Type of reranker to use (colbert, cohere, jina, cross)
        """
        if processor is None:
            return {"error": "Dense search requires an embedding model"}
        
        try:
            # Generate query vector
            query_vector = processor.get_embedding(query)
            
            # Ensure vector index exists
            if not self.has_vector_index:
                self._ensure_indexes()
            
            # If reranking is enabled, fetch more results for reranker to choose from
            overfetch_factor = 3 if rerank else 1
            fetch_limit = limit * overfetch_factor
            
            # Perform vector search
            if self.verbose:
                print(f"Performing dense vector search for: '{query}'")
                if rerank:
                    print(f"Will oversample by factor {overfetch_factor} for reranking")
            
            # Check for any unindexed data to decide on fast_search
            fast_search = True
            try:
                # Get vector index name (usually "vector_idx")
                indices = self.table.list_indices()
                vector_index = next((idx for idx in indices if idx.get("type", "") in ["IVF_PQ", "HNSW", "IVF_FLAT"]), None)
                
                if vector_index:
                    idx_name = vector_index.get("name", "vector_idx")
                    stats = self.table.index_stats(idx_name)
                    
                    if stats.num_unindexed_rows > 0:
                        if self.verbose:
                            print(f"Warning: {stats.num_unindexed_rows} unindexed rows. Setting fast_search=False to include all data.")
                        fast_search = False
            except Exception as e:
                if self.verbose:
                    print(f"Could not check for unindexed data: {e}")
            
            # Get initial search results
            search_results = self.table.search(
                query_vector.tolist(), 
                metric="cosine",
                fast_search=fast_search
            )
            
            # Apply score threshold if provided
            if score_threshold is not None:
                search_results = search_results.where(f"_distance >= {score_threshold}", prefilter=False)
            
            # Order by distance (similarity score) and get more results if we plan to rerank
            search_results = search_results.order_by("_distance", ascending=False).limit(fetch_limit)
            
            # Apply reranking if requested
            if rerank:
                # Get reranker
                reranker = self._get_reranker(reranker_type or "cross", self.verbose)
                
                if self.verbose:
                    print(f"Applying {reranker_type or 'cross'} reranker to dense search results")
                
                # Apply reranking
                search_results = search_results.rerank(reranker=reranker)
            
            # Get the final result set
            results_df = search_results.limit(limit).to_pandas()
            
            # Record hit rate metrics if available
            true_context = getattr(processor, 'expected_context', None)
            if true_context:
                hit = any(row["text"] == true_context for _, row in results_df.iterrows())
                search_key = "vector"
                if rerank:
                    search_key += f"_{reranker_type or 'cross'}"
                self._record_hit(search_key, hit)
            
            # Convert to the expected format
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
                    score=row.get("_distance", 0.0)
                )
                
                results.append(result)
            
            return results
            
        except Exception as e:
            if self.verbose:
                print(f"Error in dense search: {e}")
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
        Perform keyword-based search using full-text search index
        
        Args:
            query: Search query string
            limit: Number of results to return
            score_threshold: Minimum score threshold
            rerank: Whether to apply reranking
            reranker_type: Type of reranker to use (colbert, cohere, jina, cross)
        """
        try:
            # Make sure FTS index exists
            if not self.has_fts_index:
                self._ensure_indexes()
            
            # If reranking is enabled, fetch more results for reranker to choose from
            overfetch_factor = 3 if rerank else 1
            fetch_limit = limit * overfetch_factor
            
            # Perform FTS search
            if self.verbose:
                print(f"Performing keyword search for: '{query}'")
                if rerank:
                    print(f"Will oversample by factor {overfetch_factor} for reranking")
            
            search_results = self.table.search(
                query, 
                query_type="fts"
            )
            
            # Apply score threshold if provided
            if score_threshold is not None:
                search_results = search_results.where(f"_bm25 >= {score_threshold}", prefilter=False)
            
            # Order by BM25 score and limit results - get more if we're going to rerank
            search_results = search_results.order_by("_bm25", ascending=False).limit(fetch_limit)
            
            # Apply reranking if requested
            if rerank:
                # Get reranker
                reranker = self._get_reranker(reranker_type or "cross", self.verbose)
                
                if self.verbose:
                    print(f"Applying {reranker_type or 'cross'} reranker to keyword search results")
                
                # Apply reranking
                search_results = search_results.rerank(reranker=reranker)
            
            # Get the final result set
            results_df = search_results.limit(limit).to_pandas()
            
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
                print(f"Error in keyword search: {e}")
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