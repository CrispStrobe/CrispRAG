# Enhanced Modular Vector Database Architecture

This document explains the enhanced modular architecture that allows switching between different vector database backends and reuses common functionality through utility classes.

## Architecture Overview

The codebase has been refactored into a layered architecture:

1. **Abstract Interface Layer** - Defines common operations for all database implementations
   - `VectorDBInterface` - Abstract base class defining the database API
   - `DBFactory` - Factory class for creating database instances

2. **Utility Layer** - Common functionality shared across implementations
   - `TextProcessor` - Text processing and result formatting
   - `ResultProcessor` - Search result processing utilities
   - `SearchAlgorithms` - Search algorithms like fusion and reranking
   - `EmbeddingUtils` - Embedding generation utilities

3. **Database Implementation Layer** - Database-specific implementations
   - `QdrantManager` - Implementation for Qdrant
   - `LanceDBManager` - Implementation for LanceDB
   - `MeilisearchManager` - Implementation for Meilisearch
   - `ElasticsearchManager` - Implementation for Elasticsearch

## Utility Classes

The utility classes provide shared functionality that can be used by any database implementation:

### TextProcessor

Handles text processing and formatting of search results:

- `create_preview(text, query, context_size)` - Create a preview with context around search terms
- `create_smart_preview(text, query, context_size)` - Create a preview with highlighted search terms
- `highlight_query_terms(text, query)` - Highlight query terms in text using markdown
- `format_search_results(points, query, search_type, processor, context_size, retriever)` - Format search results with improved previews

### ResultProcessor

Utilities for processing search results:

- `get_score(result)` - Safely extract scores from different result objects

### SearchAlgorithms

Common search algorithms that work across different database backends:

- `manual_fusion(dense_results, sparse_results, limit, fusion_type)` - Combine dense and sparse search results
- `rerank_results(query, results, processor, limit)` - Rerank search results using cross-encoder approach

### EmbeddingUtils

Utilities for generating embeddings:

- `generate_sparse_vector(text)` - Generate a simple sparse vector using term frequencies

## Database Interface

All database implementations must implement the abstract interface:

```python
class VectorDBInterface(ABC):
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the database"""
        pass
    
    @abstractmethod
    def create_collection(self, recreate: bool = False) -> None:
        """Create or recreate a collection"""
        pass
    
    # ... other abstract methods ...
```

## Code Organization

The code is now organized into the following structure:

```
vector_db/
  ├── __init__.py              # Module initialization and exports
  ├── vector_db_interface.py   # Abstract interface and factory
  ├── utils.py                 # Common utility functions
  ├── qdrant_db.py             # Qdrant implementation
  ├── lancedb_manager.py       # LanceDB implementation
  ├── meilisearch_manager.py   # Meilisearch implementation
  └── elasticsearch_manager.py # Elasticsearch implementation
```

## Integration Pattern

Database implementations now delegate common functionality to the utility classes. For example, in `QdrantManager`:

```python
def _format_search_results(self, points, query, search_type, processor, context_size=300):
    """Format search results with improved preview using TextProcessor"""
    def context_retriever(file_path, chunk_index, window=1):
        return self._retrieve_context_for_chunk(file_path, chunk_index, window)
    
    return TextProcessor.format_search_results(
        points, query, search_type, processor, context_size, retriever=context_retriever
    )
```

This ensures consistent behavior across different database implementations while allowing database-specific optimizations.

## Adding New Database Backends

To add support for a new vector database:

1. Create a new class that inherits from `VectorDBInterface`
2. Implement all required methods from the abstract interface
3. Use the utility classes for common functionality
4. Register the new implementation in `vector_db/__init__.py`
5. Add database-specific command line parameters in `main.py`

## Benefits of the Enhanced Architecture

1. **DRY (Don't Repeat Yourself)** - Common logic is shared across implementations
2. **Consistent Behavior** - All implementations use the same text processing and search algorithms
3. **Maintainability** - Changes to common algorithms only need to be made in one place
4. **Extensibility** - New database backends can reuse existing utilities
5. **Separation of Concerns** - Clear separation between database operations and general utilities

This modular architecture makes the codebase more flexible, maintainable, and extensible while ensuring consistent behavior across different database backends.

# Implementation Guide for New Database Backends

This guide explains how to implement support for new vector database backends using the modular architecture.

## Step-by-Step Implementation Process

### 1. Create a New Manager Class

Create a new file named `your_db_manager.py` with a class that inherits from `VectorDBInterface`:

```python
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np

from .vector_db_interface import VectorDBInterface
from .utils import TextProcessor, ResultProcessor, SearchAlgorithms, EmbeddingUtils

class YourDBManager(VectorDBInterface):
    """Manager for YourDB vector database operations"""
    
    def __init__(self, 
                collection_name: str = "documents",
                vector_size: int = 384,
                storage_path: str = None,
                verbose: bool = False,
                dense_model_id: str = "bge-small",
                sparse_model_id: str = "distilbert-splade",
                **db_specific_args):
        """
        Initialize YourDBManager with model-specific vector configuration.
        
        Args:
            collection_name: Collection name
            vector_size: Vector dimension
            storage_path: Path to storage directory
            verbose: Verbose output
            dense_model_id: ID of dense model for naming vectors
            sparse_model_id: ID of sparse model for naming vectors
            **db_specific_args: Database-specific arguments
        """
        # Check if required dependencies are available
        try:
            import your_db_package
        except ImportError:
            raise ImportError("YourDB package not available. Install with: pip install your-db-package")
            
        # Initialize properties
        self.collection_name = collection_name
        self.vector_dim = vector_size
        self.storage_path = storage_path
        self.verbose = verbose
        self.client = None
        
        # Store model IDs
        self.dense_model_id = dense_model_id
        self.sparse_model_id = sparse_model_id
        
        # Store database-specific configuration
        self.db_config = db_specific_args
        
        # Connect to the database
        self.connect()
```

### 2. Implement Required Methods

Implement all abstract methods from the interface:

```python
def connect(self) -> None:
    """Establish connection to YourDB"""
    try:
        # Database-specific connection code
        import your_db_package
        
        if self.verbose:
            print(f"Connecting to YourDB...")
            
        self.client = your_db_package.Client(...)
        
        if self.verbose:
            print(f"Connected to YourDB successfully")
            
    except Exception as e:
        print(f"Error connecting to YourDB: {str(e)}")
        raise

def create_collection(self, recreate: bool = False) -> None:
    """Create or recreate a YourDB collection"""
    try:
        # Database-specific collection creation code
        if self.verbose:
            print(f"Creating collection '{self.collection_name}'...")
        
        # Check if collection exists
        collection_exists = self._collection_exists()
        
        if collection_exists and recreate:
            # Delete existing collection
            if self.verbose:
                print(f"Recreating collection '{self.collection_name}'...")
            self._delete_collection()
        elif collection_exists:
            if self.verbose:
                print(f"Collection '{self.collection_name}' already exists")
            return
        
        # Create collection with vector configuration
        self._create_collection_implementation()
        
        if self.verbose:
            print(f"Collection '{self.collection_name}' created successfully")
    except Exception as e:
        print(f"Error creating collection: {str(e)}")
        raise
```

### 3. Use Utility Classes

Use the utility classes for common functionality:

```python
def search(self, query: str, search_type: str = "hybrid", limit: int = 10,
          processor: Any = None, prefetch_limit: int = 50, fusion_type: str = "rrf",
          relevance_tuning: bool = True, context_size: int = 300, 
          score_threshold: float = None, rerank: bool = False) -> Dict[str, Any]:
    """
    Search the database with various options.
    
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
        
    Returns:
        Dictionary with search results and metadata
    """
    try:
        # Check for empty query
        query = query.strip()
        if not query:
            return {"error": "Empty query"}
        
        # Perform database-specific search based on search type
        if search_type == "vector" or search_type == "dense":
            points = self.search_dense(query, processor, limit, score_threshold)
        elif search_type == "sparse":
            points = self.search_sparse(query, processor, limit, score_threshold)
        elif search_type == "keyword":
            points = self.search_keyword(query, limit, score_threshold)
        else:  # Default to hybrid
            points = self.search_hybrid(query, processor, limit, prefetch_limit, 
                                      fusion_type, score_threshold, rerank)
        
        # Apply reranking if requested and not already applied
        if rerank and search_type != "hybrid" and not isinstance(points, dict):
            if self.verbose:
                print(f"Applying reranking to {len(points)} results")
            
            # Use the common reranking algorithm from SearchAlgorithms
            points = SearchAlgorithms.rerank_results(query, points, processor, limit)
        
        # Check for errors
        if isinstance(points, dict) and "error" in points:
            return points
        
        # Format search results using the common TextProcessor utility
        def context_retriever(file_path, chunk_index, window=1):
            # Implement database-specific context retrieval
            return self._get_context_for_chunk(file_path, chunk_index, window)
        
        return TextProcessor.format_search_results(
            points, query, search_type, processor, context_size, retriever=context_retriever
        )
        
    except Exception as e:
        print(f"Error during search: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
```

### 4. Register the Implementation

Register your implementation in `vector_db/__init__.py`:

```python
# Add import for your implementation
from .your_db_manager import YourDBManager

# Update __all__
__all__ = [
    'VectorDBInterface',
    'DBFactory',
    'QdrantManager',
    'LanceDBManager',
    'YourDBManager',  # Add your implementation
    'TextProcessor',
    'ResultProcessor', 
    'SearchAlgorithms',
    'EmbeddingUtils'
]

# Register available database implementations
AVAILABLE_DBS = {
    "qdrant": QdrantManager,
    "lancedb": LanceDBManager,
    "yourdb": YourDBManager,  # Add your implementation
}
```

### 5. Add Command-Line Arguments

Add database-specific command-line arguments in `main.py`:

```python
# Add your database type to the --db-type choices
parser.add_argument("--db-type", choices=["qdrant", "lancedb", "yourdb", "meilisearch", "elasticsearch"], 
                    default="qdrant", help="Vector database backend to use")

# Add database-specific parameter group for your database
yourdb_group = index_parser.add_argument_group("YourDB options")
yourdb_group.add_argument("--yourdb-host", type=str, default="localhost", 
                        help="YourDB server host")
yourdb_group.add_argument("--yourdb-port", type=int, default=1234, 
                        help="YourDB server port")
yourdb_group.add_argument("--yourdb-api-key", type=str, 
                        help="YourDB API key")

# Add the same parameters to the search parser
yourdb_search_group = search_parser.add_argument_group("YourDB options")
yourdb_search_group.add_argument("--yourdb-host", type=str, default="localhost", 
                               help="YourDB server host")
yourdb_search_group.add_argument("--yourdb-port", type=int, default=1234, 
                               help="YourDB server port")
yourdb_search_group.add_argument("--yourdb-api-key", type=str, 
                               help="YourDB API key")
```

### 6. Update Main Functions

Update the run_indexing and run_search functions to handle your database type:

```python
# In run_indexing function
elif db_type.lower() == 'yourdb':
    db_args.update({
        "yourdb_host": getattr(args, 'yourdb_host', "localhost"),
        "yourdb_port": getattr(args, 'yourdb_port', 1234),
        "yourdb_api_key": getattr(args, 'yourdb_api_key', None),
    })

# In