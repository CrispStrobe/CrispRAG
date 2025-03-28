# Modular Vector Database Architecture Reference

This guide explains the modular architecture that allows switching between different vector database backends.

## Overview

The codebase has been refactored to support multiple vector database backends through a common interface. The architecture consists of:

1. **Abstract Interface** (`VectorDBInterface`) - Defines common operations for all database implementations
2. **Database Factory** (`DBFactory`) - Creates database instances based on configuration
3. **Concrete Implementations** - Database-specific implementations that inherit from the interface
   - `QdrantManager` - Implementation for Qdrant
   - `LanceDBManager` - Implementation for LanceDB
   - `MeilisearchManager` - Implementation for Meilisearch
   - `ElasticsearchManager` - Implementation for Elasticsearch

## Command Line Usage

To specify which database backend to use, add the `--db-type` parameter to your commands:

```bash
# Indexing with Qdrant (default)
python indexer.py --db-type qdrant index /path/to/documents

# Indexing with LanceDB
python indexer.py --db-type lancedb index /path/to/documents

# Searching with Qdrant
python indexer.py --db-type qdrant search "your query here"

# Searching with LanceDB
python indexer.py --db-type lancedb search "your query here"
```

## Available Database Backends

Currently, the following vector database backends are supported:

1. **Qdrant** (`--db-type qdrant`) - Default option
   - Efficient vector database for similarity search
   - Both local and remote modes
   - Supports sparse vectors (SPLADE)

2. **LanceDB** (`--db-type lancedb`) - Lightweight vector database
   - Embedded vector database built on Lance data format
   - Great for local/edge deployments
   - Supports both local and remote modes

Other backends are planned but not yet fully implemented:

3. **Meilisearch** (`--db-type meilisearch`) - Fast search engine
   - Optimized for keyword and full-text search
   - Limited vector search capability

4. **Elasticsearch** (`--db-type elasticsearch`) - Enterprise search engine
   - Powerful search capabilities
   - Vector search via dense_vector field type

## Database-Specific Options

Each database backend has its own specific configuration options:

### Qdrant Options
```
--host HOSTNAME      Qdrant server hostname (default: localhost)
--port PORT          Qdrant server port (default: 6333)
--storage-path PATH  Path for local storage (used if host is localhost)
```

### LanceDB Options
```
--lancedb-uri URI    LanceDB URI for remote connection
--storage-path PATH  Path for local storage (used if URI not provided)
```

### Meilisearch Options
```
--meilisearch-url URL      Meilisearch server URL (default: http://localhost:7700)
--meilisearch-api-key KEY  API key for authentication
```

### Elasticsearch Options
```
--es-hosts HOSTS     Elasticsearch hosts (space-separated)
--es-api-key KEY     API key for authentication
--es-username USER   Username for basic authentication
--es-password PASS   Password for basic authentication
```

## Adding New Database Backends

To add support for a new vector database:

1. Create a new class that inherits from `VectorDBInterface`
2. Implement all required methods from the abstract interface
3. Register the new implementation in `vector_db/__init__.py`
4. Add database-specific command line parameters in `main.py`

## Common Interface Methods

All database implementations must implement these methods:

- `connect()` - Establish connection to the database
- `create_collection()` - Create or recreate a collection
- `get_collection_info()` - Get information about the collection
- `insert_embeddings()` - Insert embeddings (dense only)
- `insert_embeddings_with_sparse()` - Insert embeddings with sparse vectors
- `search()` - Main search method with various options
- `search_dense()` - Dense vector search
- `search_sparse()` - Sparse vector search
- `search_keyword()` - Keyword-based search
- `cleanup()` - Clean up resources

## Dependencies

Different database backends require different dependencies:

- **Qdrant**: `pip install qdrant-client`
- **LanceDB**: `pip install lancedb pyarrow`
- **Meilisearch**: `pip install meilisearch`
- **Elasticsearch**: `pip install elasticsearch`

Each database implementation checks for its dependencies and provides helpful error messages if they're missing.