# CrispRAG

A vector search system supporting multiple database backends and embedding providers. (Mostly for testing purposes.)

## Overview

Implements a modular system for semantic search using vector embeddings. It supports:
- Multiple database backends: Qdrant, LanceDB, Meilisearch, Chroma, Elasticsearch (others in progress)
- Multiple embedding providers: MLX, Ollama (others in progress)
- Dense and (if supported by db) sparse vector representations
- Hybrid search with multiple fusion algorithms
- Vector reranking options

## Features

- **Unified Interface**: Common interface for all vector databases
- **Modular Design**: Easily switch between different databases and embedding providers
- **Hybrid Search**: Combine dense vector, sparse vector, and keyword search results
- **Performance Tracking**: Track search performance with hit rate metrics
- **Context Retrieval**: Get surrounding context for search results

## Installation

1. Clone the repository:
```bash
git clone https://github.com/CrispStrobe/CrispRAG.git
cd CrispRAG
```

2. Install base dependencies:
```bash
pip install numpy requests psutil tqdm
```

3. Install the database backends you want to use:
```bash
# For Qdrant
pip install qdrant-client

# For LanceDB
pip install lancedb pyarrow

# For Meilisearch
pip install meilisearch

# For Elasticsearch
pip install elasticsearch

# For Milvus
pip install pymilvus
```

4. Install embedding providers:
```bash
# For MLX
pip install mlx mlx-embedding-models

# For FastEmbed
pip install fastembed
# For GPU support
pip install fastembed-gpu

# For PyTorch (fallback)
pip install torch transformers
```

## Usage

### Indexing Documents

```bash
python mlxrag.py index ./path/to/documents \
    --collection my_collection \
    --db-type qdrant \
    --use-mlx-models \
    --dense-model bge-small \
    --sparse-model distilbert-splade
```

Options:
- `--db-type`: Choose from "qdrant", "lancedb", "meilisearch", "chromadb", "elasticsearch" (WIP: "milvus")
- `--use-mlx-models`, `--use-ollama` (WIP: `--use-fastembed`): Choose embedding provider
- `--dense-model`: Dense embedding model to use
- `--sparse-model`: Sparse embedding model to use
- `--include`: File patterns to include (e.g., "*.pdf *.txt")
- `--recreate`: Recreate collection if it exists

### Searching

```bash
python mlxrag.py search "my search query" \
    --search-type hybrid \
    --db-type qdrant \
    --collection my_collection \
    --limit 10 \
    --use-mlx-models \
    --fusion rrf \
    --rerank
```

Options:
- `--search-type`: Choose from "hybrid", "vector", "sparse", "keyword"
- `--limit`: Maximum number of results to return
- `--fusion`: Fusion strategy for hybrid search ("rrf", "dbsf")
- `--rerank`: Whether to apply reranking to improve results
- `--reranker-type`: Type of reranker ("colbert", "cross", "rrf", etc.)

### List Available Models

```bash
python mlxrag.py list-models
```

### List Available Database Backends

```bash
python mlxrag.py list-dbs
```

### Examples

For milvus, you might want to create a docker-compose.yml like:

```
services:
  milvus-etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      ETCD_AUTO_COMPACTION_MODE: revision
      ETCD_AUTO_COMPACTION_RETENTION: "1000"
      ETCD_QUOTA_BACKEND_BYTES: "4294967296"
      ETCD_UNSUPPORTED_ARCH: arm64
    command: >
      etcd
      --name milvus-etcd
      --data-dir /etcd
      --listen-client-urls http://0.0.0.0:2379
      --advertise-client-urls http://milvus-etcd:2379
      --listen-peer-urls http://0.0.0.0:2380
      --initial-advertise-peer-urls http://milvus-etcd:2380
      --initial-cluster milvus-etcd=http://milvus-etcd:2380
    volumes:
      - etcd-data:/etcd
    ports:
      - "2379:2379"
      - "2380:2380"
    networks:
      - milvus-network

  minio:
    container_name: minio
    image: minio/minio:RELEASE.2023-12-02T10-51-33Z
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio-data:/data
    ports:
      - "9000:9000"
      - "9001:9001"
    networks:
      - milvus-network

  milvus:
    container_name: milvus
    image: milvusdb/milvus:v2.3.9
    command: ["milvus", "run", "standalone"]
    depends_on:
      - milvus-etcd
      - minio
    environment:
      ETCD_ENDPOINTS: milvus-etcd:2379
      MILVUS_DEPLOY_MODE: standalone
      MILVUS_ENABLE_EMBEDDED_COORDINATORS: "true"
      MILVUS_LOG_LEVEL: debug
      MINIO_ADDRESS: minio:9000
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "19530:19530"
      - "9091:9091"
    networks:
      - milvus-network

volumes:
  etcd-data:
  minio-data:

networks:
  milvus-network:
```

then:
```bash
docker compose down
docker compose up -d
python mlxrag.py --verbose --use-mlx-models --db-type milvus index ~/Documents/texte --include "*.txt *.pdf" --limit 5 --recreate
```

## Architecture

The project follows a modular architecture:

- `vector_db_interface.py`: Abstract interface and factory for vector databases
- `utils.py`: Common utilities for text processing, search algorithms, and embedding
- `mlx_utils.py`, `ollama_utils.py`, etc: Embedding providers
- Database implementations:
  - `qdrant_db.py`: Qdrant implementation
  - `lancedb_manager.py`: LanceDB implementation
  - `meilisearch_manager.py`: Meilisearch implementation
  - etc
- `mlxrag.py`: Command-line interface

## Extending the System

### Adding a New Database Backend
1. Create a new class that implements the `VectorDBInterface` abstract methods
2. Add the new database to the `DBFactory` in `vector_db_interface.py`
3. Update the `AVAILABLE_DBS` dictionary in `__init__.py`

### Adding a New Embedding Provider
1. Create a new utility file (e.g., `new_provider_utils.py`)
2. Implement the embedding provider with methods compatible with the existing ones
3. Update the `TextProcessor` class to support the new provider

## License

MIT