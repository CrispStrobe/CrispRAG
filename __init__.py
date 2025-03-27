"""
Enhanced Vector Database indexer with support for multiple backends.

This module provides support for:
- Qdrant
- LanceDB
- Meilisearch (planned)
- Elasticsearch (planned)

with flexible model selection:
- MLX embedding models from registry or custom models
- Ollama for local LLM embeddings
- FastEmbed for efficient embedding generation
- MLX for efficient dense embedding generation
- SPLADE for sparse embedding generation
"""

# Direct imports for core functionality
from vector_db_interface import VectorDBInterface, DBFactory
from qdrant_db import QdrantManager

# Import utilities
from utils import (
    TextProcessor, ResultProcessor, SearchAlgorithms, 
    FileUtils, ModelUtils, GeneralUtils, 
    TextExtractorUtils, ChunkUtils
)

# Create a registry of available database backends
AVAILABLE_DBS = {
    "qdrant": QdrantManager
}

# Try to import optional database implementations
try:
    from lancedb_manager import LanceDBManager
    AVAILABLE_DBS["lancedb"] = LanceDBManager
    if "__all__" in globals():
        __all__.append('LanceDBManager')
except ImportError:
    pass

try:
    from meilisearch_manager import MeilisearchManager
    AVAILABLE_DBS["meilisearch"] = MeilisearchManager
    if "__all__" in globals():
        __all__.append('MeilisearchManager')
except ImportError:
    pass

try:
    from elasticsearch_manager import ElasticsearchManager
    AVAILABLE_DBS["elasticsearch"] = ElasticsearchManager
    if "__all__" in globals():
        __all__.append('ElasticsearchManager')
except ImportError:
    pass

# Set up __all__ for more controlled imports
__all__ = [
    'VectorDBInterface',
    'DBFactory',
    'QdrantManager',
    'TextProcessor',
    'ResultProcessor', 
    'SearchAlgorithms',
    'FileUtils',
    'ModelUtils',
    'GeneralUtils',
    'TextExtractorUtils',
    'ChunkUtils',
    'AVAILABLE_DBS'
]