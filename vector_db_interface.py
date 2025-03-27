from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Union, Literal
import numpy as np
import os
import time
import shutil
import json


# Add this import for proper type hinting:
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from utils import TextProcessor

class VectorDBInterface(ABC):
    """Abstract interface for vector database operations"""
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the database"""
        pass
    
    @abstractmethod
    def create_collection(self, recreate: bool = False) -> None:
        """Create or recreate a collection"""
        pass
    
    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        pass
    
    @abstractmethod
    def insert_embeddings(self, embeddings_with_payloads: List[Tuple[np.ndarray, Dict[str, Any]]]) -> None:
        """Insert embeddings into the database (dense vector only)"""
        pass
    
    @abstractmethod
    def insert_embeddings_with_sparse(self, embeddings_with_sparse: List[Tuple[np.ndarray, Dict[str, Any], Tuple[List[int], List[float]]]]) -> None:
        """Insert embeddings with sparse vectors into the database"""
        pass
    
    @abstractmethod
    def search(self, query: str, search_type: str = "hybrid", limit: int = 10,
              processor: Any = None, prefetch_limit: int = 50, fusion_type: str = "rrf",
              relevance_tuning: bool = True, context_size: int = 300, 
              score_threshold: float = None, rerank: bool = False) -> Dict[str, Any]:
        """Search the database with various options"""
        pass
    
    @abstractmethod
    def search_dense(self, query: str, processor: Any, limit: int, score_threshold: float = None):
        """Perform dense vector search"""
        pass
    
    @abstractmethod
    def search_sparse(self, query: str, processor: Any, limit: int, score_threshold: float = None):
        """Perform sparse vector search"""
        pass
    
    @abstractmethod
    def search_keyword(self, query: str, limit: int = 10, score_threshold: float = None):
        """Perform keyword-based search"""
        pass
    
    @abstractmethod
    def cleanup(self, remove_storage: bool = False) -> None:
        """Clean up resources"""
        pass


class DBFactory:
    """Factory for creating vector database instances"""
    
    @staticmethod
    def create_db(db_type: str, **kwargs):
        """
        Create a database instance of the specified type with comprehensive error handling.
        
        This function dynamically imports the appropriate database module based on the
        requested type, with informative error messages if the module or dependencies
        are missing.
        
        Args:
            db_type: Type of database to create ("qdrant", "lancedb", "meilisearch", "elasticsearch")
            **kwargs: Additional arguments for database initialization
                
        Returns:
            Instance of the appropriate VectorDBInterface implementation
        
        Raises:
            ValueError: If the database type is not supported or required dependencies are missing
            ImportError: If there's an issue importing required modules
        """
        # Validate and standardize the database type
        if not db_type:
            raise ValueError("Database type must be specified")
        
        db_type = db_type.lower().strip()
        
        # Define a mapping of database types to their module and class names
        db_mapping = {
            "qdrant": {"module": "qdrant_db", "class": "QdrantManager", 
                    "dependencies": ["qdrant-client"]},
            "lancedb": {"module": "lancedb_manager", "class": "LanceDBManager", 
                    "dependencies": ["lancedb", "pyarrow"]},
            "meilisearch": {"module": "meilisearch_manager", "class": "MeilisearchManager", 
                        "dependencies": ["meilisearch"]},
            "elasticsearch": {"module": "elasticsearch_manager", "class": "ElasticsearchManager", 
                            "dependencies": ["elasticsearch"]}
        }
        
        # Check if the requested database type is supported
        if db_type not in db_mapping:
            available = ", ".join(db_mapping.keys())
            raise ValueError(f"Unsupported database type: '{db_type}'. Available types: {available}")
        
        # Get the module and class information
        db_info = db_mapping[db_type]
        module_name = db_info["module"]
        class_name = db_info["class"]
        dependencies = db_info["dependencies"]
        
        try:
            # Import specific database module based on type
            if db_type == "qdrant":
                from qdrant_db import QdrantManager
                return QdrantManager(**kwargs)
            elif db_type == "lancedb":
                try:
                    from lancedb_manager import LanceDBManager
                    return LanceDBManager(**kwargs)
                except ImportError:
                    raise ValueError("LanceDB support not available. Install with: pip install lancedb pyarrow")
            elif db_type == "meilisearch":
                try:
                    from meilisearch_manager import MeilisearchManager
                    return MeilisearchManager(**kwargs)
                except ImportError:
                    raise ValueError("Meilisearch support not available. Install with: pip install meilisearch")
            elif db_type == "elasticsearch":
                try:
                    from elasticsearch_manager import ElasticsearchManager
                    return ElasticsearchManager(**kwargs)
                except ImportError:
                    raise ValueError("Elasticsearch support not available. Install with: pip install elasticsearch")
            else:
                # This should never happen due to the earlier check, but just in case
                raise ValueError(f"Unsupported database type: {db_type}")
                
        except ImportError as e:
            # Provide helpful information about missing dependencies
            deps_str = ", ".join(dependencies)
            install_cmd = " ".join(dependencies)
            error_msg = (f"Cannot use {db_type} database: required module '{module_name}' could not be imported.\n"
                        f"Please install the required dependencies: {deps_str}\n"
                        f"Install command: pip install {install_cmd}")
            raise ValueError(error_msg) from e
            
        except AttributeError as e:
            # Handle the case where the module exists but the class doesn't
            error_msg = f"The module '{module_name}' was found, but it does not contain the required class '{class_name}'"
            raise ValueError(error_msg) from e
            
        except Exception as e:
            # Handle any other initialization errors with context
            error_msg = f"Error initializing {db_type} database: {str(e)}"
            raise ValueError(error_msg) from e
    
    @staticmethod
    def create_db_old(db_type: str, **kwargs) -> 'VectorDBInterface':
        """
        Create a database instance of the specified type
        
        Args:
            db_type: Type of database to create
            **kwargs: Additional arguments for database initialization
            
        Returns:
            Instance of the specified database type
        """
        # Import directly when needed to avoid circular imports
        db_type = db_type.lower()
        
        if db_type == "qdrant":
            from qdrant_db import QdrantManager
            return QdrantManager(**kwargs)
        elif db_type == "lancedb":
            try:
                from lancedb_manager import LanceDBManager
                return LanceDBManager(**kwargs)
            except ImportError:
                raise ValueError("LanceDB support not available. Install with: pip install lancedb pyarrow")
        elif db_type == "meilisearch":
            try:
                from meilisearch_manager import MeilisearchManager
                return MeilisearchManager(**kwargs)
            except ImportError:
                raise ValueError("Meilisearch support not available. Install with: pip install meilisearch")
        elif db_type == "elasticsearch":
            try:
                from elasticsearch_manager import ElasticsearchManager
                return ElasticsearchManager(**kwargs)
            except ImportError:
                raise ValueError("Elasticsearch support not available. Install with: pip install elasticsearch")
        else:
            # Try to get available backends if we can't find the specific one
            try:
                # First attempt, from main package
                try:
                    import sys
                    if "__init__" in sys.modules:
                        # If we have a package structure, use that
                        from __init__ import AVAILABLE_DBS
                        available = ", ".join(AVAILABLE_DBS.keys())
                except (ImportError, KeyError):
                    # Otherwise, hard-code the list
                    available = "qdrant, lancedb, meilisearch, elasticsearch"
            except:
                available = "qdrant, lancedb, meilisearch, elasticsearch"
                
            raise ValueError(f"Unsupported database type: {db_type}. Available types: {available}")