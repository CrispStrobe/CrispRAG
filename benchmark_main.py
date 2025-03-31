#!/usr/bin/env python3
"""
adapted_rag_benchmark.py: Performance benchmarking tool for vector databases and embedding providers

Adapted to:
1. Use existing document collections instead of generating synthetic data
2. Support file patterns (*.txt, *.pdf) like mlxrag.py
3. Run multiple queries in series without restarting mlxrag for each query
4. Maintain detailed performance metrics

Features:
- Benchmarking across different vector databases and embedding providers
- Supports the same parameters as mlxrag.py for consistent testing
- Detailed reporting with visualizations
- Resource usage tracking (CPU, memory, disk)
"""

import argparse
import time
import os
import sys
import json
import shutil
import random
import string
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set, Union, Callable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict
import fnmatch
import threading
import queue

# Try to import optional dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available. Install with: pip install psutil")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not available. Install with: pip install tqdm")


# ============= Configuration Classes =============

@dataclass
class BenchmarkConfig:
    # General configuration
    name: str
    output_dir: str
    verbose: bool = False
    skip_existing: bool = False
    clean: bool = False
    
    # Document configuration
    documents_dir: str
    include_patterns: List[str] = None
    limit: int = None
    
    # Query configuration
    queries: List[str] = None
    query_file: str = None
    num_queries: int = 20
    
    # Backends to test
    db_backends: List[str] = None
    embedding_backends: List[str] = None
    
    # Benchmark parameters
    index_iterations: int = 3
    search_iterations: int = 5
    
    # Resource monitoring
    monitor_resources: bool = True
    sampling_interval: float = 0.5  # seconds
    
    # Optimization
    use_server_mode: bool = True  # Run mlxrag in server mode for multiple queries


@dataclass
class BackendConfig:
    # DB backend config
    db_type: str
    storage_path: str
    collection_name: str = "benchmark"
    
    # Host configurations for remote DBs
    host: str = "localhost"
    port: int = None
    
    # Authentication configs
    username: str = None
    password: str = None
    api_key: str = None
    
    # LanceDB specific
    lancedb_uri: str = None
    
    # Meilisearch specific
    meilisearch_url: str = None
    meilisearch_api_key: str = None
    
    # Elasticsearch specific  
    es_hosts: List[str] = None
    
    # Milvus specific
    milvus_secure: bool = False
    milvus_token: str = None
    
    # ChromaDB specific
    chromadb_use_embedding_function: bool = False
    chromadb_embedding_model: str = None


@dataclass
class EmbeddingConfig:
    # Embedding backend
    provider: str  # 'mlx', 'ollama', 'fastembed'
    
    # General options
    model_name: str
    weights_path: str = None
    
    # MLX options
    dense_model: str = "bge-small"
    sparse_model: str = "distilbert-splade"
    top_k: int = 64
    custom_repo_id: str = None
    custom_ndim: int = None
    custom_pooling: str = "mean"
    custom_normalize: bool = True
    custom_max_length: int = 512
    
    # Ollama options
    ollama_host: str = "http://localhost:11434"
    
    # FastEmbed options
    fastembed_sparse_model: str = None
    fastembed_use_gpu: bool = False
    fastembed_cache_dir: str = None


@dataclass
class BenchmarkResult:
    # General info
    timestamp: str
    config_name: str 
    db_backend: str
    embedding_backend: str
    
    # Hardware info
    cpu_info: Dict[str, Any]
    memory_info: Dict[str, Any]
    
    # Document info
    document_count: int
    
    # Indexing metrics
    index_time_seconds: float
    index_throughput: float  # docs/second
    index_memory_mb: float
    index_cpu_percent: float
    disk_usage_mb: float
    
    # Search metrics
    search_time_seconds: float 
    search_throughput: float  # queries/second
    search_memory_mb: float
    search_cpu_percent: float
    
    # Quality metrics
    latency_p50: Optional[float] = None
    latency_p90: Optional[float] = None
    latency_p99: Optional[float] = None


# ============= MLXrag Server Mode Implementation =============

class MLXragServer:
    """
    A server mode wrapper for mlxrag.py that maintains a long-running process
    for executing multiple search queries without reloading models.
    """
    def __init__(self, base_args, backend_config, config):
        self.base_args = base_args
        self.backend_config = backend_config
        self.config = config
        self.process = None
        self.running = False
        self.query_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_thread = None
    
    def start(self):
        """Start the mlxrag server process"""
        if self.running:
            return
        
        # Create a custom script to act as server
        server_script = self._create_server_script()
        
        # Start the server process
        cmd = [sys.executable, server_script] + self.base_args
        cmd = [arg for arg in cmd if arg]  # Remove empty args
        
        if self.config.verbose:
            print(f"Starting MLXrag server with: {' '.join(cmd)}")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Wait for server to initialize
            ready_line = self.process.stdout.readline().strip()
            if "SERVER_READY" not in ready_line:
                print(f"Server didn't start properly. Output: {ready_line}")
                self.stop()
                return False
                
            self.running = True
            
            # Start worker thread to handle queries
            self.worker_thread = threading.Thread(target=self._worker)
            self.worker_thread.daemon = True
            self.worker_thread.start()
            
            return True
            
        except Exception as e:
            print(f"Error starting MLXrag server: {e}")
            self.stop()
            return False
    
    def _create_server_script(self):
        """Create a temporary script that runs mlxrag in server mode"""
        script_code = """
import sys
import os
import time
import subprocess
import json

def main():
    # Import necessary modules from mlxrag
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from vector_db_interface import DBFactory
        from utils import TextProcessor
    except ImportError:
        print("ERROR: Failed to import mlxrag modules")
        return 1
    
    # Parse arguments (same as mlxrag would)
    args = sys.argv[1:]
    
    # Find db_type
    db_type = "qdrant"  # default
    for i, arg in enumerate(args):
        if arg == "--db-type" and i + 1 < len(args):
            db_type = args[i + 1]
    
    # Parse --collection
    collection = "documents"  # default
    for i, arg in enumerate(args):
        if arg == "--collection" and i + 1 < len(args):
            collection = args[i + 1]
    
    # Load text processor only once
    verbose = "--verbose" in args
    
    # Determine storage path
    storage_path = "./storage"  # default
    for i, arg in enumerate(args):
        if arg == "--storage-path" and i + 1 < len(args):
            storage_path = args[i + 1]
    
    # Initialize text processor and DB
    processor_args = {
        "model_name": None,
        "weights_path": None,
        "verbose": verbose
    }
    
    # Determine which provider to use
    if "--use-mlx-models" in args:
        processor_args["use_mlx_embedding"] = True
        # Find MLX-specific args
        for i, arg in enumerate(args):
            if arg == "--dense-model" and i + 1 < len(args):
                processor_args["dense_model"] = args[i + 1]
            elif arg == "--sparse-model" and i + 1 < len(args):
                processor_args["sparse_model"] = args[i + 1]
            elif arg == "--top-k" and i + 1 < len(args):
                processor_args["top_k"] = int(args[i + 1])
            elif arg == "--custom-repo-id" and i + 1 < len(args):
                processor_args["custom_repo_id"] = args[i + 1]
    elif "--use-ollama" in args:
        processor_args["use_ollama"] = True
        for i, arg in enumerate(args):
            if arg == "--ollama-model" and i + 1 < len(args):
                processor_args["ollama_model"] = args[i + 1]
            elif arg == "--ollama-host" and i + 1 < len(args):
                processor_args["ollama_host"] = args[i + 1]
    elif "--use-fastembed" in args:
        processor_args["use_fastembed"] = True
        for i, arg in enumerate(args):
            if arg == "--fastembed-model" and i + 1 < len(args):
                processor_args["fastembed_model"] = args[i + 1]
            elif arg == "--fastembed-sparse-model" and i + 1 < len(args):
                processor_args["fastembed_sparse_model"] = args[i + 1]
            elif arg == "--fastembed-use-gpu" in args:
                processor_args["fastembed_use_gpu"] = True
    
    try:
        # Initialize the processor
        processor = TextProcessor(**processor_args)
        
        # Common database arguments
        db_args = {
            "collection_name": collection,
            "vector_size": processor.vector_size,
            "storage_path": storage_path,
            "verbose": verbose,
            "dense_model_id": processor.dense_model_id,
            "sparse_model_id": processor.sparse_model_id
        }
        
        # Add database-specific parameters
        if db_type == 'qdrant':
            for i, arg in enumerate(args):
                if arg == "--host" and i + 1 < len(args):
                    db_args["host"] = args[i + 1]
                elif arg == "--port" and i + 1 < len(args):
                    db_args["port"] = int(args[i + 1])
        elif db_type == 'lancedb':
            for i, arg in enumerate(args):
                if arg == "--lancedb-uri" and i + 1 < len(args):
                    db_args["uri"] = args[i + 1]
        elif db_type == 'milvus':
            for i, arg in enumerate(args):
                if arg == "--milvus-host" and i + 1 < len(args):
                    db_args["host"] = args[i + 1]
                elif arg == "--milvus-port" and i + 1 < len(args):
                    db_args["port"] = args[i + 1]
                elif arg == "--milvus-user" and i + 1 < len(args):
                    db_args["user"] = args[i + 1]
                elif arg == "--milvus-password" and i + 1 < len(args):
                    db_args["password"] = args[i + 1]
                elif arg == "--milvus-secure" in args:
                    db_args["secure"] = True
                elif arg == "--milvus-token" and i + 1 < len(args):
                    db_args["token"] = args[i + 1]

        # Create the database handler
        db_manager = DBFactory.create_db(db_type, **db_args)
        
        # Signal that the server is ready
        print("SERVER_READY")
        sys.stdout.flush()
        
        # Process search queries
        while True:
            line = sys.stdin.readline().strip()
            if not line or line == "EXIT":
                break
                
            try:
                # Parse the query and parameters
                query_data = json.loads(line)
                query = query_data["query"]
                search_type = query_data.get("search_type", "hybrid")
                limit = query_data.get("limit", 10)
                score_threshold = query_data.get("score_threshold", None)
                prefetch_limit = query_data.get("prefetch_limit", 50)
                fusion_type = query_data.get("fusion", "rrf")
                context_size = query_data.get("context_size", 300)
                rerank = query_data.get("rerank", False)
                reranker_type = query_data.get("reranker_type", None)
                
                # Execute search
                start_time = time.time()
                results = db_manager.search(
                    query=query,
                    search_type=search_type,
                    limit=limit,
                    processor=processor,
                    prefetch_limit=prefetch_limit,
                    fusion_type=fusion_type,
                    relevance_tuning=True,
                    context_size=context_size,
                    score_threshold=score_threshold,
                    rerank=rerank,
                    reranker_type=reranker_type
                )
                end_time = time.time()
                
                # Add timing information
                results["search_time"] = end_time - start_time
                
                # Send results back
                print(json.dumps({"status": "success", "results": results}))
                sys.stdout.flush()
                
            except Exception as e:
                print(json.dumps({"status": "error", "error": str(e)}))
                sys.stdout.flush()
    
    except Exception as e:
        print(f"SERVER_ERROR: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
        """
        
        # Create a temporary script file
        fd, script_path = tempfile.mkstemp(suffix='.py', prefix='mlxrag_server_')
        with os.fdopen(fd, 'w') as f:
            f.write(script_code)
        
        return script_path
    
    def _worker(self):
        """Worker thread to process queries and collect results"""
        while self.running:
            try:
                # Get query from queue with timeout
                try:
                    query = self.query_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Send query to server
                query_json = json.dumps(query) + "\n"
                self.process.stdin.write(query_json)
                self.process.stdin.flush()
                
                # Read response
                response_line = self.process.stdout.readline().strip()
                
                try:
                    response = json.loads(response_line)
                    self.result_queue.put(response)
                except json.JSONDecodeError:
                    self.result_queue.put({"status": "error", "error": f"Invalid response: {response_line}"})
                
                self.query_queue.task_done()
                
            except Exception as e:
                self.result_queue.put({"status": "error", "error": str(e)})
    
    def search(self, query, search_type="hybrid", limit=10, **kwargs):
        """Submit a search query to the server"""
        if not self.running:
            return {"status": "error", "error": "Server not running"}
        
        # Prepare query data
        query_data = {
            "query": query,
            "search_type": search_type,
            "limit": limit,
            **kwargs
        }
        
        # Submit query
        self.query_queue.put(query_data)
        
        # Wait for result
        result = self.result_queue.get()
        self.result_queue.task_done()
        
        return result
    
    def stop(self):
        """Stop the mlxrag server"""
        self.running = False
        
        # Send exit command
        if self.process:
            try:
                self.process.stdin.write("EXIT\n")
                self.process.stdin.flush()
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                # Force kill if needed
                try:
                    self.process.kill()
                except:
                    pass
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2)


# ============= Core Benchmark Functions =============

class ResourceMonitor:
    """Monitors CPU, memory and disk usage during benchmark"""
    
    def __init__(self, interval=0.5):
        self.interval = interval
        self.monitoring = False
        self.stats = {
            'cpu': [],
            'memory': [],
            'disk': []
        }
        self.start_time = None
        self.stop_time = None
        self._monitor_thread = None
    
    def start(self):
        """Start resource monitoring"""
        if not HAS_PSUTIL:
            print("Warning: Cannot monitor resources. psutil not available.")
            return
            
        import threading
        self.monitoring = True
        self.stats = {
            'cpu': [],
            'memory': [],
            'disk': []
        }
        self.start_time = time.time()
        
        def monitor_loop():
            while self.monitoring:
                try:
                    # Get CPU usage
                    cpu_percent = psutil.cpu_percent(interval=None)
                    
                    # Get memory usage
                    memory = psutil.virtual_memory()
                    memory_used_mb = memory.used / (1024 * 1024)
                    
                    # Get disk usage for current directory
                    disk = psutil.disk_usage(os.getcwd())
                    disk_used_mb = disk.used / (1024 * 1024)
                    
                    # Store measurements
                    timestamp = time.time() - self.start_time
                    self.stats['cpu'].append((timestamp, cpu_percent))
                    self.stats['memory'].append((timestamp, memory_used_mb))
                    self.stats['disk'].append((timestamp, disk_used_mb))
                    
                    time.sleep(self.interval)
                except Exception as e:
                    print(f"Error in resource monitoring: {e}")
        
        # Start monitoring in a background thread
        self._monitor_thread = threading.Thread(target=monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop(self):
        """Stop resource monitoring"""
        if not HAS_PSUTIL or not self.monitoring:
            return
            
        self.monitoring = False
        self.stop_time = time.time()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
    
    def get_summary(self):
        """Get summary statistics of resource usage"""
        if not self.stats['cpu']:
            return {
                'cpu_avg': 0,
                'cpu_max': 0,
                'memory_avg_mb': 0,
                'memory_max_mb': 0,
                'disk_usage_mb': 0
            }
            
        # Calculate averages and maximums
        cpu_values = [x[1] for x in self.stats['cpu']]
        memory_values = [x[1] for x in self.stats['memory']]
        
        return {
            'cpu_avg': sum(cpu_values) / len(cpu_values),
            'cpu_max': max(cpu_values),
            'memory_avg_mb': sum(memory_values) / len(memory_values),
            'memory_max_mb': max(memory_values),
            'disk_usage_mb': self.stats['disk'][-1][1] if self.stats['disk'] else 0
        }
    
    def plot(self, filename):
        """Create a plot of resource usage over time"""
        if not self.stats['cpu']:
            print("No monitoring data to plot")
            return
            
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # CPU plot
        cpu_times = [x[0] for x in self.stats['cpu']]
        cpu_values = [x[1] for x in self.stats['cpu']]
        ax1.plot(cpu_times, cpu_values, 'b-', label='CPU %')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_ylim(0, max(100, max(cpu_values) * 1.1))
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Memory plot
        mem_times = [x[0] for x in self.stats['memory']]
        mem_values = [x[1] for x in self.stats['memory']]
        ax2.plot(mem_times, mem_values, 'r-', label='Memory (MB)')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


def get_files_to_process(directory, include_patterns=None, limit=None, verbose=False):
    """Get list of files to process based on patterns and limits"""
    if verbose:
        print(f"Scanning directory: {directory}")
        if include_patterns:
            print(f"Include patterns: {include_patterns}")
        if limit:
            print(f"Limit: {limit} files")
    
    files = []
    
    # Default to all supported extensions if no patterns provided
    if not include_patterns:
        include_patterns = ["*.*"]
    
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            
            # Check if file matches any pattern
            if any(fnmatch.fnmatch(filename, pattern) for pattern in include_patterns):
                files.append(file_path)
                
                if verbose and len(files) % 100 == 0:
                    print(f"Found {len(files)} matching files so far...")
                
                # Check limit
                if limit is not None and len(files) >= limit:
                    if verbose:
                        print(f"Reached limit of {limit} files")
                    return files[:limit]
    
    if verbose:
        print(f"Found {len(files)} files to process")
    
    return files


def generate_queries(num_queries=20, query_file=None, verbose=False):
    """Generate or load benchmark queries"""
    if query_file and os.path.exists(query_file):
        # Load queries from file
        if verbose:
            print(f"Loading queries from {query_file}")
        
        with open(query_file, 'r') as f:
            try:
                queries = json.load(f)
                if isinstance(queries, list):
                    if verbose:
                        print(f"Loaded {len(queries)} queries")
                    return queries[:num_queries]
                else:
                    print("Error: Query file doesn't contain a list of queries")
            except json.JSONDecodeError:
                # Try reading as text file, one query per line
                f.seek(0)
                queries = [line.strip() for line in f if line.strip()]
                if verbose:
                    print(f"Loaded {len(queries)} queries from text file")
                return queries[:num_queries]
    
    # Generate simple queries
    if verbose:
        print(f"Generating {num_queries} sample queries")
    
    # Common search terms for different domains
    query_terms = [
        # General terms
        "information", "data", "research", "analysis", "report",
        "document", "project", "summary", "overview", "review",
        # Business terms
        "business", "strategy", "management", "marketing", "finance",
        "sales", "customer", "product", "market", "revenue",
        # Technical terms
        "technology", "software", "system", "network", "database",
        "algorithm", "code", "implementation", "architecture", "design",
        # Scientific terms
        "study", "experiment", "methodology", "results", "conclusion",
        "theory", "hypothesis", "observation", "measurement", "calculation"
    ]
    
    # Generate queries with 1-3 terms
    queries = []
    for _ in range(num_queries):
        num_terms = random.randint(1, 3)
        query = " ".join(random.sample(query_terms, num_terms))
        queries.append(query)
    
    if verbose:
        print(f"Generated {len(queries)} queries")
    
    return queries


def get_system_info():
    """Get information about the system for the benchmark report"""
    info = {
        'platform': sys.platform,
        'python_version': sys.version,
    }
    
    if HAS_PSUTIL:
        # CPU info
        cpu_info = {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
        }
        
        # Try to get CPU model name on different platforms
        if sys.platform == 'linux':
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            cpu_info['model'] = line.split(':')[1].strip()
                            break
            except:
                cpu_info['model'] = 'Unknown'
        elif sys.platform == 'darwin':  # macOS
            try:
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                       capture_output=True, text=True)
                cpu_info['model'] = result.stdout.strip()
            except:
                cpu_info['model'] = 'Unknown'
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_info = {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3)
        }
        
        info['cpu'] = cpu_info
        info['memory'] = memory_info
    else:
        info['cpu'] = {'model': 'Unknown', 'physical_cores': 'Unknown', 'logical_cores': 'Unknown'}
        info['memory'] = {'total_gb': 'Unknown', 'available_gb': 'Unknown'}
    
    return info


def run_indexing_benchmark(config: BenchmarkConfig, backend_config: BackendConfig, 
                          embedding_config: EmbeddingConfig, files: List[str]):
    """Run indexing benchmark for a specific backend and embedding configuration"""
    print(f"\nRunning indexing benchmark for {backend_config.db_type} with {embedding_config.provider}...")
    
    # Prepare command line arguments for mlxrag.py
    base_args = [
        "python", "mlxrag.py", 
        "--verbose" if config.verbose else "",
        "--storage-path", backend_config.storage_path,
        "--db-type", backend_config.db_type,
    ]
    
    # Add embedding provider flags
    if embedding_config.provider == 'mlx':
        base_args.extend([
            "--use-mlx-models",
            "--dense-model", embedding_config.dense_model,
            "--sparse-model", embedding_config.sparse_model,
            "--top-k", str(embedding_config.top_k),
        ])
        if embedding_config.custom_repo_id:
            base_args.extend(["--custom-repo-id", embedding_config.custom_repo_id])
        if embedding_config.custom_ndim:
            base_args.extend(["--custom-ndim", str(embedding_config.custom_ndim)])
        base_args.extend(["--custom-pooling", embedding_config.custom_pooling])
        if not embedding_config.custom_normalize:
            base_args.append("--no-custom-normalize")
        base_args.extend(["--custom-max-length", str(embedding_config.custom_max_length)])
    elif embedding_config.provider == 'ollama':
        base_args.extend([
            "--use-ollama",
            "--ollama-model", embedding_config.model_name,
            "--ollama-host", embedding_config.ollama_host
        ])
    elif embedding_config.provider == 'fastembed':
        base_args.extend([
            "--use-fastembed",
            "--fastembed-model", embedding_config.model_name,
        ])
        if embedding_config.fastembed_sparse_model:
            base_args.extend(["--fastembed-sparse-model", embedding_config.fastembed_sparse_model])
        if embedding_config.fastembed_use_gpu:
            base_args.append("--fastembed-use-gpu")
        if embedding_config.fastembed_cache_dir:
            base_args.extend(["--fastembed-cache-dir", embedding_config.fastembed_cache_dir])
    
    # Add DB-specific arguments
    if backend_config.db_type == 'qdrant':
        base_args.extend([
            "--host", backend_config.host,
            "--port", str(backend_config.port) if backend_config.port else "6333"
        ])
    elif backend_config.db_type == 'lancedb':
        if backend_config.lancedb_uri:
            base_args.extend(["--lancedb-uri", backend_config.lancedb_uri])
    elif backend_config.db_type == 'meilisearch':
        if backend_config.meilisearch_url:
            base_args.extend(["--meilisearch-url", backend_config.meilisearch_url])
        if backend_config.meilisearch_api_key:
            base_args.extend(["--meilisearch-api-key", backend_config.meilisearch_api_key])
    elif backend_config.db_type == 'elasticsearch':
        if backend_config.es_hosts:
            hosts_arg = []
            for host in backend_config.es_hosts:
                hosts_arg.append(host)
            base_args.extend(["--es-hosts"] + hosts_arg)
        if backend_config.api_key:
            base_args.extend(["--es-api-key", backend_config.api_key])
        if backend_config.username:
            base_args.extend(["--es-username", backend_config.username])
        if backend_config.password:
            base_args.extend(["--es-password", backend_config.password])
    elif backend_config.db_type == 'milvus':
        base_args.extend([
            "--milvus-host", backend_config.host,
            "--milvus-port", str(backend_config.port) if backend_config.port else "19530"
        ])
        if backend_config.username:
            base_args.extend(["--milvus-user", backend_config.username])
        if backend_config.password:
            base_args.extend(["--milvus-password", backend_config.password])
        if backend_config.milvus_secure:
            base_args.append("--milvus-secure")
        if backend_config.milvus_token:
            base_args.extend(["--milvus-token", backend_config.milvus_token])
    elif backend_config.db_type == 'chromadb':
        if backend_config.host:
            base_args.extend(["--chromadb-host", backend_config.host])
        if backend_config.port:
            base_args.extend(["--chromadb-port", str(backend_config.port)])
        if backend_config.chromadb_use_embedding_function:
            base_args.append("--chromadb-use-embedding-function")
        if backend_config.chromadb_embedding_model:
            base_args.extend(["--chromadb-embedding-model", backend_config.chromadb_embedding_model])
    
    # For indexing, we need to specify the document directory and include patterns
    # Get the documents directory
    doc_dir = os.path.dirname(files[0]) if files else config.documents_dir
    
    # Prepare include patterns as space-separated string
    include_patterns = " ".join(config.include_patterns) if config.include_patterns else ""
    
    # Initialize indexing command
    index_cmd = base_args + [
        "index",
        doc_dir,
        "--collection", backend_config.collection_name,
        "--recreate"
    ]
    
    # Add include patterns and limit if specified
    if include_patterns:
        index_cmd.extend(["--include", include_patterns])
    if config.limit is not None:
        index_cmd.extend(["--limit", str(config.limit)])
    
    # Remove empty strings
    index_cmd = [arg for arg in index_cmd if arg]
    
    print(f"Indexing command: {' '.join(index_cmd)}")
    
    # Initialize resource monitor
    monitor = ResourceMonitor(interval=config.sampling_interval)
    
    # Run the indexing benchmark with multiple iterations
    index_times = []
    
    for i in range(config.index_iterations):
        print(f"\nIndexing iteration {i+1}/{config.index_iterations}...")
        
        # Clear storage if needed
        storage_path = os.path.expanduser(backend_config.storage_path)
        if os.path.exists(storage_path) and i == 0 and config.clean:
            print(f"Cleaning storage directory: {storage_path}")
            try:
                shutil.rmtree(storage_path)
            except Exception as e:
                print(f"Warning: Failed to clean storage directory: {e}")
        
        # Start resource monitoring
        if config.monitor_resources:
            monitor.start()
        
        # Start timing
        start_time = time.time()
        
        # Run the command
        try:
            proc = subprocess.Popen(index_cmd, 
                                   stdout=subprocess.PIPE if not config.verbose else None,
                                   stderr=subprocess.PIPE if not config.verbose else None,
                                   universal_newlines=True)
            proc.wait()
        except Exception as e:
            print(f"Error running indexing command: {e}")
            if config.monitor_resources:
                monitor.stop()
            continue
        
        # End timing
        end_time = time.time()
        index_time = end_time - start_time
        index_times.append(index_time)
        
        # Stop resource monitoring
        if config.monitor_resources:
            monitor.stop()
        
        print(f"Indexing time: {index_time:.2f} seconds")
    
    # Calculate average indexing performance
    if index_times:
        avg_index_time = sum(index_times) / len(index_times)
        throughput = len(files) / avg_index_time  # docs per second
    else:
        avg_index_time = 0
        throughput = 0
    
    # Get resource usage summary
    if config.monitor_resources:
        resource_summary = monitor.get_summary()
        
        # Plot resource usage if verbose
        if config.verbose:
            plot_path = os.path.join(config.output_dir, 
                                    f"resources_{backend_config.db_type}_{embedding_config.provider}_index.png")
            monitor.plot(plot_path)
    else:
        resource_summary = {
            'cpu_avg': 0,
            'cpu_max': 0,
            'memory_avg_mb': 0,
            'memory_max_mb': 0,
            'disk_usage_mb': 0
        }
    
    # Calculate disk usage of the storage directory
    disk_usage_mb = 0
    storage_path = os.path.expanduser(backend_config.storage_path)
    if os.path.exists(storage_path):
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(storage_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        total_size += os.path.getsize(fp)
            disk_usage_mb = total_size / (1024 * 1024)
        except Exception as e:
            print(f"Error calculating disk usage: {e}")
    
    # Return benchmark results
    return {
        'avg_index_time': avg_index_time,
        'min_index_time': min(index_times) if index_times else 0,
        'max_index_time': max(index_times) if index_times else 0,
        'throughput': throughput,
        'cpu_percent': resource_summary['cpu_avg'],
        'memory_mb': resource_summary['memory_avg_mb'],
        'disk_usage_mb': disk_usage_mb
    }


def run_search_benchmark(config: BenchmarkConfig, backend_config: BackendConfig, 
                          embedding_config: EmbeddingConfig, queries: List[str]):
    """
    Run search benchmark for a specific backend and embedding configuration
    Uses the server mode optimization when enabled
    """
    print(f"\nRunning search benchmark for {backend_config.db_type} with {embedding_config.provider}...")
    
    # Prepare base command arguments
    base_args = [
        "--storage-path", backend_config.storage_path,
        "--db-type", backend_config.db_type,
    ]
    
    # Add embedding provider flags
    if embedding_config.provider == 'mlx':
        base_args.extend([
            "--use-mlx-models",
            "--dense-model", embedding_config.dense_model,
            "--sparse-model", embedding_config.sparse_model,
            "--top-k", str(embedding_config.top_k),
        ])
        if embedding_config.custom_repo_id:
            base_args.extend(["--custom-repo-id", embedding_config.custom_repo_id])
        if embedding_config.custom_ndim:
            base_args.extend(["--custom-ndim", str(embedding_config.custom_ndim)])
        base_args.extend(["--custom-pooling", embedding_config.custom_pooling])
        if not embedding_config.custom_normalize:
            base_args.append("--no-custom-normalize")
        base_args.extend(["--custom-max-length", str(embedding_config.custom_max_length)])
    elif embedding_config.provider == 'ollama':
        base_args.extend([
            "--use-ollama",
            "--ollama-model", embedding_config.model_name,
            "--ollama-host", embedding_config.ollama_host
        ])
    elif embedding_config.provider == 'fastembed':
        base_args.extend([
            "--use-fastembed",
            "--fastembed-model", embedding_config.model_name,
        ])
        if embedding_config.fastembed_sparse_model:
            base_args.extend(["--fastembed-sparse-model", embedding_config.fastembed_sparse_model])
        if embedding_config.fastembed_use_gpu:
            base_args.append("--fastembed-use-gpu")
        if embedding_config.fastembed_cache_dir:
            base_args.extend(["--fastembed-cache-dir", embedding_config.fastembed_cache_dir])
    
    # Add DB-specific arguments (same as indexing)
    if backend_config.db_type == 'qdrant':
        base_args.extend([
            "--host", backend_config.host,
            "--port", str(backend_config.port) if backend_config.port else "6333"
        ])
    elif backend_config.db_type == 'lancedb':
        if backend_config.lancedb_uri:
            base_args.extend(["--lancedb-uri", backend_config.lancedb_uri])
    elif backend_config.db_type == 'meilisearch':
        if backend_config.meilisearch_url:
            base_args.extend(["--meilisearch-url", backend_config.meilisearch_url])
        if backend_config.meilisearch_api_key:
            base_args.extend(["--meilisearch-api-key", backend_config.meilisearch_api_key])
    elif backend_config.db_type == 'elasticsearch':
        if backend_config.es_hosts:
            hosts_arg = []
            for host in backend_config.es_hosts:
                hosts_arg.append(host)
            base_args.extend(["--es-hosts"] + hosts_arg)
        if backend_config.api_key:
            base_args.extend(["--es-api-key", backend_config.api_key])
        if backend_config.username:
            base_args.extend(["--es-username", backend_config.username])
        if backend_config.password:
            base_args.extend(["--es-password", backend_config.password])
    elif backend_config.db_type == 'milvus':
        base_args.extend([
            "--milvus-host", backend_config.host,
            "--milvus-port", str(backend_config.port) if backend_config.port else "19530"
        ])
        if backend_config.username:
            base_args.extend(["--milvus-user", backend_config.username])
        if backend_config.password:
            base_args.extend(["--milvus-password", backend_config.password])
        if backend_config.milvus_secure:
            base_args.append("--milvus-secure")
        if backend_config.milvus_token:
            base_args.extend(["--milvus-token", backend_config.milvus_token])
    elif backend_config.db_type == 'chromadb':
        if backend_config.host:
            base_args.extend(["--chromadb-host", backend_config.host])
        if backend_config.port:
            base_args.extend(["--chromadb-port", str(backend_config.port)])
        if backend_config.chromadb_use_embedding_function:
            base_args.append("--chromadb-use-embedding-function")
        if backend_config.chromadb_embedding_model:
            base_args.extend(["--chromadb-embedding-model", backend_config.chromadb_embedding_model])
    
    # Add verbose flag if needed
    if config.verbose:
        base_args.append("--verbose")
    
    # Search parameters
    search_params = {
        "collection": backend_config.collection_name,
        "limit": 10,
        "prefetch_limit": 50,
        "fusion": "rrf",
        "context_size": 300,
        "rerank": False
    }
    
    # Initialize resource monitor
    monitor = ResourceMonitor(interval=config.sampling_interval)
    
    # Store search times for each query
    all_search_times = []
    
    # If using server mode, initialize MLXrag server
    mlxrag_server = None
    if config.use_server_mode:
        mlxrag_server = MLXragServer(base_args, backend_config, config)
        server_started = mlxrag_server.start()
        if not server_started:
            print("Failed to start MLXrag server. Falling back to per-query execution.")
            config.use_server_mode = False
    
    try:
        # Run search for each query
        for i, query in enumerate(queries[:min(config.num_queries, len(queries))]):
            query_times = []
            
            print(f"Testing query {i+1}/{min(config.num_queries, len(queries))}: '{query}'")
            
            # Start resource monitoring for first iteration
            if config.monitor_resources:
                monitor.start()
            
            # Run multiple iterations of this query
            for j in range(config.search_iterations):
                # Server mode: use the server for all queries
                if config.use_server_mode and mlxrag_server and mlxrag_server.running:
                    try:
                        # Start timing
                        start_time = time.time()
                        
                        # Execute search via server
                        result = mlxrag_server.search(
                            query=query,
                            search_type="hybrid",
                            limit=search_params["limit"],
                            prefetch_limit=search_params["prefetch_limit"],
                            fusion=search_params["fusion"],
                            context_size=search_params["context_size"],
                            rerank=search_params["rerank"]
                        )
                        
                        # End timing - use the search_time from results if available
                        if result and result.get("status") == "success" and "results" in result:
                            if "search_time" in result["results"]:
                                search_time = result["results"]["search_time"]
                            else:
                                search_time = time.time() - start_time
                        else:
                            search_time = time.time() - start_time
                            print(f"Warning: Search may have failed: {result}")
                        
                        query_times.append(search_time)
                        
                    except Exception as e:
                        print(f"Error executing search via server: {e}")
                        query_times.append(0)
                
                # Traditional mode: run mlxrag.py for each query
                else:
                    # Create search command for this query
                    cmd = ["python", "mlxrag.py"] + base_args + [
                        "search",
                        "--collection", search_params["collection"],
                        "--limit", str(search_params["limit"]),
                        "--prefetch-limit", str(search_params["prefetch_limit"]),
                        "--fusion", search_params["fusion"],
                        "--context-size", str(search_params["context_size"]),
                        "--no-color",  # Disable colors for easier parsing
                        query
                    ]
                    
                    # Add rerank if needed
                    if search_params["rerank"]:
                        cmd.append("--rerank")
                    
                    # Remove empty strings
                    cmd = [str(arg) for arg in cmd if arg]
                    
                    if config.verbose:
                        print(f"Search command: {' '.join(cmd)}")
                    
                    # Start timing
                    start_time = time.time()
                    
                    # Run the command
                    try:
                        proc = subprocess.Popen(cmd, 
                                               stdout=subprocess.PIPE if not config.verbose else None,
                                               stderr=subprocess.PIPE if not config.verbose else None,
                                               universal_newlines=True)
                        proc.wait()
                    except Exception as e:
                        print(f"Error running search command: {e}")
                        query_times.append(0)
                        continue
                    
                    # End timing
                    end_time = time.time()
                    search_time = end_time - start_time
                    query_times.append(search_time)
                
                if config.verbose:
                    print(f"  Iteration {j+1}: {search_time:.4f} seconds")
            
            # Stop resource monitoring
            if config.monitor_resources:
                monitor.stop()
            
            # Calculate average search time for this query
            if query_times:
                avg_query_time = sum(query_times) / len(query_times)
                all_search_times.append({
                    'query': query,
                    'avg_time': avg_query_time,
                    'min_time': min(query_times),
                    'max_time': max(query_times),
                    'times': query_times
                })
                print(f"  Average search time: {avg_query_time:.4f} seconds")

    finally:
        # Always stop the server if it's running
        if mlxrag_server:
            mlxrag_server.stop()
    
    # Calculate overall search performance
    total_queries = len(all_search_times)
    if total_queries > 0:
        avg_times = [q['avg_time'] for q in all_search_times]
        overall_avg_time = sum(avg_times) / total_queries
        throughput = total_queries / sum(avg_times)
        
        # Calculate percentiles
        sorted_times = sorted([time for q in all_search_times for time in q['times']])
        p50_idx = int(len(sorted_times) * 0.5)
        p90_idx = int(len(sorted_times) * 0.9)
        p99_idx = int(len(sorted_times) * 0.99)
        
        p50 = sorted_times[p50_idx] if sorted_times else 0
        p90 = sorted_times[p90_idx] if p90_idx < len(sorted_times) else 0
        p99 = sorted_times[p99_idx] if p99_idx < len(sorted_times) else 0
    else:
        overall_avg_time = 0
        throughput = 0
        p50 = p90 = p99 = 0
    
    # Get resource usage summary
    if config.monitor_resources:
        resource_summary = monitor.get_summary()
        
        # Plot resource usage if verbose
        if config.verbose:
            plot_path = os.path.join(config.output_dir, 
                                    f"resources_{backend_config.db_type}_{embedding_config.provider}_search.png")
            monitor.plot(plot_path)
    else:
        resource_summary = {
            'cpu_avg': 0,
            'cpu_max': 0,
            'memory_avg_mb': 0,
            'memory_max_mb': 0,
            'disk_usage_mb': 0
        }
    
    # Return benchmark results
    return {
        'overall_avg_time': overall_avg_time,
        'throughput': throughput,
        'latency_p50': p50,
        'latency_p90': p90,
        'latency_p99': p99,
        'cpu_percent': resource_summary['cpu_avg'],
        'memory_mb': resource_summary['memory_avg_mb'],
        'query_details': all_search_times
    }


def run_benchmark_suite(config: BenchmarkConfig):
    """Run the full benchmark suite with all configured backends and embedding providers"""
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Get system information
    system_info = get_system_info()
    print("\n====== System Information ======")
    print(f"Platform: {system_info['platform']}")
    print(f"Python: {system_info['python_version']}")
    print(f"CPU: {system_info['cpu']['model']}")
    print(f"Cores: {system_info['cpu']['physical_cores']} physical, {system_info['cpu']['logical_cores']} logical")
    print(f"Memory: {system_info['memory']['total_gb']:.2f} GB total, {system_info['memory']['available_gb']:.2f} GB available")
    
    # Find files to process based on the patterns
    print("\n====== Finding Documents to Process ======")
    files = get_files_to_process(
        config.documents_dir,
        include_patterns=config.include_patterns,
        limit=config.limit,
        verbose=config.verbose
    )
    
    if not files:
        print("Error: No files found to process. Please check the documents directory and include patterns.")
        return
    
    # Generate or load queries
    print("\n====== Preparing Benchmark Queries ======")
    if config.queries:
        queries = config.queries[:config.num_queries]
        print(f"Using {len(queries)} provided queries")
    else:
        queries = generate_queries(
            num_queries=config.num_queries,
            query_file=config.query_file,
            verbose=config.verbose
        )
    
    # Configure default backends if not specified
    if not config.db_backends:
        config.db_backends = ['qdrant']
    
    if not config.embedding_backends:
        config.embedding_backends = ['mlx']
    
    # Results storage
    results = []
    
    # Timestamp for this benchmark run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run benchmarks for all combinations
    for db_type in config.db_backends:
        for embed_type in config.embedding_backends:
            print(f"\n\n====== Benchmarking {db_type} with {embed_type} ======")
            
            # Create storage directory for this combination
            storage_path = os.path.join("benchmark_storage", f"{db_type}_{embed_type}")
            os.makedirs(storage_path, exist_ok=True)
            
            # Configure backend
            backend_config = BackendConfig(
                db_type=db_type,
                storage_path=storage_path,
                collection_name="benchmark",
                host="localhost"
            )
            
            # Set appropriate ports based on DB type
            if db_type == 'qdrant':
                backend_config.port = 6333
            elif db_type == 'milvus':
                backend_config.port = 19530
            elif db_type == 'elasticsearch':
                backend_config.es_hosts = ["http://localhost:9200"]
            elif db_type == 'meilisearch':
                backend_config.meilisearch_url = "http://localhost:7700"
            
            # Configure embedding provider
            if embed_type == 'mlx':
                embedding_config = EmbeddingConfig(
                    provider='mlx',
                    model_name='none',  # Not used for MLX
                    dense_model="bge-small",
                    sparse_model="distilbert-splade"
                )
            elif embed_type == 'ollama':
                embedding_config = EmbeddingConfig(
                    provider='ollama',
                    model_name='nomic-embed-text',
                    ollama_host="http://localhost:11434"
                )
            elif embed_type == 'fastembed':
                embedding_config = EmbeddingConfig(
                    provider='fastembed',
                    model_name='BAAI/bge-small-en-v1.5',
                    fastembed_sparse_model="prithivida/Splade_PP_en_v1"
                )
            else:
                print(f"Unknown embedding type: {embed_type}")
                continue
            
            # Run indexing benchmark
            print("\n--- Running Indexing Benchmark ---")
            index_results = run_indexing_benchmark(config, backend_config, embedding_config, files)
            
            # Run search benchmark
            print("\n--- Running Search Benchmark ---")
            search_results = run_search_benchmark(config, backend_config, embedding_config, queries)
            
            # Combine results
            benchmark_result = BenchmarkResult(
                timestamp=timestamp,
                config_name=config.name,
                db_backend=db_type,
                embedding_backend=embed_type,
                cpu_info=system_info['cpu'],
                memory_info=system_info['memory'],
                document_count=len(files),
                index_time_seconds=index_results['avg_index_time'],
                index_throughput=index_results['throughput'],
                index_memory_mb=index_results['memory_mb'],
                index_cpu_percent=index_results['cpu_percent'],
                disk_usage_mb=index_results['disk_usage_mb'],
                search_time_seconds=search_results['overall_avg_time'],
                search_throughput=search_results['throughput'],
                search_memory_mb=search_results['memory_mb'],
                search_cpu_percent=search_results['cpu_percent'],
                latency_p50=search_results['latency_p50'],
                latency_p90=search_results['latency_p90'],
                latency_p99=search_results['latency_p99']
            )
            
            # Print summary
            print("\n====== Benchmark Summary ======")
            print(f"Database: {db_type}")
            print(f"Embedding: {embed_type}")
            print(f"Documents: {len(files)}")
            print("\nIndexing Performance:")
            print(f"  Average Time: {benchmark_result.index_time_seconds:.2f} seconds")
            print(f"  Throughput: {benchmark_result.index_throughput:.2f} docs/second")
            print(f"  CPU: {benchmark_result.index_cpu_percent:.1f}%")
            print(f"  Memory: {benchmark_result.index_memory_mb:.1f} MB")
            print(f"  Disk Usage: {benchmark_result.disk_usage_mb:.1f} MB")
            
            print("\nSearch Performance:")
            print(f"  Average Time: {benchmark_result.search_time_seconds:.4f} seconds")
            print(f"  Throughput: {benchmark_result.search_throughput:.2f} queries/second")
            print(f"  Latency (p50): {benchmark_result.latency_p50:.4f} seconds")
            print(f"  Latency (p90): {benchmark_result.latency_p90:.4f} seconds")
            print(f"  Latency (p99): {benchmark_result.latency_p99:.4f} seconds")
            print(f"  CPU: {benchmark_result.search_cpu_percent:.1f}%")
            print(f"  Memory: {benchmark_result.search_memory_mb:.1f} MB")
            
            # Store result
            results.append(benchmark_result)
            
            # Save detailed results
            detailed_results = {
                'summary': asdict(benchmark_result),
                'indexing': index_results,
                'search': search_results,
                'system': system_info,
                'config': {
                    'benchmark': {k: v for k, v in vars(config).items() if not k.startswith('_')},
                    'backend': {k: v for k, v in vars(backend_config).items() if not k.startswith('_')},
                    'embedding': {k: v for k, v in vars(embedding_config).items() if not k.startswith('_')}
                },
                'files': {
                    'count': len(files),
                    'patterns': config.include_patterns
                },
                'queries': queries
            }
            
            # Save to JSON file
            result_path = os.path.join(config.output_dir, 
                                     f"results_{db_type}_{embed_type}_{timestamp}.json")
            with open(result_path, 'w') as f:
                json.dump(detailed_results, f, indent=2, default=str)
            
            print(f"\nDetailed results saved to: {result_path}")
    
    # Generate comparative report
    if len(results) > 1:
        generate_benchmark_report(config, results, timestamp)


def generate_benchmark_report(config: BenchmarkConfig, results: List[BenchmarkResult], timestamp: str):
    """Generate a comparative report of all benchmark results"""
    print("\n====== Generating Benchmark Report ======")
    
    # Create pandas DataFrame for easier analysis
    df = pd.DataFrame([asdict(r) for r in results])
    
    # Generate comparison tables
    indexing_df = df[['db_backend', 'embedding_backend', 'index_time_seconds', 'index_throughput', 
                      'index_cpu_percent', 'index_memory_mb', 'disk_usage_mb']]
    indexing_df = indexing_df.sort_values('index_throughput', ascending=False)
    
    search_df = df[['db_backend', 'embedding_backend', 'search_time_seconds', 'search_throughput',
                   'latency_p50', 'latency_p90', 'search_cpu_percent', 'search_memory_mb']]
    search_df = search_df.sort_values('search_throughput', ascending=False)
    
    # Create directory for report
    report_dir = os.path.join(config.output_dir, f"report_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)
    
    # Save DataFrames to CSV
    indexing_df.to_csv(os.path.join(report_dir, "indexing_comparison.csv"), index=False)
    search_df.to_csv(os.path.join(report_dir, "search_comparison.csv"), index=False)
    
    # Generate plots
    plt.figure(figsize=(12, 6))
    
    # Indexing throughput comparison
    plt.subplot(1, 2, 1)
    ax = indexing_df.plot.bar(x='db_backend', y='index_throughput', color='skyblue', 
                            legend=False, ax=plt.gca())
    plt.title('Indexing Throughput Comparison')
    plt.ylabel('Documents per second')
    plt.xlabel('Database Backend')
    plt.xticks(rotation=45)
    
    # Add embedding labels
    for i, row in enumerate(indexing_df.itertuples()):
        plt.annotate(row.embedding_backend, 
                   (i, row.index_throughput + 0.5),
                   ha='center')
    
    # Search throughput comparison
    plt.subplot(1, 2, 2)
    search_df.plot.bar(x='db_backend', y='search_throughput', color='lightgreen', 
                     legend=False, ax=plt.gca())
    plt.title('Search Throughput Comparison')
    plt.ylabel('Queries per second')
    plt.xlabel('Database Backend')
    plt.xticks(rotation=45)
    
    # Add embedding labels
    for i, row in enumerate(search_df.itertuples()):
        plt.annotate(row.embedding_backend, 
                   (i, row.search_throughput + 0.05),
                   ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "throughput_comparison.png"))
    
    # Generate latency comparison
    plt.figure(figsize=(10, 6))
    
    # Prepare data for grouped bar chart
    db_embedding_pairs = [f"{row.db_backend}\n{row.embedding_backend}" for row in search_df.itertuples()]
    p50_values = search_df['latency_p50'].values
    p90_values = search_df['latency_p90'].values
    
    x = np.arange(len(db_embedding_pairs))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, p50_values, width, label='p50 Latency')
    ax.bar(x + width/2, p90_values, width, label='p90 Latency')
    
    ax.set_ylabel('Latency (seconds)')
    ax.set_title('Search Latency Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(db_embedding_pairs)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "latency_comparison.png"))
    
    # Memory usage comparison
    plt.figure(figsize=(12, 6))
    
    # Indexing memory
    plt.subplot(1, 2, 1)
    indexing_df.plot.bar(x='db_backend', y='index_memory_mb', color='coral', 
                        legend=False, ax=plt.gca())
    plt.title('Indexing Memory Usage')
    plt.ylabel('Memory (MB)')
    plt.xlabel('Database Backend')
    plt.xticks(rotation=45)
    
    # Search memory
    plt.subplot(1, 2, 2)
    search_df.plot.bar(x='db_backend', y='search_memory_mb', color='mediumpurple', 
                     legend=False, ax=plt.gca())
    plt.title('Search Memory Usage')
    plt.ylabel('Memory (MB)')
    plt.xlabel('Database Backend')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "memory_comparison.png"))
    
    # Disk usage comparison
    plt.figure(figsize=(10, 6))
    indexing_df.plot.bar(x='db_backend', y='disk_usage_mb', color='teal', 
                        legend=False)
    plt.title('Disk Usage Comparison')
    plt.ylabel('Disk Usage (MB)')
    plt.xlabel('Database Backend')
    plt.xticks(rotation=45)
    
    # Add embedding labels
    for i, row in enumerate(indexing_df.itertuples()):
        plt.annotate(row.embedding_backend, 
                   (i, row.disk_usage_mb + max(indexing_df['disk_usage_mb'])*0.05),
                   ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "disk_usage_comparison.png"))
    
    # Get hardware info for the report
    hw_info = results[0].cpu_info if results else {"model": "Unknown", "physical_cores": "Unknown"}
    doc_count = results[0].document_count if results else 0
    
    # Generate HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Benchmark Report - {timestamp}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .chart {{ margin: 20px 0; max-width: 100%; }}
            .highlight {{ background-color: #e8f4f8; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>RAG Benchmark Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Benchmark: {config.name}</p>
        
        <h2>System Information</h2>
        <table>
            <tr><th>Component</th><th>Details</th></tr>
            <tr><td>Platform</td><td>{sys.platform}</td></tr>
            <tr><td>Python</td><td>{sys.version.split()[0]}</td></tr>
            <tr><td>CPU</td><td>{hw_info.get('model', 'Unknown')}</td></tr>
            <tr><td>CPU Cores</td><td>{hw_info.get('physical_cores', 'Unknown')} physical, {hw_info.get('logical_cores', 'Unknown')} logical</td></tr>
        </table>
        
        <h2>Benchmark Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Documents Directory</td><td>{config.documents_dir}</td></tr>
            <tr><td>Document Count</td><td>{doc_count}</td></tr>
            <tr><td>Include Patterns</td><td>{', '.join(config.include_patterns) if config.include_patterns else 'All files'}</td></tr>
            <tr><td>Number of Queries</td><td>{config.num_queries}</td></tr>
            <tr><td>Database Backends</td><td>{', '.join(config.db_backends)}</td></tr>
            <tr><td>Embedding Backends</td><td>{', '.join(config.embedding_backends)}</td></tr>
            <tr><td>Index Iterations</td><td>{config.index_iterations}</td></tr>
            <tr><td>Search Iterations</td><td>{config.search_iterations}</td></tr>
            <tr><td>Server Mode</td><td>{'Enabled' if config.use_server_mode else 'Disabled'}</td></tr>
        </table>
        
        <h2>Indexing Performance</h2>
        <div class="chart">
            <img src="throughput_comparison.png" alt="Throughput Comparison" style="max-width:100%;">
        </div>
        
        <table>
            <tr>
                <th>Database</th>
                <th>Embedding</th>
                <th>Time (s)</th>
                <th>Throughput (docs/s)</th>
                <th>CPU (%)</th>
                <th>Memory (MB)</th>
                <th>Disk (MB)</th>
            </tr>
    """
    
    # Add indexing rows
    for _, row in indexing_df.iterrows():
        html_report += f"""
            <tr>
                <td>{row['db_backend']}</td>
                <td>{row['embedding_backend']}</td>
                <td>{row['index_time_seconds']:.2f}</td>
                <td class="highlight">{row['index_throughput']:.2f}</td>
                <td>{row['index_cpu_percent']:.1f}</td>
                <td>{row['index_memory_mb']:.1f}</td>
                <td>{row['disk_usage_mb']:.1f}</td>
            </tr>
        """
    
    html_report += """
        </table>
        
        <h2>Search Performance</h2>
        <div class="chart">
            <img src="latency_comparison.png" alt="Latency Comparison" style="max-width:100%;">
        </div>
        
        <table>
            <tr>
                <th>Database</th>
                <th>Embedding</th>
                <th>Time (s)</th>
                <th>Throughput (q/s)</th>
                <th>P50 Latency (s)</th>
                <th>P90 Latency (s)</th>
                <th>CPU (%)</th>
                <th>Memory (MB)</th>
            </tr>
    """
    
    # Add search rows
    for _, row in search_df.iterrows():
        html_report += f"""
            <tr>
                <td>{row['db_backend']}</td>
                <td>{row['embedding_backend']}</td>
                <td>{row['search_time_seconds']:.4f}</td>
                <td class="highlight">{row['search_throughput']:.2f}</td>
                <td>{row['latency_p50']:.4f}</td>
                <td>{row['latency_p90']:.4f}</td>
                <td>{row['search_cpu_percent']:.1f}</td>
                <td>{row['search_memory_mb']:.1f}</td>
            </tr>
        """
    
    html_report += """
        </table>
        
        <h2>Resource Usage Comparison</h2>
        <div class="chart">
            <img src="memory_comparison.png" alt="Memory Usage Comparison" style="max-width:100%;">
        </div>
        <div class="chart">
            <img src="disk_usage_comparison.png" alt="Disk Usage Comparison" style="max-width:100%;">
        </div>
        
        <h2>Conclusions</h2>
        <p>Based on the benchmark results, here are the key findings:</p>
        <ul>
    """
    
    # Generate some basic conclusions
    if len(indexing_df) > 0:
        best_indexing_db = indexing_df.iloc[0]['db_backend']
        best_indexing_embed = indexing_df.iloc[0]['embedding_backend']
        best_indexing_throughput = indexing_df.iloc[0]['index_throughput']
        
        html_report += f"""
            <li>Best indexing performance: <strong>{best_indexing_db}</strong> with <strong>{best_indexing_embed}</strong> 
                ({best_indexing_throughput:.2f} docs/second)</li>
        """
    
    if len(search_df) > 0:
        best_search_db = search_df.iloc[0]['db_backend']
        best_search_embed = search_df.iloc[0]['embedding_backend']
        best_search_throughput = search_df.iloc[0]['search_throughput']
        
        html_report += f"""
            <li>Best search performance: <strong>{best_search_db}</strong> with <strong>{best_search_embed}</strong>
                ({best_search_throughput:.2f} queries/second)</li>
        """
    
    # Find lowest memory usage
    if len(search_df) > 0:
        lowest_memory_idx = search_df['search_memory_mb'].idxmin()
        lowest_memory_db = search_df.loc[lowest_memory_idx, 'db_backend']
        lowest_memory_embed = search_df.loc[lowest_memory_idx, 'embedding_backend']
        lowest_memory = search_df.loc[lowest_memory_idx, 'search_memory_mb']
        
        html_report += f"""
            <li>Lowest memory usage: <strong>{lowest_memory_db}</strong> with <strong>{lowest_memory_embed}</strong>
                ({lowest_memory:.1f} MB)</li>
        """
    
    # Find lowest disk usage
    if len(indexing_df) > 0:
        lowest_disk_idx = indexing_df['disk_usage_mb'].idxmin()
        lowest_disk_db = indexing_df.loc[lowest_disk_idx, 'db_backend']
        lowest_disk_embed = indexing_df.loc[lowest_disk_idx, 'embedding_backend']
        lowest_disk = indexing_df.loc[lowest_disk_idx, 'disk_usage_mb']
        
        html_report += f"""
            <li>Smallest storage footprint: <strong>{lowest_disk_db}</strong> with <strong>{lowest_disk_embed}</strong>
                ({lowest_disk:.1f} MB)</li>
        """
    
    html_report += """
        </ul>
        
        <p><em>Note: These benchmarks were run in a controlled environment. Your actual performance may vary based on 
        hardware, concurrent workloads, and data characteristics.</em></p>
        
        <footer>
            <p>Generated by Adapted RAG Benchmark Tool</p>
        </footer>
    </body>
    </html>
    """
    
    # Write HTML report
    with open(os.path.join(report_dir, "benchmark_report.html"), 'w') as f:
        f.write(html_report)
    
    print(f"Comparative report generated at: {os.path.join(report_dir, 'benchmark_report.html')}")


def main():
    """Main entry point for the benchmark script"""
    parser = argparse.ArgumentParser(description="Adapted RAG System Benchmark Tool")
    
    # General options
    parser.add_argument("--name", type=str, default="default", help="Benchmark name")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing benchmark results")
    parser.add_argument("--clean", action="store_true", help="Clean storage directories before indexing")
    
    # Document options
    parser.add_argument("--documents", type=str, help="Path to directory containing documents to benchmark")
    parser.add_argument("--include", type=str, help="File patterns to include (space separated, e.g. '*.txt *.pdf')")
    parser.add_argument("--limit", type=int, help="Maximum number of files to process")
    
    # Query options
    parser.add_argument("--query-file", type=str, help="Path to file containing benchmark queries")
    parser.add_argument("--num-queries", type=int, default=20, help="Number of queries to test")
    parser.add_argument("--queries", type=str, nargs="+", help="Specific queries to test")
    
    # Backend selection
    parser.add_argument("--db-backends", type=str, nargs="+", default=["qdrant"], 
                       help="Database backends to test (e.g. qdrant lancedb milvus)")
    parser.add_argument("--embedding-backends", type=str, nargs="+", default=["mlx"], 
                       help="Embedding backends to test (e.g. mlx ollama fastembed)")
    
    # Benchmark parameters
    parser.add_argument("--index-iterations", type=int, default=3, help="Number of indexing iterations")
    parser.add_argument("--search-iterations", type=int, default=5, help="Number of search iterations per query")
    
    # Resource monitoring
    parser.add_argument("--no-monitor", action="store_true", help="Disable resource monitoring")
    parser.add_argument("--sampling-interval", type=float, default=0.5, help="Resource sampling interval in seconds")
    
    # Optimization
    parser.add_argument("--no-server-mode", action="store_true", help="Disable server mode for search queries")
    
    args = parser.parse_args()
    
    # Ensure documents directory is provided
    if not args.documents:
        print("Error: Please specify a documents directory with --documents")
        parser.print_help()
        return
    
    # Parse include patterns
    include_patterns = args.include.split() if args.include else ["*.*"]
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        name=args.name,
        output_dir=args.output_dir,
        verbose=args.verbose,
        skip_existing=args.skip_existing,
        clean=args.clean,
        documents_dir=args.documents,
        include_patterns=include_patterns,
        limit=args.limit,
        queries=args.queries,
        query_file=args.query_file,
        num_queries=args.num_queries,
        db_backends=args.db_backends,
        embedding_backends=args.embedding_backends,
        index_iterations=args.index_iterations,
        search_iterations=args.search_iterations,
        monitor_resources=not args.no_monitor,
        sampling_interval=args.sampling_interval,
        use_server_mode=not args.no_server_mode
    )
    
    # Run the benchmark suite
    run_benchmark_suite(config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()