# RAG System Benchmark Tool

A comprehensive benchmark tool for evaluating the performance of different vector databases and embedding providers in the CrispRAG system.

## Overview

This benchmark tool allows you to test and compare various combinations of:

1. **Vector Database Backends**:
   - Qdrant
   - LanceDB
   - Meilisearch
   - Elasticsearch
   - ChromaDB
   - Milvus

2. **Embedding Providers**:
   - MLX Embedding Models
   - Ollama
   - FastEmbed

3. **Performance Metrics**:
   - Indexing throughput
   - Search latency (p50, p90, p99)
   - Memory usage
   - CPU utilization
   - Disk usage

## Features

- **Automated corpus generation** with configurable document size
- **Resource monitoring** to track CPU, memory, and disk usage
- **Consistent methodology** for fair comparison across backends
- **Comprehensive reporting** with charts and HTML output
- **Flexible configuration** to test various parameters

## Installation

First, clone the repository and install the required dependencies:

```bash
git clone [repository-url]
cd [repository-directory]
pip install -r requirements.txt
```

Make sure you've already installed the necessary dependencies for the database backends you want to test:

- Qdrant: `pip install qdrant-client`
- LanceDB: `pip install lancedb pyarrow`
- Meilisearch: `pip install meilisearch`
- Elasticsearch: `pip install elasticsearch`
- ChromaDB: `pip install chromadb`
- Milvus: `pip install pymilvus`

For embedding providers:
- MLX Embedding Models: `pip install mlx-embedding-models`
- Ollama: Install from [ollama.ai](https://ollama.ai)
- FastEmbed: `pip install fastembed` (or `pip install fastembed-gpu` for GPU support)

## Usage

### Basic Usage

```bash
python rag_benchmark.py --db-backends qdrant lancedb --embedding-backends mlx fastembed
```

This will run a benchmark comparing Qdrant and LanceDB with both MLX and FastEmbed embedding providers.

### Customizing the Benchmark

```bash
python rag_benchmark.py \
  --name "production-comparison" \
  --num-documents 1000 \
  --doc-size-min 1000 \
  --doc-size-max 10000 \
  --num-queries 50 \
  --db-backends qdrant milvus elasticsearch \
  --embedding-backends mlx ollama \
  --index-iterations 5 \
  --search-iterations 10 \
  --verbose
```

### Full Options List

```
General options:
  --name NAME               Benchmark name
  --output-dir OUTPUT_DIR   Output directory for results
  --verbose                 Enable verbose output
  --skip-existing           Skip existing benchmark results
  --clean                   Clean storage directories before indexing

Data generation options:
  --num-documents NUM       Number of documents to generate
  --doc-size-min MIN        Minimum document size in words
  --doc-size-max MAX        Maximum document size in words
  --corpus-dir DIR          Directory for benchmark corpus
  --num-queries NUM         Number of queries to test

Backend selection:
  --db-backends DB [DB...]  Database backends to test
  --embedding-backends EMB [EMB...]  Embedding backends to test

Benchmark parameters:
  --index-iterations NUM    Number of indexing iterations
  --search-iterations NUM   Number of search iterations per query

Resource monitoring:
  --no-monitor              Disable resource monitoring
  --sampling-interval SEC   Resource sampling interval in seconds
```

## Benchmark Process

The benchmark process consists of the following steps:

1. **Setup Phase**: 
   - Generate corpus documents (or use existing if available)
   - Generate search queries
   - Check for required dependencies

2. **Indexing Phase**:
   - For each (DB, Embedding) combination:
     - Set up database connection
     - Process and index all documents
     - Measure indexing time and resource usage

3. **Search Phase**:
   - For each (DB, Embedding) combination:
     - Run search queries multiple times
     - Measure latency, throughput, and resource usage

4. **Reporting Phase**:
   - Generate comparative charts
   - Create detailed HTML report
   - Save raw data in JSON format

## Understanding the Results

The benchmark generates several outputs:

1. **JSON Files**: Raw data for each benchmark run
2. **CSV Files**: Tabular data for comparing metrics
3. **PNG Charts**: Visual comparisons of performance metrics
4. **HTML Report**: Comprehensive report with all results

Key metrics to examine:

- **Indexing Throughput**: Higher is better (documents per second)
- **Search Latency**: Lower is better (seconds per query)
- **Resource Usage**: Lower is better (particularly important for production deployments)

## Example Report Output

The HTML report includes:
- System information
- Benchmark configuration
- Comparison tables for all metrics
- Visual charts for easier comparison
- Conclusions highlighting the best performers

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   - Ensure all required Python packages are installed
   - Make sure database services are running if testing remote connections

2. **Resource Limitations**:
   - Reduce document count if running on limited hardware
   - Disable resource monitoring if causing performance issues

3. **Database Connection Errors**:
   - Check host/port settings for remote databases
   - Verify credentials if authentication is required

### Getting Help

If you encounter issues, please:
1. Check the error message for clues
2. Review your database documentation for specific requirements
3. Open an issue in the repository with detailed information

## Extending the Benchmark

To add support for a new database or embedding provider:

1. **For a new DB backend**: 
   - Add appropriate command-line arguments to `BackendConfig`
   - Update `run_indexing_benchmark()` and `run_search_benchmark()` to handle the new backend

2. **For a new embedding provider**:
   - Add appropriate configuration in `EmbeddingConfig`
   - Update command line generation in benchmark functions

## License

MIT