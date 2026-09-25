# Parallax AI Internship — Week 2

## Chunking, Embeddings & Vector Database

This week extends the Week 1 preprocessing pipeline by implementing text chunking, embedding generation, vector database storage, semantic search, and retrieval performance benchmarking.

## Overview

The cleaned Wikipedia corpus produced in Week 1 is used as the input for the Week 2 pipeline.

The Week 2 workflow is:

```text
Cleaned Documents
        ↓
Text Chunking
        ↓
Sentence Embeddings
        ↓
ChromaDB Vector Database
        ↓
Semantic Search
        ↓
Retrieval Benchmarking
```

## Dataset

The Week 1 preprocessing pipeline produced:

- 5,998 cleaned Wikipedia documents
- Stored in:
  `data/processed/clean_corpus.parquet`

The cleaned dataset contains the following columns:

- `id`
- `title`
- `url`
- `clean_text`

## Text Chunking

A custom overlapping chunking strategy was implemented in:

```text
src/chunking.py
```

Configuration:

```text
Chunk size: 500 characters
Overlap: 100 characters
```

The overlap helps preserve context when information appears near the boundary between two chunks.

The chunking process generated:

```text
Original documents: 5,998
Total chunks: 186,349
Average chunks per document: 31.07
```

The generated chunks are stored in:

```text
data/processed/chunks.parquet
```

### Chunking Tests

Chunking behavior was tested using pytest.

The tests verify:

- Empty text handling
- Short documents
- Multi-chunk documents
- Correct overlap
- Invalid chunk sizes
- Invalid overlap values

Run:

```bash
python -m pytest tests/test_chunking.py -v
```

Result:

```text
6 passed
```

## Embedding Generation

Embeddings are generated using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Each text chunk is converted into a 384-dimensional embedding vector.

Because embedding the complete 186,349-chunk corpus on CPU would require significant processing time, 20,000 chunks were selected for the Week 2 vector database implementation.

Embedding configuration:

```text
Selected chunks: 20,000
Embedding dimension: 384
Batch size: 64
```

Measured embedding performance:

```text
Total embedding time: 438.35 seconds
Average embedding time per chunk: 21.9174 ms
Embedding throughput: 45.63 chunks/second
```

Embeddings are stored in:

```text
data/processed/embeddings.npy
```

The corresponding chunks are stored in:

```text
data/processed/embedded_chunks.parquet
```

Run embedding generation using:

```bash
python -m scripts.generate_embeddings
```

## ChromaDB

ChromaDB is used as the persistent vector database.

The database contains:

```text
20,000 embedded Wikipedia chunks
```

The persistent database is stored in:

```text
data/chroma_db
```

The ChromaDB collection name is:

```text
wikipedia_chunks
```

To populate the database, run:

```bash
python -m scripts.ingest_chroma
```

## Semantic Search

Semantic search was implemented using the same Sentence Transformer model used to generate the stored embeddings.

A user query is converted into an embedding and compared against the vectors stored in ChromaDB.

Run:

```bash
python -m scripts.search_chroma
```

Example query:

```text
What is artificial intelligence?
```

Example top result:

```text
Artificial intelligence (AI) is the intelligence of machines or software,
as opposed to the intelligence of humans or animals...
```

This demonstrates that the system can retrieve text based on semantic similarity rather than relying only on exact keyword matching.

## Retrieval Performance

A benchmark script was created to test several queries and measure retrieval latency.

Run:

```bash
python -m scripts.benchmark_retrieval
```

Queries tested included:

- What is artificial intelligence?
- How does machine learning work?
- What causes climate change?
- Who was Albert Einstein?
- What is the history of the internet?

Measured performance:

```text
Queries tested: 5
Average latency: 58.27 ms
Minimum latency: 26.99 ms
Maximum latency: 137.60 ms
```

Benchmark results are saved in:

```text
results/retrieval_benchmark.csv
```

## ChromaDB Edge Cases

Several database edge cases were tested.

The tests cover:

- Empty collections
- Adding documents successfully
- Duplicate ID handling using `upsert`
- Reading an empty collection safely

Run:

```bash
python -m pytest tests/test_chroma_edge_cases.py -v
```

Result:

```text
4 passed
```

## Project Structure

```text
parallax-ai-internship/
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── clean_corpus.parquet
│   │   ├── chunks.parquet
│   │   ├── embedded_chunks.parquet
│   │   └── embeddings.npy
│   └── chroma_db/
│
├── scripts/
│   ├── verify_env.py
│   ├── create_chunks.py
│   ├── generate_embeddings.py
│   ├── ingest_chroma.py
│   ├── search_chroma.py
│   └── benchmark_retrieval.py
│
├── src/
│   ├── preprocessing.py
│   └── chunking.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_chunking.py
│   └── test_chroma_edge_cases.py
│
├── results/
│   └── retrieval_benchmark.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Running the Project

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Create chunks:

```bash
python -m scripts.create_chunks
```

Generate embeddings:

```bash
python -m scripts.generate_embeddings
```

Populate ChromaDB:

```bash
python -m scripts.ingest_chroma
```

Run semantic search:

```bash
python -m scripts.search_chroma
```

Run the retrieval benchmark:

```bash
python -m scripts.benchmark_retrieval
```

Run tests:

```bash
python -m pytest -v
```

## Notes and Limitations

The complete cleaned corpus generated 186,349 chunks.

For this Week 2 implementation, 20,000 chunks were embedded and indexed due to CPU processing constraints.

As a result, semantic retrieval works well for topics present in the indexed subset, while some queries may return weaker results if the most relevant document was not included in the selected 20,000 chunks.

A future improvement would be to embed the complete corpus using GPU acceleration or multiprocessing.

## Week 2 Deliverable

Week 2 successfully produced:

- A tested text chunking pipeline
- Sentence Transformer embeddings
- Embedding performance measurements
- A populated persistent ChromaDB database
- Semantic search
- Retrieval latency benchmarking
- ChromaDB edge-case handling