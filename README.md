# Optimizating_Pipelines


# Embedding Pipeline Optimization

This project explores optimizations in embedding-based retrieval systems, focusing on reducing model load time, improving retrieval efficiency, and evaluating model–retriever combinations under real-world constraints.

## Motivation

In many NLP pipelines—such as entity resolution, search, or deduplication—off-the-shelf embeddings are used without considering performance trade-offs. This project began with the goal of understanding:

- How to reduce model loading time and memory usage
- What retrieval strategies scale well for large datasets
- Which embedding–retrieval combinations perform best on noisy, real-world tabular text

## Project Goals

- Design a modular framework for testing multiple embedding models and retrieval strategies
- Implement scalable retrieval methods beyond basic cosine similarity (e.g., sparse search, LSH, fuzzy match)
- Benchmark models and strategies across accuracy, speed, and memory efficiency
- Apply these methods to messy, realistic datasets (e.g., column-level text or entity names)

## Implementation Overview

- **Embedding Management**  
  Implemented lazy initialization, caching, and memory-mapped storage for large embedding matrices.

- **Retrieval Techniques**  
  - Cosine similarity (dense)
  - Sparse similarity using `sparse-dot-topn` and `csr_matrix`
  - Locality-Sensitive Hashing (MinHash + LSH Forest)
  - Fuzzy string matching (Levenshtein-based)
  - Combined scoring strategy using weighted or normalized hybrid matchers

- **Benchmarking & Monitoring**  
  Developed a performance monitor to track runtime and memory usage across loading, encoding, and retrieval phases.  
  Benchmarked 10+ models (e.g., MiniLM, GTE, E5) on real and synthetic data using Top‑1 accuracy, MRR, and latency.

## Evaluation Setup

- **Datasets:**  
  Used datasets from the [DeepMatcher (SIGMOD 2018)](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md) paper and generated synthetic noisy versions.

- **Tasks:**  
  Evaluate embedding–retriever combinations for exact match recovery and top-k accuracy on entity-style string columns.

- **Metrics:**  
  - Top-1 Accuracy  
  - Mean Reciprocal Rank (MRR)  
  - Retrieval Time (ms)  
  - Peak Memory Usage (MB)

## Results

- Achieved **reduction in model load time** using caching and lazy initialization
- Enabled retrieval on 10k+ rows with minimal memory using sparse similarity and memory mapping
- Identified trade-offs:
  - Dense cosine: higher accuracy, slower
  - Sparse retrieval: fast and lightweight, slightly less accurate
  - Fuzzy: helpful for typos, not scalable
  - Combined matcher: strong performance balance across metrics


