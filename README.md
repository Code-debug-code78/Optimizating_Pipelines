# Optimizating_Pipelines


# Embedding Pipeline Optimization

This project explores optimizations in embedding-based retrieval systems, focusing on reducing model load time, improving retrieval efficiency, and evaluating model–retriever combinations under real-world constraints.

## Motivation

In many NLP pipelines—such as entity resolution, search, or deduplication—off-the-shelf embeddings are used without considering performance trade-offs. This project began with the goal of understanding:

- How to reduce model loading time and memory usage
- What retrieval strategies scale well for large datasets
- How different models perform on noisy dataset of [DeepMatcher (SIGMOD 2018)](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md) 

## Project Goals

- Design a modular framework for testing multiple embedding models and retrieval strategies
- Implement scalable retrieval methods beyond basic cosine similarity (e.g., sparse search, LSH, fuzzy match)
- Benchmark models and strategies across accuracy, speed, and memory efficiency
- Apply these methods to messy, realistic datasets (e.g., column-level text or entity names)

## Implementation Overview

- **Embedding Management**  
  Implemented lazy initialization, caching, and memory-mapped storage for large embedding matrices.
  For this I went through [fastembed](https://qdrant.github.io/fastembed/) documentation. It is a lightweight, fast, Python library built for embedding generation.
  Following are the features of fastembed: 
  1. Light & Fast
     - Quantized model weights
     - ONNX Runtime for inference
  2.  Accuracy/Recall
     - Better than OpenAI Ada-002
     - Default is Flag Embedding, which has shown good results on the MTEB leaderboard
     - List of supported models - including multilingual models
A normal working of how fastembed works is as follows: [fastembed_check](https://github.com/Code-debug-code78/Optimizating_Pipelines/blob/main/fastembedcheck.ipynb) 
Then I ran some of the models which were supported by fastembed and compared those results: [here](https://github.com/Code-debug-code78/Optimizating_Pipelines/blob/main/fastembed%20(2).ipynb)

- **Retrieval Techniques**  
  - Cosine similarity (dense)
  - Sparse similarity using `sparse-dot-topn` and `csr_matrix`
  - Locality-Sensitive Hashing (MinHash + LSH Forest)
  - Fuzzy string matching (Levenshtein-based)
  - Combined scoring strategy using weighted or normalized hybrid matchers
The comparison can be referred [here](https://github.com/Code-debug-code78/Optimizating_Pipelines/blob/main/Combined%20strategies%20(5).ipynb)

- **Benchmarking & Monitoring**
  Next, I was comparing how different models give varied results for different types of noisy datasets. Doing this, help us know how the models can be categoried into different types based on their working.
  Developed a performance monitor to track runtime and memory usage across loading, encoding, and retrieval phases.  
  Benchmarked 10+ models (e.g., MiniLM, GTE, E5) on real and synthetic data using Top‑1 accuracy, MRR, and latency.
  For this I worked on the files : 
 [benchmark_runner.py](https://github.com/Code-debug-code78/Optimizating_Pipelines/blob/main/benchmark_runner.py)
 [scoring.py](https://github.com/Code-debug-code78/Optimizating_Pipelines/blob/main/scoring.py)
 [plot_results.py](https://github.com/Code-debug-code78/Optimizating_Pipelines/blob/main/plot_results.py)
  

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
- Optimized embedding pipelines for 10-fold faster model loading, enabling scalable, low-latency retrieval using fastembed
- Used LSH, memory-mapped vectors, and sparse similarity search for efficient embedding-based retrieval
- Benchmarked embeddings for 10+ models on noisy tabular datasets from DeepMatcher (SIGMOD’18) paper
- Identified trade-offs:
  - Dense cosine: higher accuracy, slower
  - Sparse retrieval: fast and lightweight, slightly less accurate
  - Fuzzy: helpful for typos, not scalable
  - Combined matcher: strong performance balance across metrics


