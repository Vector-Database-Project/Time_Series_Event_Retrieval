# CS-7641-Semester-Project

# Time-Series Event Retrieval with Vector Databases

## Overview
This project explores how different time-series representations and machine learning embedding methods affect the ability to retrieve similar historical events from large datasets. The goal is to build a reproducible framework that compares multiple encoding strategies and embedding approaches across different domains.

We focus on evaluating how well different pipelines perform for similarity search in time-series data when used with vector retrieval systems.

## Project Goals
- Standardize multiple time-series datasets into a common format
- Generate different signal representations (time, frequency, and time-frequency)
- Compare multiple embedding approaches
- Evaluate retrieval performance using vector similarity search
- Identify configurations that generalize well across domains

## Pipeline
The project follows three main stages:

### 1. Data Standardization
- Collect time-series datasets from multiple domains
- Clean and preprocess the data
- Split signals into fixed-length windows
- Create consistent metadata structures
- Apply normalization

### 2. Representation and Embeddings
We compare three types of signal representations:

- **Time-domain representation**
- **Frequency-domain representation (FFT / STFT)**
- **Mixed time-frequency representation**

We evaluate three embedding approaches:

- **Self-supervised embeddings** (e.g., TS2Vec)
- **Supervised embeddings** (e.g., InceptionTime)
- **Non-learning baselines** (e.g., tsfresh features or DTW)

### 3. Retrieval Evaluation
- Store embeddings in a vector search system
- Perform similarity queries
- Evaluate retrieval performance across datasets
- Compare results across representation and embedding combinations

## Project Structure (Planned)

```
data/
datasets/
src/
preprocessing/
representations/
embeddings/
retrieval/
experiments/
notebooks/
```

## Team
This project is being developed as part of a Machine Learning course project.

## Status
🚧 Work in progress — initial repository setup and pipeline design.
