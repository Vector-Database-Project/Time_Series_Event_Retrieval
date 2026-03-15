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

- **Unsupervised embeddings**
- **Supervised embeddings**
- **Non-learning baselines**

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

## Project Environment

The project environment setup is provided in the `environment/` folder through the `env.yaml` file.

For setup instructions and package details, refer to:

- [`environment/env_setup.md`](environment/env_setup.md)

## Data Processing

The data preprocessing pipeline and dataset standardization steps are documented in:

- [`data/data_processing_pipeline.md`](data/data_processing_pipeline.md)

This document outlines the preprocessing workflow, storage format, metadata structure, and split generation strategy used to prepare raw time-series datasets for training and retrieval experiments.

## Embeddings

The embedding pipeline and retrieval setup are documented in:

- [`src/embeddings/embedding_pipeline.md`](src/embeddings/embedding_pipeline.md)

This document outlines the embedding methods, input modes, run structure, vector storage, and retrieval evaluation workflow used in the ML pipeline.