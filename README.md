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


## MIT-BIH ECG chunking policy

### Time-domain preprocessing note

For the MIT-BIH preprocessing stage, beats are extracted using a fixed **beat-centered symmetric window** with:

- `pre_samples = 180`
- `post_samples = 180`

MIT-BIH signals are digitized at **360 Hz per channel**, and the beat annotations were realigned so that they **generally appear at the R-wave peak**. This makes annotation-centered extraction appropriate for waveform analysis and averaging. The same dataset documentation states that the analog signals were filtered with an approximate **0.1 to 100 Hz** passband before digitization. 
Reference("https://physionet.org/physiobank/database/html/mitdbdir/intro.htm")

This choice produces a total chunk length of:

- `N = 180 + 180 + 1 = 361 samples`
- `361 / 360 ≈ 1.003 s`

A roughly 1-second symmetric window was chosen intentionally. Standard ECG timing references place the **PR interval** around **120 to 200 ms**, the **QRS duration** below about **120 ms**, and commonly cited textbook **QTc** ranges below about **460 ms**. A ±500 ms window therefore comfortably captures the full local beat morphology around the annotated R-peak rather than only the QRS complex. 
Reference("https://physionet.org/physiobank/database/html/mitdbdir/intro.htm")

This window length was also chosen because the goal is **not** to isolate a beat completely from its temporal neighborhood. MIT-BIH contains fast and irregular episodes, including ventricular tachycardia around **174 to 177 bpm** and other arrhythmic episodes up to **189 bpm** in the record notes. A 1-second beat-centered window therefore intentionally preserves neighboring-beat transient effects when rhythms are fast, which is desirable for this retrieval setting.
Reference1("https://www.nottingham.ac.uk/nursing/practice/resources/cardiology/function/normal_duration.php")
Reference2("https://litfl.com/p-wave-ecg-library/")

### Frequency-domain preprocessing note

The saved **time-domain (TD)** representation keeps the raw ECG window unchanged. For the **frequency-domain (FD)** representation, each channel is **mean-centered per window before FFT**. This removes the constant offset and reduces the dominance of the **0 Hz / DC component** in the spectrum. This is consistent with standard spectral-analysis practice: SciPy’s `periodogram` and `welch` both use segment-level **constant detrending** as the standard detrending option for spectral estimation.
Reference1("https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html")
Reference2("https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html")

With `N = 361` and `fs = 360 Hz`, the one-sided FFT uses:

- `F = 181` bins
- frequency resolution `Δf = fs / N = 360 / 361 ≈ 0.997 Hz/bin`

This gives an interpretable frequency axis across all extracted ECG windows.