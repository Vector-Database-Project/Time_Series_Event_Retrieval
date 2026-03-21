# Gaussian Random Projection Baseline, ECG Retrieval

## Brief

This note records the first completed embedding baseline for the ECG retrieval pipeline using **Gaussian Random Projection (GRP)**.

The goal of this implementation pass was to:

- load the upstream ECG train/test split
- generate projected embeddings for the selected representation
- evaluate retrieval quality using nearest-neighbor label prediction
- save embeddings, metrics, and confusion-matrix artifacts for later analysis

This is an internal implementation note, not a final benchmark report.

---

## What Was Implemented

### Embedding stage

A Gaussian random projection embedding module was implemented for the ECG pipeline.

Current implementation behavior:

- consume the upstream split dataset from:
  - `processed/v1tts/train`
  - `processed/v1tts/test`
- load the selected representation through the runner
- concatenate all shards in a split into one full feature matrix
- fit `GaussianRandomProjection` on the **training split only**
- transform both train and test samples into lower-dimensional embeddings
- save:
  - train embeddings
  - test embeddings
  - train labels
  - test labels
  - projection components
  - run configuration metadata

### Evaluation stage

A common evaluation class was implemented to evaluate saved embedding runs.

Current evaluation behavior:

- use training embeddings as the reference pool
- use test embeddings as queries
- retrieve top-`K` nearest training embeddings for each test query
- assign the predicted label by majority vote over the retrieved labels
- compute:
  - Accuracy@K
  - Recall@K
  - F1@K
- generate and save confusion matrices for later visualization and inspection

Current reported values are for:

- `K = 1`
- `K = 5`
- `K = 10`

---

## Runs Executed

The following GRP runs were completed:

- `grp_time_064`
- `grp_time_128`
- `grp_frequency_064`
- `grp_frequency_128`

---

## Results

### Time domain, 64 components

| K | Accuracy | Macro Recall | Macro F1 |
|---|---:|---:|---:|
| 1  | 0.9799 | 0.6329 | 0.6449 |
| 5  | 0.9780 | 0.5952 | 0.6414 |
| 10 | 0.9738 | 0.5372 | 0.5788 |

### Time domain, 128 components

| K | Accuracy | Macro Recall | Macro F1 |
|---|---:|---:|---:|
| 1  | 0.9814 | 0.6379 | 0.6552 |
| 5  | 0.9789 | 0.5832 | 0.6203 |
| 10 | 0.9756 | 0.5453 | 0.5893 |

### Frequency domain, 64 components

| K | Accuracy | Macro Recall | Macro F1 |
|---|---:|---:|---:|
| 1  | 0.9642 | 0.6113 | 0.6357 |
| 5  | 0.9678 | 0.5271 | 0.5629 |
| 10 | 0.9659 | 0.5101 | 0.5508 |

### Frequency domain, 128 components

| K | Accuracy | Macro Recall | Macro F1 |
|---|---:|---:|---:|
| 1  | 0.9702 | 0.5801 | 0.5976 |
| 5  | 0.9710 | 0.5552 | 0.5931 |
| 10 | 0.9695 | 0.5265 | 0.5629 |

---

## Main Observations

### Best overall run

The strongest current result is:

- **time-domain, 128 components**
  - Accuracy@1 = **0.9814**
  - Macro Recall@1 = **0.6379**
  - Macro F1@1 = **0.6552**

This is the best recorded top-1 retrieval configuration in the current GRP baseline sweep.

### Time vs frequency

Across all recorded runs, the **time-domain representation** performs better than the **frequency-domain representation**.

In particular:

- both `time_064` and `time_128` outperform both frequency-domain runs at `K = 1`
- the time-domain runs also retain stronger macro recall and macro F1 overall
- the gap is most noticeable in the higher-dimensional comparison, where `time_128` is clearly stronger than `freq_128`

This suggests that, for the current ECG preprocessing and retrieval setup, the time-domain representation is a stronger input for the GRP baseline.

### Effect of embedding dimension

Increasing the projection dimension from `64` to `128` improves the time-domain run:

- `time_128` is better than `time_064` at `K = 1`
- `time_128` is also stronger at `K = 10`

For the frequency-domain run:

- `freq_128` improves over `freq_064`
- the gain is visible across all three reported `K` values

So, in the current baseline pass, moving from `64` to `128` dimensions appears beneficial for both representations, with a more useful payoff in the time-domain setting.

### Effect of K

For all current runs, the best values are generally seen at **`K = 1`**.

As `K` increases:

- Accuracy usually decreases or stays roughly flat
- Macro Recall usually decreases
- Macro F1 usually decreases

This suggests that the closest retrieved embedding is currently the most informative one, and expanding the neighborhood introduces more label mixing.

---

## Current Takeaways

From this first GRP baseline pass:

- the embedding pipeline is now implemented end-to-end for ECG
- the evaluation pipeline is also implemented end-to-end
- time-domain GRP is currently the strongest simple baseline
- `128` projected dimensions perform better than `64` for both time and frequency in the current runs
- `time_128` is the current best configuration among the completed GRP runs

---

## Saved Artifacts

### Embedding artifacts

Each run saves:

- train embeddings
- test embeddings
- train labels
- test labels
- projection components
- run configuration

### Evaluation artifacts

Each evaluated run saves:

- metrics JSON
- evaluation config
- top-`K` retrieved indices
- top-`K` retrieved labels
- predictions by `K`
- confusion matrices
- confusion-matrix plots
