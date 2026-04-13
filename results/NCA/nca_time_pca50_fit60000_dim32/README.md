# NCA Output Files

This folder contains saved artifacts from the Neighborhood Components Analysis (NCA) embedding pipeline.

## Purpose

These files store the final train/test embeddings and related metadata for one NCA experiment run. The embeddings are intended for downstream retrieval and evaluation.

## Pipeline Summary

For this run, the pipeline:

1. Loaded the processed train and test splits
2. Standardized features using the training split
3. Applied PCA to reduce the feature space before NCA
4. Fit NCA on a subset of the training data
5. Transformed the full training and test splits into the final embedding space
6. Saved embeddings, labels, learned components, and run metadata

## Expected Files

### `train_embeddings.npz`
Final embedding vectors for the full training split.

**Shape:** `(n_train_samples, embedding_dimension)`

### `test_embeddings.npz`
Final embedding vectors for the full test split.

**Shape:** `(n_test_samples, embedding_dimension)`

### `train_labels.npz`
Labels aligned with `train_embeddings.npz`.

### `test_labels.npz`
Labels aligned with `test_embeddings.npz`.

### `pca_components.npz`
Learned PCA component matrix used before NCA.

### `nca_components.npz`
Learned NCA linear transformation matrix.

### `run_config.json`
Metadata describing the experiment configuration, including:

- `method`
- `input_mode`
- `pca_components`
- `nca_fit_samples`
- `n_components`
- `max_iter`
- `random_state`
- data roots
- output path
- split identifier
- fit strategy

## How to Load

Each `.npz` file stores its main array under the key `"data"`.

Example:

```python
import numpy as np

Z_train = np.load("train_embeddings.npz")["data"]
y_train = np.load("train_labels.npz")["data"]