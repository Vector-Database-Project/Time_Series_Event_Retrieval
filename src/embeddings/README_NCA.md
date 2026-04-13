
## 2. `README_NCA_code.md`

```markdown
# NCA Python Code

## File

`NCA.py`

## Purpose

This script builds embeddings for ECG retrieval experiments using Neighborhood Components Analysis (NCA).

It is designed to work downstream from the existing preprocessing and train/test split pipeline. It does not generate new TD/FD representations and does not alter split membership.

## Main Function

The script:

1. Loads train and test shards from the processed dataset
2. Supports three input modes:
   - `time`
   - `frequency`
   - `mixed`
3. Standardizes the input features
4. Applies PCA before NCA
5. Fits NCA on a training subset
6. Transforms the full training and test splits
7. Saves embeddings, labels, learned components, and run metadata

## Why PCA Is Used

NCA is expensive on high-dimensional raw flattened features. PCA reduces the input feature space before NCA, which improves feasibility and reduces computation.

## Why a Training Subset Is Used for NCA

Full-sample NCA can become prohibitively memory-intensive because its optimization depends on pairwise relationships among training samples.

To make the method feasible on large datasets, this script fits NCA on a subset of the training data, then applies the learned transformation to the full training and test sets.

## Key Parameters

### `input_mode`
Which input representation to use:

- `"time"`
- `"frequency"`
- `"mixed"`

### `pca_components`
Number of PCA dimensions before NCA.

### `nca_fit_samples`
Number of training samples used to fit NCA.

### `n_components`
Final NCA embedding dimension.

### `max_iter`
Maximum number of NCA optimization iterations.

### `random_state`
Seed for reproducibility.

## Output

The script saves results under the configured output directory, typically:

```text
results/NCA/<run_name>