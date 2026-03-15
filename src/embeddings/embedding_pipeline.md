# Embedding Pipeline

## Selected Embedding Methods

### Non-Learning Baseline, GaussianRandomProjection
- Package: `scikit-learn`
- Class: `sklearn.random_projection.GaussianRandomProjection`

**Method summary**
- Gaussian Random Projection maps an input vector into a lower-dimensional space using a random Gaussian matrix.
- The method is non-learning.
- There is no supervised objective and no gradient-based training.
- The fit step only defines the random projection matrix.

**Mathematical idea**
Given an input vector \( x \in \mathbb{R}^d \), the projected embedding \( z \in \mathbb{R}^k \) is:

\[
z = xR
\]

where \( R \) is a random projection matrix and \( k \) is the target embedding dimension.

The main motivation is approximate preservation of pairwise geometry under random projection.

**Why it is used here**
- clean non-learning baseline
- fixed output dimension through `n_components`
- fast and simple
- gives a direct baseline against supervised and unsupervised learned projections

**Core API**
- `fit_transform(X_ref)`
- `transform(X_test)`

**Main hyperparameters**
- `n_components`
- `random_state`

**Implementation rule**
- flatten each event representation into one row of `X`
- run with `n_components=128`
- run with `n_components=256`
- use the projected output directly as the embedding

---

### Supervised, NeighborhoodComponentsAnalysis
- Package: `scikit-learn`
- Class: `sklearn.neighbors.NeighborhoodComponentsAnalysis`

**Method summary**
- Neighborhood Components Analysis is a supervised metric-learning method.
- It learns a linear transformation of the input space.
- The goal is to improve nearest-neighbor class structure in the transformed space.

**Mathematical idea**
Given an input vector \( x_i \), the embedding is:

\[
z_i = Lx_i
\]

where \( L \) is a learned linear transformation.

The method optimizes a supervised objective based on stochastic nearest-neighbor classification.
It learns \( L \) so that samples are more likely to have neighbors from the same class in the transformed space.

**Why it is used here**
- supervised branch of the study
- directly relevant to retrieval because it learns neighbor-aware structure
- fixed output dimension through `n_components`
- clean transformer-style API

**Core API**
- `fit(X_ref, y_ref)`
- `fit_transform(X_ref, y_ref)`
- `transform(X_test)`

**Main hyperparameters**
- `n_components`
- `init`
- `max_iter`
- `tol`
- `random_state`

**Implementation rule**
- flatten each event representation into one row of `X`
- fit only on labeled reference data
- transform both reference and test samples
- use the transformed output directly as the embedding

---

### Unsupervised, KernelPCA
- Package: `scikit-learn`
- Class: `sklearn.decomposition.KernelPCA`

**Method summary**
- Kernel PCA is an unsupervised nonlinear dimensionality reduction method.
- It extends PCA using a kernel function.
- The method performs PCA in an implicit nonlinear feature space.

**Mathematical idea**
Standard PCA finds directions of maximum variance in the original space.
Kernel PCA first computes a kernel matrix between samples, centers it, and then performs eigendecomposition in kernel space.

The final embedding is formed from the leading kernel principal components.

There is no supervised loss here.
The method is variance-driven in the transformed kernel space.

**Why it is used here**
- unsupervised branch of the study
- nonlinear counterpart to the linear methods above
- fixed output dimension through `n_components`
- clean transformer-style API

**Core API**
- `fit(X_ref)`
- `fit_transform(X_ref)`
- `transform(X_test)`

**Main hyperparameters**
- `n_components`
- `kernel`
- `gamma`
- `degree`
- `coef0`
- `eigen_solver`

**Implementation rule**
- flatten each event representation into one row of `X`
- fit on the reference split only
- transform both reference and test samples
- use the transformed output directly as the embedding

---

## Input Protocol

The unit of retrieval is a fixed event sample.

Each method will be run on the following input modes:
1. `time`
2. `frequency`
3. `mixed`

### Input adaptation
- `time`
  - flatten `[C, W]` into one vector
- `frequency`
  - flatten `[C, F, 2]` into one vector
- `mixed`
  - concatenate flattened time and flattened frequency vectors for the same sample

The encoder stage is treated as a black box.
All preprocessing decisions end before the flattened event vector is passed into the embedding method.

---

## Embedding Contract

Each run must output:
- one embedding vector per sample
- fixed dimensions only
- dimensions used in this stage:
  - `128`
  - `256`
- output dtype: `float32`

Not allowed:
- variable-length embeddings
- multiple embeddings per sample
- hierarchical embedding outputs

---

## Run Matrix

The core run matrix is:

### Methods
- `GaussianRandomProjection`
- `NeighborhoodComponentsAnalysis`
- `KernelPCA`

### Input modes
- `time`
- `frequency`
- `mixed`

### Embedding dimensions
- `128`
- `256`

This gives:
- `3 × 3 × 2 = 18` core runs

The main study axes are:
- embedding method
- input mode
- embedding dimension

---

## Split and Fit Strategy

A common reference/test split will be used for all methods.

### Reference split
- used to fit the method when fitting is required
- used to populate the retrieval backend

### Withheld test split
- never inserted into the reference index
- only embedded and used as query input

### Per method
- GaussianRandomProjection
  - `fit_transform(X_ref)`
  - `transform(X_test)`
- NeighborhoodComponentsAnalysis
  - `fit_transform(X_ref, y_ref)`
  - `transform(X_test)`
- KernelPCA
  - `fit_transform(X_ref)`
  - `transform(X_test)`

---

## Retrieval Flow

For each run:
1. build event vectors for the selected input mode
2. fit the method on the reference split if required
3. transform the reference split into embeddings
4. store reference embeddings in the vector database
5. transform each withheld test sample
6. use each test embedding as a query against the reference pool
7. evaluate top-k retrieval quality

The vector database is only the retrieval backend.
The canonical output of each run is the saved embedding artifact plus metadata.

---

## Stored Output Per Run

Each run must produce:
- embeddings
- `sample_id`
- `source_id`
- label
- method tag
- input mode tag
- embedding dimension tag

Optional additional metadata:
- run id
- split id
- package version
- method hyperparameters

---

## Feature Handling

Allowed:
- removal of dead features
- removal of zero-variance features

Excluded:
- additional dimensionality reduction stacked after the selected embedding method

---

## Normalization Check

Normalization is not part of the main ablation matrix.

Plan:
- run a small representative check
- compare normalized vs unnormalized input handling
- if normalization is not clearly useful, drop it before the main runs

This check is only a gate before the 18 core runs.
It is not one of the final study axes.
