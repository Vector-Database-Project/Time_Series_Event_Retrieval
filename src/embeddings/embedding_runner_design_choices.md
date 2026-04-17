# Embedding Runner Design Choices

## Brief

This document explains the design choices behind the unified embedding runner used for the ECG benchmark.

The goal of this stage is not to discuss results. The goal is to define a clean and reproducible execution policy for generating embedding artifacts across the selected representation and method combinations.

---

## Why a single unified runner is used

The project evaluates multiple embedding methods under the same downstream retrieval setup. Running each method through separate hand-edited scripts would make the experiment fragile and hard to reproduce.

A single unified runner is used so that:

- all methods read the same train and test splits
- all methods use the same representation definitions
- all runs are created with a consistent naming convention
- all outputs are written to a consistent folder structure
- the full comparison matrix can be reproduced from one entry point

This reduces manual mistakes and ensures that method comparisons are driven by actual modeling differences rather than by inconsistencies in execution.

---

## Why the full representation-method matrix is run

The benchmark compares three representation choices:

- time
- frequency
- mixed

and three embedding methods:

- Gaussian Random Projection
- UMAP
- Supervised UMAP

Each method is run at two embedding dimensions:

- 64
- 128

This creates the full matrix:

- 3 methods
- 3 input representations
- 2 embedding dimensions

for a total of 18 runs.

This matrix is intentional. The project is not evaluating one isolated model. It is evaluating how retrieval behavior changes as both the input representation and the embedding method change under a fixed preprocessing and split policy.

Running the full matrix keeps the comparison symmetric. No method is given a narrower or easier subset of representation choices.

---

## Why the same train and test roots are reused for all runs

All embedding runs consume the same preprocessed benchmark:

- the same retained label space
- the same leakage-aware train and test split
- the same balanced class counts
- the same time-domain and frequency-domain source artifacts

This is important because the embedding stage should only vary the representation and embedding method. The underlying data split must remain fixed.

If the split changed between methods, then observed differences could not be attributed cleanly to the embedding policy.

---

## Why time, frequency, and mixed are all included

The benchmark is designed around three representation views of the same ECG beat windows.

### Time representation

The time representation preserves the raw local waveform structure in the beat-centered extraction window. This is the most direct view of the signal and keeps morphological information in its original form.

### Frequency representation

The frequency representation emphasizes spectral structure after the same beat window is mean-centered and transformed. This gives a different view of local shape and periodic content than the raw waveform alone.

### Mixed representation

The mixed representation concatenates the time-domain and frequency-domain features into one joint input vector. This keeps both views available to the embedding method rather than forcing a single-view choice.

Including all three is a design choice to test representation dependence under one consistent benchmark, not to assume in advance that one view is universally best.

---

## Why 64 and 128 dimensions are used

Two embedding sizes are used:

- 64
- 128

The purpose is to compare a smaller and a larger embedding budget without turning the experiment into a broad hyperparameter sweep.

This gives:

- a lower-dimensional compact embedding setting
- a higher-dimensional less compressed setting

Using exactly two dimensions keeps the benchmark manageable while still allowing the study to observe how much the chosen embedding size changes the representation learned by each method.

The selected dimensions are also shared across methods so that dimensionality itself remains comparable across the matrix.

---

## Why Gaussian Random Projection is included

Gaussian Random Projection is included as the non-learning baseline.

This matters because the benchmark is not only about whether complex methods work. It is also about whether they provide value beyond a simple, generic dimensionality reduction baseline.

A random projection baseline is useful because:

- it is simple
- it is fast
- it does not use labels
- it does not rely on neighborhood optimization
- it provides a clean non-learned reference point

Including it makes the comparison more disciplined. Improvements from UMAP-based methods are easier to interpret when they are measured against a strong simple baseline rather than only against other learned methods.

---

## Why UMAP is included

UMAP is included as the unsupervised manifold-learning method.

It is useful in this benchmark because it learns a low-dimensional embedding by preserving local neighborhood structure from the input space without using labels during fitting.

This makes it a suitable unsupervised contrast point against both:

- the non-learning Gaussian RP baseline
- the label-aware supervised UMAP variant

Including UMAP allows the study to ask whether neighborhood-preserving unsupervised structure alone is sufficient to produce useful embedding spaces for downstream retrieval.

---

## Why Supervised UMAP is included

Supervised UMAP is included as the label-aware counterpart to standard UMAP.

It uses the training labels during fitting and therefore represents a different modeling regime from both:

- Gaussian RP, which is non-learning
- standard UMAP, which is unsupervised

This is important for the benchmark because it introduces the supervised method family into the comparison while keeping the underlying UMAP formulation closely related to the unsupervised version.

That makes the comparison cleaner. The effect of using labels can be studied without completely changing the type of embedding model.

---

## Why mixed-mode loading had to be handled explicitly

The UMAP and supervised UMAP components already support the three input modes directly.

Gaussian Random Projection, however, is naturally defined only on a feature matrix and does not contain explicit mixed-mode loading logic in the same way as the UMAP wrappers.

The unified runner therefore uses a shared loading policy that:

- loads time-domain features when `input_mode = time`
- loads frequency-domain features when `input_mode = frequency`
- concatenates aligned time-domain and frequency-domain features when `input_mode = mixed`

This design keeps the representation definition identical across methods. Mixed input means the same thing for every method.

---

## Why train embeddings and test embeddings are both saved

Each run saves:

- train embeddings
- test embeddings
- train labels
- test labels
- run configuration metadata

This is done so that the embedding stage stays cleanly separated from the downstream retrieval and evaluation stage.

The embedding runner is responsible only for constructing the embedding artifacts. It should not mix embedding generation with retrieval scoring or final analysis.

Saving both train and test embeddings makes later retrieval evaluation reproducible and avoids rerunning the embedding stage unnecessarily.

---

## Why consistent run naming is important

Each run is stored under a method- and configuration-specific folder name.

Examples:

- `grp_time_064`
- `umap_frequency_nn15_dim128`
- `sup_umap_mixed_nn15_dim64`

This naming policy is deliberate. It ensures that the output directory itself records the main experimental identity of the run.

That makes it easier to:

- inspect outputs manually
- script downstream evaluation
- compare runs without ambiguity
- avoid accidental overwrite between configurations

---

## Why low-memory mode is disabled for UMAP runs

For the current ECG benchmark, the dataset size is small enough that the UMAP runs do not need the more conservative low-memory execution path.

The runner therefore sets:

- `low_memory = False`

for both standard UMAP and supervised UMAP.

This is a deliberate design choice based on dataset scale. Since the benchmark has already been reduced to a compact balanced train and test split, using the full-memory path is appropriate and keeps the execution policy straightforward.

---

## Why overwrite is disabled by default

The runner skips already completed runs unless overwrite is explicitly enabled.

This is done to protect existing artifacts and to make repeated execution safer.

When running a full matrix, some runs may already be complete while others may still be pending. Skipping completed runs by default allows the matrix to be resumed without forcing all finished experiments to be recomputed.

---

## Why a run summary file is written

The runner writes a summary JSON file over the full matrix execution.

This is useful because the full experiment set contains many independent runs. A summary file gives one central place to record:

- which runs were attempted
- which runs completed successfully
- which runs failed
- where each run was saved

This improves auditability and makes it easier to continue downstream evaluation without manually reconstructing execution status from folders alone.

---

## Design summary

The unified embedding runner is designed to enforce one clean policy across the ECG benchmark:

- fixed train and test benchmark inputs
- full symmetric representation-method comparison matrix
- shared embedding dimensions
- method-appropriate but representation-consistent loading
- separate saved artifacts for later retrieval evaluation
- reproducible configuration recording

The key idea is to make the embedding stage systematic and reproducible so that later differences in retrieval behavior can be attributed to controlled experimental choices rather than inconsistent execution.