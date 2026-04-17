# ECG Embedding Retrieval Metrics

## Brief

This document summarizes the observed retrieval results for the ECG `N/S/V` benchmark across the full embedding matrix.

The purpose of this note is to record the major metric trends and the main conclusions supported by the current runs. It is not intended to claim broad generalization beyond this benchmark.

---

## Evaluation setting

The evaluated run matrix contains:

- 3 methods
  - Gaussian Random Projection
  - UMAP
  - Supervised UMAP
- 3 input representations
  - time
  - frequency
  - mixed
- 2 embedding dimensions
  - 64
  - 128

This gives a total of 18 evaluated runs.

Each run was evaluated using retrieval with:

- `K = 1, 5, 10`
- `l2` distance
- metrics:
  - accuracy
  - macro precision
  - macro recall
  - macro F1

---

## Main result

The strongest overall result in the current benchmark is:

- **GRP + frequency + 64 dimensions**

At `K = 10`, this run achieved:

- accuracy: `0.7253`
- macro precision: `0.7406`
- macro recall: `0.7253`
- macro F1: `0.7287`

This was the best result across all tested method, representation, and dimension combinations.

---

## Best run per method family

### Gaussian Random Projection

Best run:

- `grp_frequency_064`

Metrics:

- `K = 1`: accuracy `0.6640`, macro F1 `0.6666`
- `K = 5`: accuracy `0.7200`, macro F1 `0.7224`
- `K = 10`: accuracy `0.7253`, macro F1 `0.7287`

### UMAP

Best run:

- `umap_mixed_nn15_dim64`

Metrics:

- `K = 1`: accuracy `0.5840`, macro F1 `0.5866`
- `K = 5`: accuracy `0.5907`, macro F1 `0.5930`
- `K = 10`: accuracy `0.5680`, macro F1 `0.5709`

A close alternative was:

- `umap_frequency_nn15_dim128`

which reached macro F1 `0.5914` at `K = 5`.

### Supervised UMAP

Best run:

- `sup_umap_mixed_nn15_dim128`

Metrics:

- `K = 1`: accuracy `0.5573`, macro F1 `0.5490`
- `K = 5`: accuracy `0.5560`, macro F1 `0.5472`
- `K = 10`: accuracy `0.5573`, macro F1 `0.5484`

Overall, the supervised UMAP family underperformed both GRP and the better unsupervised UMAP runs in this benchmark.

---

## Representation-level trends

### Frequency representation

Frequency was the strongest representation overall.

This is supported by two consistent observations:

- the best run in the entire matrix was frequency-based
- several of the stronger non-best runs also came from the frequency setting

Notable frequency results:

- `grp_frequency_064`: macro F1 `0.7287` at `K = 10`
- `grp_frequency_128`: macro F1 `0.6880` at `K = 10`
- `umap_frequency_nn15_dim128`: macro F1 `0.5914` at `K = 5`
- `sup_umap_frequency_nn15_dim64`: macro F1 `0.5462` at `K = 1`

Across the full matrix, frequency was consistently more reliable than raw time representation and usually stronger than direct mixed concatenation.

### Mixed representation

Mixed representation produced intermediate results.

It sometimes improved over time-only inputs, but it did not surpass the strongest frequency-only runs. In the current setup, direct concatenation of time and frequency features did not produce the best-performing retrieval space.

### Time representation

Time representation was the weakest overall.

This pattern was especially clear for the supervised UMAP runs, where time-based results were the lowest in the full matrix.

Notable weak results:

- `sup_umap_time_nn15_dim128`: macro F1 `0.4329` at `K = 10`
- `sup_umap_time_nn15_dim64`: macro F1 `0.4402` at `K = 10`

This suggests that, under the current leakage-aware benchmark, raw time-domain morphology was less stable for retrieval than the corresponding frequency-domain view.

---

## Method-level trends

### Gaussian Random Projection was strongest

The non-learning GRP baseline outperformed both UMAP and supervised UMAP.

This is an important result because it shows that, in the current benchmark, a simple linear random projection preserved retrieval-relevant structure better than the more complex manifold-learning alternatives.

The best six runs in the matrix were all GRP-based, and the strongest two runs were the frequency-domain GRP variants.

### UMAP was competitive but not best

Standard UMAP performed better than supervised UMAP in the current benchmark, but it did not beat GRP.

Its best results were in the high `0.58` to low `0.59` macro F1 range, well below the strongest GRP frequency result.

This suggests that the unsupervised manifold geometry learned by UMAP did not translate into the best retrieval neighborhoods under this ECG setup.

### Supervised UMAP was weakest

Supervised UMAP was the weakest method family overall.

In particular, the time-domain supervised UMAP runs were clearly the worst in the full matrix. Even the better supervised UMAP mixed and frequency runs remained below the stronger UMAP runs and well below the GRP baseline.

This indicates that using training labels inside the UMAP fitting procedure did not improve retrieval behavior for unseen test queries in the current record-level split benchmark.

---

## Embedding dimension trends

The comparison between `64` and `128` dimensions does not support a simple “larger is better” conclusion.

Observed pattern:

- several of the strongest runs used `64` dimensions
- in multiple cases, `128` was slightly worse than `64`
- larger embeddings did not reliably improve retrieval performance

Examples:

- `grp_frequency_064` outperformed `grp_frequency_128`
- `umap_mixed_nn15_dim64` was stronger than `umap_mixed_nn15_dim128`
- `sup_umap_frequency_nn15_dim64` was slightly stronger than `sup_umap_frequency_nn15_dim128`

This suggests that, in the current benchmark, the smaller embedding size was often sufficient and could even act as a more stable compressed representation.

---

## Top-K behavior

The effect of increasing `K` was not uniform across runs.

### When K helped

For the strongest GRP frequency runs, larger `K` improved performance.

Example:

- `grp_frequency_064`
  - macro F1 at `K = 1`: `0.6666`
  - macro F1 at `K = 5`: `0.7224`
  - macro F1 at `K = 10`: `0.7287`

This indicates that the local retrieval neighborhoods of the strongest run were class-consistent enough for majority voting to become more reliable as more neighbors were used.

### When K did not help

For many weaker runs, increasing `K` did not improve the metrics and sometimes reduced them.

Examples:

- `grp_time_064` dropped from macro F1 `0.5889` at `K = 1` to `0.5083` at `K = 10`
- `umap_time_nn15_dim64` dropped from macro F1 `0.5042` at `K = 1` to `0.4884` at `K = 10`

This indicates that the retrieved neighborhoods in these runs became less class-pure as more neighbors were considered.

---

## Ranked summary by best observed macro F1

Using the best macro F1 achieved by each run over `K = 1, 5, 10`, the strongest runs were:

1. `grp_frequency_064` -> `0.7287`
2. `grp_frequency_128` -> `0.6880`
3. `grp_frequency_064` at `K = 5` already reached `0.7224`
4. `grp_frequency_128` at `K = 5` reached `0.6772`
5. `grp_frequency_064` at `K = 1` reached `0.6666`
6. `grp_mixed_064` -> `0.6319`
7. `grp_mixed_128` -> `0.5985`
8. `umap_mixed_nn15_dim64` -> `0.5930`
9. `umap_frequency_nn15_dim128` -> `0.5914`
10. `grp_time_064` -> `0.5889`

The exact ordering after the strongest GRP frequency runs is less important than the overall pattern:

- GRP dominates the top of the ranking
- frequency is the strongest representation family
- time-only learned embeddings remain weak

---

## Supported conclusions

The current results support the following benchmark-specific conclusions:

### 1. Representation choice mattered strongly

The difference between time, frequency, and mixed inputs was large and consistent enough to affect the final ranking substantially.

### 2. Frequency-domain preprocessing was the strongest tested representation

Under the current ECG benchmark, frequency-domain inputs were the most effective of the tested preprocessing choices.

### 3. Greater model complexity did not guarantee better retrieval

The simplest method family, Gaussian Random Projection, outperformed both UMAP-based alternatives.

### 4. Supervision inside UMAP did not help this retrieval setup

Using labels during UMAP fitting did not improve performance and instead produced the weakest overall family of runs.

### 5. Smaller embeddings were often sufficient

The comparison between 64 and 128 dimensions suggests that a larger embedding budget was not necessary to obtain the best retrieval behavior in this benchmark.

---

## How to phrase the result safely

The current results are strong enough to support the following type of statement:

> In the current leakage-aware balanced ECG benchmark, frequency-domain preprocessing consistently outperformed time-domain and direct mixed representations across the evaluated embedding families. The best overall retrieval performance was obtained by a simple Gaussian Random Projection baseline, indicating that representation choice had a larger effect than embedding-model complexity in the tested setup.

This statement is supported by the observed runs and stays within the scope of the experiment.

---

## Scope note

These conclusions apply to:

- the current ECG dataset
- the current `N/S/V` benchmark definition
- the current leakage-aware balanced split
- the tested embedding families and dimensions
- the current retrieval evaluator based on nearest-neighbor voting

The results should therefore be interpreted as a benchmark finding for this project setting, not as a universal claim about all time-series embedding problems.

---

## Final takeaway

The benchmark did not support the idea that more complex embedding methods automatically improve retrieval. Instead, it showed a cleaner and more useful result:

- **frequency-domain preprocessing was the strongest tested representation choice**
- **Gaussian Random Projection was the strongest tested embedding method**
- **UMAP-based learned embeddings did not outperform the simple baseline in this setting**

This is a valid and meaningful result for the project because it gives a concrete direction for further investigation and provides a controlled comparison across representation and embedding choices.

