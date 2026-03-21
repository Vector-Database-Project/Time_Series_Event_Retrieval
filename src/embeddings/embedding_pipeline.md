# Embedding Pipeline

## Brief

This file defines the current **implementation plan** for the embedding stage.

It is a downstream planning file only.
Upstream preprocessing and split generation are already defined in `data_processing_pipeline.md` and are treated as fixed inputs here.

This file should only capture:
- what the embedding stage consumes
- what runs are planned
- what outputs must be saved
- what constraints must be respected

---

## Upstream Contract

The embedding stage will consume:

- `data/ecg/processed/v1tts/train`
- `data/ecg/processed/v1tts/test`

Assumptions:
- train/test split is already finalized upstream
- TD, FD, and label shards are aligned within each split
- preprocessing is not re-done here
- no new TD or FD representations are generated here

---

## Input Modes

Planned input modes:
- `time`
- `frequency`
- `mixed`

Mode rules:
- `time`: load TD shards and flatten to `X`
- `frequency`: load FD shards and flatten to `X`
- `mixed`: load aligned TD and FD shards, flatten both, concatenate row-wise

Mixed mode is assembled in the embedding stage.

---

## Loader Contract

The embedding stage will use the existing shard-based extraction path.

Per split and shard index, it should load:
- features as `X`
- labels as `y`

Requirements:
- one row per sample
- one label per sample
- matching sample order across aligned shards

Any alignment mismatch should fail loudly.

---

## Planned Methods

Initial planned methods:
- `GaussianRandomProjection`
- `NeighborhoodComponentsAnalysis`
- `KernelPCA`

This is the current baseline set and can be revised after implementation.

---

## Planned Run Matrix

Each run is defined by:
- one input mode
- one embedding method
- one embedding dimension setting

Current plan:
- input modes: `time`, `frequency`, `mixed`
- methods: `GaussianRandomProjection`, `NeighborhoodComponentsAnalysis`, `KernelPCA`
- embedding dimensions: `TBD`

---

## Fit Policy

- fit on training split only
- transform both training and test splits
- use training embeddings as the retrieval pool
- use test embeddings as queries

The test split must not be used during fitting.

---

## Planned Execution Flow

For each run:
1. choose input mode
2. load train split
3. load test split
4. build `X_train, y_train`
5. build `X_test, y_test`
6. fit embedding method on training data
7. transform train and test data
8. index training embeddings
9. query with test embeddings
10. evaluate retrieval
11. save run artifacts

---

## Output Contract

Each run should save:
- train embeddings
- test embeddings
- train labels
- test labels
- method tag
- input mode tag
- embedding dimension
- split identifier
- run configuration metadata

Final artifact layout will be fixed during implementation.

---

## Constraints

The embedding stage must not:
- redefine preprocessing
- regenerate TD or FD data
- alter train/test membership
- fit on test data
- combine unmatched TD and FD shards
- ignore alignment errors

---

## Pending Before Execution

To be finalized:
- embedding dimension values
- artifact folder structure
- retrieval backend choice
- evaluation metrics
- full-split vs shard-wise execution strategy

---

## Status

This is a planning file for implementation.

It should be updated after execution to reflect:
- what was actually run
- finalized settings
- saved outputs
- any deviations from plan