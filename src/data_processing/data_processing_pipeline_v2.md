# Data Processing Pipeline for Leakage-Aware Balanced ECG Benchmark

## Brief

This document describes the ECG preprocessing pipeline used to build a leakage-aware balanced benchmark for coarse beat classification.

The pipeline starts from raw MIT-BIH style ECG records and produces fixed-length beat-centered windows in both time-domain and frequency-domain form. It retains only endogenous beat classes that map cleanly into the coarse label space `N`, `S`, and `V`, then constructs balanced train and test splits using record-aware assignment and within-record random downsampling.

The main goals of this pipeline are:

- convert raw ECG records into fixed-length beat-centered windows
- preserve aligned time-domain and frequency-domain representations
- collapse detailed beat annotations into a small defensible coarse label space
- avoid train/test leakage by assigning whole records to a single split
- build balanced train and test sets while keeping representation across contributing records
- save the processed outputs in a consistent format for downstream embedding and retrieval experiments

---

## Current Scope

The implementation covers:

- raw ECG record discovery from `raw/`
- annotation parsing and filtering
- coarse label collapsing into `N`, `S`, and `V`
- fixed-length time-domain window extraction
- frequency-domain conversion from the extracted time-domain windows
- record-aware train/test split construction
- balanced within-split sampling
- saving processed train and test datasets and split metadata

---

## Folder Structure

```text
data/
  ecg/
    raw/
      *.hea
      *.dat
      *.atr

    processed/
      v1tts/
        split_config.json
        label_map.json
        frequency_bins.npz

        train/
          time_domain_data/
            td_shard_000.npz
            td_shard_001.npz
            ...
          frequency_domain_data/
            fd_shard_000.npz
            fd_shard_001.npz
            ...
          labels/
            label_shard_000.npz
            label_shard_001.npz
            ...
          metadata/
            metadata_shard_000.parquet
            metadata_shard_001.parquet
            ...

        test/
          time_domain_data/
            td_shard_000.npz
            td_shard_001.npz
            ...
          frequency_domain_data/
            fd_shard_000.npz
            fd_shard_001.npz
            ...
          labels/
            label_shard_000.npz
            label_shard_001.npz
            ...
          metadata/
            metadata_shard_000.parquet
            metadata_shard_001.parquet
            ...
```

If sharding is not needed for a small run, the same structure may be used with a single shard per split.

---

## Raw ECG Record Discovery

Each ECG record is defined by a shared record key and is expected to have three files:

- `.hea` for header information
- `.dat` for signal data
- `.atr` for beat annotations

The raw data directory is scanned and a registry is built from record keys. A record is considered valid only if all three required files are present.

The record key is the source identifier used for leakage-aware split construction.

---

## Retained Label Space

### Raw annotation vocabulary

The raw annotation stream may contain multiple beat types, rhythm markers, boundary markers, and artifact-related symbols.

As part of parsing:

- `.` and `·` are normalized to `N`
- only retained endogenous beat symbols are mapped into the final benchmark label space
- all other symbols are dropped

### Coarse benchmark classes

The retained coarse label mapping is:

- `N -> 0`
- `S -> 1`
- `V -> 2`

The detailed symbol collapse is:

- `N` group: `N`, `L`, `R`, `e`, `j`
- `S` group: `A`, `a`, `J`, `S`
- `V` group: `V`, `E`

The following symbols are dropped from this benchmark:

- ventricular-normal fusion beat: `F`
- paced or paced-fusion beats: `/`, `f`
- unclassifiable beat: `Q`
- ventricular flutter or fibrillation markers: `[`, `!`, `]`
- blocked or non-conducted event marker: `x`
- artifact-like annotation: `|`

This produces a coarse endogenous-beat benchmark focused on normal or conduction-side beats, supraventricular ectopic beats, and ventricular-origin beats.

---

## Annotation Parsing and Filtering

For each valid record:

- read the annotation stream from the `.atr` file
- read the signal length from the header or from the loaded signal if needed
- normalize annotation symbols where required
- keep only annotations that map into the retained `N`, `S`, and `V` space
- discard any annotation that does not have enough signal context on both sides to support the fixed extraction window

Boundary filtering is applied before extraction:

- discard annotations with insufficient pre-window context
- discard annotations with insufficient post-window context

This guarantees that every kept annotation can produce a complete fixed-length window without padding.

---

## MIT-BIH ECG Chunking Policy

### Time-domain preprocessing note

Beats are extracted using a fixed beat-centered symmetric window with:

- `pre_samples = 180`
- `post_samples = 180`

This gives a total chunk length of:

- `N = 180 + 180 + 1 = 361 samples`
- `361 / 360 ≈ 1.003 s`

The window is centered on the retained annotation sample.

This policy intentionally preserves the full local beat morphology together with limited temporal neighborhood context around the annotated beat, rather than isolating only the QRS complex.

### Time-domain output shape

Each extracted time-domain beat window is stored in channel-first form:

- shape `[C, T]`
- for this dataset, `[2, 361]`

Only finite windows are kept.

---

## Frequency-Domain Preprocessing Note

The saved time-domain representation keeps the extracted ECG window unchanged.

For the frequency-domain representation, each channel is mean-centered per window before FFT. This reduces the dominance of the DC component and produces a cleaner local spectrum.

With `N = 361` and `fs = 360 Hz`, the one-sided FFT uses:

- `F = 181` bins
- frequency resolution `Δf = fs / N = 360 / 361 ≈ 0.997 Hz/bin`

For each time-domain window:

- apply per-channel mean centering
- compute one-sided real FFT along the time axis
- save magnitude and phase for each channel and frequency bin

### Frequency-domain output shape

Each extracted frequency-domain beat window is stored as:

- shape `[C, F, 2]`
- for this dataset, `[2, 181, 2]`
- the last dimension stores `[magnitude, phase]`

The frequency axis is shared across all windows and is saved once as `frequency_bins.npz`.

---

## Record-Level Inventory Pass

Before constructing the train and test splits, the pipeline performs an inventory pass over all valid records.

For each record, it computes:

- record ID
- retained beat centers
- original retained symbols
- coarse mapped labels
- per-record counts for `N`, `S`, and `V`

This inventory is used to build leakage-aware splits before final sampling.

---

## Leakage-Aware Train and Test Split Policy

### Split objective

The split is driven by the rarest retained class, `S`.

The target is:

- `2250` samples per class for `train`
- `250` samples per class for `test`

This gives a balanced `2250 + 250` allocation per class.

### Record-level assignment

Train and test assignment is performed at the record level.

A record may belong to only one split.

No individual beat from a train record may appear in test, and no individual beat from a test record may appear in train.

### Seeded shuffled record order

To avoid bias from raw record ordering while keeping the split reproducible:

- create the list of valid record IDs
- shuffle the record IDs using a fixed random seed
- use this shuffled order for cumulative split assignment

### Cumulative `S`-driven assignment

Split assignment proceeds as follows:

- add whole shuffled records to the train split until cumulative retained `S` count reaches or slightly exceeds `2250`
- from the remaining shuffled records, add whole shuffled records to the test split until cumulative retained `S` count reaches or slightly exceeds `250`
- allow small overshoot caused by the final included record in each split

Only records selected by this procedure are used in the derived balanced benchmark.

Any remaining records outside the selected train and test record pools are not used in this derived split version.

---

## Balanced Within-Split Sampling

After record assignment, each split contains all retained candidate `N`, `S`, and `V` beats from its selected records.

Exact class balancing is then applied separately inside `train` and `test`.

### Split targets

The per-class keep targets are:

- train: `2250`
- test: `250`

for each of `N`, `S`, and `V`.

### Even-per-record sampling rule

Within each split and each class:

- group candidate beats by record ID
- identify the records that contribute at least one beat of that class
- divide the class target as evenly as possible across those contributing records
- sample randomly within each record to meet the provisional per-record quota

This prevents the final balanced subset from being dominated by one record with many more beats than the others.

### Feasibility guard and redistribution

Strictly even per-record allocation may be impossible if a record contributes fewer beats than its provisional quota.

In that case:

- keep all available beats from that record for that class
- compute the leftover quota
- redistribute the leftover quota across the remaining contributing records
- repeat until the split reaches the target count for that class

The final selection therefore remains:

- record-aware
- within-record random
- as even across contributing records as the available data allows

### Sampling reproducibility

All random within-record sampling uses a fixed random seed so that the final balanced train and test subsets are reproducible.

---

## Final Saved Outputs

For each split, the pipeline saves aligned arrays for:

- time-domain windows
- frequency-domain windows
- coarse labels
- sample-level metadata

### Time-domain data

Saved under:

```text
time_domain_data/
```

Each shard contains a `data` array with shape:

- `[N_samples, 2, 361]`

Stored as `float32`.

### Frequency-domain data

Saved under:

```text
frequency_domain_data/
```

Each shard contains a `data` array with shape:

- `[N_samples, 2, 181, 2]`

Stored as `float32`.

### Labels

Saved under:

```text
labels/
```

Each shard contains a `data` array with shape:

- `[N_samples]`

Stored as `int64`.

The label map is saved once as:

```text
label_map.json
```

with:

```json
{
  "N": 0,
  "S": 1,
  "V": 2
}
```

### Metadata

Saved under:

```text
metadata/
```

Each metadata shard should include at least:

- `record_id`
- `center_sample`
- `original_symbol`
- `coarse_label_name`
- `coarse_label_id`
- `split`

This metadata is required for traceability, leakage auditing, and debugging.

### Shared split metadata

The root split directory also stores:

- `split_config.json`
- `frequency_bins.npz`

`split_config.json` should record the key preprocessing and split settings, including:

- sampling rate
- window sizes
- retained symbol mapping
- dropped symbol list
- train and test per-class targets
- record shuffle seed
- within-record sampling seed
- selected train record IDs
- selected test record IDs

---

## Processing Summary

At a high level, the pipeline runs in this order:

- discover valid raw ECG records
- parse annotations and normalize symbols
- retain only annotations that map into `N`, `S`, and `V`
- apply boundary checks for fixed beat-centered extraction
- build a record-level inventory of retained beat counts
- shuffle record IDs with a fixed seed
- accumulate whole records into train until retained `S >= 2250`
- accumulate whole remaining records into test until retained `S >= 250`
- extract all candidate time-domain windows from selected records
- convert retained time-domain windows into frequency-domain windows
- balance `N`, `S`, and `V` separately inside each split using even-per-record random sampling
- save aligned train and test artifacts and metadata

---

## Resulting Benchmark Definition

The final derived benchmark is:

- coarse 3-class ECG beat classification
- label space `N`, `S`, `V`
- record-aware train and test separation
- balanced train set with `2250` samples per class
- balanced test set with `250` samples per class
- identical chunking and FFT policy across splits

This benchmark is intended to provide a leakage-aware and class-balanced ECG preprocessing base for downstream embedding and retrieval experiments.

