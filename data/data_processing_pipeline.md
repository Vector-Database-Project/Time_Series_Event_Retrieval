# Data Preprocessing Pipeline for Time-Series Event Retrieval

## Brief

This processed dataset must support:

- supervised training
- unsupervised representation learning
- retrieval experiments
- leakage-aware train, validation, and test splits
- grouped cross-validation later if needed

The preprocessing pipeline should standardize different datasets into a common processed format so that ECG, gear vibration, IMU, and other time-series sources can all be used in one shared training and evaluation pipeline.

---

## Final Output

The downstream model pipeline should be able to assume that every processed sample has:

- a fixed-length signal window
- a consistent channel layout
- one primary label
- one source ID
- one metadata row in the sample information file

The preprocessing team should deliver:

- unchanged raw data
- processed time-domain signal windows
- processed frequency-domain signal representations
- a sample-level information file
- a source-level information file
- a label map
- grouped split definitions later

---

## Folder Structure

Each dataset should follow the same folder layout so that preprocessing, representation generation, training, and evaluation can all use a common pipeline.

```text
data/
  <dataset_name>/
    raw/
      ...
    processed/
      v1/
        source_info.parquet
        sample_info.parquet
        label_map.json
        time_domain_data/
          td_shard_000.npz
          td_shard_001.npz
          ...
        frequency_domain_data/
          fd_shard_000.npz
          fd_shard_001.npz
          ...
```

### Main idea

- `raw/` stores the original dataset files exactly as they were downloaded or received
- `processed/v1/` stores the first cleaned and standardized version of the dataset
- `time_domain_data/` stores the canonical time-domain samples
- `frequency_domain_data/` stores the frequency-domain versions of those same samples
- `v1/` means the first preprocessing version
- if the preprocessing logic changes later, create a new version such as `v2/` instead of overwriting old outputs

This keeps the preprocessing pipeline reproducible and easier to debug.

---

## Folder and File Descriptions

### `raw/`

This folder contains the original dataset files.

**Rules:**
- do not edit files in this folder
- do not rename files unless absolutely necessary
- do not save cleaned outputs here

This folder remains the source of truth for the original data.

---

### `processed/v1/`

This folder contains the first cleaned and model-ready version of the dataset.

Everything inside this folder must follow one clear preprocessing policy, including:

- sample extraction rule
- label assignment rule
- metadata format
- storage format
- frequency representation generation rule

If the preprocessing policy changes later, create a new folder such as `processed/v2/`.

---

### `source_info.parquet`

This file stores **one row per raw source sequence**.

A source sequence could be:

- one ECG record
- one machine run
- one sensor recording
- one continuous raw file

This file is used for:

- source-level metadata
- bookkeeping
- debugging
- later grouped split generation

Typical information stored here includes:

- dataset name
- source ID
- sequence ID
- sampling frequency
- number of channels
- channel names
- sequence duration
- raw file path
- optional notes

---

### `sample_info.parquet`

This file stores **one row per processed sample**.

A processed sample is usually:

- one fixed-length time-series window
- one primary label
- one source ID
- one location inside a saved shard file

This is the main indexing file in the preprocessing pipeline.

The model loader should use this file to locate and load samples from disk.

Typical information stored here includes:

- sample ID
- source ID
- sequence ID
- label
- start and end sample index
- sampling frequency
- number of channels
- shard file name
- row index inside the shard file
- optional quality flags

---

### `label_map.json`

This file stores the mapping between label names and integer IDs.

Example:

```json
{
  "healthy": 0,
  "initial_decay": 1,
  "failure_trend": 2,
  "failure": 3
}
```

This keeps label handling consistent across preprocessing, training, and evaluation.

---

### `time_domain_data/`

This folder stores the processed **time-domain** signal samples.

Samples should be saved in `.npz` shard files, not as one separate file per sample.

Example:

```text
time_domain_data/
  td_shard_000.npz
  td_shard_001.npz
```

Each shard file should contain many samples together.

This is preferred because it:

- reduces file clutter
- loads faster than millions of tiny files
- is easier to manage and debug

Each stored time-domain sample should have shape:

```python
[C, W]
```

where:

- `C` = number of channels
- `W` = fixed window length

Each shard should usually store:

```python
x           # shape [N, C, W]
sample_ids  # shape [N]
```

where:

- `N` = number of samples inside that shard

The `time_domain_data/` folder should be treated as the canonical processed dataset.

---

### `frequency_domain_data/`

This folder stores the processed **frequency-domain** representations generated from the time-domain samples.

Example:

```text
frequency_domain_data/
  fd_shard_000.npz
  fd_shard_001.npz
```

These representations are used to capture:

- dominant frequency content
- periodic behavior
- harmonic structure
- spectral changes linked to system condition

### FFT Setup

FFT generation is done **after** the time-domain samples are created.

For each dataset version:

- select a **random balanced subset** from `time_domain_data/`
- run FFT on that subset
- inspect the dominant frequencies
- decide the **frequency range** and **bin resolution**

This inspection step is **manual** and is used to fix the FFT setup for that dataset version.

Once the FFT setup is fixed, run the full automated FFT generation script on all time-domain samples.

The script should take:
- selected frequency range
- selected bin resolution

and generate frequency-domain shards with:
- magnitude
- phase

### Rules

- FFT setup must be decided only after time-domain samples are created
- use one fixed FFT setup per dataset version
- keep the same frequency-bin layout across all `fd_shard_*.npz` files
- `frequency_domain_data/` must always be generated from the samples stored in `time_domain_data/`
- every frequency-domain sample must correspond to an existing time-domain sample
- `sample_ids` must stay aligned between `time_domain_data/` and `frequency_domain_data/`
- shard naming should stay aligned
- the time-domain sample remains the reference version of the data
- if the FFT setup changes later, create a new processed version such as `v2/`

This keeps comparisons fair across:
- time-domain models
- frequency-domain models
- mixed-representation models

---

## File Formats

Use the following formats throughout preprocessing:

- `.npz` for time-domain and frequency-domain shard files
- `.parquet` for `source_info.parquet` and `sample_info.parquet`
- `.json` for `label_map.json` and split files later

---

## Sample Definition

A sample is:
- one fixed-length time-series window
- one primary label
- one source ID
- one row in `sample_info.parquet`

Use the following extraction rules:

- for **point annotations**, extract one fixed-length window centered on the annotation
- for **interval annotations**, extract fixed-length windows inside the labeled interval using a fixed stride

Rules:
- all samples in the same dataset version must use the same window length
- channel ordering must stay consistent
- annotation boundaries must not be crossed silently

---

## Data Shapes

Use these shapes consistently:

```python
raw signal:                  [T, C]
time-domain sample:          [C, W]
frequency-domain sample:     [C, F, 2]
time-domain shard:           [N, C, W]
frequency-domain shard:      [N, C, F, 2]
```

where:
- `T` = total time samples in the raw sequence
- `C` = number of channels
- `W` = fixed window length
- `N` = number of samples in the shard
- `F` = number of frequency bins used for that dataset version
- `2` = magnitude and phase

Store numeric arrays as `float32`.

---

## Processing Flow

Follow this order:

- read raw data and annotations
- build `source_info.parquet`
- normalize annotations into a consistent internal format
- extract fixed-length time-domain samples
- validate samples
- save time-domain shards in `time_domain_data/`
- build `sample_info.parquet`
- generate FFT-based frequency-domain samples from `time_domain_data/`
- save frequency-domain shards in `frequency_domain_data/`

---

## Validation Checks

Before finalizing the dataset, check:

- all source IDs are valid
- all sample IDs are valid and unique
- all sample shapes are consistent
- labels exist in `label_map.json`
- shard references in `sample_info.parquet` are valid
- every frequency-domain sample matches a time-domain sample

---

## Split Handling Note

Split generation will be added later.

When splits are created:
- grouping must happen at the source level
- samples from the same source must stay in the same split
- train and test sets must not share source IDs