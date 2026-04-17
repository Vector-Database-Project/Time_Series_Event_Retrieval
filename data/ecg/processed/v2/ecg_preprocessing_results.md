# ECG Preprocessing Results

## Brief

This note records the observed output of the leakage-aware balanced ECG preprocessing pipeline for the coarse `N/S/V` benchmark.

The goal of this note is to document:

- the retained coarse label space
- the final selected train and test records
- the inventory counts before within-split balancing
- the final balanced class counts after sampling
- the set of records left unused in this derived benchmark

---

## Retained coarse label space

The preprocessing pipeline retains only the endogenous beat groups:

- `N -> 0`
- `S -> 1`
- `V -> 2`

with the detailed annotation collapse:

- `N`: `N`, `L`, `R`, `e`, `j`
- `S`: `A`, `a`, `J`, `S`
- `V`: `V`, `E`

The following symbols are dropped from this benchmark:

- `F`, `/`, `f`, `Q`, `[`, `!`, `]`, `x`, `|`

---

## Fixed preprocessing settings

The run used the following fixed preprocessing parameters:

- sampling rate: `360 Hz`
- pre-annotation window: `180` samples
- post-annotation window: `180` samples
- total window length: `361` samples
- record shuffle seed: `42`
- train within-split sampling seed: `123`
- test within-split sampling seed: `456`

The balanced split targets were:

- train: `2250` samples per class
- test: `250` samples per class

---

## Selected train records

The following records were selected into the train split by seeded shuffled cumulative assignment:

`219, 209, 232, 214, 105, 234, 231, 201, 122, 207, 118, 230, 119, 107, 104, 220, 208`

### Train inventory before balancing

Before within-split downsampling, the selected train records contributed:

- `N = 29205`
- `S = 2250`
- `V = 2289`

This means:

- the train split hit the `S` target exactly at `2250`
- `N` had substantial excess and required heavy downsampling
- `V` was only slightly above the target and required only light downsampling

### Final train counts after balancing

After even-per-record random sampling within the selected train records, the final train split was:

- `N = 2250`
- `S = 2250`
- `V = 2250`

Total train samples:

- `6750`

---

## Selected test records

The following records were selected into the test split from the remaining record pool:

`123, 212, 109, 202, 200, 221, 117, 222`

### Test inventory before balancing

Before within-split downsampling, the selected test records contributed:

- `N = 16389`
- `S = 295`
- `V = 1282`

This means:

- the test split exceeded the `S` target modestly
- `N` again had substantial excess and required heavy downsampling
- `V` had comfortable excess and required downsampling

### Final test counts after balancing

After even-per-record random sampling within the selected test records, the final test split was:

- `N = 250`
- `S = 250`
- `V = 250`

Total test samples:

- `750`

---

## Unused records in this derived benchmark

The following records were not used in the final derived balanced benchmark:

`100, 101, 102, 103, 106, 108, 111, 112, 113, 114, 115, 116, 121, 124, 203, 205, 210, 213, 215, 217, 223, 228, 233`

These records remain outside the selected train and test record pools for this specific balanced `N/S/V` benchmark definition.

---

## What this run shows

This run confirms the intended behavior of the preprocessing pipeline:

- train and test were constructed at the record level, not by random beat-level splitting
- the retained benchmark uses only the coarse endogenous beat groups `N`, `S`, and `V`
- the `S` class successfully drove the split construction because it is the limiting retained class
- both splits had enough excess `N` and `V` samples to support exact balancing after record selection
- the final saved benchmark is exactly balanced within both train and test

---

## Reporting notes

For later reporting, the most important observations from this run are:

- the benchmark is leakage-aware at the record level
- the final train set contains `6750` samples and the final test set contains `750` samples
- class balancing was achieved by downsampling after record-aware split construction, not by random splitting first
- the selected train split required especially heavy reduction of `N`
- the selected train split was already nearly at the target for `V`
- the selected test split had enough surplus in all three classes to support exact balancing cleanly

A concise way to summarize this in the report is:

> We constructed a leakage-aware balanced coarse ECG benchmark by retaining only `N`, `S`, and `V` beats, assigning whole records to train and test using a seeded shuffled cumulative `S`-count rule, and then applying even-per-record random downsampling within each split. The final derived benchmark contained 2250 samples per class in train and 250 samples per class in test.
