# Data Processing Pipeline for Time-Series Event Retrieval

## Brief

This document describes the **currently implemented** data processing pipeline for the ECG dataset.

The goal of this stage is simple:

- convert raw ECG records into fixed-length event windows
- build aligned time-domain and frequency-domain representations
- save them in a consistent processed format
- create a derived train/test split version for downstream embedding and retrieval experiments

It uses a **stratified random train/test split on labels** after the full processed dataset is created.

---

## Current Scope

The implementation currently covers:

- raw ECG record discovery from `raw/`
- annotation parsing and filtering
- fixed-length time-domain window extraction
- frequency-domain conversion from the extracted time-domain windows
- saving the processed dataset to `processed/v1/`
- building a derived train/test split dataset at `processed/v1tts/`

This document reflects the implemented code, not the earlier broader design notes.

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
      v1/
        time_domain_data/
          td_shard.npz
        frequency_domain_data/
          fd_shard.npz
        labels/
          label_shard.npz

        label_map.json
        frequency_bins.npz

      v1tts/
        split_config.json
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
```

## MIT-BIH ECG chunking policy

### Time-domain preprocessing note

For the MIT-BIH preprocessing stage, beats are extracted using a fixed **beat-centered symmetric window** with:

- `pre_samples = 180`
- `post_samples = 180`

MIT-BIH signals are digitized at **360 Hz per channel**, and the beat annotations were realigned so that they **generally appear at the R-wave peak**. This makes annotation-centered extraction appropriate for waveform analysis and averaging. The same dataset documentation states that the analog signals were filtered with an approximate **0.1 to 100 Hz** passband before digitization. 
Reference("https://physionet.org/physiobank/database/html/mitdbdir/intro.htm")

This choice produces a total chunk length of:

- `N = 180 + 180 + 1 = 361 samples`
- `361 / 360 ≈ 1.003 s`

A roughly 1-second symmetric window was chosen intentionally. Standard ECG timing references place the **PR interval** around **120 to 200 ms**, the **QRS duration** below about **120 ms**, and commonly cited textbook **QTc** ranges below about **460 ms**. A ±500 ms window therefore comfortably captures the full local beat morphology around the annotated R-peak rather than only the QRS complex. 
Reference("https://physionet.org/physiobank/database/html/mitdbdir/intro.htm")

This window length was also chosen because the goal is **not** to isolate a beat completely from its temporal neighborhood. MIT-BIH contains fast and irregular episodes, including ventricular tachycardia around **174 to 177 bpm** and other arrhythmic episodes up to **189 bpm** in the record notes. A 1-second beat-centered window therefore intentionally preserves neighboring-beat transient effects when rhythms are fast, which is desirable for this retrieval setting.
Reference1("https://www.nottingham.ac.uk/nursing/practice/resources/cardiology/function/normal_duration.php")
Reference2("https://litfl.com/p-wave-ecg-library/")

### Frequency-domain preprocessing note

The saved **time-domain (TD)** representation keeps the raw ECG window unchanged. For the **frequency-domain (FD)** representation, each channel is **mean-centered per window before FFT**. This removes the constant offset and reduces the dominance of the **0 Hz / DC component** in the spectrum. This is consistent with standard spectral-analysis practice: SciPy’s `periodogram` and `welch` both use segment-level **constant detrending** as the standard detrending option for spectral estimation.
Reference1("https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html")
Reference2("https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html")

With `N = 361` and `fs = 360 Hz`, the one-sided FFT uses:

- `F = 181` bins
- frequency resolution `Δf = fs / N = 360 / 361 ≈ 0.997 Hz/bin`

This gives an interpretable frequency axis across all extracted ECG windows.