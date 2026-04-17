# Environment Setup

The project environment is defined in:

```text
environment/env.yaml
```

## Create the environment

Run this from the project root:

```bash
conda env create -f environment/env.yaml
```

## Activate the environment

```bash
conda activate vec-db
```

## Update the environment

If `env.yaml` changes later:

```bash
conda env update -f environment/env.yaml --prune
```

## Current packages

| Package | Use |
|---|---|
|`python` | 3.11 |
| `numpy` | numerical arrays and sample storage |
| `pandas` | metadata tables |
| `pyarrow` | Parquet support |
| `scipy` | signal processing |
| `scikit-learn` | grouped splits and baseline ML |
| `matplotlib` | plotting |
| `jupyterlab` | notebooks |
| `ipykernel` | Jupyter kernel support |
| `tqdm` | progress bars |
| `pyyaml` | config files |
| `wfdb` | waveform dataset reading |
| `pip`  | pip dependent packages install |
| `umap-learn` | package for unsupervised and supervised learning modules |

## Note

This environment will grow as the project evolves.  
Keep `env.yaml` updated whenever dependencies are added or changed.