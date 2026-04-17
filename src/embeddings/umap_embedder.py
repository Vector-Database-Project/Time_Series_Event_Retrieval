from pathlib import Path
import json
from tqdm.auto import tqdm

import numpy as np
import umap

from data_extractor import DataExtractor


class UMAPEmbedder:
    def __init__(
        self,
        train_root,
        test_root,
        output_root,
        input_mode="time",
        n_components=32,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        low_memory=True,
        verbose=True,
        n_jobs=-1,
    ):
        self.train_root = Path(train_root)
        self.test_root = Path(test_root)
        self.output_root = Path(output_root)

        self.input_mode = input_mode
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.low_memory = low_memory
        self.verbose = verbose
        self.n_jobs = n_jobs

        valid_modes = {"time", "frequency", "mixed"}
        if self.input_mode not in valid_modes:
            raise ValueError(
                f"input_mode must be one of {valid_modes}, got '{self.input_mode}'."
            )

        self.train_extractor = DataExtractor(self.train_root)
        self.test_extractor = DataExtractor(self.test_root)

        self.model = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            low_memory=self.low_memory,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
        )

    def _discover_shard_indices(self, split_root):
        split_root = Path(split_root)
        label_dir = split_root / "labels"

        if not label_dir.exists():
            raise FileNotFoundError(f"Labels folder not found: {label_dir}")

        shard_files = sorted(label_dir.glob("label_shard_*.npz"))

        if not shard_files:
            raise FileNotFoundError(f"No label shards found in: {label_dir}")

        shard_indices = []
        for path in shard_files:
            shard_str = path.stem.split("_")[-1]
            shard_indices.append(int(shard_str))

        return shard_indices

    def _validate_X_y(self, X_shard, y_shard, split_name, shard_idx, source_name):
        if X_shard.shape[0] != y_shard.shape[0]:
            raise ValueError(
                f"Mismatch in {split_name} shard {shard_idx:03d} ({source_name}): "
                f"X has {X_shard.shape[0]} samples, y has {y_shard.shape[0]} labels."
            )

    def _load_single_mode_split(self, extractor, extract_fn_name, split_root, split_name):
        if not hasattr(extractor, extract_fn_name):
            raise AttributeError(
                f"{extractor.__class__.__name__} has no method '{extract_fn_name}'."
            )

        extract_fn = getattr(extractor, extract_fn_name)
        shard_indices = self._discover_shard_indices(split_root)

        X_parts = []
        y_parts = []

        for shard_idx in tqdm(
            shard_indices,
            total=len(shard_indices),
            desc=f"Loading {split_name} shards ({self.input_mode})",
            unit="shard",
        ):
            X_shard, y_shard = extract_fn(shard_idx)

            self._validate_X_y(
                X_shard=X_shard,
                y_shard=y_shard,
                split_name=split_name,
                shard_idx=shard_idx,
                source_name=extract_fn_name,
            )

            X_parts.append(X_shard)
            y_parts.append(y_shard)

        X = np.concatenate(X_parts, axis=0).astype(np.float32, copy=False)
        y = np.concatenate(y_parts, axis=0).astype(np.int64, copy=False)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Mismatch after concatenation for {split_name}: "
                f"X has {X.shape[0]} samples, y has {y.shape[0]} labels."
            )

        return X, y

    def _load_mixed_mode_split(self, extractor, split_root, split_name):
        if not hasattr(extractor, "TD_extractor"):
            raise AttributeError(
                f"{extractor.__class__.__name__} has no method 'TD_extractor'."
            )
        if not hasattr(extractor, "FD_extractor"):
            raise AttributeError(
                f"{extractor.__class__.__name__} has no method 'FD_extractor'."
            )

        td_extract_fn = getattr(extractor, "TD_extractor")
        fd_extract_fn = getattr(extractor, "FD_extractor")

        shard_indices = self._discover_shard_indices(split_root)

        X_parts = []
        y_parts = []

        for shard_idx in tqdm(
            shard_indices,
            total=len(shard_indices),
            desc=f"Loading {split_name} shards (mixed)",
            unit="shard",
        ):
            X_td, y_td = td_extract_fn(shard_idx)
            X_fd, y_fd = fd_extract_fn(shard_idx)

            self._validate_X_y(
                X_shard=X_td,
                y_shard=y_td,
                split_name=split_name,
                shard_idx=shard_idx,
                source_name="TD_extractor",
            )
            self._validate_X_y(
                X_shard=X_fd,
                y_shard=y_fd,
                split_name=split_name,
                shard_idx=shard_idx,
                source_name="FD_extractor",
            )

            if X_td.shape[0] != X_fd.shape[0]:
                raise ValueError(
                    f"Mixed-mode row mismatch in {split_name} shard {shard_idx:03d}: "
                    f"TD has {X_td.shape[0]} rows, FD has {X_fd.shape[0]} rows."
                )

            if not np.array_equal(y_td, y_fd):
                raise ValueError(
                    f"Mixed-mode alignment failure in {split_name} shard {shard_idx:03d}: "
                    f"TD labels and FD labels are not identical."
                )

            X_mixed = np.concatenate([X_td, X_fd], axis=1)

            X_parts.append(X_mixed)
            y_parts.append(y_td)

        X = np.concatenate(X_parts, axis=0).astype(np.float32, copy=False)
        y = np.concatenate(y_parts, axis=0).astype(np.int64, copy=False)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Mismatch after concatenation for {split_name}: "
                f"X has {X.shape[0]} samples, y has {y.shape[0]} labels."
            )

        return X, y

    def _load_split(self, extractor, split_root, split_name):
        if self.input_mode == "time":
            return self._load_single_mode_split(
                extractor=extractor,
                extract_fn_name="TD_extractor",
                split_root=split_root,
                split_name=split_name,
            )

        if self.input_mode == "frequency":
            return self._load_single_mode_split(
                extractor=extractor,
                extract_fn_name="FD_extractor",
                split_root=split_root,
                split_name=split_name,
            )

        if self.input_mode == "mixed":
            return self._load_mixed_mode_split(
                extractor=extractor,
                split_root=split_root,
                split_name=split_name,
            )

        raise ValueError(f"Unsupported input_mode: {self.input_mode}")

    def load_train_split(self):
        return self._load_split(
            extractor=self.train_extractor,
            split_root=self.train_root,
            split_name="train",
        )

    def load_test_split(self):
        return self._load_split(
            extractor=self.test_extractor,
            split_root=self.test_root,
            split_name="test",
        )

    def fit(self, X):
        self.model.fit(X)
        return self

    def transform(self, X):
        Z = self.model.transform(X)
        return np.asarray(Z, dtype=np.float32)

    def fit_transform(self, X):
        Z = self.model.fit_transform(X)
        return np.asarray(Z, dtype=np.float32)

    def _save_npz_array(self, path, array):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, data=array)

    def save_outputs(
        self,
        Z_train,
        Z_test,
        y_train,
        y_test,
        run_name,
        extra_config=None,
    ):
        run_root = self.output_root / run_name
        run_root.mkdir(parents=True, exist_ok=True)

        save_map = {
            "train_embeddings": (run_root / "train_embeddings.npz", Z_train),
            "test_embeddings": (run_root / "test_embeddings.npz", Z_test),
            "train_labels": (run_root / "train_labels.npz", y_train),
            "test_labels": (run_root / "test_labels.npz", y_test),
        }

        for _, (path, array) in tqdm(
            save_map.items(),
            total=len(save_map),
            desc="Saving artifacts",
            unit="file",
        ):
            self._save_npz_array(path, array)

        run_config = {
            "method": "UMAP",
            "supervised": False,
            "input_mode": self.input_mode,
            "n_components": int(self.n_components),
            "n_neighbors": int(self.n_neighbors),
            "min_dist": float(self.min_dist),
            "metric": self.metric,
            "low_memory": bool(self.low_memory),
            "verbose": bool(self.verbose),
            "n_jobs": int(self.n_jobs),
            "train_root": str(self.train_root),
            "test_root": str(self.test_root),
            "output_root": str(run_root),
        }

        if extra_config is not None:
            run_config.update(extra_config)

        with open(run_root / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=4)

        return run_root
