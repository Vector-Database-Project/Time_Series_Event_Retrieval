from pathlib import Path
import json
from tqdm.auto import tqdm

import numpy as np
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler

from data_extractor import DataExtractor


class NeighborhoodComponentsAnalysisEmbedder:
    def __init__(
        self,
        train_root,
        test_root,
        output_root,
        input_mode="time",
        nca_fit_samples=60000,
        n_components=32,
        init="identity",
        max_iter=10,
        tol=1e-5,
        random_state=42,
        verbose=1,
    ):
        """
        Embedding pipeline for Neighborhood Components Analysis (NCA)
        using subset-fit / full-transform, with no PCA.

        Parameters
        ----------
        train_root : str or Path
            Root folder for the processed training split.
        test_root : str or Path
            Root folder for the processed test split.
        output_root : str or Path
            Root folder where run artifacts will be saved.
        input_mode : {"time", "frequency", "mixed"}
            Which representation to load:
            - "time": TD only
            - "frequency": FD only
            - "mixed": TD + FD concatenated column-wise
        nca_fit_samples : int
            Number of training samples used to fit NCA.
            After fitting, the learned transform is applied to full train/test.
        n_components : int
            Final NCA embedding dimension.
        init : {"auto", "pca", "lda", "identity", "random"} or ndarray
            Initialization passed to sklearn NCA.
        max_iter : int
            Maximum number of optimization iterations.
        tol : float
            Optimization stopping tolerance.
        random_state : int
            Random seed for reproducibility.
        verbose : int
            Passed to sklearn NCA.
        """
        self.train_root = Path(train_root)
        self.test_root = Path(test_root)
        self.output_root = Path(output_root)

        self.input_mode = input_mode
        self.nca_fit_samples = nca_fit_samples
        self.n_components = n_components
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

        valid_modes = {"time", "frequency", "mixed"}
        if self.input_mode not in valid_modes:
            raise ValueError(
                f"input_mode must be one of {valid_modes}, got '{self.input_mode}'."
            )

        self.train_extractor = DataExtractor(self.train_root)
        self.test_extractor = DataExtractor(self.test_root)

        self.scaler = StandardScaler()
        self.model = NeighborhoodComponentsAnalysis(
            n_components=self.n_components,
            init=self.init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            verbose=self.verbose,
        )

    def _discover_shard_indices(self, split_root):
        """
        Find all shard indices available in a split using the labels folder
        as the source of truth.
        """
        split_root = Path(split_root)
        label_dir = split_root / "labels"

        if not label_dir.exists():
            raise FileNotFoundError(f"Labels folder not found: {label_dir}")

        shard_files = sorted(label_dir.glob("label_shard_*.npz"))

        if not shard_files:
            raise FileNotFoundError(f"No label shards found in: {label_dir}")

        shard_indices = []
        for path in shard_files:
            stem = path.stem
            shard_str = stem.split("_")[-1]
            shard_indices.append(int(shard_str))

        return shard_indices

    def _validate_X_y(self, X_shard, y_shard, split_name, shard_idx, source_name):
        """
        Basic alignment check for one feature shard and its labels.
        """
        if X_shard.shape[0] != y_shard.shape[0]:
            raise ValueError(
                f"Mismatch in {split_name} shard {shard_idx:03d} ({source_name}): "
                f"X has {X_shard.shape[0]} samples, y has {y_shard.shape[0]} labels."
            )

    def _load_single_mode_split(self, extractor, extract_fn_name, split_root, split_name):
        """
        Load all shards for one split using a single extractor path:
        - TD_extractor for time mode
        - FD_extractor for frequency mode
        """
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
        """
        Load all shards for one split in mixed mode.

        For each shard index:
        - load TD shard
        - load FD shard
        - confirm row counts match
        - confirm labels match exactly
        - concatenate TD and FD feature matrices column-wise
        """
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

            if y_td.shape[0] != y_fd.shape[0]:
                raise ValueError(
                    f"Mixed-mode label length mismatch in {split_name} shard {shard_idx:03d}: "
                    f"TD labels have length {y_td.shape[0]}, FD labels have length {y_fd.shape[0]}."
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
        """
        Dispatch split loading based on input mode.
        """
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

    def _make_subset_indices(self, y_train):
        """
        Choose a reproducible random subset of training indices for NCA fitting.
        """
        n_train = y_train.shape[0]
        n_fit = min(self.nca_fit_samples, n_train)

        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(n_train, size=n_fit, replace=False)
        idx.sort()
        return idx

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def transform(self, X):
        Z = self.model.transform(X)
        return np.asarray(Z, dtype=np.float32)

    def fit_transform(self, X, y):
        Z = self.model.fit_transform(X, y)
        return np.asarray(Z, dtype=np.float32)

    def standardize_train_test(self, X_train, X_test):
        """
        Fit scaler on train and transform both splits.

        Returns float32 arrays to reduce RAM usage.
        """
        X_train_scaled = self.scaler.fit_transform(X_train).astype(np.float32, copy=False)
        X_test_scaled = self.scaler.transform(X_test).astype(np.float32, copy=False)
        return X_train_scaled, X_test_scaled

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
        split_identifier="v1tts",
        extra_config=None,
    ):
        """
        Save embeddings, labels, NCA components, and run metadata.
        """
        run_root = self.output_root / run_name
        run_root.mkdir(parents=True, exist_ok=True)

        save_map = {
            "train_embeddings": (run_root / "train_embeddings.npz", Z_train),
            "test_embeddings": (run_root / "test_embeddings.npz", Z_test),
            "train_labels": (run_root / "train_labels.npz", y_train),
            "test_labels": (run_root / "test_labels.npz", y_test),
        }

        if hasattr(self.model, "components_"):
            save_map["nca_components"] = (
                run_root / "nca_components.npz",
                np.asarray(self.model.components_, dtype=np.float32),
            )

        for _, (path, array) in tqdm(
            save_map.items(),
            total=len(save_map),
            desc="Saving artifacts",
            unit="file",
        ):
            self._save_npz_array(path, array)

        run_config = {
            "method": "NeighborhoodComponentsAnalysis",
            "input_mode": self.input_mode,
            "nca_fit_samples": int(self.nca_fit_samples),
            "n_components": int(self.n_components),
            "init": self.init if isinstance(self.init, str) else "array_init",
            "max_iter": int(self.max_iter),
            "tol": float(self.tol),
            "random_state": int(self.random_state),
            "verbose": int(self.verbose),
            "train_root": str(self.train_root),
            "test_root": str(self.test_root),
            "output_root": str(run_root),
            "split_identifier": split_identifier,
            "fit_strategy": "fit_nca_on_train_subset_transform_full_train_test",
        }

        if extra_config is not None:
            run_config.update(extra_config)

        with open(run_root / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=4)

        return run_root

    def run(
        self,
        run_name=None,
        split_identifier="v1tts",
        extra_config=None,
    ):
        """
        End-to-end execution for one NCA run.

        Flow:
        1. load full train split
        2. load full test split
        3. standardize on full train, transform full test
        4. choose a train subset for NCA fitting
        5. fit NCA on the standardized train subset only
        6. transform full train and full test using the learned NCA transform
        7. save outputs
        """
        print(f"Loading training split in '{self.input_mode}' mode...")
        X_train, y_train = self.load_train_split()

        print(f"Loading test split in '{self.input_mode}' mode...")
        X_test, y_test = self.load_test_split()

        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError(
                "Feature dimension mismatch between train and test after loading: "
                f"train has {X_train.shape[1]} columns, "
                f"test has {X_test.shape[1]} columns."
            )

        print("Raw shapes:")
        print("X_train:", X_train.shape, "y_train:", y_train.shape)
        print("X_test: ", X_test.shape, "y_test: ", y_test.shape)

        print("Standardizing using training split...")
        X_train_scaled, X_test_scaled = self.standardize_train_test(X_train, X_test)

        if self.n_components > X_train_scaled.shape[1]:
            raise ValueError(
                f"n_components={self.n_components} exceeds input feature dimension "
                f"{X_train_scaled.shape[1]}."
            )

        subset_idx = self._make_subset_indices(y_train)
        X_train_fit = X_train_scaled[subset_idx]
        y_train_fit = y_train[subset_idx]

        print(f"Fitting NCA on a training subset of size {len(subset_idx)}...")
        print("NCA fit subset shapes:")
        print("X_train_fit:", X_train_fit.shape, "y_train_fit:", y_train_fit.shape)

        self.fit(X_train_fit, y_train_fit)

        print("Transforming full training split...")
        Z_train = self.transform(X_train_scaled)

        print("Transforming full test split...")
        Z_test = self.transform(X_test_scaled)

        if run_name is None:
            run_name = (
                f"nca_{self.input_mode}_"
                f"fit{self.nca_fit_samples}_"
                f"dim{self.n_components}"
            )

        run_root = self.save_outputs(
            Z_train=Z_train,
            Z_test=Z_test,
            y_train=y_train,
            y_test=y_test,
            run_name=run_name,
            split_identifier=split_identifier,
            extra_config=extra_config,
        )

        print("Run complete.")
        print(f"Saved outputs to: {run_root}")

        return {
            "X_train_shape": X_train.shape,
            "X_test_shape": X_test.shape,
            "X_train_scaled_shape": X_train_scaled.shape,
            "X_test_scaled_shape": X_test_scaled.shape,
            "X_train_fit_shape": X_train_fit.shape,
            "Z_train_shape": Z_train.shape,
            "Z_test_shape": Z_test.shape,
            "run_root": run_root,
        }
    
    
