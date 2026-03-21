from pathlib import Path
import json
from tqdm.auto import tqdm

import numpy as np
from sklearn.random_projection import GaussianRandomProjection

from data_extractor import DataExtractor


class GaussianRandomProjectionEmbedder:
    def __init__(
        self,
        train_root,
        test_root,
        output_root,
        n_components=64,
        random_state=42,
    ):
        # Root folders for the already-created train/test splits
        self.train_root = Path(train_root)
        self.test_root = Path(test_root)
        self.output_root = Path(output_root)

        # Gaussian RP config
        self.n_components = n_components
        self.random_state = random_state

        # One extractor per split root
        # These are used later to load TD or FD shards
        self.train_extractor = DataExtractor(self.train_root)
        self.test_extractor = DataExtractor(self.test_root)

        # Sklearn projection model
        self.model = GaussianRandomProjection(
            n_components=self.n_components,
            random_state=self.random_state,
        )

    def _discover_shard_indices(self, split_root):
        """
        Find all shard indices available in a split.

        We use the labels folder as the source of truth and collect shard ids
        from files like:
            label_shard_000.npz
            label_shard_001.npz
            ...
        """
        split_root = Path(split_root)
        label_dir = split_root / "labels"

        # Make sure the expected labels folder exists
        if not label_dir.exists():
            raise FileNotFoundError(f"Labels folder not found: {label_dir}")

        # Collect all label shard files in sorted order
        shard_files = sorted(label_dir.glob("label_shard_*.npz"))

        # Fail early if the split is empty or path is wrong
        if not shard_files:
            raise FileNotFoundError(f"No label shards found in: {label_dir}")

        # Extract the numeric shard id from each filename
        # Example: label_shard_003.npz -> 3
        shard_indices = []
        for path in shard_files:
            stem = path.stem
            shard_str = stem.split("_")[-1]
            shard_indices.append(int(shard_str))

        return shard_indices

    def _load_split(self, extractor, extract_fn_name, split_root, split_name):
        """
        Load all shards for one split and concatenate them into full X and y.

        Parameters
        ----------
        extractor : DataExtractor
            Extractor tied to either the train split or test split.
        extract_fn_name : str
            Name of the extractor method to call.
            Example:
                "TD_extractor"
                "FD_extractor"
        split_root : Path or str
            Root folder of the split.
        split_name : str
            Just used for progress-bar text and error messages.
        """
        # Confirm the requested extractor method actually exists
        if not hasattr(extractor, extract_fn_name):
            raise AttributeError(
                f"{extractor.__class__.__name__} has no method '{extract_fn_name}'."
            )

        # Get the actual extractor function from the object
        extract_fn = getattr(extractor, extract_fn_name)

        # Discover which shard ids exist in this split
        shard_indices = self._discover_shard_indices(split_root)

        X_parts = []
        y_parts = []

        # Load one shard at a time and collect them
        for shard_idx in tqdm(
            shard_indices,
            total=len(shard_indices),
            desc=f"Loading {split_name} shards",
            unit="shard",
        ):
            X_shard, y_shard = extract_fn(shard_idx)

            # Safety check, each row in X must have one matching label
            if X_shard.shape[0] != y_shard.shape[0]:
                raise ValueError(
                    f"Mismatch in {split_name} shard {shard_idx:03d}: "
                    f"X has {X_shard.shape[0]} samples, y has {y_shard.shape[0]} labels."
                )

            X_parts.append(X_shard)
            y_parts.append(y_shard)

        # Merge all shard-level arrays into one full split matrix/vector
        X = np.concatenate(X_parts, axis=0).astype(np.float32, copy=False)
        y = np.concatenate(y_parts, axis=0).astype(np.int64, copy=False)

        # Final safety check after concatenation
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Mismatch after concatenation for {split_name}: "
                f"X has {X.shape[0]} samples, y has {y.shape[0]} labels."
            )

        return X, y

    def load_train_split(self, extract_fn_name):
        """
        Convenience wrapper to load the full training split.

        Example:
            load_train_split("TD_extractor")
            load_train_split("FD_extractor")
        """
        return self._load_split(
            extractor=self.train_extractor,
            extract_fn_name=extract_fn_name,
            split_root=self.train_root,
            split_name="train",
        )

    def load_test_split(self, extract_fn_name):
        """
        Convenience wrapper to load the full test split.

        Example:
            load_test_split("TD_extractor")
            load_test_split("FD_extractor")
        """
        return self._load_split(
            extractor=self.test_extractor,
            extract_fn_name=extract_fn_name,
            split_root=self.test_root,
            split_name="test",
        )

    def fit(self, X):
        """
        Fit the Gaussian RP model on training data.

        For GaussianRandomProjection, this mainly fixes the random projection
        matrix using the input feature dimension and random_state.
        """
        self.model.fit(X)
        return self

    def transform(self, X):
        """
        Project input features into the lower-dimensional embedding space
        using the already-fitted projection matrix.
        """
        Z = self.model.transform(X)
        return np.asarray(Z, dtype=np.float32)

    def fit_transform(self, X):
        """
        Fit the projection model and immediately transform the same data.

        This is typically used for the training split.
        """
        Z = self.model.fit_transform(X)
        return np.asarray(Z, dtype=np.float32)

    def _save_npz_array(self, path, array):
        """
        Save one array as a compressed .npz file.

        The array is stored under the default key: 'data'
        """
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
        """
        Save embeddings, labels, and run metadata for one experiment run.

        Saved artifacts:
        - train_embeddings.npz
        - test_embeddings.npz
        - train_labels.npz
        - test_labels.npz
        - projection_components.npz (if available)
        - run_config.json
        """
        # Each run gets its own output folder
        run_root = self.output_root / run_name
        run_root.mkdir(parents=True, exist_ok=True)

        # Map output names to the arrays that should be written
        save_map = {
            "train_embeddings": (run_root / "train_embeddings.npz", Z_train),
            "test_embeddings": (run_root / "test_embeddings.npz", Z_test),
            "train_labels": (run_root / "train_labels.npz", y_train),
            "test_labels": (run_root / "test_labels.npz", y_test),
        }

        # Save the learned/random projection matrix too, if sklearn exposes it
        if hasattr(self.model, "components_"):
            save_map["projection_components"] = (
                run_root / "projection_components.npz",
                self.model.components_,
            )

        # Save all array artifacts with a small progress bar
        for _, (path, array) in tqdm(
            save_map.items(),
            total=len(save_map),
            desc="Saving artifacts",
            unit="file",
        ):
            self._save_npz_array(path, array)

        # Save run configuration for reproducibility
        run_config = {
            "method": "GaussianRandomProjection",
            "n_components": int(self.n_components),
            "random_state": int(self.random_state),
            "train_root": str(self.train_root),
            "test_root": str(self.test_root),
            "output_root": str(run_root),
        }

        # Allow the runner to inject extra metadata like representation name
        if extra_config is not None:
            run_config.update(extra_config)

        with open(run_root / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=4)

        return run_root