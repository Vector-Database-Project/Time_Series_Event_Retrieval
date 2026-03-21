from pathlib import Path
import numpy as np


class DataExtractor:
    def __init__(self, processed_root):
        """
            Initialize the extractor with the root directory of one processed dataset version.
            The root is converted to a Path object so folder scaffolding works cleanly.
        """
        self.processed_root = Path(processed_root)

    def load_labels(self, shard_idx):
        """
            Load one label shard from the processed labels folder using the given shard index.
            The shard index can be passed as an int or numeric string and is normalized to 3 digits.
            Returns the label array as int64.
        """
        shard_idx = int(shard_idx)
        label_path = self.processed_root / "labels" / f"label_shard_{shard_idx:03d}.npz"

        if not label_path.exists():
            raise FileNotFoundError(f"Label shard not found: {label_path}.")

        with np.load(label_path) as data:
            if len(data.files) != 1:
                raise ValueError(
                    f"Expected exactly one array in {label_path}, found {len(data.files)}: {data.files}"
                )

            y = data[data.files[0]]

        return y.astype(np.int64, copy=False)

    def build_matrix(self, data_array):
        """
            Build a 2D feature matrix from a loaded shard array by flattening each sample.
            Converts input of shape [N, ...] into shape [N, D] so it matches standard ML APIs.
            Returns the matrix as float32.
        """
        if data_array is None:
            raise ValueError("Input data_array is None.")

        if data_array.ndim < 2:
            raise ValueError("Input data_array must have at least 2 dimensions: [N, ...].")

        num_samples = data_array.shape[0]
        X = data_array.reshape(num_samples, -1)

        return X.astype(np.float32, copy=False)

    def pass_data(self, path, shard_idx):
        """
            Load a shard from the given path and build a feature matrix from it of the form [N, D].
            Also loads the matching label shard.
            Returns X, y.
        """
        if not path.exists():
            raise FileNotFoundError(f"Data shard not found: {path}.")

        with np.load(path) as data:
            if len(data.files) != 1:
                raise ValueError(
                    f"Expected exactly one array in {path}, found {len(data.files)}: {data.files}"
                )

            data_array = data[data.files[0]]

        X = self.build_matrix(data_array)
        y = self.load_labels(shard_idx)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Mismatch between samples ({X.shape[0]}) and labels ({y.shape[0]}) "
                f"for shard {shard_idx:03d}."
            )

        return X, y

    def TD_extractor(self, shard_idx):
        """
            Call pass_data with the path to the time-domain data shard corresponding to the given shard index.
            The shard index can be passed as an int or numeric string and is normalized to 3 digits.
            Returns X, y for the time-domain data.
        """
        shard_idx = int(shard_idx)
        td_path = self.processed_root / "time_domain_data" / f"td_shard_{shard_idx:03d}.npz"

        return self.pass_data(td_path, shard_idx)

    def FD_extractor(self, shard_idx):
        """
            Call pass_data with the path to the frequency-domain data shard corresponding to the given shard index.
            The shard index can be passed as an int or numeric string and is normalized to 3 digits.
            Returns X, y for the frequency-domain data.
        """
        shard_idx = int(shard_idx)
        fd_path = self.processed_root / "frequency_domain_data" / f"fd_shard_{shard_idx:03d}.npz"

        return self.pass_data(fd_path, shard_idx)