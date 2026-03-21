import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


class TestTrainSplitter:
    def __init__(self, processed_root, output_root, test_size=0.2, random_state=42, max_shard_mb=40):
        """
        Initialize the splitter with input/output paths and split settings.

        Parameters
        ----------
        processed_root : str or Path
            Root directory of the processed dataset version containing:
            - time_domain_data/td_shard.npz
            - frequency_domain_data/fd_shard.npz
            - labels/label_shard.npz

        output_root : str or Path
            Root directory where the split dataset will be written with nested
            train/ and test/ folders.

        test_size : float
            Fraction of samples to place in the test split.

        random_state : int
            Random seed used by sklearn train_test_split.

        max_shard_mb : int or float
            Maximum allowed size per saved TD/FD shard in MB.
            Sharding is computed conservatively using raw numpy bytes.
        """
        self.processed_root = Path(processed_root)
        self.output_root = Path(output_root)
        self.test_size = test_size
        self.random_state = random_state
        self.max_shard_mb = max_shard_mb

    def _load_npz_array(self, path):
        """
        Load a single numpy array from a .npz file.

        Expects exactly one stored array inside the file and returns that array.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with np.load(path) as data:
            if len(data.files) != 1:
                raise ValueError(
                    f"Expected exactly one array in {path}, found {len(data.files)}: {data.files}"
                )
            array = data[data.files[0]]

        return array

    def load_full_data(self):
        """
        Load the full TD, FD, and label arrays from the processed dataset root.

        Expected input file layout
        --------------------------
        - time_domain_data/td_shard.npz
        - frequency_domain_data/fd_shard.npz
        - labels/label_shard.npz

        Returns
        -------
        td_data : np.ndarray
        fd_data : np.ndarray
        labels : np.ndarray
        """
        file_map = {
            "td_data": self.processed_root / "time_domain_data" / "td_shard.npz",
            "fd_data": self.processed_root / "frequency_domain_data" / "fd_shard.npz",
            "labels": self.processed_root / "labels" / "label_shard.npz",
        }

        loaded = {}
        for name, path in tqdm(
            file_map.items(),
            total=len(file_map),
            desc="Loading processed arrays",
            unit="file",
        ):
            loaded[name] = self._load_npz_array(path)

        td_data = loaded["td_data"]
        fd_data = loaded["fd_data"]
        labels = loaded["labels"].astype(np.int64, copy=False)

        if not (td_data.shape[0] == fd_data.shape[0] == labels.shape[0]):
            raise ValueError(
                "Mismatch in number of samples: "
                f"td_data={td_data.shape[0]}, "
                f"fd_data={fd_data.shape[0]}, "
                f"labels={labels.shape[0]}"
            )

        return td_data, fd_data, labels

    def split_data(self, labels):
        """
        Generate stratified train and test indices from the label array.

        Returns
        -------
        train_idx : np.ndarray
        test_idx : np.ndarray
        """
        indices = np.arange(labels.shape[0])

        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels,
            shuffle=True,
        )

        return np.sort(train_idx), np.sort(test_idx)

    def make_output_dirs(self):
        """
        Create the nested train/test folder structure used by the split dataset.
        """
        for split_name in ["train", "test"]:
            (self.output_root / split_name / "time_domain_data").mkdir(parents=True, exist_ok=True)
            (self.output_root / split_name / "frequency_domain_data").mkdir(parents=True, exist_ok=True)
            (self.output_root / split_name / "labels").mkdir(parents=True, exist_ok=True)

    def _save_npz_array(self, path, array):
        """
        Save a single numpy array to a compressed .npz file, creating parent directories if needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, data=array)

    def _samples_per_shard(self, td_array, fd_array, labels_array):
        """
        Compute the maximum number of aligned samples that can be saved in one shard
        without exceeding the configured shard size limit.

        The same sample boundaries are used for TD, FD, and labels, so the shard size
        is determined by the most restrictive per-sample byte cost among the arrays.
        """
        max_bytes = int(self.max_shard_mb * 1024 * 1024)

        td_bytes_per_sample = td_array[0].nbytes
        fd_bytes_per_sample = fd_array[0].nbytes
        label_bytes_per_sample = labels_array[0].nbytes

        max_bytes_per_sample = max(td_bytes_per_sample, fd_bytes_per_sample, label_bytes_per_sample)

        if max_bytes_per_sample <= 0:
            raise ValueError("Computed bytes per sample is non-positive, cannot determine shard size.")

        samples_per_shard = max_bytes // max_bytes_per_sample

        if samples_per_shard < 1:
            raise ValueError(
                "A single sample exceeds the configured shard size limit. "
                f"Single-sample bytes={max_bytes_per_sample}, max shard bytes={max_bytes}."
            )

        return int(samples_per_shard)

    def _save_split_in_shards(self, td_array, fd_array, labels_array, split_name):
        """
        Save one split into multiple aligned TD, FD, and label shards.

        Output file naming
        ------------------
        - train/time_domain_data/td_shard_000.npz
        - train/frequency_domain_data/fd_shard_000.npz
        - train/labels/label_shard_000.npz
        - and similarly for test/
        """
        if not (td_array.shape[0] == fd_array.shape[0] == labels_array.shape[0]):
            raise ValueError(
                f"Split arrays are misaligned for {split_name}: "
                f"td={td_array.shape[0]}, fd={fd_array.shape[0]}, labels={labels_array.shape[0]}"
            )

        num_samples = td_array.shape[0]
        samples_per_shard = self._samples_per_shard(td_array, fd_array, labels_array)
        num_shards = int(np.ceil(num_samples / samples_per_shard))

        shard_ranges = [
            (start, min(start + samples_per_shard, num_samples))
            for start in range(0, num_samples, samples_per_shard)
        ]

        for shard_idx, (start, end) in enumerate(
            tqdm(
                shard_ranges,
                total=len(shard_ranges),
                desc=f"Saving {split_name} shards",
                unit="shard",
            )
        ):
            td_shard = td_array[start:end]
            fd_shard = fd_array[start:end]
            label_shard = labels_array[start:end]

            td_path = self.output_root / split_name / "time_domain_data" / f"td_shard_{shard_idx:03d}.npz"
            fd_path = self.output_root / split_name / "frequency_domain_data" / f"fd_shard_{shard_idx:03d}.npz"
            label_path = self.output_root / split_name / "labels" / f"label_shard_{shard_idx:03d}.npz"

            self._save_npz_array(td_path, td_shard)
            self._save_npz_array(fd_path, fd_shard)
            self._save_npz_array(label_path, label_shard)

        return {
            "num_samples": int(num_samples),
            "samples_per_shard": int(samples_per_shard),
            "num_shards": int(num_shards),
        }

    def split_and_save(self):
        """
        Run the full split pipeline.

        Steps
        -----
        1. Load full TD, FD, and label arrays
        2. Generate stratified train/test indices from labels
        3. Slice TD, FD, and labels consistently
        4. Save train and test outputs into nested folders with size-limited shards
        5. Write split_config.json to the output root
        """
        td_data, fd_data, labels = self.load_full_data()
        train_idx, test_idx = self.split_data(labels)

        td_train, td_test = td_data[train_idx], td_data[test_idx]
        fd_train, fd_test = fd_data[train_idx], fd_data[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        self.make_output_dirs()

        train_stats = self._save_split_in_shards(td_train, fd_train, y_train, split_name="train")
        test_stats = self._save_split_in_shards(td_test, fd_test, y_test, split_name="test")

        split_config = {
            "source_processed_root": str(self.processed_root),
            "output_root": str(self.output_root),
            "split_type": "stratified_train_test",
            "test_size": self.test_size,
            "random_state": self.random_state,
            "max_shard_mb": self.max_shard_mb,
            "num_total_samples": int(labels.shape[0]),
            "num_train_samples": int(y_train.shape[0]),
            "num_test_samples": int(y_test.shape[0]),
            "train_shards": train_stats,
            "test_shards": test_stats,
        }

        config_path = self.output_root / "split_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(split_config, f, indent=4)