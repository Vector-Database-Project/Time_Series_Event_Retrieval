from pathlib import Path
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm


class ECGBalancedSplitBuilder:
    """
    Builds a leakage-aware balanced N/S/V train-test split directly from raw ECG records.

    Pipeline summary:
    - discover valid raw ECG records
    - parse retained endogenous beat annotations only
    - inventory per-record beat counts for coarse classes N/S/V
    - shuffle record IDs with a fixed seed
    - assign whole records to train until cumulative S >= 2250
    - assign whole remaining records to test until cumulative S >= 250
    - within each split, downsample N/S/V evenly across contributing records
    - extract TD windows, convert to FD, save aligned artifacts and metadata
    """

    def __init__(self):
        self.repo_root = Path(__file__).resolve().parents[2]

        self.raw_data_dir = self.repo_root / "data" / "ecg" / "raw"
        self.processed_data_dir = self.repo_root / "data" / "ecg" / "processed" / "v2"

        self.train_dir = self.processed_data_dir / "train"
        self.test_dir = self.processed_data_dir / "test"

        self.train_td_dir = self.train_dir / "time_domain_data"
        self.train_fd_dir = self.train_dir / "frequency_domain_data"
        self.train_labels_dir = self.train_dir / "labels"
        self.train_metadata_dir = self.train_dir / "metadata"

        self.test_td_dir = self.test_dir / "time_domain_data"
        self.test_fd_dir = self.test_dir / "frequency_domain_data"
        self.test_labels_dir = self.test_dir / "labels"
        self.test_metadata_dir = self.test_dir / "metadata"

        self.valid_extensions = {".hea", ".dat", ".atr"}

        self.window_pre_annotation = 180
        self.window_post_annotation = 180
        self.sampling_rate = 360

        self.record_shuffle_seed = 42
        self.train_sampling_seed = 123
        self.test_sampling_seed = 456

        self.train_target_per_class = 2250
        self.test_target_per_class = 250

        self.coarse_label_map = {
            "N": 0,
            "S": 1,
            "V": 2,
        }

        self.detail_to_coarse = {
            "N": "N",
            "L": "N",
            "R": "N",
            "e": "N",
            "j": "N",
            "A": "S",
            "a": "S",
            "J": "S",
            "S": "S",
            "V": "V",
            "E": "V",
        }

        self.dropped_symbols = [
            "F", "/", "f", "Q", "[", "!", "]", "x", "|"
        ]

        self.registry = {}
        self.record_inventory = {}
        self.frequency_bins = None

    def ensure_output_dirs(self):
        dirs = [
            self.processed_data_dir,
            self.train_td_dir,
            self.train_fd_dir,
            self.train_labels_dir,
            self.train_metadata_dir,
            self.test_td_dir,
            self.test_fd_dir,
            self.test_labels_dir,
            self.test_metadata_dir,
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)

    def build_registry(self):
        for file_path in self.raw_data_dir.iterdir():
            if not file_path.is_file():
                continue
            if file_path.suffix not in self.valid_extensions:
                continue

            record_key = file_path.stem
            if record_key not in self.registry:
                self.registry[record_key] = {
                    "hea": None,
                    "dat": None,
                    "atr": None,
                }

            slot = file_path.suffix[1:]
            self.registry[record_key][slot] = file_path

    def get_valid_record_ids(self):
        valid_record_ids = []
        for record_id, entry in self.registry.items():
            if entry["hea"] is not None and entry["dat"] is not None and entry["atr"] is not None:
                valid_record_ids.append(record_id)
        return sorted(valid_record_ids)

    def get_signal_length(self, record_id):
        record_entry = self.registry[record_id]
        record_base_path = str(record_entry["hea"].with_suffix(""))
        header = wfdb.rdheader(record_base_path)

        if hasattr(header, "sig_len") and header.sig_len is not None:
            return header.sig_len

        signal, _ = wfdb.rdsamp(record_base_path)
        return signal.shape[0]

    def parse_retained_annotations(self, record_id):
        record_entry = self.registry.get(record_id)
        if record_entry is None:
            raise ValueError(f"Record '{record_id}' not found in registry.")

        if record_entry["hea"] is None or record_entry["dat"] is None or record_entry["atr"] is None:
            raise ValueError(f"Record '{record_id}' is missing required files.")

        record_base_path = str(record_entry["hea"].with_suffix(""))
        ann = wfdb.rdann(record_base_path, "atr")
        signal_length = self.get_signal_length(record_id)

        retained = []

        for center_sample, symbol in zip(ann.sample, ann.symbol):
            if symbol in {".", "·"}:
                symbol = "N"

            coarse_name = self.detail_to_coarse.get(symbol)
            if coarse_name is None:
                continue

            if center_sample - self.window_pre_annotation < 0:
                continue
            if center_sample + self.window_post_annotation >= signal_length:
                continue

            retained.append(
                {
                    "record_id": record_id,
                    "center_sample": int(center_sample),
                    "original_symbol": symbol,
                    "coarse_label_name": coarse_name,
                    "coarse_label_id": self.coarse_label_map[coarse_name],
                }
            )

        return retained

    def build_record_inventory(self):
        if not self.registry:
            self.build_registry()

        self.record_inventory = {}
        valid_record_ids = self.get_valid_record_ids()

        for record_id in tqdm(valid_record_ids, desc="Inventory pass", unit="record"):
            retained_annotations = self.parse_retained_annotations(record_id)

            counts = {"N": 0, "S": 0, "V": 0}
            for item in retained_annotations:
                counts[item["coarse_label_name"]] += 1

            self.record_inventory[record_id] = {
                "annotations": retained_annotations,
                "counts": counts,
            }

    def shuffled_record_ids(self):
        record_ids = list(self.record_inventory.keys())
        rng = np.random.default_rng(self.record_shuffle_seed)
        rng.shuffle(record_ids)
        return record_ids

    def cumulative_select_by_s(self, record_ids, target_s):
        selected = []
        cumulative_s = 0

        for record_id in record_ids:
            selected.append(record_id)
            cumulative_s += self.record_inventory[record_id]["counts"]["S"]
            if cumulative_s >= target_s:
                break

        if cumulative_s < target_s:
            raise RuntimeError(
                f"Could not reach target S count {target_s}. Only collected {cumulative_s}."
            )

        return selected

    def split_records(self):
        shuffled_ids = self.shuffled_record_ids()

        train_record_ids = self.cumulative_select_by_s(
            record_ids=shuffled_ids,
            target_s=self.train_target_per_class,
        )

        remaining_ids = [rid for rid in shuffled_ids if rid not in set(train_record_ids)]

        test_record_ids = self.cumulative_select_by_s(
            record_ids=remaining_ids,
            target_s=self.test_target_per_class,
        )

        train_summary = self.summarize_records(train_record_ids)
        test_summary = self.summarize_records(test_record_ids)

        for class_name in ["N", "S", "V"]:
            if train_summary[class_name] < self.train_target_per_class:
                raise RuntimeError(
                    f"Train split does not have enough {class_name} beats. "
                    f"Have {train_summary[class_name]}, need {self.train_target_per_class}."
                )
            if test_summary[class_name] < self.test_target_per_class:
                raise RuntimeError(
                    f"Test split does not have enough {class_name} beats. "
                    f"Have {test_summary[class_name]}, need {self.test_target_per_class}."
                )

        return train_record_ids, test_record_ids

    def summarize_records(self, record_ids):
        summary = {"N": 0, "S": 0, "V": 0}
        for record_id in record_ids:
            counts = self.record_inventory[record_id]["counts"]
            for class_name in summary:
                summary[class_name] += counts[class_name]
        return summary

    def compute_even_quotas(self, counts_by_record, target_count):
        counts_by_record = {k: int(v) for k, v in counts_by_record.items() if v > 0}

        if not counts_by_record:
            raise RuntimeError("No contributing records available for class allocation.")

        total_available = sum(counts_by_record.values())
        if total_available < target_count:
            raise RuntimeError(
                f"Not enough samples to allocate target {target_count}. Only {total_available} available."
            )

        quotas = {record_id: 0 for record_id in counts_by_record}
        active = set(counts_by_record.keys())
        remaining_target = int(target_count)

        while remaining_target > 0:
            if not active:
                raise RuntimeError("Ran out of active records during quota allocation.")

            ordered_active = sorted(active)
            base_quota = remaining_target // len(ordered_active)
            remainder = remaining_target % len(ordered_active)

            proposed = {}
            for idx, record_id in enumerate(ordered_active):
                proposed_quota = base_quota + (1 if idx < remainder else 0)
                proposed[record_id] = proposed_quota

            capped_records = []
            for record_id in ordered_active:
                capacity_left = counts_by_record[record_id] - quotas[record_id]
                if capacity_left <= proposed[record_id]:
                    quotas[record_id] += capacity_left
                    remaining_target -= capacity_left
                    capped_records.append(record_id)

            if capped_records:
                for record_id in capped_records:
                    active.remove(record_id)
                continue

            for record_id in ordered_active:
                quotas[record_id] += proposed[record_id]
            remaining_target = 0

        return quotas

    def select_balanced_annotations(self, record_ids, target_per_class, sampling_seed):
        selected_annotations = []

        class_to_record_annotations = {
            "N": defaultdict(list),
            "S": defaultdict(list),
            "V": defaultdict(list),
        }

        for record_id in record_ids:
            annotations = self.record_inventory[record_id]["annotations"]
            for ann in annotations:
                class_to_record_annotations[ann["coarse_label_name"]][record_id].append(ann)

        rng = np.random.default_rng(sampling_seed)

        for class_name in ["N", "S", "V"]:
            counts_by_record = {
                record_id: len(ann_list)
                for record_id, ann_list in class_to_record_annotations[class_name].items()
            }

            quotas = self.compute_even_quotas(
                counts_by_record=counts_by_record,
                target_count=target_per_class,
            )

            for record_id in sorted(quotas.keys()):
                quota = quotas[record_id]
                ann_list = class_to_record_annotations[class_name][record_id]
                if quota == 0:
                    continue

                if quota > len(ann_list):
                    raise RuntimeError(
                        f"Quota {quota} exceeds available {len(ann_list)} for record {record_id}, class {class_name}."
                    )

                chosen_idx = rng.choice(len(ann_list), size=quota, replace=False)
                for idx in chosen_idx:
                    selected_annotations.append(ann_list[int(idx)])

        rng.shuffle(selected_annotations)
        return selected_annotations

    def group_selected_annotations_by_record(self, selected_annotations):
        grouped = defaultdict(list)
        for ann in selected_annotations:
            grouped[ann["record_id"]].append(ann)
        return grouped

    def center_window_for_fd(self, td_window):
        return td_window - np.mean(td_window, axis=1, keepdims=True)

    def compute_fd_window(self, td_window):
        expected_window_length = self.window_pre_annotation + self.window_post_annotation + 1
        if self.frequency_bins is None:
            self.frequency_bins = np.fft.rfftfreq(expected_window_length, d=1 / self.sampling_rate)

        td_window_centered = self.center_window_for_fd(td_window)
        fft_vals = np.fft.rfft(td_window_centered, axis=1)
        magnitude = np.abs(fft_vals)
        phase = np.angle(fft_vals)
        return np.stack([magnitude, phase], axis=-1)

    def extract_split_arrays(self, selected_annotations, split_name):
        td_windows = []
        fd_windows = []
        labels = []
        metadata_rows = []

        grouped = self.group_selected_annotations_by_record(selected_annotations)
        expected_window_length = self.window_pre_annotation + self.window_post_annotation + 1

        for record_id in tqdm(sorted(grouped.keys()), desc=f"Extracting {split_name}", unit="record"):
            record_entry = self.registry[record_id]
            record_base_path = str(record_entry["hea"].with_suffix(""))
            signal, _ = wfdb.rdsamp(record_base_path)

            if signal.shape[1] != 2:
                raise RuntimeError(
                    f"Expected 2 channels for record {record_id}, got {signal.shape[1]}."
                )

            for ann in grouped[record_id]:
                center = ann["center_sample"]
                start = center - self.window_pre_annotation
                end = center + self.window_post_annotation + 1
                window = signal[start:end, :]

                if window.shape[0] != expected_window_length:
                    raise RuntimeError(
                        f"Unexpected window length for record {record_id} at sample {center}. "
                        f"Expected {expected_window_length}, got {window.shape[0]}."
                    )

                if not np.isfinite(window).all():
                    raise RuntimeError(
                        f"Encountered non-finite window in split {split_name} for record {record_id}, sample {center}."
                    )

                td_window = window.T.astype(np.float32)
                fd_window = self.compute_fd_window(td_window).astype(np.float32)

                td_windows.append(td_window)
                fd_windows.append(fd_window)
                labels.append(ann["coarse_label_id"])
                metadata_rows.append(
                    {
                        "record_id": ann["record_id"],
                        "center_sample": ann["center_sample"],
                        "original_symbol": ann["original_symbol"],
                        "coarse_label_name": ann["coarse_label_name"],
                        "coarse_label_id": ann["coarse_label_id"],
                        "split": split_name,
                    }
                )

        td_array = np.asarray(td_windows, dtype=np.float32)
        fd_array = np.asarray(fd_windows, dtype=np.float32)
        label_array = np.asarray(labels, dtype=np.int64)
        metadata_df = pd.DataFrame(metadata_rows)

        if len(td_array) != len(label_array):
            raise RuntimeError("Mismatch between TD windows and labels.")
        if len(fd_array) != len(label_array):
            raise RuntimeError("Mismatch between FD windows and labels.")
        if len(metadata_df) != len(label_array):
            raise RuntimeError("Mismatch between metadata rows and labels.")

        return td_array, fd_array, label_array, metadata_df

    def save_split(self, split_name, td_array, fd_array, label_array, metadata_df):
        if split_name == "train":
            td_dir = self.train_td_dir
            fd_dir = self.train_fd_dir
            labels_dir = self.train_labels_dir
            metadata_dir = self.train_metadata_dir
        elif split_name == "test":
            td_dir = self.test_td_dir
            fd_dir = self.test_fd_dir
            labels_dir = self.test_labels_dir
            metadata_dir = self.test_metadata_dir
        else:
            raise ValueError(f"Unsupported split name: {split_name}")

        np.savez(td_dir / "td_shard_000.npz", data=td_array)
        np.savez(fd_dir / "fd_shard_000.npz", data=fd_array)
        np.savez(labels_dir / "label_shard_000.npz", data=label_array)
        metadata_df.to_parquet(metadata_dir / "metadata_shard_000.parquet", index=False)

    def save_shared_outputs(self, train_record_ids, test_record_ids, train_selected, test_selected):
        with open(self.processed_data_dir / "label_map.json", "w") as f:
            json.dump(self.coarse_label_map, f, indent=2)

        if self.frequency_bins is None:
            expected_window_length = self.window_pre_annotation + self.window_post_annotation + 1
            self.frequency_bins = np.fft.rfftfreq(expected_window_length, d=1 / self.sampling_rate)
        np.savez(self.processed_data_dir / "frequency_bins.npz", data=self.frequency_bins)

        train_inventory_counts = self.summarize_records(train_record_ids)
        test_inventory_counts = self.summarize_records(test_record_ids)

        split_config = {
            "sampling_rate": self.sampling_rate,
            "window_pre_annotation": self.window_pre_annotation,
            "window_post_annotation": self.window_post_annotation,
            "window_length": self.window_pre_annotation + self.window_post_annotation + 1,
            "coarse_label_map": self.coarse_label_map,
            "detail_to_coarse": self.detail_to_coarse,
            "dropped_symbols": self.dropped_symbols,
            "record_shuffle_seed": self.record_shuffle_seed,
            "train_sampling_seed": self.train_sampling_seed,
            "test_sampling_seed": self.test_sampling_seed,
            "train_target_per_class": self.train_target_per_class,
            "test_target_per_class": self.test_target_per_class,
            "train_record_ids": train_record_ids,
            "test_record_ids": test_record_ids,
            "unused_record_ids": [
                rid for rid in self.record_inventory.keys()
                if rid not in set(train_record_ids) and rid not in set(test_record_ids)
            ],
            "train_inventory_counts": train_inventory_counts,
            "test_inventory_counts": test_inventory_counts,
            "train_final_counts": self.label_counts_from_array(train_selected),
            "test_final_counts": self.label_counts_from_array(test_selected),
        }

        with open(self.processed_data_dir / "split_config.json", "w") as f:
            json.dump(split_config, f, indent=2)

    def label_counts_from_array(self, label_array):
        inv_map = {v: k for k, v in self.coarse_label_map.items()}
        unique, counts = np.unique(label_array, return_counts=True)
        out = {"N": 0, "S": 0, "V": 0}
        for label_id, count in zip(unique, counts):
            out[inv_map[int(label_id)]] = int(count)
        return out

    def run(self):
        self.ensure_output_dirs()
        self.build_registry()
        self.build_record_inventory()

        train_record_ids, test_record_ids = self.split_records()

        train_selected_annotations = self.select_balanced_annotations(
            record_ids=train_record_ids,
            target_per_class=self.train_target_per_class,
            sampling_seed=self.train_sampling_seed,
        )
        test_selected_annotations = self.select_balanced_annotations(
            record_ids=test_record_ids,
            target_per_class=self.test_target_per_class,
            sampling_seed=self.test_sampling_seed,
        )

        train_td, train_fd, train_labels, train_metadata = self.extract_split_arrays(
            selected_annotations=train_selected_annotations,
            split_name="train",
        )
        test_td, test_fd, test_labels, test_metadata = self.extract_split_arrays(
            selected_annotations=test_selected_annotations,
            split_name="test",
        )

        self.save_split("train", train_td, train_fd, train_labels, train_metadata)
        self.save_split("test", test_td, test_fd, test_labels, test_metadata)
        self.save_shared_outputs(
            train_record_ids=train_record_ids,
            test_record_ids=test_record_ids,
            train_selected=train_labels,
            test_selected=test_labels,
        )

        print("Done.")
        print(f"Train records: {train_record_ids}")
        print(f"Test records: {test_record_ids}")
        print(f"Train counts: {self.label_counts_from_array(train_labels)}")
        print(f"Test counts: {self.label_counts_from_array(test_labels)}")


if __name__ == "__main__":
    builder = ECGBalancedSplitBuilder()
    builder.run()
