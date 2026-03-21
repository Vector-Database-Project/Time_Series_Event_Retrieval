from pathlib import Path
import json
import numpy as np
import wfdb
from tqdm import tqdm

class ECGDataCleaner:
    """
        Preprocesses ECG data with beat annotations, by chuking into symmetric windows around each beat.
    """
    def __init__(self):
        self.repo_root = Path(__file__).resolve().parents[2]

        self.raw_data_dir = (
            self.repo_root
            / "data"
            / "ecg"
            / "raw"
        )

        self.processed_data_dir = (
            self.repo_root
            / "data"
            / "ecg"
            / "processed"
            / "v1"
        )

        self.time_domain_dir = self.processed_data_dir / "time_domain_data"
        self.frequency_domain_dir = self.processed_data_dir / "frequency_domain_data"
        self.labels_dir = self.processed_data_dir / "labels"

        self.valid_extensions = {".hea", ".dat", ".atr"}

        self.registry = {}
        self.label_vocab = [
            "N", "L", "R", "A", "a", "J", "S", "V", "F",
            "[", "!", "]", "e", "j", "E", "/", "f", "x", "Q", "|"
        ]

        self.label_map = {label: idx for idx, label in enumerate(self.label_vocab)}

        # These parameters are specific to this ECG dataset. Adjust as needed for other datasets.
        self.window_pre_annotation = 180
        self.window_post_annotation = 180
        # citation details in readme.md file in the root of the repo
        self.sampling_rate = 360

        self.total_annotations_received = 0
        self.total_annotations_saved_td = 0
        self.total_annotations_saved_fd = 0

        self.label_ids_all = []
        self.td_windows_all = []
        self.fd_windows_all = []
        self.frequency_bins = None

    def build_registry(self):
        """
            Scans the raw data directory and builds a registry mapping record keys 
            to their corresponding .hea, .dat, and .atr files.
        """
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

    def parse_annotations(self, record_key):
        """
            Parses the annotation file for a given record key, filters out unwanted annotations, 
            and ensures that the remaining annotations have enough surrounding signal data for windowing.
        """
        record_entry = self.registry.get(record_key)

        if record_entry is None:
            raise ValueError(f"Record key '{record_key}' not found in registry.")

        if record_entry["hea"] is None or record_entry["dat"] is None or record_entry["atr"] is None:
            raise ValueError(f"Record '{record_key}' is missing required files.")
        
        record_base_path = str(record_entry["hea"].with_suffix(""))

        ann = wfdb.rdann(record_base_path, "atr")
        signal_length = self.get_signal_length(record_key)

        samples = list(ann.sample)
        symbols = list(ann.symbol)

        self.total_annotations_received += len(samples)

        filtered_samples = []
        filtered_symbols = []

        for sample, symbol in zip(samples, symbols):
            if symbol in {".","·"}:
                symbol = "N"

            if symbol in self.label_map:
                filtered_samples.append(sample)
                filtered_symbols.append(symbol)
        
        samples = filtered_samples
        symbols = filtered_symbols

        while samples and samples[0] - self.window_pre_annotation < 0:
            samples.pop(0)
            symbols.pop(0)

        while samples and samples[-1] + self.window_post_annotation >= signal_length:
            samples.pop()
            symbols.pop()

        return samples, symbols

    def get_signal_length(self, record_key):
        """
            Retrieves the length of the ECG signal for a given record key.
        """
        record_entry = self.registry[record_key]
        record_base_path = str(record_entry["hea"].with_suffix(""))

        header = wfdb.rdheader(record_base_path)

        if hasattr(header, "sig_len") and header.sig_len is not None:
            return header.sig_len

        signal, _ = wfdb.rdsamp(record_base_path)
        return signal.shape[0]

    def extract_time_domain_windows(self, record_key):
        """
            Extracts time-domain windows around each beat annotation for a given record key.
        """
        record_entry = self.registry[record_key]
        record_base_path = str(record_entry["hea"].with_suffix(""))
        signal, _ = wfdb.rdsamp(record_base_path)

        if signal.shape[1] != 2:
            raise RuntimeError(
                f"Expected 2 channels for record {record_key}, got {signal.shape[1]}."
            )

        samples, symbols = self.parse_annotations(record_key)

        expected_window_length = (
            self.window_pre_annotation + self.window_post_annotation + 1
        )

        for center, symbol in zip(samples, symbols):
            start = center - self.window_pre_annotation
            end = center + self.window_post_annotation + 1

            window = signal[start:end, :]

            if window.shape[0] != expected_window_length:
                raise RuntimeError(
                    f"Unexpected window length for record {record_key} at sample {center}. "
                    f"Expected {expected_window_length}, got {window.shape[0]}."
                    f"Filtering logic during parsing failed. Clean up the filetering logic."
                )

            if not np.isfinite(window).all():
                continue

            self.td_windows_all.append(window.T)
            self.label_ids_all.append(self.label_map[symbol])
            self.total_annotations_saved_td += 1

    def td_data_processing(self):
        """
            Processes all records in the registry to extract time-domain windows and their corresponding labels.
        """
        if not self.registry:
            self.build_registry()

        for record_key in tqdm(self.registry, desc="TD Processing", unit="record"):
            record_entry = self.registry[record_key]

            if (
                record_entry["hea"] is None 
                or record_entry["dat"] is None 
                or record_entry["atr"] is None):
                
                print(f"Skipping record {record_key} due to missing files.")
                continue
            try:
                self.extract_time_domain_windows(record_key)
            except Exception as e:
                print(f"Error processing record {record_key}: {e}")

    def center_window_for_fd(self, td_window):
        """
            Centers the time-domain window by removing the mean across each channel. For cleaner FFT.
        """
        return td_window - np.mean(td_window, axis=1, keepdims=True)

    def fd_data_processing(self):
        """
            Converts all extracted time-domain windows into frequency-domain windows.
            Stores frequency data in shape [N, C, F, 2], where the last dimension is
            [magnitude, phase].
        """
        if not self.td_windows_all:
            raise RuntimeError("No time-domain windows found. Run td_data_processing() first.")

        self.fd_windows_all = []

        expected_window_length = (
            self.window_pre_annotation + self.window_post_annotation + 1
        )
        self.frequency_bins = np.fft.rfftfreq(expected_window_length, d=1/self.sampling_rate)

        for td_window in tqdm(self.td_windows_all, desc="FD Processing", unit="window"):
            td_window_centered = self.center_window_for_fd(td_window)
            fft_vals = np.fft.rfft(td_window_centered, axis=1)   #[C, F]
            magnitude = np.abs(fft_vals)                #[C, F]
            phase = np.angle(fft_vals)                  #[C, F]

            fd_window = np.stack([magnitude, phase], axis=-1)  #[C, F, 2]

            self.fd_windows_all.append(fd_window)
            self.total_annotations_saved_fd += 1

    def save_processed_data(self):
        """
            Run and saves time-domain windows and frequency-domain windows 
            along with their corresponding labels to disk in a structured format.
        """
        self.td_windows_all = []
        self.fd_windows_all = []
        self.label_ids_all = []

        self.total_annotations_received = 0
        self.total_annotations_saved_td = 0
        self.total_annotations_saved_fd = 0

        self.td_data_processing()
        self.fd_data_processing()

        td_data_array = np.array(self.td_windows_all, dtype=np.float32)
        fd_data_array = np.array(self.fd_windows_all, dtype=np.float32)
        label_ids_array = np.array(self.label_ids_all, dtype=np.int64)

        if len(td_data_array) != len(label_ids_array):
            raise RuntimeError(
                f"Mismatch between number of time-domain windows and labels. "
                f"TD windows: {len(td_data_array)}, Labels: {len(label_ids_array)}. "
                f"Check the time domain data processing logic for consistency."
            )

        if len(fd_data_array) != len(label_ids_array):
            raise RuntimeError(
                f"Mismatch between number of frequency-domain windows and labels. "
                f"FD windows: {len(fd_data_array)}, Labels: {len(label_ids_array)}. "
                f"Check the frequency domain data processing logic for consistency."
            )
        
        np.savez(self.time_domain_dir / "td_shard.npz", data=td_data_array)
        np.savez(self.frequency_domain_dir / "fd_shard.npz", data=fd_data_array)
        np.savez(self.labels_dir / "label_shard.npz", data=label_ids_array)
        np.savez(self.processed_data_dir / "frequency_bins.npz", data=self.frequency_bins)

        with open(self.processed_data_dir / "label_map.json", "w") as f:
            json.dump(self.label_map, f, indent=2)


if __name__ == "__main__":
    cleaner = ECGDataCleaner()
    cleaner.save_processed_data()

    print(f"Total annotations received: {cleaner.total_annotations_received}")
    print(f"Total annotations saved (TD): {cleaner.total_annotations_saved_td}")
    print(f"Total annotations saved (FD): {cleaner.total_annotations_saved_fd}")