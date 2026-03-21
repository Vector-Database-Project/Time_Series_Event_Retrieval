from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wfdb


LABEL_VOCAB = [
    "N", "L", "R", "A", "a", "J", "S", "V", "F",
    "[", "!", "]", "e", "j", "E", "/", "f", "x", "Q", "|"
]
LABEL_MAP = {label: idx for idx, label in enumerate(LABEL_VOCAB)}

WINDOW_PRE_ANNOTATION = 180
WINDOW_POST_ANNOTATION = 180
SAMPLING_RATE = 360
EXPECTED_WINDOW_LENGTH = WINDOW_PRE_ANNOTATION + WINDOW_POST_ANNOTATION + 1


def resolve_repo_root() -> Path:
    """Resolve repo root assuming this script lives in repo_root/src/representations/."""
    return Path(__file__).resolve().parents[2]


def read_record_signal(record_base_path: Path) -> tuple[np.ndarray, dict]:
    """Load ECG signal using wfdb.rdsamp."""
    signal, metadata = wfdb.rdsamp(str(record_base_path))
    if signal.ndim != 2 or signal.shape[1] != 2:
        raise RuntimeError(
            f"Expected a 2-channel ECG record, got shape {signal.shape}."
        )
    return signal, metadata


def read_annotations(record_base_path: Path) -> tuple[list[int], list[str]]:
    """Load annotations using the same symbol handling as the cleaner."""
    ann = wfdb.rdann(str(record_base_path), "atr")
    samples = []
    symbols = []

    for sample, symbol in zip(ann.sample, ann.symbol):
        if symbol in {".", "·"}:
            symbol = "N"
        if symbol in LABEL_MAP:
            samples.append(int(sample))
            symbols.append(symbol)

    return samples, symbols


def filter_valid_annotations(
    samples: list[int],
    symbols: list[str],
    signal_length: int,
    window_pre: int = WINDOW_PRE_ANNOTATION,
    window_post: int = WINDOW_POST_ANNOTATION,
) -> tuple[list[int], list[str]]:
    """Apply the same edge filtering used in the cleaner."""
    valid_samples = []
    valid_symbols = []

    for sample, symbol in zip(samples, symbols):
        if sample - window_pre < 0:
            continue
        if sample + window_post >= signal_length:
            continue
        valid_samples.append(sample)
        valid_symbols.append(symbol)

    return valid_samples, valid_symbols


def center_window_for_fd(td_window: np.ndarray) -> np.ndarray:
    """Match the implemented FD preprocessing, mean-center each channel."""
    return td_window - np.mean(td_window, axis=1, keepdims=True)


def build_single_sample_view(
    signal: np.ndarray,
    center_sample: int,
    symbol: str,
    lead_index: int,
    raw_context_extra: int,
) -> dict:
    """Create all arrays needed for the pipeline-evolution figure."""
    td_start = center_sample - WINDOW_PRE_ANNOTATION
    td_end = center_sample + WINDOW_POST_ANNOTATION + 1

    raw_start = max(0, td_start - raw_context_extra)
    raw_end = min(signal.shape[0], td_end + raw_context_extra)

    raw_context = signal[raw_start:raw_end, :]
    td_window = signal[td_start:td_end, :].T  # [C, T]

    if td_window.shape != (2, EXPECTED_WINDOW_LENGTH):
        raise RuntimeError(
            f"Unexpected TD window shape {td_window.shape}, expected (2, {EXPECTED_WINDOW_LENGTH})."
        )

    td_window_centered = center_window_for_fd(td_window)
    fft_vals = np.fft.rfft(td_window_centered, axis=1)
    freq_bins = np.fft.rfftfreq(EXPECTED_WINDOW_LENGTH, d=1 / SAMPLING_RATE)
    magnitude = np.abs(fft_vals)
    phase = np.angle(fft_vals)

    raw_x = (np.arange(raw_start, raw_end) - center_sample) / SAMPLING_RATE
    td_x = (np.arange(td_start, td_end) - center_sample) / SAMPLING_RATE

    td_left_sec = (td_start - center_sample) / SAMPLING_RATE
    td_right_sec = ((td_end - 1) - center_sample) / SAMPLING_RATE

    return {
        "symbol": symbol,
        "lead_index": lead_index,
        "raw_context": raw_context,
        "raw_x": raw_x,
        "td_window": td_window,
        "td_window_centered": td_window_centered,
        "td_x": td_x,
        "freq_bins": freq_bins,
        "magnitude": magnitude,
        "phase": phase,
        "td_left_sec": td_left_sec,
        "td_right_sec": td_right_sec,
        "center_sample": center_sample,
        "raw_start": raw_start,
        "raw_end": raw_end,
        "td_start": td_start,
        "td_end": td_end,
    }


def format_lead_name(metadata: dict, lead_index: int) -> str:
    sig_names = metadata.get("sig_name", None)
    if sig_names is None or lead_index >= len(sig_names):
        return f"Lead {lead_index}"
    return str(sig_names[lead_index])


def make_figure(view: dict, lead_name: str, output_path: Path) -> None:
    """Create the 4-panel figure showing the sample through the pipeline."""
    lead = view["lead_index"]
    symbol = view["symbol"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    ax_raw, ax_td, ax_mag, ax_phase = axes.flat

    raw_y = view["raw_context"][:, lead]
    td_y = view["td_window"][lead]
    td_centered_y = view["td_window_centered"][lead]
    mag_y = view["magnitude"][lead]
    phase_y = view["phase"][lead]

    ax_raw.plot(view["raw_x"], raw_y, linewidth=1.5)
    ax_raw.axvline(0.0, linestyle="--", linewidth=1.0, label="Selected annotation")
    ax_raw.axvline(view["td_left_sec"], linestyle=":", linewidth=1.2, label="TD clip bounds")
    ax_raw.axvline(view["td_right_sec"], linestyle=":", linewidth=1.2)
    ax_raw.set_title("1. Raw local context around selected event")
    ax_raw.set_xlabel("Time relative to event (s)")
    ax_raw.set_ylabel(f"Amplitude ({lead_name})")
    ax_raw.legend(loc="best", fontsize=8)
    ax_raw.grid(alpha=0.3)

    ax_td.plot(view["td_x"], td_y, linewidth=1.5, label="Clipped TD chunk")
    ax_td.plot(
        view["td_x"],
        td_centered_y,
        linestyle="--",
        linewidth=1.2,
        label="Mean-centered before FFT",
    )
    ax_td.axvline(0.0, linestyle=":", linewidth=1.0)
    ax_td.set_title("2. Fixed time-domain chunk")
    ax_td.set_xlabel("Time relative to event (s)")
    ax_td.set_ylabel(f"Amplitude ({lead_name})")
    ax_td.legend(loc="best", fontsize=8)
    ax_td.grid(alpha=0.3)

    ax_mag.plot(view["freq_bins"], mag_y, linewidth=1.5)
    ax_mag.set_title("3. Frequency-domain magnitude")
    ax_mag.set_xlabel("Frequency (Hz)")
    ax_mag.set_ylabel("Magnitude")
    ax_mag.grid(alpha=0.3)

    ax_phase.plot(view["freq_bins"], phase_y, linewidth=1.5)
    ax_phase.set_title("4. Frequency-domain phase")
    ax_phase.set_xlabel("Frequency (Hz)")
    ax_phase.set_ylabel("Phase (rad)")
    ax_phase.grid(alpha=0.3)

    fig.suptitle(
        "Evolution of one ECG event through the preprocessing pipeline\n"
        f"Record 100, first valid annotation, label '{symbol}', {EXPECTED_WINDOW_LENGTH}-sample TD chunk at {SAMPLING_RATE} Hz",
        fontsize=13,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def print_summary(view: dict, lead_name: str) -> None:
    """Print small, useful facts for progress updates or figure captions."""
    lead = view["lead_index"]
    td_y = view["td_window"][lead]
    td_centered_y = view["td_window_centered"][lead]
    freq_bins = view["freq_bins"]
    mag_y = view["magnitude"][lead]

    dominant_idx = int(np.argmax(mag_y[1:]) + 1) if mag_y.size > 1 else 0
    dominant_freq = float(freq_bins[dominant_idx]) if dominant_idx < freq_bins.size else 0.0

    print("Selected event summary")
    print("-" * 24)
    print(f"Record:              100")
    print(f"Annotation sample:   {view['center_sample']}")
    print(f"Beat label:          {view['symbol']}")
    print(f"Lead used in figure: {lead_name} (index {lead})")
    print(f"TD chunk length:     {EXPECTED_WINDOW_LENGTH} samples")
    print(f"TD duration:         {EXPECTED_WINDOW_LENGTH / SAMPLING_RATE:.4f} s")
    print(f"Sampling rate:       {SAMPLING_RATE} Hz")
    print(f"Raw context range:   [{view['raw_start']}, {view['raw_end']}) samples")
    print(f"TD clip range:       [{view['td_start']}, {view['td_end']}) samples")
    print(f"TD mean before FFT:  {np.mean(td_y):.6f}")
    print(f"TD mean after cent.: {np.mean(td_centered_y):.6f}")
    print(f"FFT bins stored:     {freq_bins.size}")
    print(f"Dominant non-DC bin: {dominant_freq:.3f} Hz")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 4-panel figure showing one ECG event through the preprocessing pipeline."
    )
    parser.add_argument(
        "--record",
        type=str,
        default="100",
        help="MIT-BIH record key to visualize. Default: 100",
    )
    parser.add_argument(
        "--lead-index",
        type=int,
        default=0,
        choices=[0, 1],
        help="Which ECG lead to plot. Default: 0",
    )
    parser.add_argument(
        "--raw-context-extra",
        type=int,
        default=180,
        help="Extra samples to show before and after the TD clip bounds in the raw-context panel. Default: 180",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional output path. Default: repo_root/outputs/representations/"
            "ecg_pipeline_evolution_record_<record>_lead_<lead>.png"
        ),
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    repo_root = resolve_repo_root()
    raw_dir = repo_root / "data" / "ecg" / "raw"
    record_base_path = raw_dir / args.record

    for suffix in [".hea", ".dat", ".atr"]:
        required_path = record_base_path.with_suffix(suffix)
        if not required_path.exists():
            raise FileNotFoundError(f"Missing required ECG file: {required_path}")

    signal, metadata = read_record_signal(record_base_path)
    samples, symbols = read_annotations(record_base_path)
    valid_samples, valid_symbols = filter_valid_annotations(
        samples=samples,
        symbols=symbols,
        signal_length=signal.shape[0],
    )

    if not valid_samples:
        raise RuntimeError(f"No valid annotations found for record {args.record} after filtering.")

    center_sample = valid_samples[0]
    symbol = valid_symbols[0]

    view = build_single_sample_view(
        signal=signal,
        center_sample=center_sample,
        symbol=symbol,
        lead_index=args.lead_index,
        raw_context_extra=args.raw_context_extra,
    )

    lead_name = format_lead_name(metadata, args.lead_index)

    if args.output is None:
        output_path = (
            repo_root
            / "src"
            / "representations"
            / "assets"
            / f"ecg_pipeline_evolution_record_{args.record}_lead_{args.lead_index}.png"
        )
    else:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = repo_root / output_path

    make_figure(view=view, lead_name=lead_name, output_path=output_path)
    print_summary(view=view, lead_name=lead_name)
    print(f"\nSaved figure to: {output_path}")


if __name__ == "__main__":
    main()
