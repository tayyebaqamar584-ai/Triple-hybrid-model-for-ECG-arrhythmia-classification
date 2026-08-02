"""Create a simple preprocessing visualization for ECG signals.

The script loads a raw MIT-BIH record, applies a configurable band-pass filter,
then standardizes the signal and saves a comparison figure showing the raw,
filtered, and normalized waveforms.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple, cast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wfdb
from scipy import signal as scipy_signal


@dataclass
class FilterConfig:
    lowcut_hz: float = 0.5
    highcut_hz: float = 40.0
    order: int = 3
    sampling_rate_hz: int = 360


def load_record_signal(record_id: int, base_dir: Path | None = None) -> Tuple[np.ndarray, int]:
    """Load the first ECG lead from a MIT-BIH record."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[1] / "raw_data" / "mit-bih-arrhythmia-database-1.0.0"
    path = str(base_dir / str(record_id))
    record = cast(Any, wfdb.rdrecord(path))
    raw_p_signal = getattr(record, 'p_signal', None)
    if raw_p_signal is None:
        raise RuntimeError(f"Record {record_id} did not contain p_signal")
    fs_value = getattr(record, 'fs', None)
    if fs_value is None:
        raise RuntimeError(f"Record {record_id} did not contain fs")
    signal = np.asarray(raw_p_signal[:, 0], dtype=float)
    return signal, int(fs_value)


def bandpass_filter(signal: np.ndarray, config: FilterConfig) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter."""
    if len(signal) < 10:
        return signal.copy()
    sos = scipy_signal.butter(
        config.order,
        [config.lowcut_hz, config.highcut_hz],
        btype="bandpass",
        fs=config.sampling_rate_hz,
        output="sos",
    )
    return scipy_signal.sosfiltfilt(sos, signal)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Standardize the signal to zero mean and unit variance."""
    if signal.size == 0:
        return signal.copy()
    std = float(signal.std())
    if std < 1e-12:
        return np.zeros_like(signal, dtype=float)
    return (signal - float(signal.mean())) / std


def prepare_signal(signal: np.ndarray, config: FilterConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the raw, filtered and normalized signal arrays."""
    filtered = bandpass_filter(signal, config)
    normalized = normalize_signal(filtered)
    return signal, filtered, normalized


def plot_preprocessing_steps(
    raw_signal: np.ndarray,
    filtered_signal: np.ndarray,
    normalized_signal: np.ndarray,
    output_path: Path,
    record_id: int,
    config: FilterConfig,
) -> Path:
    """Save a comparison plot of the raw, filtered and normalized ECG signal."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_len = min(len(raw_signal), len(filtered_signal), len(normalized_signal))
    x = np.arange(max_len)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"ECG preprocessing example — record {record_id}", fontsize=14)

    axes[0].plot(x, raw_signal[:max_len], color="tab:blue", linewidth=1.0)
    axes[0].set_ylabel("Raw signal")
    axes[0].set_title("Raw ECG")

    axes[1].plot(x, filtered_signal[:max_len], color="tab:orange", linewidth=1.0)
    axes[1].set_ylabel("Filtered signal")
    axes[1].set_title(
        f"Band-pass filter: {config.lowcut_hz:.1f}-{config.highcut_hz:.1f} Hz, order {config.order}"
    )

    axes[2].plot(x, normalized_signal[:max_len], color="tab:green", linewidth=1.0)
    axes[2].set_ylabel("Normalized signal")
    axes[2].set_title("Standardized signal (z-score)")
    axes[2].set_xlabel("Samples")

    for ax in axes:
        ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(str(output_path), dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize ECG preprocessing steps")
    parser.add_argument("--record", type=int, default=100, help="MIT-BIH record to plot")
    parser.add_argument(
        "--output",
        type=str,
        default="results_beatwise/plots/preprocessing_signal_demo.png",
        help="Path for the saved figure",
    )
    parser.add_argument("--lowcut", type=float, default=0.5, help="Low cutoff in Hz")
    parser.add_argument("--highcut", type=float, default=40.0, help="High cutoff in Hz")
    parser.add_argument("--order", type=int, default=3, help="Butterworth filter order")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_path = (root / args.output).resolve()
    config = FilterConfig(
        lowcut_hz=args.lowcut,
        highcut_hz=args.highcut,
        order=args.order,
    )

    raw_signal, sampling_rate = load_record_signal(args.record, root / "raw_data" / "mit-bih-arrhythmia-database-1.0.0")
    config.sampling_rate_hz = sampling_rate
    raw_signal, filtered_signal, normalized_signal = prepare_signal(raw_signal, config)
    plot_preprocessing_steps(raw_signal, filtered_signal, normalized_signal, output_path, args.record, config)
    print(f"Saved preprocessing plot to {output_path}")


if __name__ == "__main__":
    main()
