import numpy as np
from soundsig.signal import lowpass_filter, highpass_filter, bandpass_filter


def compute_ampenv(
        data,
        sampling_rate: int,
        f0: float,
        f1: float,
        rectify_lowpass: float
    ) -> np.ndarray:
    """Compute an amplitude envelope of a signal
    """
    filtered = highpass_filter(data.T, fs, f0, filter_order=10).T
    filtered = lowpass_filter(filtered.T, fs, f1, filter_order=10).T

    # Rectify and lowpass filter
    filtered = np.abs(filtered)
    filtered = lowpass_filter(filtered.T, fs, rectify_lowpass).T

    return filtered
