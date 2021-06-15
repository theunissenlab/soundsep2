import numpy as np
from soundsig.signal import lowpass_filter, highpass_filter, bandpass_filter


def filter_and_ampenv(
        data,
        sampling_rate: int,
        f0: float,
        f1: float,
        rectify_lowpass: float
    ) -> np.ndarray:
    """Compute an amplitude envelope of a signal
    """
    filtered = highpass_filter(data.T, sampling_rate, f0, filter_order=10).T
    filtered = lowpass_filter(filtered.T, sampling_rate, f1, filter_order=10).T

    # Rectify and lowpass filter
    rectified = np.abs(filtered)
    ampenv = lowpass_filter(rectified.T, sampling_rate, rectify_lowpass, filter_order=10).T

    return filtered, ampenv
