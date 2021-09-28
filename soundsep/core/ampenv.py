import numpy as np
from scipy.signal import filter_design, filtfilt


def lowpass_filter(s, sample_rate, cutoff_freq, filter_order=5, rescale=False):
    """
    From https://github.com/theunissenlab/soundsig/blob/7d3eb40d7e701ade915bf8a8b2eaef34e3561bd2/soundsig/signal.py

        Lowpass filter a signal s, with sample rate sample_rate.
        s: the signal (n_channels x n_timepoints)
        sample_rate: the sample rate in Hz of the signal
        cutoff_freq: the cutoff frequency of the filter
        filter_order: the order of the filter...
        Returns the low-pass filtered signal s.
    """
    #create a butterworth filter
    nyq = sample_rate / 2.0
    b,a = filter_design.butter(filter_order, cutoff_freq / nyq)

    #filter the signal
    filtered_s = filtfilt(b, a, s)

    if rescale:
        #rescale filtered signal
        filtered_s /= filtered_s.max()
        filtered_s *= s.max()

    return filtered_s


def highpass_filter(s, sample_rate, cutoff_freq, filter_order=5, rescale=False):
    """
    From https://github.com/theunissenlab/soundsig/blob/7d3eb40d7e701ade915bf8a8b2eaef34e3561bd2/soundsig/signal.py

        Highpass filter a signal s, with sample rate sample_rate.
        s: the signal (n_channels x n_timepoints)
        sample_rate: the sample rate in Hz of the signal
        cutoff_freq: the cutoff frequency of the filter
        filter_order: the order of the filter...
        Returns the high-pass filtered signal s.
    """
    #create a butterworth filter
    nyq = sample_rate / 2.0
    b,a = filter_design.butter(filter_order, cutoff_freq / nyq, btype='high')

    #filter the signal
    filtered_s = filtfilt(b, a, s)

    if rescale:
        #rescale filtered signal
        filtered_s /= filtered_s.max()
        filtered_s *= s.max()

    return filtered_s


def filter_and_ampenv(
        data,
        sampling_rate: int,
        f0: float,
        f1: float,
        rectify_lowpass: float
    ) -> np.ndarray:
    """Compute an amplitude envelope of a signal
    """
    filtered = highpass_filter(data.T, sampling_rate, f0, filter_order=5).T
    filtered = lowpass_filter(filtered.T, sampling_rate, f1, filter_order=5).T

    # Rectify and lowpass filter
    rectified = np.abs(filtered)
    ampenv = lowpass_filter(rectified.T, sampling_rate, rectify_lowpass, filter_order=5).T

    return filtered.astype(np.float32), ampenv.astype(np.float32)
