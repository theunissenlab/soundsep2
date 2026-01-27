import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filter_design, filtfilt


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

def advanced_filter_and_ampenv(data, fs, params):
    # -------- Extract parameters --------
    software_gain = params['software_gain']
    signal_low    = params['signal_low']
    signal_high   = params['signal_high']
    noise_low     = params['noise_low']
    noise_high    = params['noise_high']
    signal_gain   = params['signal_gain']
    noise_gain    = params['noise_gain']
    smooth_ms     = params['smooth_ms']
    threshold     = params['threshold']
    min_gap_sec   = params['min_gap_sec']
    min_dur_sec   = params['min_dur_sec']
    max_dur_sec   = params['max_dur_sec']
    
    if fs != params['fs']:
        raise ValueError(f"Sampling rate mismatch: data fs={fs}, params fs={params['fs']}")

    # -------- Precompute constants --------
    smooth_samples = int(smooth_ms * fs / 1000)

    # -------- Step 1: mean subtract + gain --------
    data = data.astype(np.float32)
    data = software_gain * (data - np.mean(data))

    # -------- Step 2: bandpass filters --------
    hb, ha = butter(5, [signal_low, signal_high], 'bandpass', fs=fs)
    lb, la = butter(5, [noise_low, noise_high], 'bandpass', fs=fs)
    sig = filtfilt(hb, ha, data)
    noi = filtfilt(lb, la, data)


    # -------- Step 3–4: energy-based combination --------
    rms_signal = np.sqrt(np.mean(sig**2))
    rms_noise  = np.sqrt(np.mean(noi**2))

    amp = (signal_gain * (sig**2 - rms_signal)
         - noise_gain  * (noi**2 - rms_noise))
    
    #amp = (signal_gain * sig**2) - (noise_gain  * noi**2)
    amp = np.maximum(amp, 0)

    # -------- Step 5: smoothing --------
    amp_smooth = uniform_filter1d(amp, size=smooth_samples, mode='nearest')
    return amp_smooth


def advanced_seg(data, fs, params, return_amp=False):
    """
    Segment a single audio waveform into onset/offset pairs of vocalizations
    using band-specific energy detection (no chunking).

    Parameters
    ----------
    data : np.ndarray or np.memmap
        Audio waveform (mono).
    fs : float
        Sampling rate in Hz.
    params : dict
        Dictionary containing all segmentation parameters.
    return_amp : bool, optional
        If True, also returns the smoothed amplitude envelope used for detection.

    Returns
    -------
    onsets, offsets : np.ndarray
        Arrays of onset and offset indices (samples).
    amp_full (optional) : np.ndarray
        Full smoothed amplitude envelope.
    """

    # -------- Extract parameters --------
    software_gain = params['software_gain']
    signal_low    = params['signal_low']
    signal_high   = params['signal_high']
    noise_low     = params['noise_low']
    noise_high    = params['noise_high']
    signal_gain   = params['signal_gain']
    noise_gain    = params['noise_gain']
    smooth_ms     = params['smooth_ms']
    threshold     = params['threshold']
    min_gap_sec   = params['min_gap_sec']
    min_dur_sec   = params['min_dur_sec']
    max_dur_sec   = params['max_dur_sec']
    min_gap_samp  = int(min_gap_sec * fs)
    min_dur_samp  = int(min_dur_sec * fs)
    max_dur_samp  = int(max_dur_sec * fs)
    
    
    amp_smooth = advanced_filter_and_ampenv(data, fs, params)

    # -------- Step 6: threshold crossings --------
    above = amp_smooth > threshold
    diff = np.diff(above.astype(np.int8))
    onsets  = np.where(diff == 1)[0]
    offsets = np.where(diff == -1)[0]

    # Handle boundary conditions
    if len(onsets) == 0 or len(offsets) == 0:
        return (np.array([]), np.array([]), amp_smooth) if return_amp else (np.array([]), np.array([]))

    if offsets[0] < onsets[0]:
        offsets = offsets[1:]
    if len(onsets) > len(offsets):
        onsets = onsets[:-1]

    # -------- Step 7: merge short gaps --------
    valid_on, valid_off = [onsets[0]], []
    for i in range(1, len(onsets)):
        gap = onsets[i] - offsets[i-1]
        if gap < min_gap_samp:
            continue  # merge
        valid_off.append(offsets[i-1])
        valid_on.append(onsets[i])
    valid_off.append(offsets[-1])

    valid_on = np.array(valid_on)
    valid_off = np.array(valid_off)

    # -------- Step 8: filter by duration --------
    dur = valid_off - valid_on
    keep = (dur > min_dur_samp) & (dur < max_dur_samp)
    on_final = valid_on[keep]
    off_final = valid_off[keep]

    # -------- Step 9: enforce pairing correctness --------
    if len(off_final) and len(on_final):
        off_final = off_final[off_final > on_final[0]]
    if len(on_final) and len(off_final) and on_final[-1] > off_final[-1]:
        on_final = on_final[:-1]
    n_pairs = min(len(on_final), len(off_final))
    on_final = on_final[:n_pairs]
    off_final = off_final[:n_pairs]
    valid_pairs = off_final > on_final
    on_final = on_final[valid_pairs]
    off_final = off_final[valid_pairs]


    # -------- Return --------
    if len(on_final) == 0 or len(off_final) == 0:
        return (np.array([]), np.array([]), amp_smooth) if return_amp else (np.array([]), np.array([]))
    return (on_final, off_final, amp_smooth) if return_amp else (on_final, off_final)