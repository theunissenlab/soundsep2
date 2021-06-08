import numpy as np
from scipy.fft import fft, fftfreq, next_fast_len


def _create_gaussian_window(half_window: int, nstd: float):
    """Create a gaussian window

    Returns gaussian window with array size 2*half_window.

    [ half_window | half_window ]
    [             _             ]
    [          .'   '.          ]
    [         .       .         ]
    [......-'           '-......]
                
                  --- 1 std
                  ------ 2 std
                  --------- 3 std

    
    The steepness of the decline on each side is determined
    by nstd

    :param half_window: Half width of the gaussian window
    :type half_window: int
    :param nstd: Number of standard deviations of the gaussian the window should
        cover. Larger means that the edges will fall more quickly, weighting points
        closer to the center. Smaller values of nstd means the fall-off will be
        more gradual and edge points will be weighted more cloesly to the center.
    :type nstd: float


    """
    t = np.arange(-half_window, half_window + 1)
    std = 2.0 * float(half_window) / float(nstd)
    return (
        np.exp(-t**2 / (2.0 * std**2))
        / (std * np.sqrt(2 * np.pi))
    )


def _iter_gaussian_windows(data, start_index, window_size, window_step, n_windows):
    """Create an iterator of gaussian tapered windows

    :param data: Dataset to iterate over
    :type data: np.ndarray
    :param start_index: Center index of the first gaussian window
    :type start_index: int
    :param window_size: Full width of each window. If even, will be incremented by 1
        so that the windows will be symmetric.
    :type window_size: int
    :param window_step: Number of samples between each window center
    :type window_step: int
    :param n_windows: Total number of windows to iterate over
    :type n_windows: int
    """
    if window_size % 2 == 0:
        window_size += 1

    half_window = window_size // 2

    gaussian_window = _create_gaussian_window(half_window, 6)

    for i in range(n_windows):
        window_center = (start_index + window_step * i)
        window_start = max(0, window_center - half_window)
        window_end = min(len(data), window_center - half_window + window_size)

        if window_start >= len(data):
            break

        window_data = data[window_start:window_end]
        if window_start == 0:
            window_data = window_data * gaussian_window[-window_data.size:]
        elif window_end == len(data):
            window_data = window_data * gaussian_window[:window_data.size]
        else:
            window_data = window_data * gaussian_window

        yield window_data


def stft_gen(data, start_index, window_size, window_step, n_windows):
    """Generator for samples in stft

    Returns an iterator over numpy arrays representing windows centered starting at
    start_index and increasing by window_step. The iterator will yield n_windows times
    unless it reaches the end of data. The ith iteration will yield the fft of the
    ith window.

    :param data: Dataset to iterate over
    :type data: np.ndarray
    :param start_index: Center index of the first gaussian window
    :type start_index: int
    :param window_size: Full width of each window. If even, will be incremented by 1
        so that the windows will be symmetric.
    :type window_size: int
    :param window_step: Number of samples between each window center
    :type window_step: int
    :param n_windows: Total number of windows to iterate over
    :type n_windows: int
    """
    window_iterator = _iter_gaussian_windows(
        data,
        start_index,
        window_size,
        window_step,
        n_windows
    )

    if window_size % 2:
        window_size += 1
    fft_len = next_fast_len(window_size)

    for window in window_iterator:
        yield fft(window, n=fft_len, overwrite_x=1)


def stft_freq(window_size: int, sampling_rate: float):
    """Return the freq array for a stft with the given window_size

    :param window_size: Full width of each window. If even, will be incremented by 1
        so that the windows will be symmetric.
    :type window_size: int
    :param sampling_rate: Sampling rate of the dataset
    :type sampling_rate: float
    """
    if window_size % 2:
        window_size += 1
    fft_len = next_fast_len(window_size)

    return fftfreq(fft_len, d=1.0 / sampling_rate)


def spectral_derivative(spec):
    dy, dx = np.gradient(np.abs(spec))
    theta = np.arctan2(dy, dx)
    dspec = np.abs(dy) * np.sin(theta) + np.abs(dx) * np.cos(theta)

    return dspec
