from dataclasses import dataclass

import numpy as np


@dataclass
class StftParameters:
    """Define a STFT"""
    hop: int = 22
    half_window: int = 302
    sample_rate: int = 44100
    fmin: float = 250.0
    fmax: float = 10000.0

    @property
    def nfft(self):
        return 2 * self.half_window + 1


def compute_pad_for_windowing(
        array_length: int,
        first_window_pos: int,
        last_window_pos: int,
        half_window: int
        ):
    """Prepares a signal for stft from sig_i0 to sig_i1

    Arguments
    ---------
    array_length : int
        The full length of the signal
    first_window_pos : int
        The index in the signal of the first window center
    last_window_pos : int
        The index in the signal of the last window center
    half_window : int
        The half window size in samples

    Returns
    -------
    before : int
        Padding to add before the signal
    after : int
        Padding to add after the signal
    """
    wanted_start = first_window_pos - half_window
    wanted_stop = last_window_pos + half_window + 1

    if wanted_start <= 0:
        pad_start = int(np.abs(wanted_start))
        wanted_start = 0
    else:
        pad_start = 0

    if wanted_stop >= array_length:
        pad_stop = int(wanted_stop - array_length)
        wanted_stop = array_length
    else:
        pad_stop = 0

    return pad_start, pad_stop, wanted_start, wanted_stop
