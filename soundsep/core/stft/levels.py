"""Code for generating multi-scale STFTs
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from soundsep.core.stft.lattice import Bound, BoundedLattice, Lattice


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


def iter_lattice_windows(
        arr: np.ndarray,
        bounded_lattice: 'BoundedLattice',
        scale_factor: int,
        half_window: int
    ):
    """A generator over windows whose centers lie on a given lattice

    The lattice can be scaled up

    The windows are views into the original array so should relatively fast

    Yields
    ------
    window_idx : StftIndex
        Index in the BoundedLattice's native coordinate system
    window : np.ndarray (view into arr)
        A view into arr of a window centered on window_idx. The central coordinate
        of this window in the signal coordinates should be ``window_idx * scale``.
    """
    scaled_lattice = bounded_lattice * scale_factor
    window_centers = np.array(list(scaled_lattice))
    pad_start, pad_stop, start_index, stop_index = compute_pad_for_windowing(
        len(arr),
        window_centers[0] - scaled_lattice.bound.start,
        window_centers[-1] - scaled_lattice.bound.start,
        half_window
    )
    arr = arr[start_index:stop_index]
    padding = ((pad_start, pad_stop),) + tuple([(0, 0) for _ in range(arr.ndim - 1)])

    padded_arr = np.pad(arr, padding, mode="reflect")
    windows = sliding_window_view(padded_arr, 2 * half_window + 1, axis=0)

    for window_idx, window in zip(
            bounded_lattice,
            windows[scaled_lattice.to_slice(relative_to=scaled_lattice.bound.start + half_window - pad_start)]
            ):
        yield window_idx, window


def gen_layers(n: int):
    """Generate a set of n layers with offsets such that every sample is filled once and only once

    Example:

      ```
      . . . . . . . . . .
       .   .   .   .   .
         .       .
             .       .
      ```
    """
    layers = [Lattice(offset=0, step=2)]
    for _ in range(n - 1):
        next_lattice = layers[-1] * 2
        next_lattice.offset += 1
        layers.append(next_lattice)

    # The last lattice needs to have half the step size
    layers[-1].step //= 2

    return layers


def gen_bounded_layers(n: int, size: int):
    """Generate a set of bounded lattices with offsets that fill in each other where valid

    Doubles the extent of each layer's bounds
    """
    layers = [BoundedLattice(offset=0, step=2, bound=Bound(start=0, stop=size))]
    for _ in range(n - 1):
        next_lattice = layers[-1] * 2
        next_lattice.offset += 1
        next_lattice.bound += 1
        layers.append(next_lattice)

    # The last lattice needs to have half the step size
    layers[-1].step //= 2

    return layers


def _get_layer_offset(layer: int):
    return pow(2, layer - 1) - 1


def _get_layer_step(layer: int, max_layer: int):
    return pow(2, min(layer, max_layer - 1))
