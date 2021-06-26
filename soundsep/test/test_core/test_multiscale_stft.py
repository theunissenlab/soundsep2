import unittest

from typing import List, Optional
from dataclasses import dataclass
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike

from soundsep.core import lattice
from soundsep.core.lattice import Lattice


@dataclass
class Bound:
    start: int
    stop: int


# How to make sure our indices are always consistent?
class ReferenceFrame(int):
    parent: Optional['ReferenceFrame']
    lattice: Lattice
    start: int  # Start index on this lattice
    stop: int   # Stop index on this lattice

    @staticmethod
    def make_root(n_samples):
        return ReferenceFrame(
            parent=None,
            lattice=Lattice(offset=0, step=1),
            start=0,
            stop=n_samples
        )


@dataclass
class StftParameters:
    hop: int
    half_window: int

    @property
    def n_fft(self):
        return 2 * self.half_window + 1


class _LocalIndex(int):
    _lattices = {}

    def __new__(cls, lattice: Lattice, value: int):
        return int.__new__(cls, value)

    def __init__(self, lattice: Lattice, value: int):
        self._lattice = Lattice
        super().__init__()


def LocalIndex(lattice: Lattice):
    """Type factory for indexes on lattices
    """
    if lattice in LocalIndex._lattices:
        return LocalIndex._lattices[lattice]
    else:
        LocalIndex._lattices[lattice] = type(
            "LocalIndex_{}_{}".format(lattice.offset, lattice.step),
            (_LocalIndex,),
            {}
        )


def compute_pad_for_windowing(arr: ArrayLike, window_centers: List[int], half_window: int):
    """Prepares a signal for stft from sig_i0 to sig_i1

    i0 and i1 are the first 
    """
    wanted_start = window_centers[0] - half_window
    wanted_stop = window_centers[-1] + half_window + 1

    if wanted_start < 0:
        pad_start = int(np.abs(wanted_start))
    else:
        pad_start = 0

    if wanted_stop > len(arr):
        pad_stop = int(wanted_stop - len(arr))
    else:
        pad_stop = 0

    return pad_start, pad_stop


def pad_for_windowing(arr: ArrayLike, window_centers: List[int], half_window: int):
    pad_start, pad_stop = compute_pad_for_windowing(arr, window_centers, half_window)
    padding = ((pad_start, pad_stop),) + tuple([(0, 0) for _ in range(arr.ndim - 1)])

    return np.pad(arr, padding, mode="reflect")


StftIndex = TypeVar("StftIndex", bound=_LocalIndex)


def faux_stft(
        arr: ArrayLike, 
        win0: StftIndex,
        win1: StftIndex,
        layers: List[Lattice],
        stft_params: StftParameters
        ):
    output = np.zeros((win1 - win0, stft_params.half_window))
    window_centers = np.arange(win0.to_project_index(), win1.to_project_index())
    arr = pad_for_windowing(arr, window_centers, stft_params.half_window)

    windows = sliding_window_view(
        arr,
        stft_params.n_fft,
        axis=0
    )

    for i, layer in enumerate(layers):
        start, stop, step = lattice.overlapping_slice(win0, win1, layer, relative_to=win0)
        windows[slice(
            start * stft_params.hop,
            stop * stft_params.hop,
            step * stft_params.hop
        )]
        yield output


class Layer(Lattice, Bound):
    pass


class Layers:
    """Manager for lattices defined at multiple scales with multiple extents
    """
    def __init__(self, lattices: List[Lattice]):
        self.layers = lattices


def _get_layer_offset(layer: int):
    return pow(2, layer - 1) - 1


def _get_layer_step(layer: int, max_layer: int):
    return pow(2, min(layer, max_layer - 1))


class TestMultiScaleStft(unittest.TestCase):

    def test_get_layer_step(self):
        self.assertEqual(_get_layer_step(1, 1), 1)
        self.assertEqual(_get_layer_step(1, 2), 2)
        self.assertEqual(_get_layer_step(1, 4), 2)
        self.assertEqual(_get_layer_step(2, 2), 2)
        self.assertEqual(_get_layer_step(2, 3), 4)
        self.assertEqual(_get_layer_step(3, 3), 4)
        self.assertEqual(_get_layer_step(3, 4), 8)

    def test_get_layer_offset(self):
        self.assertEqual(_get_layer_offset(1), 0)
        self.assertEqual(_get_layer_offset(2), 1)
        self.assertEqual(_get_layer_offset(3), 3)
        self.assertEqual(_get_layer_offset(4), 7)
        self.assertEqual(_get_layer_offset(5), 15)

    def test_lattice_floor(self):
        self.assertEqual(lattice.floor(0, Lattice(0, 1)), 0)
        self.assertEqual(lattice.floor(0, Lattice(2, 1)), 0)
        self.assertEqual(lattice.floor(3, Lattice(0, 1)), 3)
        self.assertEqual(lattice.floor(3, Lattice(2, 1)), 3)
        self.assertEqual(lattice.floor(2, Lattice(3, 1)), 2)

        self.assertEqual(lattice.floor(0, Lattice(0, 2)), 0)
        self.assertEqual(lattice.floor(0, Lattice(3, 2)), -1)
        self.assertEqual(lattice.floor(0, Lattice(4, 2)), 0)
        self.assertEqual(lattice.floor(5, Lattice(4, 2)), 4)
        self.assertEqual(lattice.floor(8, Lattice(4, 2)), 8)
        self.assertEqual(lattice.floor(-2, Lattice(4, 2)), -2)
        self.assertEqual(lattice.floor(-3, Lattice(4, 2)), -4)

        self.assertEqual(lattice.floor(0, Lattice(0, 3)), 0)
        self.assertEqual(lattice.floor(0, Lattice(2, 3)), -1)
        self.assertEqual(lattice.floor(0, Lattice(4, 3)), -2)
        self.assertEqual(lattice.floor(5, Lattice(4, 3)), 4)
        self.assertEqual(lattice.floor(7, Lattice(4, 3)), 7)
        self.assertEqual(lattice.floor(8, Lattice(4, 3)), 7)
        self.assertEqual(lattice.floor(-2, Lattice(4, 3)), -2)
        self.assertEqual(lattice.floor(-3, Lattice(4, 3)), -5)

    def test_lattice_ceil(self):
        self.assertEqual(lattice.ceil(0, Lattice(0, 1)), 0)
        self.assertEqual(lattice.ceil(0, Lattice(2, 1)), 0)
        self.assertEqual(lattice.ceil(3, Lattice(0, 1)), 3)
        self.assertEqual(lattice.ceil(3, Lattice(2, 1)), 3)
        self.assertEqual(lattice.ceil(2, Lattice(3, 1)), 2)

        self.assertEqual(lattice.ceil(0, Lattice(0, 2)), 0)
        self.assertEqual(lattice.ceil(0, Lattice(3, 2)), 1)
        self.assertEqual(lattice.ceil(0, Lattice(4, 2)), 0)
        self.assertEqual(lattice.ceil(5, Lattice(4, 2)), 6)
        self.assertEqual(lattice.ceil(8, Lattice(4, 2)), 8)
        self.assertEqual(lattice.ceil(-2, Lattice(4, 2)), -2)
        self.assertEqual(lattice.ceil(-3, Lattice(4, 2)), -2)

        self.assertEqual(lattice.ceil(0, Lattice(0, 3)), 0)
        self.assertEqual(lattice.ceil(0, Lattice(2, 3)), 2)
        self.assertEqual(lattice.ceil(0, Lattice(4, 3)), 1)
        self.assertEqual(lattice.ceil(5, Lattice(4, 3)), 7)
        self.assertEqual(lattice.ceil(7, Lattice(4, 3)), 7)
        self.assertEqual(lattice.ceil(8, Lattice(4, 3)), 10)
        self.assertEqual(lattice.ceil(-2, Lattice(4, 3)), -2)
        self.assertEqual(lattice.ceil(-3, Lattice(4, 3)), -2)

    def test_overlap(self):
        l = lattice.Lattice(offset=3, step=4)
        np.testing.assert_array_equal(lattice.overlapping_range(0, 10, l), [3, 7])
        np.testing.assert_array_equal(lattice.overlapping_range(-3, 10, l), [-1, 3, 7])
        np.testing.assert_array_equal(lattice.overlapping_range(-3, 7, l), [-1, 3])
        np.testing.assert_array_equal(lattice.overlapping_range(-3, 2, l), [-1])
        np.testing.assert_array_equal(lattice.overlapping_range(0, 2, l), [])
        np.testing.assert_array_equal(lattice.overlapping_range(-1, 9, l), [-1, 3, 7])

    def test_flood_fill(self):
        layers = []
        for layer_idx in [4, 3, 2, 1]:
            layers.append(lattice.Lattice(
                offset=_get_layer_offset(layer_idx),
                step=_get_layer_step(layer_idx, 4)
            ))

        for subresult in lattice.test_flood(10, 30, layers):
            print(subresult)

        layers = []
        for layer_idx in [3, 2, 1]:
            layers.append(Lattice(
                offset=_get_layer_offset(layer_idx),
                step=_get_layer_step(layer_idx, 3)
            ))

        for subresult in lattice.test_flood(0, 100, layers):
            print(subresult)

    def test_compute_pad_for_windowing(self):
        input_arr = np.ones(90)
        pad_start, pad_stop = compute_pad_for_windowing(input_arr, [10, 50, 80], half_window=5)
        self.assertEqual(pad_start, 0)
        self.assertEqual(pad_stop, 0)

        input_arr = np.ones(90)
        pad_start, pad_stop = compute_pad_for_windowing(input_arr, [10, 50, 80], half_window=10)
        self.assertEqual(pad_start, 0)
        self.assertEqual(pad_stop, 1)

        input_arr = np.ones(90)
        pad_start, pad_stop = compute_pad_for_windowing(input_arr, [10, 50, 80], half_window=15)
        self.assertEqual(pad_start, 5)
        self.assertEqual(pad_stop, 6)

        input_arr = np.ones(90)
        pad_start, pad_stop = compute_pad_for_windowing(input_arr, [1, 50, 80], half_window=20)
        self.assertEqual(pad_start, 19)
        self.assertEqual(pad_stop, 11)

    def test_pad_for_windowing(self):
        input_arr = np.ones(90)
        output_arr = pad_for_windowing(input_arr, [10, 50, 80], half_window=5)
        self.assertEqual(output_arr.shape, (90,))

        input_arr = np.ones(90)
        output_arr = pad_for_windowing(input_arr, [10, 50, 80], half_window=10)
        self.assertEqual(output_arr.shape, (91,))

        input_arr = np.ones(90)
        output_arr = pad_for_windowing(input_arr, [10, 50, 80], half_window=15)
        self.assertEqual(output_arr.shape, (101,))

        input_arr = np.ones((90, 10))
        output_arr = pad_for_windowing(input_arr, [10, 50, 80], half_window=15)
        self.assertEqual(output_arr.shape, (101, 10))

        input_arr = np.ones((90, 10, 50))
        output_arr = pad_for_windowing(input_arr, [10, 50, 80], half_window=15)
        self.assertEqual(output_arr.shape, (101, 10, 50))
