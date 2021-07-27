import logging
from dataclasses import dataclass
from typing import List

import numpy as np

from soundsep.core.utils import DuplicatedRingBuffer, ceil_div, get_blocks_of_ones

from .lattice import (
    Bound,
    BoundedLattice,
    ceil as lattice_ceil,
    floor as lattice_floor,
    overlapping_slice,
    overlapping_range
)
from .levels import gen_bounded_layers
from .params import StftParameters


def create_cache_from_layers(layers: List[BoundedLattice]):
    BoundedLattice


logger = logging.getLogger(__name__)


class CacheLayer:

    def __init__(self, lattice: 'BoundedLattice'):
        self.lattice = lattice
        self.data = DuplicatedRingBuffer(np.zeros((len(self.lattice), 0, 0)))
        self.stale = DuplicatedRingBuffer(np.ones((len(self.lattice), 0), dtype=np.bool))

    def data_range(self) -> int:
        """

        Return the size of the range spanned by this caching layer in StftIndex samples
        """
        return len(self.data) * self.lattice.step

    def get_lim(self) -> int:
        """Return start (inclusive), stop (non-inclusive) tuple in StftIndex"""
        return self.lattice[0], self.lattice[len(self.lattice)]

    def set_shape(self, channels: int, features: int):
        """Set the size of the buffered data

        Arguments
        ---------
        shape : Tuple[int, int]
            A tuple representing the
        """
        self.data = DuplicatedRingBuffer(np.zeros((len(self.lattice), channels, features)))
        self.stale = DuplicatedRingBuffer(np.ones((len(self.lattice), channels), dtype=np.bool))

    def set_data(self, idx: 'StftIndex', data: np.ndarray):
        i = self.lattice.to_position(idx)
        self.data[i] = data
        self.stale[i] = False

    def get_bounds_from_central_range(self, i0: int, i1: int, full_size: int) -> 'Bound':
        """Find a new bound centered on (i0, i1), but on the current lattice

        The max i has to be provided since this object doesn't have knowledge
        of the size of the full data array
        """
        layer_size = self.data_range()
        samples_before = ceil_div(layer_size - (i1 - i0), 2)
        potential_start = lattice_floor(i0 - samples_before, self.lattice)
        assert potential_start in self.lattice.without_bound()

        start = max(self.lattice.offset, potential_start)

        if layer_size >= full_size:
            stop = start + layer_size
        elif start + layer_size > full_size:
            stop = lattice_floor(full_size, self.lattice)
            start = stop - layer_size
        else:
            stop = start + layer_size

        assert start in self.lattice.without_bound()

        return Bound(start, stop)

    def set_central_range(self, i0: int, i1: int, full_size: int):
        """Center this CacheLayer on the given position

        The size of the lattice bounds must remain constant
        """
        new_bound = self.get_bounds_from_central_range(i0, i1, full_size)
        if self.lattice.bound.start == new_bound.start:
           return

        offset = (new_bound.start - self.lattice.bound.start) // self.lattice.step
        self.data.roll(offset, fill=0)
        self.stale.roll(offset, fill=True)
        self.lattice.bound = new_bound

    def get_primary_jobs(self, i0: int, i1: int):
        """Return a list of (int, int, BoundedLattice) describing the jobs that must be run.
        """
        request_ranges = [
            (i0, i1)
        ]

        jobs = []
        for start, stop in request_ranges:
            slice_ = slice(*overlapping_slice(start, stop, self.lattice))
            data_start = self.lattice.to_position(slice_.start)
            data_stop = self.lattice.to_position(slice_.stop)
            for a, b in get_blocks_of_ones(self.stale[data_start:data_stop, 0]):
                jobs.append((start + int(a) * self.lattice.step, start + int(b) * self.lattice.step))
        return jobs

    def get_secondary_jobs(self, i0: int, i1: int, full_size):
        cache_bound = self.get_bounds_from_central_range(i0, i1, full_size)
        request_ranges = [
            (i1, cache_bound.stop),
            (cache_bound.start, i0)
        ]

        jobs = []

        for start, stop in request_ranges:
            slice_ = slice(*overlapping_slice(start, stop, self.lattice))
            data_start = self.lattice.to_position(slice_.start)
            data_stop = self.lattice.to_position(slice_.stop)
            for a, b in get_blocks_of_ones(self.stale[data_start:data_stop, 0]):
                jobs.append((start + int(a) * self.lattice.step, start + int(b) * self.lattice.step))

        return jobs


class StftCache:
    def __init__(self, layers: List[CacheLayer]):
        self.layers = layers

    @property
    def n_levels(self) -> int:
        return len(self.layers)

    def choose_level(self, read_size: int, fraction_of_cache: float = 1.0):
        """Choose level to read up to
        """
        for i, layer in enumerate(self.layers):
            # If the read_size is within the bounds of the layer, we can read up to that layer
            if read_size < layer.data_range() * fraction_of_cache:
                return i
        else:
            return self.n_levels - 1

    def read(self, i0: int, i1: int, level: int):
        """Read data from the given StftIndex coordinates"""
        arr = np.zeros(shape=(i1 - i0, self.layers[0].data.shape[1], self.layers[0].data.shape[2]))
        stale_mask = np.ones(shape=(i1 - i0, self.layers[0].stale.shape[1]), dtype=np.bool)

        first_offset = self.layers[-1].lattice.offset
        for layer in self.layers[level:]:
            layer_lim = layer.get_lim()
            start, stop, step = overlapping_slice(i0, i1, layer.lattice)
            layer_selector = slice((start - layer_lim[0]) // step, (stop - layer_lim[0]) // step)
            fill_data = layer.data[layer_selector]

            stale_mask[start - i0:stop - i0:step] &= layer.stale[layer_selector]
            first_offset = min(first_offset, start - i0)
            arr[start - i0:stop - i0:step] = fill_data

        every_other = pow(2, level)
        return np.arange(i0 + first_offset, i1, every_other), arr[first_offset::every_other], stale_mask[first_offset::every_other]

    def set_shape(self, channels: int, features: int):
        for layer in self.layers:
            layer.set_shape(channels, features)

    def set_central_range(self, i0: int, i1: int, full_size: int):
        """

        Coordinate layers by moving them such that they all center on the given
        pos. Then,
        """
        for layer in self.layers:
            layer.set_central_range(i0, i1, full_size)
