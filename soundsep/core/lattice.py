from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class Lattice:
    offset: int
    step: int

    def __hash__(self):
        return (offset, step)


def floor(idx: int, lattice: Lattice):
    """
    Returns floor of the given index when projected onto a lattice defined by (offset, step)
    """
    return lattice.offset + lattice.step * ((idx - lattice.offset) // lattice.step)


def ceil(idx: int, lattice: Lattice):
    """
    Returns ceil of the given index when projected onto a lattice defined by (offset, step)
    """
    return lattice.offset + lattice.step * ((idx - lattice.offset + lattice.step - 1) // lattice.step)


def overlapping_range(i0: int, i1: int, lattice: Lattice, relative_to: Optional[int] = 0):
    """
    Returns the indicies where the lattice overlaps the range i0, i1
    """
    start, stop, step = overlapping_slice(i0, i1, lattice, relative_to=relative_to)
    return np.arange(start, stop, step)


def overlapping_slice(i0: int, i1: int, lattice: Lattice, relative_to: Optional[int] = 0):
    """Gets the coordinates where a lattice overlaps the range [i0, i1)

    Returns the slice values (start, stop, step) where start and stop are in the
    global coordinates of i0
    """
    start = ceil(i0, lattice)
    stop = ceil(i1, lattice)
    return start - relative_to, stop - relative_to, lattice.step


def test_flood(i0: int, i1: int, layers: List[Lattice]):
    """Test function to fill a range at given indices

    Create an array with each layer of a lattice filling in its index in its own spots
    """
    output = np.zeros((i1 - i0))
    for i, layer in enumerate(layers):
        slice_ = slice(*overlapping_slice(i0, i1, layer, relative_to=i0))
        output[slice_] = i + 1
        yield output

