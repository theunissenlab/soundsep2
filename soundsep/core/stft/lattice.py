import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from soundsep.core.utils import ceil_div


@dataclass
class Bound:
    start: int
    stop: int

    def __sub__(self, other: int):
        return Bound(self.start - other, self.stop - other)

    def __add__(self, other: int):
        return Bound(self.start + other, self.stop + other)

    def __mul__(self, other: int):
        return Bound(self.start * other, self.stop * other)

    def __floordiv__(self, other: int):
        return Bound(self.start // other, self.stop // other)

    def __eq__(self, other: 'Bound'):
        return self.start == other.start and self.stop == other.stop

    def __contains__(self, idx: int):
        return self.start <= idx < self.stop


@dataclass
class Lattice:
    offset: int
    step: int

    def __hash__(self):
        return (offset, step)

    def __eq__(self, other: 'Lattice'):
        return self.offset == other.offset and self.step == other.step

    def __mul__(self, other: int):
        return self.scale_up(other)

    def __floordiv__(self, other: int):
        return self.scale_down(other)

    def __iter__(self):
        raise NotImplementedError("Cannot iterate over infinite lattice. Use BoundedLattice instead")
        # next_ = offset
        # while True:
        #     yield next_
        #     next_ += step

    def scale_up(self, scale_factor: int):
        """Scales the Lattice up relative to zero"""
        return Lattice(offset=self.offset * scale_factor, step=self.step * scale_factor)

    def scale_down(self, scale_factor: int):
        """Scale the Lattice down relative to zero

        If scale_factor does not divide evenly into step and/or offset, does
        floor division but shows a warning.
        """
        if self.step % scale_factor or self.offset % scale_factor:
            warnings.warn("Scaling down a lattice by an uneven multiple of step or offset")
        return Lattice(offset=self.offset // scale_factor, step=self.step // scale_factor)

    def __contains__(self, idx: int):
        return (idx - self.offset) % self.step == 0

    def with_bound(self, bound: 'Bound') -> 'BoundedLattice':
        return BoundedLattice(self.offset, self.step, bound)


@dataclass
class BoundedLattice(Lattice):
    # TODO: enforce that the bound's starting inde
    bound: Bound

    def __len__(self):
        return ceil_div(self.bound.stop - ceil(max(self.bound.start, self.offset), self), self.step)

    def __iter__(self):
        for i in overlapping_range(self.bound.start, self.bound.stop, self):
            yield i

    def __eq__(self, other: 'BoundedLattice'):
        return super().__eq__(other) and self.bound == other.bound

    def __getitem__(self, idx: 'Union[int, slice]'):
        if isinstance(idx, int):
            return ceil(self.bound.start + idx * self.step, self)
        elif isinstance(idx, slice):
            return list(self)[slice]

    def to_position(self, idx: int):
        """Map a index in StftIndex coordinates to a integer index [0, len(self))
        """
        if (idx - self.offset) % self.step:
            raise ValueError("Given index does not lie on the lattice: {} {}".format(idx, self))

        return (idx - self[0]) // self.step

    def scale_up(self, scale_factor: int):
        """Scales the Lattice up relative to zero"""
        return BoundedLattice(
            offset=self.offset * scale_factor,
            step=self.step * scale_factor,
            bound=self.bound * scale_factor,
        )

    def scale_down(self, scale_factor: int):
        """Scale the Lattice down relative to zero

        If scale_factor does not divide evenly into step and/or offset, does
        floor division but shows a warning.
        """
        if self.step % scale_factor or self.offset % scale_factor:
            warnings.warn("Scaling down a lattice by an uneven multiple of step or offset")
        return BoundedLattice(
            offset=self.offset // scale_factor,
            step=self.step // scale_factor,
            bound=self.bound // scale_factor,
        )

    def to_slice(self, relative_to: int = 0) -> slice:
        return slice(*overlapping_slice(self.bound.start, self.bound.stop, self, relative_to))

    def without_bound(self) -> 'Lattice':
        return Lattice(offset=self.offset, step=self.step)

    def __contains__(self, idx: int):
        return super().__contains__(idx) and self.bound.__contains__(idx)


def floor(idx: int, lattice: Lattice):
    """
    Returns floor of the given index when projected onto a lattice defined by (offset, step)
    """
    return lattice.offset + lattice.step * ((idx - lattice.offset) // lattice.step)


def ceil(idx: int, lattice: Lattice):
    """
    Returns ceil of the given index when projected onto a lattice defined by (offset, step)
    """
    return lattice.offset + lattice.step * ceil_div(idx - lattice.offset, lattice.step)


def overlapping_range(i0: int, i1: int, lattice: Lattice, relative_to: Optional[int] = 0):
    """
    Returns the indicies where the lattice overlaps the range i0, i1
    """
    start, stop, step = overlapping_slice(i0, i1, lattice, relative_to=relative_to)
    return range(start, stop, step)


def overlapping_slice(i0: int, i1: int, lattice: Lattice, relative_to: Optional[int] = 0):
    """Gets the coordinates where a lattice overlaps the range [i0, i1)

    Returns the slice values (start, stop, step) where start and stop are in the
    global coordinates of i0
    """
    if isinstance(lattice, BoundedLattice):
        start = ceil(max(i0, lattice.bound.start), lattice)
        stop = ceil(max(start, min(i1, lattice.bound.stop)), lattice)
    else:
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
