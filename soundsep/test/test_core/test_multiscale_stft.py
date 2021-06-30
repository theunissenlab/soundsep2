import unittest
import warnings

import numpy as np

from soundsep.core.stft import lattice
from soundsep.core.stft.cache import _CacheLayer, StftCache
from soundsep.core.stft.lattice import Bound, BoundedLattice, Lattice
from soundsep.core.stft.levels import (
    _get_layer_offset,
    _get_layer_step,
    compute_pad_for_windowing,
    iter_lattice_windows,
    gen_layers,
    gen_bounded_layers
)
from soundsep.core.stft.params import StftParameters
from soundsep.core.utils import ceil_div


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


class TestMultiScaleStft(unittest.TestCase):

    def test_get_layer_step(self):
        self.assertEqual(_get_layer_step(1, 1), 1)
        self.assertEqual(_get_layer_step(1, 2), 2)
        self.assertEqual(_get_layer_step(1, 4), 2)
        self.assertEqual(_get_layer_step(2, 2), 2)
        self.assertEqual(_get_layer_step(2, 3), 4)
        self.assertEqual(_get_layer_step(3, 3), 4)
        self.assertEqual(_get_layer_step(3, 4), 8)

    def test_gen_layers(self):
        layers = gen_layers(5)
        self.assertEqual(len(layers), 5)
        self.assertEqual(layers[0], Lattice(offset=_get_layer_offset(1), step=_get_layer_step(1, 5)))
        self.assertEqual(layers[1], Lattice(offset=_get_layer_offset(2), step=_get_layer_step(2, 5)))
        self.assertEqual(layers[2], Lattice(offset=_get_layer_offset(3), step=_get_layer_step(3, 5)))
        self.assertEqual(layers[3], Lattice(offset=_get_layer_offset(4), step=_get_layer_step(4, 5)))
        self.assertEqual(layers[4], Lattice(offset=_get_layer_offset(5), step=_get_layer_step(5, 5)))

        layers = gen_layers(1)
        self.assertEqual(len(layers), 1)
        self.assertEqual(layers[0], Lattice(offset=_get_layer_offset(1), step=_get_layer_step(1, 1)))

    def test_gen_bounded_layers(self):
        layers = gen_bounded_layers(5, 100)
        self.assertEqual(len(layers), 5)
        self.assertEqual(layers[0], BoundedLattice(
            offset=_get_layer_offset(1),
            step=_get_layer_step(1, 5),
            bound=Bound(0, 100),
        ))
        print(layers)
        self.assertEqual(layers[1], BoundedLattice(
            offset=_get_layer_offset(2),
            step=_get_layer_step(2, 5),
            bound=Bound(1, 201),
        ))
        self.assertEqual(layers[2], BoundedLattice(
            offset=_get_layer_offset(3),
            step=_get_layer_step(3, 5),
            bound=Bound(3, 403),
        ))
        self.assertEqual(layers[3], BoundedLattice(
            offset=_get_layer_offset(4),
            step=_get_layer_step(4, 5),
            bound=Bound(7, 807),
        ))
        self.assertEqual(layers[4], BoundedLattice(
            offset=_get_layer_offset(5),
            step=_get_layer_step(5, 5),
            bound=Bound(15, 1615),
        ))

        layers = gen_bounded_layers(1, 100)
        self.assertEqual(len(layers), 1)
        self.assertEqual(layers[0], BoundedLattice(
            offset=_get_layer_offset(1),
            step=_get_layer_step(1, 1),
            bound=Bound(0, 100),
        ))

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
        pad_start, pad_stop = compute_pad_for_windowing(len(input_arr), 10, 80, half_window=5)
        self.assertEqual(pad_start, 0)
        self.assertEqual(pad_stop, 0)

        input_arr = np.ones(90)
        pad_start, pad_stop = compute_pad_for_windowing(len(input_arr), 10, 80, half_window=10)
        self.assertEqual(pad_start, 0)
        self.assertEqual(pad_stop, 1)

        input_arr = np.ones(90)
        pad_start, pad_stop = compute_pad_for_windowing(len(input_arr), 10, 80, half_window=15)
        self.assertEqual(pad_start, 5)
        self.assertEqual(pad_stop, 6)

        input_arr = np.ones(90)
        pad_start, pad_stop = compute_pad_for_windowing(len(input_arr), 1, 80, half_window=20)
        self.assertEqual(pad_start, 19)
        self.assertEqual(pad_stop, 11)

    def test_lattice_windows(self):
        input_arr = np.arange(100)
        output_view = list(iter_lattice_windows(
            input_arr,
            BoundedLattice(offset=2, step=14, bound=Bound(start=0, stop=50)),
            scale_factor=1,
            half_window=5,
        ))

        output_view = list(iter_lattice_windows(
            input_arr,
            BoundedLattice(offset=2, step=14, bound=Bound(start=0, stop=50)),
            scale_factor=4,
            half_window=5,
        ))

    def test_scale_up_lattice(self):
        self.assertEqual(Lattice(offset=3, step=5).scale_up(7), Lattice(offset=21, step=35))
        self.assertEqual(Lattice(offset=3, step=5).scale_up(1), Lattice(offset=3, step=5))
        self.assertEqual(Lattice(offset=0, step=4).scale_up(2), Lattice(offset=0, step=8))

        # Test the operator too
        self.assertEqual(Lattice(offset=3, step=5) * 7, Lattice(offset=21, step=35))
        self.assertEqual(Lattice(offset=3, step=5) * 1, Lattice(offset=3, step=5))
        self.assertEqual(Lattice(offset=0, step=4) * 2, Lattice(offset=0, step=8))

    def test_scale_down_lattice(self):
        self.assertEqual(Lattice(offset=4, step=8).scale_down(2), Lattice(offset=2, step=4))
        self.assertEqual(Lattice(offset=4, step=8).scale_down(4), Lattice(offset=1, step=2))
        with self.assertWarns(UserWarning):
            self.assertEqual(Lattice(offset=6, step=9).scale_down(4), Lattice(offset=1, step=2),
                "Scaling down lattice by uneven factor should floor")

        # Test the operator too
        self.assertEqual(Lattice(offset=4, step=8) // 2, Lattice(offset=2, step=4))
        self.assertEqual(Lattice(offset=4, step=8) // 4, Lattice(offset=1, step=2))
        with self.assertWarns(UserWarning):
            self.assertEqual(Lattice(offset=6, step=9) // 4, Lattice(offset=1, step=2),
                "Scaling down lattice by uneven factor should floor")

    def test_with_bound(self):
        self.assertEqual(
            Lattice(offset=3, step=5).with_bound(Bound(start=2, stop=13)),
            BoundedLattice(offset=3, step=5, bound=Bound(start=2, stop=13))
        )

    def test_lattice_contains(self):
        self.assertTrue(2 in Lattice(offset=2, step=3))
        self.assertFalse(3 in Lattice(offset=2, step=3))
        self.assertFalse(4 in Lattice(offset=2, step=3))
        self.assertTrue(5 in Lattice(offset=2, step=3))
        self.assertFalse(6 in Lattice(offset=2, step=3))
        self.assertFalse(7 in Lattice(offset=2, step=3))
        self.assertTrue(8 in Lattice(offset=2, step=3))

        self.assertFalse(2 in Lattice(offset=2, step=3).with_bound(Bound(start=4, stop=8)))
        self.assertFalse(3 in Lattice(offset=2, step=3).with_bound(Bound(start=4, stop=8)))
        self.assertFalse(4 in Lattice(offset=2, step=3).with_bound(Bound(start=4, stop=8)))
        self.assertTrue(5 in Lattice(offset=2, step=3).with_bound(Bound(start=4, stop=8)))
        self.assertFalse(6 in Lattice(offset=2, step=3).with_bound(Bound(start=4, stop=8)))
        self.assertFalse(7 in Lattice(offset=2, step=3).with_bound(Bound(start=4, stop=8)))
        self.assertFalse(8 in Lattice(offset=2, step=3).with_bound(Bound(start=4, stop=8)))

    def test_bounded_lattice_with_bound(self):
        """Test that the with_bound() method of BoundedLattice overwrites the bounds"""
        self.assertEqual(
            BoundedLattice(offset=5, step=4, bound=Bound(start=10, stop=100)).with_bound(Bound(start=5, stop=10)),
            BoundedLattice(offset=5, step=4, bound=Bound(start=5, stop=10))
        )

    def test_get_bounds_from_central_range(self):
        """Test method that centers a cache layer on a given region"""
        layer = _CacheLayer(lattice=BoundedLattice(offset=0, step=1, bound=Bound(start=0, stop=6)))
        self.assertEqual(
            layer.get_bounds_from_central_range(0, 2, full_size=20),
            Bound(0, 6),
            "Ranges too close to 0 should be aligned to lattice start"
        )
        self.assertEqual(
            layer.get_bounds_from_central_range(5, 7, full_size=20),
            Bound(3, 9),
        )
        self.assertEqual(
            layer.get_bounds_from_central_range(5, 8, full_size=20),
            Bound(3, 9),
            "Center ranges that are uneven should err on the left side"
        )
        self.assertEqual(
            layer.get_bounds_from_central_range(17, 19, full_size=20),
            Bound(14, 20),
            "Ranges close to the max index should be aligned to the max index"
        )

        self.assertEqual(
            layer.get_bounds_from_central_range(1, 2, full_size=5),
            Bound(0, 6),
            "When layer size (6) is larger than the full size of data (5), bound should go from 0 to layer size"
        )
        self.assertEqual(
            layer.get_bounds_from_central_range(4, 14, full_size=20),
            Bound(6, 12),
            "When requested center region is larger than the layer size (6), should center within the requested region"
        )

        # Now test a more interesting one
        # This layer is size 7 and spans 14
        layer = _CacheLayer(lattice=BoundedLattice(offset=1, step=2, bound=Bound(start=0, stop=15)))
        self.assertEqual(
            layer.get_bounds_from_central_range(10, 16, full_size=50),
            Bound(5, 19),
        )
        self.assertEqual(
            layer.get_bounds_from_central_range(11, 17, full_size=50),
            Bound(7, 21),
            "The new bound start must align to the layer's lattice"
        )
        self.assertEqual(
            layer.get_bounds_from_central_range(5, 7, full_size=50),
            Bound(1, 15),
            "Ranges too close to 0 should be aligned to lattice start"
        )
        self.assertEqual(
            layer.get_bounds_from_central_range(46, 49, full_size=50),
            Bound(35, 49),
            "Ranges close to the max index should be aligned to the max index"
        )

    def test_set_central_range(self):
        layer = _CacheLayer(lattice=BoundedLattice(offset=1, step=2, bound=Bound(start=0, stop=15)))

        layer.set_central_range(10, 16, full_size=50)
        # This layer should noe be aligned to Bound(5, 19)
        self.assertEqual(layer.lattice.bound, Bound(5, 19))

    def test_cache(self):
        layers = gen_bounded_layers(5, 100)
        params = StftParameters(hop=6, half_window=4)

        cache = StftCache([_CacheLayer(l) for l in layers])
        cache.set_shape(1, 1)
        for i, layer in enumerate(cache.layers):
            layer.data[:] = i + 1

        out_arr = cache.read(90, 100, 0)
        self.assertTrue(np.all(out_arr != 0.0))

        cache.set_central_range(50, 100, 500)

        self.assertEqual(cache.layers[0].lattice.bound, Bound(24, 124))
        self.assertEqual(cache.layers[1].lattice.bound, Bound(1, 201))
        self.assertEqual(cache.layers[2].lattice.bound, Bound(3, 403))
        self.assertEqual(cache.layers[3].lattice.bound, Bound(7, 807))
        self.assertEqual(cache.layers[4].lattice.bound, Bound(15, 1615))

        t, out_arr = cache.read(22, 42, 0)
        self.assertEqual(np.sum(out_arr == 0.0), 1)

        cache.set_central_range(250, 300, 500)
        self.assertEqual(cache.layers[0].lattice.bound, Bound(224, 324))
        self.assertEqual(cache.layers[1].lattice.bound, Bound(173, 373))
        self.assertEqual(cache.layers[2].lattice.bound, Bound(75, 475))
        self.assertEqual(cache.layers[3].lattice.bound, Bound(7, 807))
        self.assertEqual(cache.layers[4].lattice.bound, Bound(15, 1615))
