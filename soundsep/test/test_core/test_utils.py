import unittest

import numpy as np

from soundsep.core.utils import get_blocks_of_ones, hhmmss


class TestGetBlocksOfOnes(unittest.TestCase):

    def test_empty(self):
        arr = np.array([])
        result = get_blocks_of_ones(arr)
        np.testing.assert_array_equal(result, np.zeros((0, 2)))

    def test_starts_with_one(self):
        arr = np.array([1, 1, 0, 0, 0, 1, 1, 1, 0])
        result = get_blocks_of_ones(arr)
        np.testing.assert_array_equal(result, [(0, 2), (5, 8)])

    def test_ends_with_one(self):
        arr = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1])
        result = get_blocks_of_ones(arr)
        np.testing.assert_array_equal(result, [(1, 3), (6, 9)])

    def test_starts_and_ends_with_one(self):
        arr = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1])
        result = get_blocks_of_ones(arr)
        np.testing.assert_array_equal(result, [(0, 3), (6, 9)])

    def test_all_ones(self):
        arr = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        result = get_blocks_of_ones(arr)
        np.testing.assert_array_equal(result, [(0, 9)])

    def test_all_zeros(self):
        arr = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        result = get_blocks_of_ones(arr)
        np.testing.assert_array_equal(result, np.zeros((0, 2)))

    def test_one(self):
        arr = np.array([1])
        result = get_blocks_of_ones(arr)
        np.testing.assert_array_equal(result, [(0, 1)])

        arr = np.array([0])
        result = get_blocks_of_ones(arr)
        np.testing.assert_array_equal(result, np.zeros((0, 2)))


class TestTimeFormatting(unittest.TestCase):

    def test_hhmmss(self):
        t = 0.0
        result = hhmmss(t)
        self.assertEqual(result, "0:00:00")

        t = 59.0
        result = hhmmss(t)
        self.assertEqual(result, "0:00:59")

        t = 60.0
        result = hhmmss(t)
        self.assertEqual(result, "0:01:00")

        t = 60.0 * 60.0 - 1.0
        result = hhmmss(t)
        self.assertEqual(result, "0:59:59")

        t = 60.0 * 60.0
        result = hhmmss(t)
        self.assertEqual(result, "1:00:00")

        t = 10 * 60.0 * 60.0
        result = hhmmss(t)
        self.assertEqual(result, "10:00:00")

        t = 60.0 * 60.0 * 100
        result = hhmmss(t)
        self.assertEqual(result, "100:00:00")

    def test_hhmmss_decimals(self):
        t = 60.0 + 0.1289
        result = hhmmss(t, 1)
        self.assertEqual(result, "0:01:00.1")
        result = hhmmss(t, 2)
        self.assertEqual(result, "0:01:00.13")
        result = hhmmss(t, 3)
        self.assertEqual(result, "0:01:00.129")
        result = hhmmss(t, 4)
        self.assertEqual(result, "0:01:00.1289")
        result = hhmmss(t, 5)
        self.assertEqual(result, "0:01:00.12890")

        t = 60.0 + 30.1289
        result = hhmmss(t, 1)
        self.assertEqual(result, "0:01:30.1")
        result = hhmmss(t, 2)
        self.assertEqual(result, "0:01:30.13")
        result = hhmmss(t, 3)
        self.assertEqual(result, "0:01:30.129")
        result = hhmmss(t, 4)
        self.assertEqual(result, "0:01:30.1289")
        result = hhmmss(t, 5)
        self.assertEqual(result, "0:01:30.12890")
