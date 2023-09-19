import unittest

import numpy as np

from soundsep.core.utils import DuplicatedRingBuffer


class TestRingBuffer(unittest.TestCase):
    def test_len(self):
        base = np.arange(100)
        buffer = DuplicatedRingBuffer(base)
        self.assertEqual(len(buffer), 100)

        base = np.zeros((250, 100, 30, 11))
        buffer = DuplicatedRingBuffer(base)
        self.assertEqual(len(buffer), 250)

    def test_shape(self):
        base = np.arange(100)
        buffer = DuplicatedRingBuffer(base)
        self.assertEqual(buffer.shape, (100,))
        self.assertEqual(buffer[:].shape, (100,))

        base = np.zeros((250, 100, 30, 11))
        buffer = DuplicatedRingBuffer(base)
        self.assertEqual(buffer.shape, (250, 100, 30, 11))
        self.assertEqual(buffer[:].shape, (250, 100, 30, 11))

    def test_size(self):
        base = np.arange(100)
        buffer = DuplicatedRingBuffer(base)
        self.assertEqual(buffer.size, 100)
        self.assertEqual(buffer[:].size, 100)

        base = np.zeros((250, 100, 30, 11))
        buffer = DuplicatedRingBuffer(base)
        self.assertEqual(buffer.size, 250 * 100 * 30 * 11)
        self.assertEqual(buffer[:].size, 250 * 100 * 30 * 11)

    def test_convert_index(self):
        self.assertEqual(DuplicatedRingBuffer._convert_index(0, 10, 100), 10)
        self.assertEqual(DuplicatedRingBuffer._convert_index(20, 0, 100), 20)
        self.assertEqual(DuplicatedRingBuffer._convert_index(0, 0, 100), 0)
        self.assertEqual(DuplicatedRingBuffer._convert_index(25, 25, 100), 50)

        # Test wrap around
        self.assertEqual(DuplicatedRingBuffer._convert_index(80, 25, 100), 5)
        self.assertEqual(DuplicatedRingBuffer._convert_index(90, 90, 100), 80)

        # Test negative indices
        self.assertEqual(DuplicatedRingBuffer._convert_index(-20, 50, 100), 30)
        self.assertEqual(DuplicatedRingBuffer._convert_index(-20, 10, 100), 90)

    def test_duplicated_index(self):
        self.assertEqual(DuplicatedRingBuffer._duplicated_index(0, 100), 100)
        self.assertEqual(DuplicatedRingBuffer._duplicated_index(20, 100), 120)
        self.assertEqual(DuplicatedRingBuffer._duplicated_index(99, 100), 199)
        self.assertEqual(DuplicatedRingBuffer._duplicated_index(-1, 100), 99, "The internal array of size 200 should duplicate data at 199 and 99")

    def test_convert_slice(self):
        self.assertEqual(DuplicatedRingBuffer._convert_slice(slice(None, None), 0, 100), slice(0, 100, 1))
        self.assertEqual(DuplicatedRingBuffer._convert_slice(slice(None, None), 90, 100), slice(90, 190, 1))

        self.assertEqual(DuplicatedRingBuffer._convert_slice(slice(5, None), 90, 100), slice(95, 190, 1))
        self.assertEqual(DuplicatedRingBuffer._convert_slice(slice(20, None), 90, 100), slice(110, 190, 1))

        self.assertEqual(DuplicatedRingBuffer._convert_slice(slice(-5, None), 90, 100), slice(185, 190, 1))
        self.assertEqual(DuplicatedRingBuffer._convert_slice(slice(-95, None), 90, 100), slice(95, 190, 1))

        self.assertEqual(DuplicatedRingBuffer._convert_slice(slice(None, 5), 90, 100), slice(90, 95, 1))
        self.assertEqual(DuplicatedRingBuffer._convert_slice(slice(None, 95), 90, 100), slice(90, 185, 1))

        self.assertEqual(DuplicatedRingBuffer._convert_slice(slice(None, -10), 90, 100), slice(90, 180, 1))
        self.assertEqual(DuplicatedRingBuffer._convert_slice(slice(None, -80), 90, 100), slice(90, 110, 1))

        self.assertEqual(DuplicatedRingBuffer._convert_slice(slice(10, 20), 90, 100), slice(100, 110, 1))
        self.assertEqual(DuplicatedRingBuffer._convert_slice(slice(-50, 60), 90, 100), slice(140, 150, 1))
        self.assertEqual(DuplicatedRingBuffer._convert_slice(slice(-50, 40), 90, 100), slice(140, 130, 1))
        self.assertEqual(DuplicatedRingBuffer._convert_slice(slice(20, -40), 90, 100), slice(110, 150, 1))
        self.assertEqual(DuplicatedRingBuffer._convert_slice(slice(20, 20), 90, 100), slice(110, 110, 1))

        self.assertEqual(DuplicatedRingBuffer._convert_slice(slice(None, None, 2), 0, 100), slice(0, 100, 2))
        self.assertEqual(DuplicatedRingBuffer._convert_slice(slice(20, -40, 3), 90, 100), slice(110, 150, 3))

    def test_duplicated_slice(self):
        self.assertEqual(DuplicatedRingBuffer._duplicated_slice(slice(60, 90, 1), 100), (slice(160, 190, 1), slice(0, 0, 1)))
        self.assertEqual(DuplicatedRingBuffer._duplicated_slice(slice(160, 190, 1), 100), (slice(60, 90, 1), slice(0, 0, 1)))
        self.assertEqual(DuplicatedRingBuffer._duplicated_slice(slice(60, 110, 1), 100), (slice(160, 200, 1), slice(0, 10, 1)))

        # Test that steps are preserved
        self.assertEqual(DuplicatedRingBuffer._duplicated_slice(slice(60, 90, 3), 100), (slice(160, 190, 3), slice(0, 0, 3)))
        self.assertEqual(DuplicatedRingBuffer._duplicated_slice(slice(160, 190, 4), 100), (slice(60, 90, 4), slice(0, 0, 4)))

        # Test that steps are preserved across the 2N boundary
        self.assertEqual(DuplicatedRingBuffer._duplicated_slice(slice(92, 110, 2), 100), (slice(192, 200, 2), slice(0, 10, 2)))
        self.assertEqual(DuplicatedRingBuffer._duplicated_slice(slice(92, 110, 3), 100), (slice(192, 200, 3), slice(1, 10, 3)))
        self.assertEqual(DuplicatedRingBuffer._duplicated_slice(slice(92, 110, 4), 100), (slice(192, 200, 4), slice(0, 10, 4)))
        self.assertEqual(DuplicatedRingBuffer._duplicated_slice(slice(92, 110, 5), 100), (slice(192, 200, 5), slice(2, 10, 5)))
        self.assertEqual(DuplicatedRingBuffer._duplicated_slice(slice(92, 110, 6), 100), (slice(192, 200, 6), slice(4, 10, 6)))
        self.assertEqual(DuplicatedRingBuffer._duplicated_slice(slice(92, 110, 7), 100), (slice(192, 200, 7), slice(6, 10, 7)))
        self.assertEqual(DuplicatedRingBuffer._duplicated_slice(slice(92, 110, 8), 100), (slice(192, 200, 8), slice(0, 10, 8)))

    def test_roll(self):
        base = np.arange(100)
        buffer = DuplicatedRingBuffer(base)

        buffer.roll(10, -1)
        np.testing.assert_array_equal(buffer[:90], np.arange(10, 100))
        np.testing.assert_array_equal(buffer[-10:], [-1] * 10)

        buffer.roll(70, -1)
        np.testing.assert_array_equal(buffer[:20], np.arange(80, 100))
        np.testing.assert_array_equal(buffer[20:], [-1] * 80)

        buffer.roll(15, -1)
        np.testing.assert_array_equal(buffer[:5], np.arange(95, 100))
        np.testing.assert_array_equal(buffer[5:], [-1] * 95)

        buffer.roll(15, -1)
        np.testing.assert_array_equal(buffer[:], [-1] * 100)

        # now test rolling backward
        base = np.arange(100)
        buffer = DuplicatedRingBuffer(base)

        buffer.roll(-10, -1)
        np.testing.assert_array_equal(buffer[:10], [-1] * 10)
        np.testing.assert_array_equal(buffer[10:], np.arange(90))

        buffer.roll(-70, -1)
        np.testing.assert_array_equal(buffer[:80], [-1] * 80)
        np.testing.assert_array_equal(buffer[80:], np.arange(20))

        buffer.roll(-30, -1)
        np.testing.assert_array_equal(buffer[:], [-1] * 100)

        # test roll a huge amount
        base = np.arange(100)
        buffer = DuplicatedRingBuffer(base)
        buffer.roll(10000, -1)
        np.testing.assert_array_equal(buffer[:], [-1] * 100)

        base = np.arange(100)
        buffer = DuplicatedRingBuffer(base)
        buffer.roll(-10000, -1)
        np.testing.assert_array_equal(buffer[:], [-1] * 100)

    def test_roll_dtype(self):
        """Test that rolling an array and filling matches the existing dtype"""
        base = np.ones(100)
        int_buffer = DuplicatedRingBuffer(base.astype(int))
        assert int_buffer._data.dtype == int
        int_buffer.roll(10, -1.0)
        self.assertEqual(int_buffer[:].dtype, int)

        base = np.ones(100)
        bool_buffer = DuplicatedRingBuffer(base.astype(np.bool))
        assert bool_buffer._data.dtype == np.bool
        bool_buffer.roll(10, 0.0)
        self.assertEqual(bool_buffer[:].dtype, np.bool)
        np.testing.assert_array_equal(bool_buffer[-10:], [False] * 10)
        bool_buffer.roll(10, -1.0)
        np.testing.assert_array_equal(bool_buffer[-10:], [True] * 10)
