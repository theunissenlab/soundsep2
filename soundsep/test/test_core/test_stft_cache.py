import unittest
from unittest import mock

import numpy as np
from PyQt5 import QtWidgets as widgets
from PyQt5 import QtTest

from soundsep.app.services import StftCache, StftConfig
from soundsep.core.models import Project, ProjectIndex, StftIndex
from soundsep.test.base_qt_test import test_app


class DummyProject(Project):
    def __init__(self, frames, channels, sampling_rate):
        self._frames = frames
        self._channels = channels
        self._sampling_rate = sampling_rate
        self._data = np.repeat(np.arange(frames)[:, None], channels, axis=1)

    def __getitem__(self, slices):
        return self._data[slices]

    @property
    def channels(self) -> int:
        """int: Number of channels in all Blocks of this Project"""
        return self._channels

    @property
    def sampling_rate(self) -> int:
        """int: Sampling rate of audio in this Project"""
        return self._sampling_rate

    @property
    def frames(self) -> int:
        """int: Total number of samples in the entire Project"""
        return self._frames



class TestStftCache(unittest.TestCase):

    def setUp(self):
        self.config = StftConfig(250, 44)
        self.project = DummyProject(frames=2000000, channels=2, sampling_rate=1000)
        self.cache = StftCache(self.project, 4410, 8820, self.config)

    def stftidx(self, i):
        return StftIndex(self.project, self.config.step, i)

    def tearDown(self):
        self.cache._worker.cancel()

    # @mock.patch("soundsep.core.stft_cache.fft")
    def test_threading(self):
        self.assertEqual(self.cache._start_ptr, self.stftidx(0))
        expected_freq_channels = 2 * self.config.window + 1
        # mock_fft.return_value = np.ones((expected_freq_channels,))
        self.assertEqual(self.cache._data.shape, (8820 * 2 + 4410, 2, expected_freq_channels))

        self.cache.set_position(self.stftidx(10000))
        QtTest.QTest.qWait(100)
        a, b = self.cache.read()
        print(np.sum(b == False), len(b))
        QtTest.QTest.qWait(100)
        a, b = self.cache.read()
        print(np.sum(b == False), len(b))
        QtTest.QTest.qWait(100)
        a, b = self.cache.read()
        print(np.sum(b == False), len(b))

        self.cache.set_position(self.stftidx(11000))
        QtTest.QTest.qWait(100)
        a, b = self.cache.read()
        print(np.sum(b == False), len(b))
        QtTest.QTest.qWait(100)
        a, b = self.cache.read()
        print(np.sum(b == False), len(b))
        QtTest.QTest.qWait(500)
        a, b = self.cache.read()
        print("AFter 500", np.sum(b == False), print(len(b)))

        self.cache.set_position(self.stftidx(10000))
        QtTest.QTest.qWait(100)
        a, b = self.cache.read()
        print(np.sum(b == False), len(b))
        QtTest.QTest.qWait(100)
        a, b = self.cache.read()
        print(np.sum(b == False), len(b))
        QtTest.QTest.qWait(100)
        a, b = self.cache.read()
        print(np.sum(b == False), len(b))

        self.cache.set_position(self.stftidx(20000))
        QtTest.QTest.qWait(100)
        a, b = self.cache.read()
        print("Big Jump", np.sum(b == False), len(b))
        QtTest.QTest.qWait(100)
        a, b = self.cache.read()
        print(np.sum(b == False), len(b))
        QtTest.QTest.qWait(100)
        a, b = self.cache.read()
        print(np.sum(b == False), len(b))

