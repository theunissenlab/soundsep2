import unittest
from unittest import mock
from contextlib import contextmanager

import numpy as np

from soundsep.core.io import AudioFile


def mock_soundfile_with(*args, **kwargs):
    @contextmanager
    def _mock_soundfile(*args, **kwargs):
        yield mock.Mock(samplerate=44100, channels=2, frames=22050)
    return _mock_soundfile


class TestAudioFile(unittest.TestCase):

    def setUp(self):
        # Set up a mocked soundfile with known attrs
        self.sampling_rate = 44100
        self.channels = 2
        self.frames = 22050
        self.MockSoundFile = mock_soundfile_with(
            samplerate=self.sampling_rate,
            channels=self.channels,
            frames=self.frames
        )

    def test_init(self):
        with mock.patch("soundfile.SoundFile", self.MockSoundFile):
            f = AudioFile("asdf.wav")

        self.assertEqual(f.sampling_rate, self.sampling_rate)
        self.assertEqual(f.channels, self.channels)
        self.assertEqual(f.frames, self.frames)
        self.assertEqual(f.path, "asdf.wav")

    def test_set_max_frame(self):
        with mock.patch("soundfile.SoundFile", self.MockSoundFile):
            f = AudioFile("asdf.wav")

        self.assertEqual(f.frames, self.frames)

        with self.assertRaises(ValueError):
            f.set_max_frame(-10)

        with self.assertRaises(ValueError):
            f.set_max_frame(2.2)

        with self.assertRaises(RuntimeError):
            f.set_max_frame(self.frames + 1)

        f.set_max_frame(20000)
        self.assertEqual(f.frames, 20000, "frames property should read max_frame if set")

        f.clear_max_frame()
        self.assertEqual(f.frames, self.frames, "frames property should be default after max_frames is cleared")

    def test_equality(self):
        with mock.patch("soundfile.SoundFile", self.MockSoundFile):
            f1 = AudioFile("asdf.wav")
            f2 = AudioFile("asdf.wav")
            f3 = AudioFile("notasdf.wav")
            f4 = AudioFile("not/asdf.wav")

        self.assertEqual(f1, f2, "Two AudioFiles should be == if and only if they have the same path")
        self.assertNotEqual(f1, f3, "Two AudioFiles should be == if and only if they have the same path")
        self.assertNotEqual(f1, f4, "Two AudioFiles should be == if and only if they have the same path")

    def test_read(self):
        with mock.patch("soundfile.SoundFile", self.MockSoundFile):
            f1 = AudioFile("asdf.wav")

        with mock.patch("soundfile.read") as mock_read:
            with self.assertRaises(ValueError):
                f1.read(-5, 10, channel=-1)

            with self.assertRaises(ValueError):
                f1.read(-5, 10, channel=2)

        with mock.patch("soundfile.read") as mock_read:
            result = f1.read(-5, 10, channel=0)
            mock_read.assert_called_with("asdf.wav", 15, -5, dtype=np.float64)
            
        with mock.patch("soundfile.read") as mock_read:
            result = f1.read(10, self.frames + 100, channel=0)
            mock_read.assert_called_with("asdf.wav", self.frames - 10, 10, dtype=np.float64)

    def test_read_shape(self):
        with mock.patch("soundfile.SoundFile", self.MockSoundFile):
            f1 = AudioFile("asdf.wav")
            
        with mock.patch("soundfile.read") as mock_read:
            mock_read.return_value = np.random.random((10, 2))

            result0 = f1.read(10, 20, channel=0)
            np.testing.assert_array_equal(result0, mock_read.return_value[:, 0])

            result1 = f1.read(10, 20, channel=1)
            np.testing.assert_array_equal(result1, mock_read.return_value[:, 1])

