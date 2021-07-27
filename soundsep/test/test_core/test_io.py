import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from soundsep.core.io import (
    common_subsequence,
    group_files_by_pattern,
    guess_filename_pattern,
    search_for_wavs
)


def mock_soundfile_metadata():
    @contextmanager
    def _mock_soundfile(path):
        if path == str(Path("foo/ba(r/rec:s1_ch0-00123.wav")):
            obj = mock.MagicMock()
            obj.frames = 20
            obj.samplerate = 44100
            obj.channels = 1
        elif path == str(Path("foo/bar/rec:s1_ch1-0231|2.wav")):
            obj = mock.MagicMock()
            obj.frames = 20
            obj.samplerate = 44100
            obj.channels = 2
        elif path == str(Path("foo/ba)r/rec:s2_ch0-12|345.wav")):
            obj = mock.MagicMock()
            obj.frames = 44
            obj.samplerate = 44100
            obj.channels = 1
        elif path == str(Path("foo/bar/rec:s2_ch1-42358.wav")):
            obj = mock.MagicMock()
            obj.frames = 44
            obj.samplerate = 44100
            obj.channels = 2

        yield obj
    return _mock_soundfile


class TestIoFunctions(unittest.TestCase):
    def test_search_for_wavs(self):
        """Test that search for wavs returns absolute paths"""
        with tempfile.TemporaryDirectory() as temp_base:
            temp_base = Path(temp_base)

            # Create a dummy filesystem
            (temp_base / "findme3.wav").touch()
            (temp_base / "dontfindme2.txt").touch()
            (temp_base / "subdir1").mkdir()
            (temp_base / "subdir1" / "findme1.wav").touch()
            (temp_base / "subdir1" / "dontfindme1.txt").touch()
            (temp_base / "subdir2").mkdir()
            (temp_base / "subdir2" / "findme2.wav").touch()

            result = search_for_wavs(temp_base, recursive=False)
            self.assertEqual(result, [temp_base / "findme3.wav"])

            result_recursive = search_for_wavs(temp_base, recursive=True)
            # This tests that the contents of the lists are the same
            self.assertCountEqual(result_recursive,
                [
                    temp_base / "findme3.wav",
                    temp_base / "subdir1" / "findme1.wav",
                    temp_base / "subdir2" / "findme2.wav"
                ]
            )

    @mock.patch("soundfile.SoundFile", mock_soundfile_metadata())
    def test_guess_filename_pattern(self):
        """Test that search for wavs returns absolute paths"""
        base_directory = Path("foo/")
        filelist = [
            "foo/ba(r/rec:s1_ch0-00123.wav",
            "foo/bar/rec:s1_ch1-0231|2.wav",
            "foo/ba)r/rec:s2_ch0-12|345.wav",
            "foo/bar/rec:s2_ch1-42358.wav",
        ]
        filelist = [str(Path(s)) for s in filelist]

        block_keys, filename_pattern = guess_filename_pattern(base_directory, filelist)

        self.assertEqual(block_keys, ["var2"])
        self.assertEqual(filename_pattern, str(Path("{var0}/rec:{var2}_{var3}-{var4}.wav")))

    def test_common_subsequence(self):
        a = [
            "aqbcdf",
            "abecde",
            "gbrcdefg",
            "bcacd",
        ]
        result = common_subsequence(a)
        self.assertListEqual(result, [
            ("b", "c", "d"),
        ])

class TestGroupFilesByPattern(unittest.TestCase):

    def setUp(self):
        self.base_directory = Path("foo/")
        self.filelist = [
            Path("foo/bar/baz_001_ch0.wav"),
            Path("foo/bar/baz_001_ch1.wav"),
            Path("foo/bar/baz_002_ch0.wav"),
            Path("foo/bar/baz_002_ch1.wav"),
            Path("foo/bar/baz_003_ch0.wav"),
        ]

    @mock.patch("soundsep.core.io.AudioFile", autospec=True)
    def test_group_files_by_pattern_no_keys(self, mock_audiofile):
        """Test that group files without any particular pattern works"""
        groups, errors = group_files_by_pattern(
            base_directory=self.base_directory,
            filelist=self.filelist,
            filename_pattern=None,
            block_keys=None,
            channel_keys=None
        )

        self.assertEqual(len(groups), 1)
        mock_audiofile.assert_has_calls([mock.call(f) for f in self.filelist])
        self.assertEqual(groups[0][0], None)
        for obj in groups[0][1]:
            self.assertIsNone(obj["block_id"])
            self.assertIsNone(obj["channel_id"])

        self.assertEqual(len(errors), 0)

    @mock.patch("soundsep.core.io.AudioFile", autospec=True)
    def test_group_files_by_pattern_block_keys(self, mock_audiofile):
        groups, errors = group_files_by_pattern(
            base_directory=self.base_directory,
            filelist=self.filelist,
            filename_pattern=str(Path("bar/baz_{timestamp}_ch{channel}.wav")),
            block_keys=["timestamp"],
            channel_keys=None
        )

        self.assertEqual(len(groups), 3)
        self.assertEqual(groups[0][0], ("001",))
        self.assertEqual(len(groups[0][1]), 2)
        for obj in groups[0][1]:
            self.assertEqual(obj["block_id"], ("001",))
            self.assertIsNone(obj["channel_id"])

        self.assertEqual(groups[1][0], ("002",))
        self.assertEqual(len(groups[1][1]), 2)
        for obj in groups[1][1]:
            self.assertEqual(obj["block_id"], ("002",))
            self.assertIsNone(obj["channel_id"])

        self.assertEqual(groups[2][0], ("003",))
        self.assertEqual(len(groups[2][1]), 1)
        for obj in groups[2][1]:
            self.assertEqual(obj["block_id"], ("003",))
            self.assertIsNone(obj["channel_id"])

        self.assertEqual(len(errors), 0)

    @mock.patch("soundsep.core.io.AudioFile", autospec=True)
    def test_group_files_by_pattern_multiple_block_keys(self, mock_audiofile):
        groups, errors = group_files_by_pattern(
            base_directory=self.base_directory,
            filelist=self.filelist,
            filename_pattern=str(Path("bar/baz_{timestamp}_ch{channel}.wav")),
            block_keys=["timestamp", "channel"],
            channel_keys=None
        )

        self.assertEqual(len(groups), 5)
        self.assertEqual(groups[0][0], ("001", "0"))
        self.assertEqual(len(groups[0][1]), 1)
        for obj in groups[0][1]:
            self.assertEqual(obj["block_id"], ("001", "0"))
            self.assertIsNone(obj["channel_id"])

        self.assertEqual(groups[1][0], ("001", "1"))
        self.assertEqual(len(groups[1][1]), 1)
        for obj in groups[1][1]:
            self.assertEqual(obj["block_id"], ("001", "1"))
            self.assertIsNone(obj["channel_id"])

        self.assertEqual(groups[2][0], ("002", "0"))
        self.assertEqual(len(groups[2][1]), 1)
        for obj in groups[2][1]:
            self.assertEqual(obj["block_id"], ("002", "0"))
            self.assertIsNone(obj["channel_id"])

        self.assertEqual(groups[3][0], ("002", "1"))
        self.assertEqual(len(groups[3][1]), 1)
        for obj in groups[3][1]:
            self.assertEqual(obj["block_id"], ("002", "1"))
            self.assertIsNone(obj["channel_id"])

        self.assertEqual(groups[4][0], ("003", "0"))
        self.assertEqual(len(groups[4][1]), 1)
        for obj in groups[4][1]:
            self.assertEqual(obj["block_id"], ("003", "0"))
            self.assertIsNone(obj["channel_id"])

        self.assertEqual(len(errors), 0)

    @mock.patch("soundsep.core.io.AudioFile", autospec=True)
    def test_group_files_bad_path(self, mock_audiofile):
        """Test that it errors when the filename_pattern matches too much or too little of the file paths"""

        # Includes part of the base directory in the filename pattern
        groups, errors = group_files_by_pattern(
            base_directory=self.base_directory,
            filelist=self.filelist,
            filename_pattern=str(Path("foo/bar/baz_{timestamp}_ch{channel}.wav")),
            block_keys=None,
            channel_keys=None
        )

        self.assertEqual(len(groups), 0)
        self.assertEqual(len(errors), 5)
        self.assertCountEqual([e[1] for e in errors], [None] * 5)

        # Only includes the basename even though base directory doesn't include "bar"
        groups, errors = group_files_by_pattern(
            base_directory=self.base_directory,
            filelist=self.filelist,
            filename_pattern="baz_{timestamp}_ch{channel}.wav",
            block_keys=None,
            channel_keys=None
        )

        self.assertEqual(len(groups), 0)
        self.assertEqual(len(errors), 5)
        self.assertCountEqual([e[1] for e in errors], [None] * 5)

    @mock.patch("soundsep.core.io.AudioFile", autospec=True)
    def test_group_files_by_pattern_with_channel_keys(self, mock_audiofile):
        groups, errors = group_files_by_pattern(
            base_directory=self.base_directory,
            filelist=self.filelist,
            filename_pattern=str(Path("bar/baz_{timestamp}_ch{channel}.wav")),
            block_keys=None,
            channel_keys=["channel"]
        )

        self.assertEqual(len(groups), 1)
        self.assertIsNone(groups[0][0])
        self.assertEqual(len(groups[0][1]), 5)
        self.assertCountEqual([obj["block_id"] for obj in groups[0][1]], [None] * 5)
        self.assertListEqual([obj["channel_id"] for obj in groups[0][1]],
            [("0",), ("0",), ("0",), ("1",), ("1",)])

        self.assertEqual(len(errors), 0)

    @mock.patch("soundsep.core.io.AudioFile", autospec=True)
    def test_group_files_by_pattern_with_block_and_channel_keys(self, mock_audiofile):
        groups, errors = group_files_by_pattern(
            base_directory=self.base_directory,
            filelist=self.filelist,
            filename_pattern=str(Path("bar/baz_{timestamp}_ch{channel}.wav")),
            block_keys=["timestamp"],
            channel_keys=["channel"]
        )

        self.assertEqual(len(groups), 3)

        self.assertEqual(groups[0][0], ("001",))
        self.assertEqual(len(groups[0][1]), 2)
        self.assertEqual(groups[0][1][0]["block_id"], ("001",))
        self.assertEqual(groups[0][1][0]["channel_id"], ("0",))
        self.assertEqual(groups[0][1][1]["block_id"], ("001",))
        self.assertEqual(groups[0][1][1]["channel_id"], ("1",))

        self.assertEqual(groups[1][0], ("002",))
        self.assertEqual(len(groups[1][1]), 2)
        self.assertEqual(groups[1][1][0]["block_id"], ("002",))
        self.assertEqual(groups[1][1][0]["channel_id"], ("0",))
        self.assertEqual(groups[1][1][1]["block_id"], ("002",))
        self.assertEqual(groups[1][1][1]["channel_id"], ("1",))

        self.assertEqual(groups[2][0], ("003",))
        self.assertEqual(len(groups[2][1]), 1)
        self.assertEqual(groups[2][1][0]["block_id"], ("003",))
        self.assertEqual(groups[2][1][0]["channel_id"], ("0",))

        self.assertEqual(len(errors), 0)
