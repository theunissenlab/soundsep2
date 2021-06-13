import unittest
import warnings
from unittest import mock
from contextlib import contextmanager

import numpy as np

from soundsep.core.models import AudioFile, Block, BlockIndex, Project, ProjectIndex


def mock_soundfile_with(samplerate=44100, channels=1, frames=44100):
    @contextmanager
    def _mock_soundfile(*args, **kwargs):
        yield mock.Mock(samplerate=samplerate, channels=channels, frames=frames)
    return _mock_soundfile


class TestIndex(unittest.TestCase):
    """Test the BlockIndex and ProjectIndex types
    """
    def setUp(self):
        with mock.patch("soundfile.SoundFile", mock_soundfile_with(frames=100)):
            b1f1 = AudioFile("asdf_1.wav")
        with mock.patch("soundfile.SoundFile", mock_soundfile_with(frames=100)):
            b1f2 = AudioFile("asdf_2.wav")

        with mock.patch("soundfile.SoundFile", mock_soundfile_with(frames=100)):
            b2f1 = AudioFile("foo_1.wav")
        with mock.patch("soundfile.SoundFile", mock_soundfile_with(frames=100)):
            b2f2 = AudioFile("foo_2.wav")

        self.block1 = Block([b1f1, b1f2], fix_uneven_frame_counts=False)
        self.block2 = Block([b2f1, b2f2], fix_uneven_frame_counts=False)
        self.project = Project([self.block1, self.block2])
        self.project2 = Project([self.block1])

    def test_project_index(self):
        pidx = ProjectIndex(self.project, 10)

        self.assertEqual(repr(pidx), "ProjectIndex<10>")
        self.assertEqual(pidx.project, self.project)

    def test_block_index(self):
        bidx = BlockIndex(self.block1, 10)

        self.assertEqual(repr(bidx), "BlockIndex<10>")
        self.assertEqual(bidx.block, self.block1)

    def test_equality_relations(self):
        """Tests equality relations between Indexes.

        Note that __eq__ and __ne__ are separate methods, so we test both using assertTrue and assertFalse"""
        # Generate some examples
        # p0 == p1, b0 == b1
        p0 = ProjectIndex(self.project, 10)
        p1 = ProjectIndex(self.project, 10)
        p2 = ProjectIndex(self.project2, 10)
        p3 = ProjectIndex(self.project, 20)
        b0 = BlockIndex(self.block1, 10)
        b1 = BlockIndex(self.block1, 10)
        b2 = BlockIndex(self.block2, 10)
        b3 = BlockIndex(self.block1, 20)

        self.assertTrue(p0 == p1)
        self.assertFalse(p0 != p1)
        self.assertTrue(p0 != p2)
        self.assertFalse(p0 == p2)
        self.assertTrue(p0 != p3)
        self.assertFalse(p0 == p3)
        with self.assertRaises(TypeError):
            p0 == b0
        with self.assertRaises(TypeError):
            p0 != b0
        with self.assertRaises(TypeError):
            p0 == 10
        with self.assertRaises(TypeError):
            p0 != 10

        self.assertTrue(b0 == b1)
        self.assertFalse(b0 != b1)
        self.assertTrue(b0 != b2)
        self.assertFalse(b0 == b2)
        self.assertTrue(b0 != b3)
        self.assertFalse(b0 == b3)
        with self.assertRaises(TypeError):
            b0 == p0
        with self.assertRaises(TypeError):
            b0 != p0
        with self.assertRaises(TypeError):
            b0 == 10
        with self.assertRaises(TypeError):
            b0 != 10

    def test_add(self):
        p0 = ProjectIndex(self.project, 10)
        self.assertEqual(p0 + 10, ProjectIndex(self.project, 20))
        self.assertEqual(p0 + 300, ProjectIndex(self.project, 310))

        b0 = BlockIndex(self.block1, 10)
        self.assertEqual(b0 + 10, BlockIndex(self.block1, 20))
        self.assertEqual(b0 + 300, BlockIndex(self.block1, 310))

    def test_subtract(self):
        p0 = ProjectIndex(self.project, 10)
        p1 = ProjectIndex(self.project, 6)
        self.assertEqual(p0 - p1, 4)
        self.assertEqual(p1 - p0, -4)
        self.assertEqual(p0 - 2, ProjectIndex(self.project, 8))
        self.assertEqual(p0 - 300, ProjectIndex(self.project, -290))

        b0 = BlockIndex(self.block1, 10)
        b1 = BlockIndex(self.block1, 6)
        self.assertEqual(b0 - b1, 4)
        self.assertEqual(b1 - b0, -4)
        self.assertEqual(b0 - 2, BlockIndex(self.block1, 8))
        self.assertEqual(b0 - 300, BlockIndex(self.block1, -290))

    def test_comparisons(self):
        """Test that <, <=, >, and >= work for indices"""
        p0 = ProjectIndex(self.project, 10)
        p1 = ProjectIndex(self.project, 10)
        p2 = ProjectIndex(self.project2, 10)
        p3 = ProjectIndex(self.project, 20)
        b0 = BlockIndex(self.block1, 10)
        b1 = BlockIndex(self.block1, 10)
        b2 = BlockIndex(self.block2, 10)
        b3 = BlockIndex(self.block1, 20)

        self.assertTrue(p0 <= p1)
        self.assertFalse(p0 < p1)
        self.assertTrue(p0 >= p1)
        self.assertFalse(p0 > p1)

        with self.assertRaises(ValueError, msg="ProjectIndex with different Project cannot be compared"):
            p0 < p2
        with self.assertRaises(ValueError, msg="ProjectIndex with different Project cannot be compared"):
            p0 <= p2
        with self.assertRaises(ValueError, msg="ProjectIndex with different Project cannot be compared"):
            p0 > p2
        with self.assertRaises(ValueError, msg="ProjectIndex with different Project cannot be compared"):
            p0 >= p2

        self.assertTrue(p0 < p3)
        self.assertTrue(p0 <= p3)
        self.assertFalse(p0 > p3)
        self.assertFalse(p0 >= p3)

        for other in [b0, 10]:
            with self.assertRaises(TypeError, msg="ProjectIndex cannot be compared to different type {}".format(other)):
                p0 < other
            with self.assertRaises(TypeError, msg="ProjectIndex cannot be compared to different type {}".format(other)):
                p0 <= other
            with self.assertRaises(TypeError, msg="ProjectIndex cannot be compared to different type {}".format(other)):
                p0 > other
            with self.assertRaises(TypeError, msg="ProjectIndex cannot be compared to different type {}".format(other)):
                p0 >= other

        self.assertTrue(b0 <= b1)
        self.assertFalse(b0 < b1)
        self.assertTrue(b0 >= b1)
        self.assertFalse(b0 > b1)

        with self.assertRaises(ValueError, msg="BlockIndex with different block cannot be compared"):
            b0 < b2
        with self.assertRaises(ValueError, msg="BlockIndex with different block cannot be compared"):
            b0 <= b2
        with self.assertRaises(ValueError, msg="BlockIndex with different block cannot be compared"):
            b0 > b2
        with self.assertRaises(ValueError, msg="BlockIndex with different block cannot be compared"):
            b0 >= b2

        self.assertTrue(b0 < b3)
        self.assertTrue(b0 <= b3)
        self.assertFalse(b0 > b3)
        self.assertFalse(b0 >= b3)

        for other in [p0, 10]:
            with self.assertRaises(TypeError, msg="BlockIndex cannot be compared to different type {}".format(other)):
                b0 < other
            with self.assertRaises(TypeError, msg="BlockIndex cannot be compared to different type {}".format(other)):
                b0 <= other
            with self.assertRaises(TypeError, msg="BlockIndex cannot be compared to different type {}".format(other)):
                b0 > other
            with self.assertRaises(TypeError, msg="BlockIndex cannot be compared to different type {}".format(other)):
                b0 >= other


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
            mock_read.return_value = np.random.random((10, 2)), f1.sampling_rate
            result = f1.read(-5, 10, channel=0)
            mock_read.assert_called_with("asdf.wav", 15, -5, dtype=np.float64)

        with mock.patch("soundfile.read") as mock_read:
            mock_read.return_value = np.random.random((self.frames - 10, self.channels)), f1.sampling_rate
            result = f1.read(10, self.frames + 100, channel=0)
            mock_read.assert_called_with("asdf.wav", self.frames - 10, 10, dtype=np.float64)

    def test_read_shape(self):
        with mock.patch("soundfile.SoundFile", self.MockSoundFile):
            f1 = AudioFile("asdf.wav")

        with mock.patch("soundfile.read") as mock_read:
            mock_read.return_value = np.random.random((10, 2)), f1.sampling_rate

            result0 = f1.read(10, 20, channel=0)
            np.testing.assert_array_equal(result0, mock_read.return_value[0][:, 0])

            result1 = f1.read(10, 20, channel=1)
            np.testing.assert_array_equal(result1, mock_read.return_value[0][:, 1])


class TestBlock(unittest.TestCase):

    def test_make_channel_mapping(self):
        """Test that make_channel_mapping assigns one channel for every channel in the AudioFiles

        Test with audio files with 2, 1 and 3 channels each. 6 channels should be created and assigned
        in order.
        """
        with mock.patch("soundfile.SoundFile", mock_soundfile_with(channels=2)):
            f1 = AudioFile("asdf_1.wav")

        with mock.patch("soundfile.SoundFile", mock_soundfile_with(channels=1)):
            f2 = AudioFile("asdf_2.wav")

        with mock.patch("soundfile.SoundFile", mock_soundfile_with(channels=3)):
            f3 = AudioFile("asdf_3.wav")

        mapping = Block.make_channel_mapping([f1, f2, f3])

        self.assertEqual(mapping[0], (f1, 0))
        self.assertEqual(mapping[1], (f1, 1))
        self.assertEqual(mapping[2], (f2, 0))
        self.assertEqual(mapping[3], (f3, 0))
        self.assertEqual(mapping[4], (f3, 1))
        self.assertEqual(mapping[5], (f3, 2))
        self.assertEqual(len(mapping), 6)

    def test_init(self):
        expect_sr = 44100
        expect_frames = 2345

        with mock.patch("soundfile.SoundFile", mock_soundfile_with(samplerate=expect_sr, channels=1, frames=expect_frames)):
            f1 = AudioFile("asdf_1.wav")
        with mock.patch("soundfile.SoundFile", mock_soundfile_with(samplerate=expect_sr, channels=2, frames=expect_frames)):
            f2 = AudioFile("asdf_2.wav")

        block = Block([f1, f2], fix_uneven_frame_counts=False)
        self.assertEqual(block.sampling_rate, expect_sr)
        self.assertEqual(block.frames, expect_frames)
        self.assertEqual(block.channels, 3)

    def test_fix_uneven_frame_counts(self):
        """Test that fix uneven frame counts events out frames across files
        """
        expect_sr = 44100
        expect_frames = 2345

        with mock.patch("soundfile.SoundFile", mock_soundfile_with(samplerate=expect_sr, channels=1, frames=expect_frames)):
            f1 = AudioFile("asdf_1.wav")
            assert f1.frames == expect_frames
        with mock.patch("soundfile.SoundFile", mock_soundfile_with(samplerate=expect_sr, channels=2, frames=expect_frames + 100)):
            f2 = AudioFile("asdf_2.wav")
            assert f2.frames == expect_frames + 100

        with self.assertRaises(ValueError):
            Block([f1, f2], fix_uneven_frame_counts=False)

        block = Block([f1, f2], fix_uneven_frame_counts=True)
        self.assertEqual(block.frames, expect_frames)

    def test_block_requires_equal_rates(self):
        with mock.patch("soundfile.SoundFile", mock_soundfile_with(samplerate=1)):
            f1 = AudioFile("asdf_1.wav")
        with mock.patch("soundfile.SoundFile", mock_soundfile_with(samplerate=2)):
            f2 = AudioFile("asdf_2.wav")

        with self.assertRaises(ValueError):
            Block([f1, f2], fix_uneven_frame_counts=False)

    def test_block_equality(self):
        with mock.patch("soundfile.SoundFile", mock_soundfile_with()):
            f1 = AudioFile("asdf_1.wav")
        with mock.patch("soundfile.SoundFile", mock_soundfile_with()):
            f2 = AudioFile("asdf_2.wav")
        with mock.patch("soundfile.SoundFile", mock_soundfile_with()):
            f3 = AudioFile("asdf_2.wav")  # Has the same path as f2

        block_12 = Block([f1, f2], fix_uneven_frame_counts=False)
        block_23 = Block([f2, f3], fix_uneven_frame_counts=False)
        block_13 = Block([f1, f3], fix_uneven_frame_counts=False)

        self.assertEqual(block_12, block_13)
        self.assertNotEqual(block_12, block_23)
        self.assertNotEqual(block_13, block_23)

    def test_get_channel_info(self):
        with mock.patch("soundfile.SoundFile", mock_soundfile_with(channels=2)):
            f1 = AudioFile("asdf_1.wav")
        with mock.patch("soundfile.SoundFile", mock_soundfile_with(channels=2)):
            f2 = AudioFile("asdf_2.wav")

        block = Block([f1, f2], fix_uneven_frame_counts=False)

        self.assertEqual(block.get_channel_info(0), ("asdf_1.wav", 0))
        self.assertEqual(block.get_channel_info(1), ("asdf_1.wav", 1))
        self.assertEqual(block.get_channel_info(2), ("asdf_2.wav", 0))
        self.assertEqual(block.get_channel_info(3), ("asdf_2.wav", 1))

    def test_block_read(self):
        with mock.patch("soundfile.SoundFile", mock_soundfile_with(channels=2)):
            f1 = AudioFile("asdf_1.wav")
        with mock.patch("soundfile.SoundFile", mock_soundfile_with(channels=2)):
            f2 = AudioFile("asdf_2.wav")

        block = Block([f1, f2], fix_uneven_frame_counts=False)

        with mock.patch("soundfile.read") as mock_read:
            mock_read.return_value = np.random.random((10, 2)), f1.sampling_rate

            result = block.read(10, 20, [0, 1])

            mock_read.assert_has_calls([
                mock.call("asdf_1.wav", 10, 10, dtype=np.float64),
                mock.call("asdf_1.wav", 10, 10, dtype=np.float64),
            ])
            self.assertEqual(result.shape, (10, 2))

    def test_block_read_with_max_frames(self):
        """Test a block read where the max frames is pinned to 15"""
        with mock.patch("soundfile.SoundFile", mock_soundfile_with(channels=2, frames=15)):
            f1 = AudioFile("asdf_1.wav")
        with mock.patch("soundfile.SoundFile", mock_soundfile_with(channels=2)):
            f2 = AudioFile("asdf_2.wav")

        block = Block([f1, f2], fix_uneven_frame_counts=True)

        with mock.patch("soundfile.read") as mock_read:
            mock_read.return_value = np.random.random((5, 2)), f1.sampling_rate

            result = block.read(10, 20, [0, 3, 1])

            mock_read.assert_has_calls([
                mock.call("asdf_1.wav", 5, 10, dtype=np.float64),
                mock.call("asdf_2.wav", 5, 10, dtype=np.float64),
                mock.call("asdf_1.wav", 5, 10, dtype=np.float64),
            ])
            self.assertEqual(result.shape, (5,  3))


class TestProject(unittest.TestCase):

    def setUp(self):
        """Set up a set of blocks and audio files

        Each block in the project should have two files, one with 1 channel and one with 2
        """
        with mock.patch("soundfile.SoundFile", mock_soundfile_with(channels=1, frames=100)):
            b1f1 = AudioFile("asdf_1.wav")
        with mock.patch("soundfile.SoundFile", mock_soundfile_with(channels=2, frames=100)):
            b1f2 = AudioFile("asdf_2.wav")

        with mock.patch("soundfile.SoundFile", mock_soundfile_with(channels=1, frames=100)):
            b2f1 = AudioFile("foo_1.wav")
        with mock.patch("soundfile.SoundFile", mock_soundfile_with(channels=2, frames=100)):
            b2f2 = AudioFile("foo_2.wav")

        with mock.patch("soundfile.SoundFile", mock_soundfile_with(channels=1, frames=200)):
            b3f1 = AudioFile("bar_1.wav")
        with mock.patch("soundfile.SoundFile", mock_soundfile_with(channels=2, frames=200)):
            b3f2 = AudioFile("bar_2.wav")

        # Block 4 will have a different sampling rate
        with mock.patch("soundfile.SoundFile", mock_soundfile_with(channels=1, samplerate=b3f1.sampling_rate + 1)):
            b4f1 = AudioFile("bad_1.wav")
        with mock.patch("soundfile.SoundFile", mock_soundfile_with(channels=2, samplerate=b3f1.sampling_rate + 1)):
            b4f2 = AudioFile("bad_2.wav")

        # Block 5 will have a different number of files but same number of channels (should be fine, but could raise an warnign)
        with mock.patch("soundfile.SoundFile", mock_soundfile_with(channels=1)):
            b5f1 = AudioFile("woof_1.wav")
            b5f2 = AudioFile("woof_2.wav")
            b5f3 = AudioFile("woof_3.wav")

        # Block 6 will have a different number of channels in one file
        with mock.patch("soundfile.SoundFile", mock_soundfile_with(channels=1)):
            b6f1 = AudioFile("baz_1.wav")
        with mock.patch("soundfile.SoundFile", mock_soundfile_with(channels=1)):
            b6f2 = AudioFile("baz_2.wav")

        self.block1 = Block([b1f1, b1f2], fix_uneven_frame_counts=False)
        self.block2 = Block([b2f1, b2f2], fix_uneven_frame_counts=False)
        self.block3 = Block([b3f1, b3f2], fix_uneven_frame_counts=False)
        self.block4 = Block([b4f1, b4f2], fix_uneven_frame_counts=False)
        self.block5 = Block([b5f1, b5f2, b5f3], fix_uneven_frame_counts=False)
        self.block6 = Block([b6f1, b6f2], fix_uneven_frame_counts=False)

    def test_init(self):
        project = Project([self.block1, self.block2, self.block3])

        self.assertEqual(project.sampling_rate, self.block3.sampling_rate)
        self.assertEqual(project.frames, self.block1.frames + self.block2.frames + self.block3.frames)
        self.assertEqual(project.channels, self.block1.channels)
        self.assertEqual([self.block1, self.block2, self.block3], project.blocks)

    def test_mismatched_blocks(self):
        """Test multiple ways blocks can be mismatched"""
        with self.assertRaises(ValueError):
            project = Project([self.block1, self.block2, self.block3, self.block4])

        with self.assertWarns(Warning):
            project = Project([self.block1, self.block2, self.block3, self.block5])

        with self.assertRaises(ValueError):
            project = Project([self.block1, self.block2, self.block3, self.block6])

    def test_iter_blocks(self):
        project = Project([self.block1, self.block2, self.block3])

        result = list(project.iter_blocks())

        self.assertEqual(
            result,
            [
                ((ProjectIndex(project, 0), ProjectIndex(project, self.block1.frames)), self.block1),
                ((ProjectIndex(project, self.block1.frames), ProjectIndex(project, 2 * self.block1.frames)), self.block2),
                ((ProjectIndex(project, 2 * self.block1.frames), ProjectIndex(project, 4 * self.block1.frames)), self.block3),
            ],
        )

    def test_read_by_project_indices(self):
        project = Project([self.block1, self.block2, self.block3])

        # Test that we can read within each block
        with mock.patch.object(self.block1, "read") as mock_read1, mock.patch.object(self.block2, "read") as mock_read2, mock.patch.object(self.block3, "read") as mock_read3:
            i0 = ProjectIndex(project, 40)
            i1 = ProjectIndex(project, 80)
            project._read_by_project_indices(i0, i1, channels=[0])

            mock_read1.assert_called_with(40, 80, channels=[0])

    def test_read(self):
        """Test that read can be called with any indices"""
        project = Project([self.block1, self.block2])

        b0 = BlockIndex(self.block2, 10)
        b1 = BlockIndex(self.block2, 50)
        p0 = ProjectIndex(project, 110)
        p1 = ProjectIndex(project, 150)


        with mock.patch.object(project, "_read_by_project_indices") as mock_fn:
            project.read(b0, b1, channels=[0])
            mock_fn.assert_called_with(
                ProjectIndex(project, 110),
                ProjectIndex(project, 150),
                [0]
            )

        with mock.patch.object(project, "_read_by_project_indices") as mock_fn:
            project.read(p0, p1, channels=[0])
            mock_fn.assert_called_with(
                ProjectIndex(project, 110),
                ProjectIndex(project, 150),
                [0]
            )

        with mock.patch.object(project, "_read_by_project_indices") as mock_fn:
            project.read(b0, p1, channels=[0])
            mock_fn.assert_called_with(
                ProjectIndex(project, 110),
                ProjectIndex(project, 150),
                [0]
            )

        with mock.patch.object(project, "_read_by_project_indices") as mock_fn:
            project.read(p0, b1, channels=[0])
            mock_fn.assert_called_with(
                ProjectIndex(project, 110),
                ProjectIndex(project, 150),
                [0]
            )

    def test_normalize_slice_both_none(self):
        """Test that normalizing a slice with both sides None gives the whole Project"""
        project = Project([self.block1, self.block2])
        self.assertEqual(
            project.normalize_slice(slice(None, None)),
            slice(ProjectIndex(project, 0), ProjectIndex(project, project.frames))
        )

    def test_normalize_slice_both_project_index(self):
        project = Project([self.block1, self.block2])
        i0 = ProjectIndex(project, 25)
        i1 = ProjectIndex(project, 99)
        self.assertEqual(
            project.normalize_slice(slice(i0, i1)),
            slice(ProjectIndex(project, 25), ProjectIndex(project, 99))
        )

    def test_normalize_slice_both_block_index(self):
        project = Project([self.block1, self.block2])
        i0 = BlockIndex(self.block1, 25)
        i1 = BlockIndex(self.block2, 1)
        self.assertEqual(
            project.normalize_slice(slice(i0, i1)),
            slice(ProjectIndex(project, 25), ProjectIndex(project, 101))
        )

    def test_normalize_slice_one_project_index(self):
        project = Project([self.block1, self.block2])
        i0 = ProjectIndex(project, 25)
        self.assertEqual(
            project.normalize_slice(slice(i0, None)),
            slice(ProjectIndex(project, 25), ProjectIndex(project, project.frames))
        )
        self.assertEqual(
            project.normalize_slice(slice(None, i0)),
            slice(ProjectIndex(project, 0), ProjectIndex(project, 25))
        )

    def test_normalize_slice_one_block_index(self):
        project = Project([self.block1, self.block2])
        i0 = BlockIndex(self.block1, 25)
        self.assertEqual(
            project.normalize_slice(slice(i0, None)),
            slice(ProjectIndex(project, 25), ProjectIndex(project, 100))
        )
        self.assertEqual(
            project.normalize_slice(slice(None, i0)),
            slice(ProjectIndex(project, 0), ProjectIndex(project, 25))
        )

    def test_normalize_slice_one_block_one_project(self):
        project = Project([self.block1, self.block2])

        i0 = BlockIndex(self.block1, 25)
        i1 = ProjectIndex(project, 101)
        self.assertEqual(
            project.normalize_slice(slice(i0, i1)),
            slice(ProjectIndex(project, 25), ProjectIndex(project, 101))
        )

        i0 = ProjectIndex(project, 25)
        i1 = BlockIndex(self.block2, 1)
        self.assertEqual(
            project.normalize_slice(slice(i0, i1)),
            slice(ProjectIndex(project, 25), ProjectIndex(project, 101))
        )

    @mock.patch("soundsep.core.models.Project._read_by_project_indices")
    def test_getitem_all_channels(self, mock_fn):
        """Test the primary data access on the Project through bracket notation
        """
        p = Project([self.block1, self.block2])

        p0 = ProjectIndex(p, 40)
        p1 = ProjectIndex(p, 160)
        b0 = BlockIndex(self.block1, 40)
        b1 = BlockIndex(self.block2, 60)

        p[p0:p1]
        mock_fn.assert_called_with(ProjectIndex(p, 40), ProjectIndex(p, 160), channels=[0, 1, 2])

        p[b0:b1]
        mock_fn.assert_called_with(ProjectIndex(p, 40), ProjectIndex(p, 160), channels=[0, 1, 2])

        p[b0:p1]
        mock_fn.assert_called_with(ProjectIndex(p, 40), ProjectIndex(p, 160), channels=[0, 1, 2])

        p[p0:b1]
        mock_fn.assert_called_with(ProjectIndex(p, 40), ProjectIndex(p, 160), channels=[0, 1, 2])

    @mock.patch("soundsep.core.models.Block.read_one")
    @mock.patch("soundsep.core.models.Project._read_by_project_indices")
    def test_getitem_one_sample(self, mock_fn, mock_read_one):
        p = Project([self.block1, self.block2])

        p0 = ProjectIndex(p, 40)

        p[p0]
        mock_fn.assert_not_called()
        mock_read_one.assert_called_with(BlockIndex(self.block1, 40), channels=[0, 1, 2])

        p[p0, :2]
        mock_fn.assert_not_called()
        mock_read_one.assert_called_with(BlockIndex(self.block1, 40), channels=[0, 1])

        p[p0, 2]
        mock_fn.assert_not_called()
        mock_read_one.assert_called_with(BlockIndex(self.block1, 40), channels=[2])

    @mock.patch("soundsep.core.models.Project._read_by_project_indices")
    def test_getitem_select_channels(self, mock_fn):
        """Test the primary data access on the Project through bracket notation
        """
        p = Project([self.block1, self.block2])

        p[:, [0, 1]]
        mock_fn.assert_called_with(ProjectIndex(p, 0), ProjectIndex(p, p.frames), channels=[0, 1])

        p[:, [0, 1, 1]]
        mock_fn.assert_called_with(ProjectIndex(p, 0), ProjectIndex(p, p.frames), channels=[0, 1, 1])

        p[:, range(1, 3)]
        mock_fn.assert_called_with(ProjectIndex(p, 0), ProjectIndex(p, p.frames), channels=[1, 2])

        p[:, :2]
        mock_fn.assert_called_with(ProjectIndex(p, 0), ProjectIndex(p, p.frames), channels=[0, 1])

        p[:, :]
        mock_fn.assert_called_with(ProjectIndex(p, 0), ProjectIndex(p, p.frames), channels=[0, 1, 2])

        p[:, 1:]
        mock_fn.assert_called_with(ProjectIndex(p, 0), ProjectIndex(p, p.frames), channels=[1, 2])

        p[:, -2:]
        mock_fn.assert_called_with(ProjectIndex(p, 0), ProjectIndex(p, p.frames), channels=[1, 2])

        p[:, :-2]
        mock_fn.assert_called_with(ProjectIndex(p, 0), ProjectIndex(p, p.frames), channels=[0])

        p[:, -4:-1]
        mock_fn.assert_called_with(ProjectIndex(p, 0), ProjectIndex(p, p.frames), channels=[0, 1])

        p[:, 1]
        mock_fn.assert_called_with(ProjectIndex(p, 0), ProjectIndex(p, p.frames), channels=[1])

    def test_to_block_index(self):
        """Test conversion of an index to a block index"""
        project = Project([self.block1, self.block2])

        p0 = ProjectIndex(project, 40)
        p1 = ProjectIndex(project, 160)
        b0 = BlockIndex(self.block1, 40)
        b1 = BlockIndex(self.block2, 60)

        self.assertEqual(project.to_block_index(p0), b0, "ProjectIndex should map to the correct BlockIndex")
        self.assertEqual(project.to_block_index(p1), b1, "ProjectIndex should map to the correct BlockIndex")
        self.assertEqual(project.to_block_index(b0), b0, "BlockIndex should be mapped to itself")

    def test_to_project_index(self):
        """Test conversion of an index to a project index"""
        project = Project([self.block1, self.block2])

        p0 = ProjectIndex(project, 40)
        p1 = ProjectIndex(project, 160)
        b0 = BlockIndex(self.block1, 40)
        b1 = BlockIndex(self.block2, 60)

        self.assertEqual(project.to_project_index(b0), p0, "BlockIndex should map to the correct ProjectIndex")
        self.assertEqual(project.to_project_index(b1), p1, "BlockIndex should map to the correct ProjectIndex")
        self.assertEqual(project.to_project_index(p0), p0, "ProjectIndex should be mapped to itself")

    def test_get_block_boundaries_by_project_index(self):
        project = Project([self.block1, self.block2, self.block3])

        i0 = ProjectIndex(project, 0)
        i1 = ProjectIndex(project, project.frames)
        self.assertEqual(
            project.get_block_boundaries(i0, i1),
            [
                ProjectIndex(project, 0),
                ProjectIndex(project, 100),
                ProjectIndex(project, 200),
                ProjectIndex(project, 400)
            ]
        )

        i0 = ProjectIndex(project, 40)
        i1 = ProjectIndex(project, 200)
        self.assertEqual(
            project.get_block_boundaries(i0, i1),
            [
                ProjectIndex(project, 100),
                ProjectIndex(project, 200),
            ]
        )

    def test_get_block_boundaries_by_block_index(self):
        project = Project([self.block1, self.block2, self.block3])

        i0 = BlockIndex(self.block1, 0)
        i1 = BlockIndex(self.block3, self.block3.frames)
        self.assertEqual(
            project.get_block_boundaries(i0, i1),
            [
                ProjectIndex(project, 0),
                ProjectIndex(project, 100),
                ProjectIndex(project, 200),
                ProjectIndex(project, 400)
            ]
        )

        i0 = BlockIndex(self.block1, 40)
        i1 = BlockIndex(self.block2, 80)
        self.assertEqual(
            project.get_block_boundaries(i0, i1),
            [
                ProjectIndex(project, 100),
            ]
        )

    def test_get_block_boundaries_by_mixed_index(self):
        project = Project([self.block1, self.block2, self.block3])

        i0 = BlockIndex(self.block1, 40)
        i1 = ProjectIndex(project, 300)
        self.assertEqual(
            project.get_block_boundaries(i0, i1),
            [
                ProjectIndex(project, 100),
                ProjectIndex(project, 200),
            ]
        )

        i0 = ProjectIndex(project, 100)
        i1 = BlockIndex(self.block3, 10)
        self.assertEqual(
            project.get_block_boundaries(i0, i1),
            [
                ProjectIndex(project, 100),
                ProjectIndex(project, 200),
            ]
        )
