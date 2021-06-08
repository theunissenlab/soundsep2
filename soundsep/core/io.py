"""Classes to organize and link together wav files

One of the goals in this project is to make it easy to read in audio data
that is recorded in a variety of ways.

(i)   Audio data on multiple channels will be recorded simultaneously (the primary
        purpose of SoundSep is to deal with this kind of data) but each channel may
        be stored in separate files. These classes should unify those cases
        under a single API for reading
(ii)  Audio data recorded simultaneously may or may not be perfectly synchronized
        and may include an uneven number of samples. As long as the user is okay
        with this, our data structures should allow for a bit of fudging to
        align the samples
(iii) Audio data recorded simulataneously may not even be recorded at the same
        sampling rate. We won't support this in the first version of SoundSep, but
        design the indexing strategy with this in mind. The issue with supporting this
        is that the Block structure (see below) wants to abstract the fact that there
        are multiple Files but can't do that if different channels have different sampling
        rates.
(iv)  Some users may break up long recordings into multiple files by time, where
        boundaries between files are arbitrary and data is continuous between them.
        Other users may want to analyze an arbitrary set of files, where the boundaries
        between files are real and slices made across boundaries make no sense. We 
        should handle both cases.

In SoundSep we want to handle these cases. To do this, we will provide users with
configuration options to tell us how these should be handled.

Configuration Options
---------------------
FileOrganization: FileOrganization (i)
FixUnevenFrameCounts: bool (ii)
AllowCrossBlockSlices: bool (iv)

The solution used in SoundSep is a hierarchical structure

|Project                               |
|--------------------------------------|
|Block   |Block   |Block     |...      |
|--------------------------------------|
|File    |File    |File      |...      |
|File    |File    |File      |...      |
|...     |...     |...       |...      |

Each File (represented by the _AudioFile) class points to a .wav file on disk and stores
its sampling rate, length (number of frames), and number of channels.

Each Block contains a list of _AudioFiles, and it is required that all the contained AudioFiles_
share the same sampling rate and number of frames. If FixUnevenFrameCounts is True,
a Block can be initialized with uneven frame counts, and each file will be artificially truncated
at the shared min frame count (note that the audio files themselves are never edited).

The idea of the Block is to abstract the fact that there are potentially multiple wav files recorded
simultaneously and allow reading from them as if they were a single file. And so each channel in each
File is represented as a single channel in the Block. The Block can unambiguously refer to its channels
by a tuple of (_AudioFile, int).

Finally, the Project is what links the Blocks together. It knows how the blocks link to each other
(the order) and also whether they should be considered to be continuous in time (if AllowCrossBlockSlices
is True). The last requirement of the Project is that the channel ordering in each block is consistent.
Since only the Project knows about all the Blocks at once, it is required to define a function that
orders the _AudioFiles of each Block in a consistent manner. This ordering function will utilize filenames
and file naming convention must be well documented and controlled by the user. The API of the Project
will allow for reading data by slices and viewing the breakpoints between Blocks by index.

Indexing for each of these structures will be managed by specific Index types to keep things organized.
* An _AudioFile is indexed by integers.
* A Block is indexed by integers.
* A Project can be indexed by a global index (ProjectIndex(int)), or a BlockIndex(Block, int) that points
    to a specific index within a Block. Since the Project knows the order and sizes of the blocks,
    it can easily translate between the two indices.

One question is, how much of the underlying solution do we want to expose to (1) the user
and (2) a developer of a plugin? While it would be nice to keep Project as the only public
facing object and the Block and AudioFile as private, usability requirements make it more likely that
making all of these objects public will be necessary.

For users
Users will be responsible for preparing and organizing their data files so that they can be read 
    by the program. This means they need at least implicit understanding of the way a Project can use
    filenames to link corresponding channels across files/Blocks. For example, a typical filename
    convention might look like "Recording_{timestamp}_ch{n}.wav", where the timestamp can be used as
    a Block identified, and n could be used as a secondary sorting method to keep channel order consistent

For developers
Developers who want to create plugins or modify the code for specialized projects will need to be able
    to store references to positions in the data where events or annotations occur. The cleanest way
    to do this is to refer to a (Block, int) index. Thus they need to at least understand the Block
    organization and have access to those object identifiers.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from functools import wraps
from typing import List, Union

import numpy as np
import soundfile


def match_type(fn):
    """Decorator for wrapping methods with one argument to enforce types match
    """
    @wraps
    def _wrapped(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError("Mismatching types for {}: {} and {}".format(
                fn.__name__,
                self.__class__,
                other.__class__
            ))
        return fn(self, other)
    return _wrapped


class AudioFile:
    """Container for a audio file on disk

    Attributes
    ----------
    max_frame : int
        Last frame that can be read from. Reads beyond the given frame will be cut off.

    Methods
    -------
    read(i0: int, i1: int, channel: int) -> numpy.ndarray
    """

    def __init__(self, path):
        """

        Arguments
        ---------
        path : str
            Full path to the audio file on disk
        """
        self._path = path
        self._max_frame = None

        with soundfile.SoundFile(path) as f:
            self._sampling_rate = f.samplerate
            self._channels = f.channels
            self._actual_frames = f.frames

    def __repr__(self):
        return "<AudioFile object: {}; {} Hz; {} Ch; {} frames>".format(
            os.path.basename(self._path),
            self.sampling_rate,
            self.channels,
            self.frames
        )

    def set_max_frame(self, frames):
        """Set the maximum frame to read from the file
        
        This can be used to force multiple AudioFiles to behave as if they have the
        same duration. Reads beyond the given frame will be cut off.

        Arguments
        ---------
        frames : int, optional
            Truncate reads from this file to force_frames (treat this as the length
            of the file rather than its actual length). 
        """
        if not isinstance(frames, int) or frames <= 0:
            raise ValueError("max_frame must be a positive integer or None")
        if frames > self._actual_frames:
            raise RuntimeError("Cannot force AudioFile to use more frames than on disk")

        self._max_frame = frames

    def clear_max_frame(self):
        self._max_frame = None

    def __eq__(self, other_file):
        if isinstance(other_file, AudioFile):
            return self.path == other_file.path
        else:
            raise ValueError("Can only compare AudioFile equality with other AudioFiles")

    @property
    def path(self):
        """str: Full path to audio file"""
        return self._path

    @property
    def sampling_rate(self):
        """int: Sampling rate of the audio file"""
        return self._sampling_rate

    @property
    def frames(self):
        """int: Number of readable samples in audio file"""
        return self._max_frame or self._actual_frames 

    @property
    def channels(self):
        """int: Number of samples in audio file"""
        return self._channels

    def read(self, i0: int, i1: int, channel: int):
        """Read samples from i0 to i1 on channel

        Arguments
        ---------
        i0 : int
            Starting index to read from (inclusive)
        i1 : int
            Ending index to read until (exclusive)
        channel : int
            Channel index to read

        Returns
        -------
        ndarray
            A 2D array of shape (frames: int, 1) containing data from the requested channel.
            The first dimension is the sample index, the second dimension is the channel
            axis, although this function only returns one channel.
        """
        read_start = i0
        read_stop = min(i1, self.frames)

        result = soundfile.read(self.path, read_stop - read_start, read_start, dtype=np.float64)
        
        if result.ndim == 1:
            result = result[:, None]

        return result


class Block:
    """Wrapper for simultaneously recorded audio files

    Methods
    -------
    read(i0: int, i1: int, channels: list[int]) -> numpy.ndarray
        Reads data from i0 to i1 in the block on the selected channels
    """

    @staticmethod
    def make_channel_mapping(files: List[AudioFile]):
        """Produce a mapping channel_block -> (AudioFile, channel_file)

        Given n AudioFile objects and k_i channels on the ith AudioFile,
        there are sum(k) total channels across the files.

        make_channel_mapping produces a dict that maps

        x: int -> (y: AudioFile, z: int)

        Where x is in [0, sum(k)) and references the zth channel of AudioFile y.
        """
        i = 0
        mapping = {}
        for f in files:
            for c in range(f.channels):
                mapping[i] = (f, c)
                i += 1

        return mapping

    def __init__(self, audio_files: List[AudioFile], fix_uneven_frame_counts: bool):
        if not len(audio_files):
            raise ValueError("Cannot instantiate Block with no files")

        self._files = audio_files

        # Validate that all the files match up
        rates = [f.sampling_rate for f in self._files]
        frames = [f.frames for f in self._files]

        if not all([r == rates[0] for r in rates]):
            raise ValueError("Cannot instantiate Block with files of different rates: {}".format(rates))

        if not all([f == frames[0] for f in frames]):
            if fix_uneven_frame_counts:
                min_frame = np.min(frames)
                for f in self._files:
                    f.set_max_frame(min_frame)
            else:
                raise ValueError("Cannot instantiate Block with files of different lengths: {}".format(frames))

        self._sampling_rate = rates[0]
        self._frames = frames[0]
        self._channel_mapping = Block.make_channel_mapping(self._files)

    def __repr__(self):
        return "<Block: {} files; {} Hz; {} Ch; {} frames>".format(
            len(self._files),
            self.sampling_rate,
            self.channels,
            self.frames,
        )

    def __eq__(self, other_block):
        if isinstance(other_block, Block):
            if len(other_block._files) != len(self._files):
                return False
            return all([f1 == f2 for f1, f2 in zip(self._files, other_block._files)])
        else:
            raise ValueError("Can only compare Block equality with other Block")

    @property
    def channel_mapping(self):
        return self._channel_mapping

    def get_channel_info(self, channel: int):
        """Get the original file path and index of the Block's channel

        Arguments
        ---------
        channel : int
            Block's channel to request info from

        Returns
        -------
        path : str
            Path to file corresponding to requested channel
        index : int
            Index of channel in original file
        """
        original_file, original_channel = self._channel_mapping[channel]
        return original_file.path, original_channel

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @property
    def frames(self):
        return self._frames

    @property
    def channels(self):
        return len(self._channel_mapping)

    def read(self, i0: int, i1: int, channels: List[int]):
        output = np.zeros((i1 - i0, len(channels)))
        for i, ch in enumerate(channels):
            (audio_file, audio_file_ch) = self.channel_mapping[ch]
            output[:, i] = audio_file.read(i0, i1, audio_file_ch)

        return output

    def read_one(self, i, channels:List[int]):
        return self.read(i, i+1, channels)


class Project:
    """Data access to audio data in a series of Blocks

    Attributes
    ----------
    blocks : List[Block]
    channels : int
    sampling_rate : int
    channels : int
    frames : int
        Number of samples in entire project

    Methods
    -------
    .iter_blocks()
    .read(start, stop, channels)
    .to_block_index(index)
    .to_project_index(index)

    Examples
    --------
    Data can be read from a project using the .read() method or square bracket
    access notation. Data can be accessed using either BlockIndex or ProjectIndex
    values, but not raw integers, to enforce consistency.

    >>> project = Project(...)
    >>> start = BlockIndex(block, 10)
    >>> end = BlockIndex(block, 20)

    Accessing one frame can be done with a single index
    >>> project[start]

    Accessing a range can be done with a slice object. The following are equivalent
    >>> project[start:end]
    >>> project.read(start, end)  # Equivalent to the line above

    Selecting the channels to read can be specified as an argument to .read() or
    as the second slice in the bracket notation. The following are all equivalent

    >>> project[start:end, :4]
    >>> project[start:end, range(4)]
    >>> project.read(start, end, channels=[0, 1, 2, 3])

    Slice behavior using BlockIndex and ProjectIndex follows the scheme that if
    one end is a BlockIndex and the other is None, the None will be interpreted
    as the end of that block. Otherwise, None values refer to the endpoints of
    the entire Project.
    """

    def __init__(self, blocks: List[Block]):# , allow_cross_block_slices: bool):
        """Initialize the Project
        """
        self._blocks = blocks
        # Enforce that all WavBlocks have the same channel configuration

    def __repr__(self):
        return "<Project: {} blocks; {} Hz; {} Ch; {} frames>".format(
            len(self.blocks),
            self.sampling_rate,
            self.channels,
            self.frames,
        )

    @property
    def channels(self):
        return self._blocks[0].channels

    @property
    def sampling_rate(self):
        return self._blocks[0].sampling_rate

    @property
    def frames(self):
        return np.sum([b.frames for b in self.blocks])

    @property
    def blocks(self):
        return self._blocks
    
    @property
    def iter_blocks(self):
        """Iterate over ((start_idx, stop_idx), Block) pairs"""
        frame = 0
        for block in self.blocks:
            yield ((ProjectIndex(self, frame), ProjectIndex(self, frame + block.frames)), block)
            frame += block.frames

    def read(
            self,
            start: Union['BlockIndex', 'ProjectIndex'],
            stop: Union['BlockIndex', 'ProjectIndex'],
            channels: List[int]
        ):
        """Reads data of a start->stop slice, can use ProjectIndex or BlockIndex values"""
        if isinstance(start, BlockIndex):
            start = self.convert_block2project(start)
        elif isinstance(start, ProjectIndex):
            pass
        else:
            raise TypeError("Can only read from project using BlockIndex or ProjectIndex")

        if isinstance(stop, BlockIndex):
            stop = self.convert_block2project(stop)
        elif isinstance(start, ProjectIndex):
            pass
        else:
            raise TypeError("Can only read from project using BlockIndex or ProjectIndex")
        
        return self._read_by_project_indices(start, stop, channels)

    def _read_by_project_indices(self, start: 'ProjectIndex', stop: 'ProjectIndex', channels: List[int]):
        """Reads slice's data from one or more Blocks in project"""
        out_data = []
        for (i0, i1), block in self.iter_blocks():
            if i1 < self.start:
                continue
            elif i0 > self.stop:
                break
            else:
                # block.read() takes normal ints; if we enforce it to take BlockIndex instead, we will
                # need to cast these its
                block_read_start = max(self.start - i0, 0)
                block_read_stop = min(self.stop - i0, block.frames)
                out_data.append(block.read(block_read_start, block_read_stop, channels=channels))

        return np.concatenate(out_data)

    def _normalize_slice(self, slice_: slice) -> slice:
        """Convert slice of ProjectIndex or BlockIndex values to a slice with explicit endpoints

        This function makes the None values in a slice explicit relative to the Block or Project,
        depending on the other values.

        Example
        -------
        >>> project._normalize_slice(slice(BlockIndex(block, 3), None))
        slice(ProjectIndex<3>, ProjectIndex<10>, None)

        Arguments
        ---------
        slice_ : slice
            A slice(start, stop, step) where start and stop are ProjectIndex, BlockIndex, or both. The
            step attribute of the slice is ignored.

        Returns
        -------
        normalized_slice : slice
            A slice whose start and stop values are both ProjectIndex values corresponding to the input
            slice_. The step attribute is preserved. (note that step will typically be ignored).
            If the input had None, it is filled in with the Block endpoint if the other element was
            a BlockIndex, or ProjectIndex otherwise. If both start and stop were None, returns
            a slice representing the entire Project.
        """
        if slice_.start is None and slice_.stop is None:
            start = ProjectIndex(self, 0)
            stop = ProjectIndex(self, self.frames)
        elif slice_.start is None:
            if isinstance(slice_.stop, BlockIndex):
                start = self.to_project_index(BlockIndex(slice_.stop.block, 0))
                stop = self.to_project_index(slice_.stop)
            elif isinstance(slice_.stop, ProjectIndex):
                start = ProjectIndex(self, 0)
                stop = slice_.stop
            else:
                raise TypeError("Can only normalize slices with ProjectIndex, BlockIndex, or None values")
        elif slice_.stop is None:
            if isinstance(slice_.start, BlockIndex):
                start = self.to_project_index(slice_.start)
                stop = self.to_project_index(BlockIndex(slice_.start.block, slice_.start.block.frames))
            elif isinstance(slice_.start, ProjectIndex):
                start = slice_.start
                stop = ProjectIndex(self, slice_.start.block.frames)
            else:
                raise TypeError("Can only normalize slices with ProjectIndex, BlockIndex, or None values")
        else:
            start = self.to_project_index(slice_.start)
            stop = self.to_project_index(slice_.stop)

        return slice(start, stop, None)

    def __getitem__(self, slices):
        """Main data access of Project via Block coordinates or Project coordinates

        See Project class documentation for usage examples

        First parameters to the slice selects the indices, and the second parameter (optional)
        selects the channels. The second parameter can either be an int, Python slice object,
        or iterable.
        """
        # This function must handle four cases: (int, int), (int, slice), (slice, int), (slice, slice)
        # The second value could be an iterable as well
        if isinstance(slices, tuple):
            if not len(slices, 2):
                raise ValueError("Invalid index into Project of length {}".format(len(slices)))

            s1, s2 = slices

            if isinstance(s1, slice):
                s1 = self._normalize_slice(s1)

            if isinstance(s2, slice):
                s2 = s2.indices(self.channels)
                s2 = list(range(s2.start, s2.stop, s2.step))

            if isinstance(s1, BaseIndex) and isinstance(s2, int):
                index = self.to_block_index(s1)
                return index.block.read_one(index, channels=[s2])
            elif isinstance(s1, BaseIndex) and isinstance(s2, Iterable):
                index = self.to_block_index(s1)
                return index.block.read_one(index, channels=list(s2))
            elif isinstance(s1, slice) and isinstance(s2, int):
                return self._read_by_project_indices(s1.start, s1.stop, channels=[s2])
            elif isinstance(s1, slice) and isinstance(s2, Iterable):
                return self._read_by_project_indices(s1.start, s1.stop, channels=list(s2))
            else:
                raise TypeError("Invalid types for Project __getitem__ access: {} and {}".format(s1, s2))
        elif isinstance(slices, BaseIndex):
            index = self.to_block_index(slices)
            return index.block.read_one(index, list(range(self.channels)))
        elif isinstance(slices, slice):
            slice_ = self._normalize_slice(slices)
            return self._read_by_project_indices(slice_.start, slice_.stop, channels=list(range(self.channels)))

        raise ValueError("Invalid index into Project {}".format(slices))

    def to_block_index(self, index: Union['BlockIndex', 'ProjectIndex']) -> 'BlockIndex':
        """Convert a BlockIndex/ProjectIndex to a BlockIndex
        
        Arguments
        ---------
        index : ProjectIndex or BlockIndex

        Returns
        -------
        block_index : BlockIndex
            Index relative to a Block in the Project corresponding to the given index
        """
        if isinstance(index, BlockIndex):
            return index
        elif isinstance(index, ProjectIndex):
            for (i0, i1), block in project.iter_blocks():
                if i1 > project_index:
                    return BlockIndex(block, project_index - i0)
            raise ValueError("Could not find BlockIndex in Project")
        else:
            raise TypeError("Cannot covert type {} to BlockIndex".format(type(index)))


    def to_project_index(self, index: Union['BlockIndex', 'ProjectIndex']) -> 'ProjectIndex':
        """Convert a BlockIndex/ProjectIndex to a ProjectIndex

        Arguments
        ---------
        index : ProjectIndex or BlockIndex

        Returns
        -------
        project_index : ProjectIndex
            Index to data relative to the entire Project
        """
        if isinstance(index, ProjectIndex):
            return index
        elif isinstance(index, BlockIndex):
            for (i0, i1), block in project.iter_blocks():
                if block == block_index.block:
                    return ProjectIndex(self, i0 + block_index)
            raise ValueError("Could not find BlockIndex in Project")
        else:
            raise TypeError("Cannot covert type {} to ProjectIndex".format(type(index)))


# TODO: re-evaluate if inheriting from int is worth it or if it will be more likely to cause
# more problems than later (due to all the methods they will inherit)
class BaseIndex(int):

    ObjectType = None

    def __new__(cls, source_object, value: int):
        if not isinstance(source_object, cls.ObjectType):
            raise TypeError("Index of type {} must be instantiated with {}".format(cls, cls.ObjectType))

        return int.__new__(cls, value)

    def __init__(cls, source_object, value: int):
        if value < -source_object.frames:
            value = -source_object.frames
        elif value > source_object.frames:
            value = source_object.frames

        self._source_object = source_object

        super().__init__(value)

    def __repr__(self):
        return "{}<{}>".format(self.__class__, int.__repr__(self))

    __lt__ = match_type(int.__lt__)
    __gt__ = match_type(int.__gt__)
    __le__ = match_type(int.__le__)
    __ge__ = match_type(int.__ge__)

    @match_type
    def __eq__(self, other):
        return (other._source_object is self._source_object) and self == other

    def __add__(self, other: int):
        """BaseIndex + int -> BaseIndex"""
        return self.__class__(self._source_object, super().__add__(self, other))

    def __sub__(self, other: int):
        """BaseIndex - int -> BaseIndex | BaseIndex - BaseIndex -> int"""
        if isinstance(other, self.__class__):
            return super().__sub__(self, other)
        elif isinstance(other, int):
            return self.__class__(self._source_object, super().__sub__(self, other))


class ProjectIndex(BaseIndex):
    """An integer index that is a global index into a specific Project

    The range of values are clamped to the bounds of the Project itself

    Arguments
    ---------
    source_object : soundsep.io.Project
        Project for which the index is valid
    value : int
        Frame within the Project that the index refers to

    Example
    -------
    >>> pidx = ProjectIndex(project, 10)
    """

    ObjectType = Project

    @property
    def project(self):
        return self._source_object


class BlockIndex(BaseIndex):
    """An integer index that is local to a specific Block

    The range of values are clamped to the bounds of the Block

    Arguments
    ---------
    source_object : soundsep.io.Block
        Project for which the index is valid
    value : int
        Frame within the Project that the index refers to

    Example
    -------
    >>> bidx = BlockIndex(block, 10)
    """
    ObjectType = Block

    @property
    def block(self):
        return self._source_object


