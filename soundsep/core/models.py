"""Data structures for abstracting the audio file organization on disk and indexing into data
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterable
from functools import wraps
from typing import Dict, List, Tuple, Union

import numpy as np
import soundfile


class AudioFile:
    """Container for a audio file on disk

    Arguments
    ---------
    path : str
        Full path to the audio file on disk
    """

    def __init__(self, path):
        self._path = path
        self._max_frame = None
        self._file = None

        with soundfile.SoundFile(path) as f:
            self._sampling_rate = f.samplerate
            self._channels = f.channels
            self._actual_frames = f.frames

    def is_open(self):
        return self._file is not None and not self._file.closed

    def is_closed(self):
        return self._file is None or self._file.closed

    def open(self):
        if not self.is_open():
            self._file = soundfile.SoundFile(self._path, "r")

    def close(self):
        if self.is_open():
            self._file.close()

    def __repr__(self):
        return "<AudioFile: {}; {} Hz; {} Ch; {} frames>".format(
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
            raise ValueError("max_frame must be a positive integer or None: got {}".format(frames))
        if frames > self._actual_frames:
            raise RuntimeError("Cannot force AudioFile to use more frames than on disk")

        self._max_frame = frames

    def clear_max_frame(self):
        self._max_frame = None

    def __eq__(self, other_file) -> bool:
        if isinstance(other_file, AudioFile):
            return self.path == other_file.path
        else:
            raise ValueError("Can only compare AudioFile equality with other AudioFiles")

    def __hash__(self):
        return id(self)

    @property
    def path(self) -> str:
        """str: Full path to audio file"""
        return self._path

    @property
    def sampling_rate(self) -> int:
        """int: Sampling rate of the audio file"""
        return self._sampling_rate

    @property
    def frames(self) -> int:
        """int: Number of readable samples in audio file"""
        return self._max_frame or self._actual_frames

    @property
    def channels(self) -> int:
        """int: Number of channels in audio file"""
        return self._channels

    def read(self, i0: int, i1: int) -> np.ndarray:
        """Read samples from i0 to i1 on channel

        Arguments
        ---------
        i0 : int
            Starting index to read from (inclusive)
        i1 : int
            Ending index to read until (exclusive)

        Returns
        -------
        data : ndarray
            A 2D array of shape (frames: int, channels: int) containing data from the requested channel.
            The first dimension is the sample index, the second dimension is the channel
            axis.
        """
        read_start = i0
        read_stop = min(i1, self.frames)

        if self.is_closed():
            self.open()

        self._file.seek(read_start)
        return self._file.read(read_stop - read_start, dtype=np.float32, always_2d=True)


class Block:
    """Wrapper for simultaneously recorded audio files

    Arguments
    ---------
    audio_files : List[AudioFile]
        List of audio files in channel order to include in the Block
    fix_uneven_frame_counts : bool
        If set to True, will treat all AudioFiles in Block as having the same
        number of samples as the shortest file.
    """

    @staticmethod
    def make_channel_mapping(files: List[AudioFile]) -> Dict[int, Tuple[AudioFile, int]]:
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
                min_frame = int(np.min(frames))
                for f in self._files:
                    f.set_max_frame(min_frame)
            else:
                raise ValueError("Cannot instantiate Block with files of different lengths: {}".format(frames))

        self._sampling_rate = rates[0]
        self._frames = frames[0]
        self._channel_mapping = Block.make_channel_mapping(self._files)

    def __repr__(self):
        return "<Block: {} files; {} Ch>".format(
            len(self._files),
            self.channels,
        )

    def __eq__(self, other_block: 'Block') -> bool:
        if isinstance(other_block, Block):
            if len(other_block._files) != len(self._files):
                return False
            return all([f1 == f2 for f1, f2 in zip(self._files, other_block._files)])
        else:
            raise ValueError("Can only compare Block equality with other Block")

    @property
    def channel_mapping(self) -> Dict[int, Tuple[AudioFile, int]]:
        """Dict[int, Tuple[AudioFile, int]]: Mapping from channel number in Block to channel number in an AudioFile"""
        return self._channel_mapping

    def get_channel_info(self, channel: int) -> Tuple[str, int]:
        """Get the original file path and index of the Block's channel

        Arguments
        ---------
        channel : int
            Block's channel to request info from

        Returns
        -------
        info : Tuple[str, int]
            A tuple where the first element is the path to file corresponding to
            the requested channel, and the second element is the channel index
            within that file corresponding to that channel.
        """
        original_file, original_channel = self._channel_mapping[channel]
        return original_file.path, original_channel

    def lookup_channel_info(self, audio_file: AudioFile, channel: int) -> int:
        for block_channel, (f, c) in self._channel_mapping.items():
            if f == audio_file and c == channel:
                return block_channel
        raise ValueError("Audio file and channel not found in block")

    @property
    def sampling_rate(self) -> int:
        """int: Sampling rate of the Block"""
        return self._sampling_rate

    @property
    def frames(self) -> int:
        """int: Number of readable samples in Block"""
        return self._frames

    @property
    def channels(self) -> int:
        """int: Number of channels in Block"""
        return len(self._channel_mapping)

    def close_files(self):
        for f in self._files:
            if f.is_open():
                f.close()

    def read(self, i0: int, i1: int, channels: List[int]) -> np.ndarray:
        """Read data from i0 to i1 in the block on the selected channels

        Arguments
        ---------
        i0 : int
            Starting index to read data from relative to the block (inclusive)
        i1 : int
            Last index to read data from relative to the block (exclusive)
        channels : List[int]
            List of channel numbers within block to read from

        Returns
        -------
        data : np.ndarray
            A 2D floating point array of shape (i1 - i0, len(channels))
        """
        i1 = min(i1, self.frames)
        output = np.zeros((i1 - i0, len(channels)))

        audio_files_needed = set([self.channel_mapping[ch][0] for ch in channels])
        data_by_audio_file = {}

        for f in audio_files_needed:
            data_by_audio_file[f] = f.read(i0, i1)

        for i, ch in enumerate(channels):
            (audio_file, audio_file_ch) = self.channel_mapping[ch]
            output[:, i] = data_by_audio_file[audio_file][:, audio_file_ch]

        return output

    def read_one(self, i, channels: List[int]) -> np.typing.ArrayLike:
        """Read a single sample at index i

        Arguments
        ---------
        i : int
            Index to read data from relative to the block

        Returns
        -------
        data : np.ndarray
            A 2D floating point array of shape (1, len(channels))

        """
        return self.read(i, i+1, channels)


class Project:
    """Data access to audio data in a series of Blocks

    Arguments
    ---------
    blocks : List[Block]
        An ordered list of Blocks to read in the Project. The indices determine
        the order data will be read, so the Project should be initialized with Blocks
        in the correct order (if relevant). All Blocks must have the same number of
        channels and sampling rate.

    Examples
    --------
    Data can be read from a project using the .read() method or square bracket
    access notation. Data can be accessed using either BlockIndex or ProjectIndex
    values to enforce consistency. When accessing with bracket notation, integers
    are interpreted as ProjectIndexes.

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

        rates = [b.sampling_rate for b in self._blocks]
        channels = [b.channels for b in self._blocks]
        channel_profiles = [tuple([f.channels for f in b._files]) for b in self._blocks]

        if not all([r == rates[0] for r in rates]):
            raise ValueError("Cannot instantiate Project with Blocks of different rates: {}".format(rates))

        if not all([c == channels[0] for c in channels]):
            raise ValueError("Cannot instantiate Project with Blocks with different channel counts: {}".format(channels))

        if not all([c == channel_profiles[0] for c in channel_profiles]):
            warnings.warn("Blocks in Project have mismatched channel profiles but same number of channels per block.")

    def __repr__(self):
        return "<Project: {} blocks>".format(len(self.blocks))

    @property
    def channels(self) -> int:
        """int: Number of channels in all Blocks of this Project"""
        return self._blocks[0].channels

    @property
    def sampling_rate(self) -> int:
        """int: Sampling rate of audio in this Project"""
        return self._blocks[0].sampling_rate

    @property
    def frames(self) -> int:
        """int: Total number of samples in the entire Project"""
        return int(np.sum([b.frames for b in self.blocks]))

    @property
    def blocks(self) -> List[Block]:
        """List[Block]: An ordered list of the Blocks in the Project"""
        return self._blocks

    def iter_blocks(self) -> Iterable[Tuple['ProjectIndex', 'ProjectIndex', 'Block']]:
        """Iterate over blocks in the Project

        Returns
        -------
        generator
            An iterator that yields tuples of the form ((start, stop), block), where
            start and stop are ProjectIndex instances ponting to the startpoint (inclusive)
            and endpoint (exclusive) of the Block, block.
        """
        frame = 0
        for block in self.blocks:
            yield ((ProjectIndex(self, frame), ProjectIndex(self, frame + block.frames)), block)
            frame += block.frames

    def close_files(self):
        for block in self.blocks:
            block.close_files()

    def read(
            self,
            start: Union['BlockIndex', 'ProjectIndex'],
            stop: Union['BlockIndex', 'ProjectIndex'],
            channels: List[int]
        ) -> np.ndarray:
        """Reads data of a start->stop slice, can use ProjectIndex or BlockIndex values

        Arguments
        ---------
        start : BlockIndex or ProjectIndex
        stop : BlockIndex or ProjectIndex
        channels : List[int]

        Returns
        -------
        data : np.ndarray
            A 2D array of shape (stop - start, len(channels)) representing data between
            the start and stop indices.
        """
        start = self.to_project_index(start)
        stop = self.to_project_index(stop)
        return self._read_by_project_indices(start, stop, channels)

    def _read_by_project_indices(
            self,
            start: 'ProjectIndex',
            stop: 'ProjectIndex',
            channels: List[int]
        ) -> np.typing.ArrayLike:
        """Reads slice's data from one or more Blocks in project"""
        if not isinstance(start, ProjectIndex) or not isinstance(stop, ProjectIndex):
            raise RuntimeError("_read_by_project_indicies should never be called with anything but ProjectIndex instances")

        out_data = []
        for (i0, i1), block in self.iter_blocks():
            if i1 < start:
                continue
            elif i0 > stop:
                break
            else:
                # block.read() takes normal ints; if we enforce it to take BlockIndex instead, we will
                # need to cast these its
                block_read_start = max(start - i0, 0)
                block_read_stop = min(stop - i0, block.frames)
                out_data.append(block.read(block_read_start, block_read_stop, channels=channels))

        return np.concatenate(out_data)

    def _normalize_slice(self, slice_: slice, cast_int_to_project_index: bool = False) -> slice:
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
        cast_int_to_project_index : bool
            Automatically cast ints to ProjectIndex instances (defaults to False)

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
            elif cast_int_to_project_index and isinstance(slice_.stop, int):
                start = ProjectIndex(self, 0)
                stop = ProjectIndex(self, slice_.stop)
            else:
                raise TypeError("Can only normalize slices with ProjectIndex, BlockIndex, or None values")
        elif slice_.stop is None:
            if isinstance(slice_.start, BlockIndex):
                start = self.to_project_index(slice_.start)
                stop = self.to_project_index(BlockIndex(slice_.start.block, slice_.start.block.frames))
            elif isinstance(slice_.start, ProjectIndex):
                start = slice_.start
                stop = ProjectIndex(self, self.frames)
            elif cast_int_to_project_index and isinstance(slice_.start, int):
                start = ProjectIndex(self, slice_.start)
                stop = ProjectIndex(self, self.frames)
            else:
                raise TypeError("Can only normalize slices with ProjectIndex, BlockIndex, or None values")
        else:
            start = self.to_project_index(slice_.start, cast_int_to_project_index=cast_int_to_project_index)
            stop = self.to_project_index(slice_.stop, cast_int_to_project_index=cast_int_to_project_index)

        return slice(start, stop, None)

    def __getitem__(self, slices) -> np.typing.ArrayLike:
        """Main data access of Project via Block coordinates or Project coordinates

        See Project class documentation for usage examples

        First parameters to the slice selects the indices, and the second parameter (optional)
        selects the channels. The second parameter can either be an int, Python slice object,
        or iterable.

        New in 0.1.4: When using __getitem__ (square bracket notation), automatically interpret
        integers as ProjectIndex
        """
        # This function must handle four cases: (int, int), (int, slice), (slice, int), (slice, slice)
        # The second value could be an iterable as well
        if isinstance(slices, tuple):
            if not len(slices) == 2:
                raise ValueError("Invalid index into Project of length {}".format(len(slices)))

            s1, s2 = slices

            if isinstance(s1, slice):
                s1 = self._normalize_slice(s1, cast_int_to_project_index=True)

            if isinstance(s2, slice):
                s2 = s2.indices(self.channels)
                s2 = list(range(s2[0], s2[1], s2[2]))

            if isinstance(s1, BaseIndex) and isinstance(s2, int):
                index = self.to_block_index(s1)
                return index.block.read_one(index, channels=[s2])[:, 0]
            elif isinstance(s1, BaseIndex) and isinstance(s2, Iterable):
                index = self.to_block_index(s1)
                return index.block.read_one(index, channels=list(s2))
            elif isinstance(s1, slice) and isinstance(s2, int):
                return self._read_by_project_indices(s1.start, s1.stop, channels=[s2])[:, 0]
            elif isinstance(s1, slice) and isinstance(s2, Iterable):
                return self._read_by_project_indices(s1.start, s1.stop, channels=list(s2))
            else:
                raise TypeError("Invalid types for Project __getitem__ access: {} and {}".format(s1, s2))
        elif isinstance(slices, BaseIndex):
            index = self.to_block_index(slices)
            return index.block.read_one(index, channels=list(range(self.channels)))
        elif isinstance(slices, int):
            index = self.to_block_index(ProjectIndex(self, slices))
            return index.block.read_one(index, channels=list(range(self.channels)))
        elif isinstance(slices, slice):
            slice_ = self._normalize_slice(slices, cast_int_to_project_index=True)
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
            for (i0, i1), block in self.iter_blocks():
                if i1 > index:
                    return BlockIndex(block, index - i0)
            raise ValueError("Could not find BlockIndex in Project")
        else:
            raise TypeError("Cannot covert type {} to BlockIndex".format(type(index)))

    def to_project_index(self, index: Union['BlockIndex', 'ProjectIndex'], cast_int_to_project_index: bool = False) -> 'ProjectIndex':
        """Convert a BlockIndex/ProjectIndex to a ProjectIndex

        Arguments
        ---------
        index : ProjectIndex or BlockIndex
        cast_int_to_project_index : bool
            Automatically cast ints to ProjectIndex instances (defaults to False)

        Returns
        -------
        project_index : ProjectIndex
            Index to data relative to the entire Project
        """
        if isinstance(index, ProjectIndex):
            return index
        elif isinstance(index, BlockIndex):
            for (i0, i1), block in self.iter_blocks():
                if block == index.block:
                    # Casting to int to turn i0 and index into pure ints
                    return ProjectIndex(self, int(i0 + int(index)))
            raise ValueError("Could not find BlockIndex in Project")
        elif cast_int_to_project_index and isinstance(index, int):
            return ProjectIndex(self, index)
        else:
            raise TypeError("Cannot covert type {} to ProjectIndex.".format(type(index)))

    def get_block_boundaries(
            self,
            from_: Union['BlockIndex', 'ProjectIndex'],
            to: Union['BlockIndex', 'ProjectIndex']
        ) -> List['ProjectIndex']:
        """Get the boundaries between blocks that lie between two points

        Arguments
        ---------
        from_ : BlockIndex or ProjectIndex
            First point to search for boundaries (inclusive)
        to : BlockIndex or ProjectIndex
            Last point to search for boundaries (inclusive)

        Returns
        -------
        boundaries : List[ProjectIndex]
            List of project index values that point to the edges of Blocks. These
            include the endpoints, so if block 1 is length 10 and block 2 is length 10,
            the boundaries would defined to be at [0, 10, 20]
        """
        from_ = self.to_project_index(from_)
        to = self.to_project_index(to)

        bounds = []

        if from_ == ProjectIndex(self, 0):
            bounds.append(from_)

        for (_, i1), block in self.iter_blocks():
            if from_ <= i1 <= to:
                bounds.append(i1)

        return bounds


# TODO: re-evaluate if inheriting from int is worth it or if it will be more likely to cause
# more problems than later (due to all the methods they will inherit)
def _match_type(fn, require_same_source=False):
    """Decorator for wrapping methods with one argument to enforce types match
    """
    @wraps(fn)
    def _wrapped(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError("Mismatching types for {}: {} and {}".format(
                fn.__name__,
                self.__class__.__name__,
                other.__class__.__name__
            ))
        if require_same_source and not (self._source_object == other._source_object):
            raise ValueError("Cannot call {} on {} with mismatched targets {} and {}".format(
                fn.__name__,
                self.__class__.__name__,
                self._source_object,
                other._source_object
            ))

        return fn(self, other)
    return _wrapped


class BaseIndex(int):

    ObjectType = None

    def __new__(cls, source_object, value: int):
        if not isinstance(source_object, cls.ObjectType):
            raise TypeError("Index of type {} must be instantiated with {}".format(cls, cls.ObjectType))

        return int.__new__(cls, value)

    def __init__(self, source_object, value: int):
        self._source_object = source_object
        self._args = [source_object]
        super().__init__()

    def __repr__(self):
        return "{}<{}>".format(self.__class__.__name__, int.__repr__(self))

    __lt__ = _match_type(int.__lt__, require_same_source=True)
    __gt__ = _match_type(int.__gt__, require_same_source=True)
    __le__ = _match_type(int.__le__, require_same_source=True)
    __ge__ = _match_type(int.__ge__, require_same_source=True)

    @_match_type
    def __eq__(self, other):
        return (other._source_object == self._source_object) and super().__eq__(other)

    @_match_type
    def __ne__(self, other):
        return (other._source_object != self._source_object) or super().__ne__(other)

    def __add__(self, other: int):
        """BaseIndex + int -> BaseIndex"""
        if isinstance(other, BaseIndex) or not isinstance(other, int):
            raise TypeError("Cannot add {} to {}".format(type(self).__name__, type(other).__name__))

        args = self._args + [super().__add__(other)]
        return self.__class__(*args)

    def __sub__(self, other: int):
        """BaseIndex - int -> BaseIndex | BaseIndex - BaseIndex -> int"""
        if isinstance(other, BaseIndex):
            if type(self) != type(other):
                raise TypeError("Cannot subtract {} from {}".format(type(other).__name__, type(self).__name__))
            return super().__sub__(other)
        elif isinstance(other, int):
            args = self._args + [super().__sub__(other)]
            return self.__class__(*args)
        else:
            raise TypeError("Cannot subtract {} from {}".format(type(other).__name__, type(self).__name__))

    @classmethod
    def range(cls, start, stop):
        for i in range(start, stop):
            args = start._args + [i]
            yield cls(*args)


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

    def to_timestamp(self):
        return int(self) / self.project.sampling_rate


class StftIndex(BaseIndex):
    """An integer index to a lattice on ProjectIndex separated by a given step

    Arguments
    ---------
    source_object : (soundsep.io.Project, int)
        A tuple of the Project for which the index is valid and the Stft step size
    value : int
        Frame within the Project that the index refers to

    Example
    -------
    >>> sidx = StftIndex(project, 50, 10)
    """

    ObjectType = Project

    def __new__(cls, project, step, value: int):
        if not isinstance(project, Project):
            raise TypeError("Cannot instantiate StftIndex without Project")
        if not isinstance(step, int):
            raise TypeError("Cannot instantiate StftIndex without step size")

        return int.__new__(cls, value)

    def __init__(self, project, step, value: int):
        super().__init__((project, step), value)
        self._step = step
        self._args = [project, step]

    @property
    def project(self):
        return self._source_object[0]

    @property
    def step(self):
        return self._source_object[1]

    def to_project_index(self):
        return ProjectIndex(self.project, int(self * self.step))

    def to_timestamp(self):
        return self.to_project_index().to_timestamp()


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

    def to_file_timestamp(self):
        """Return the timestamp of this index relative to its file"""
        return float(self) / float(self.block.sampling_rate)


class Source:
    """A source represents one source of auditory objects and is associated with one channel

    Arguments
    ---------
    project : Project
    channel : int
    index : int
        Kinda weird to put it here, but a way to keep track of a source like an "id"
    """

    def __init__(self, project: Project, name: str, channel: int, index: int):
        if channel >= project.channels:
            raise IndexError("Cannot assign channel greater than number of channels in project")

        self._project = project
        self.channel = channel
        # TODO: there is nothing that enforces uniqueness of the Source names
        self.name = name
        self.index = index

    @property
    def project(self) -> Project:
        """Project associated with this Source"""
        return self._project
