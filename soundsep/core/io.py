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

import os
from typing import List

import numpy as np
import soundfile


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

        with soundfile.soundfile(path) as f:
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
