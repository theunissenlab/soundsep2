"""Data structures for abstracting the audio file organization on disk and indexing into data
"""

from __future__ import annotations

import datetime as dt
import os
import struct
import warnings
from collections.abc import Iterable
from functools import wraps
from pathlib import Path
from typing import Dict, List, Tuple, Union
import bisect

import numpy as np
import soundfile

try:
    from pynwb import NWBHDF5IO
    import h5py
    HAS_NWB = True
except ImportError:
    HAS_NWB = False
    h5py = None


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


def load_dat(filename: str | os.PathLike, loaddata: bool = True) -> tuple[np.ndarray, int, int, dt.datetime]:
    """
    Minimal acquisition-GUI .dat loader (format -4 only).

    Returns
    -------
    data : np.ndarray
        Contiguous float32 audio vector (empty if loaddata=False).
        Shape is (n_samples, n_channels) for multi-channel files.
    fs : int
        Sampling rate (Hz), inferred from footer timing fields.
    n_channels : int
        Number of channels in the file.
    start_time : datetime.datetime
        Acquisition start timestamp from the header.
    """
    filename = Path(filename)

    with filename.open("rb") as f:
        fmt = struct.unpack("<d", f.read(8))[0]
        if fmt != -4:
            raise ValueError(f"Only format -4 supported (got {fmt}).")

        # acquisition time
        acq_time = struct.unpack("<6d", f.read(48))
        start_time = dt.datetime(*map(int, acq_time))

        # skip file-created time
        f.read(48)

        # channel info
        n_chan = int(struct.unpack("<d", f.read(8))[0])
        f.read(8 * n_chan)  # channel IDs

        # scales and offsets (unused)
        f.read(8 * n_chan)  # scales
        f.read(8 * n_chan)  # offsets

        # dtype string encoded as doubles -> chars until sentinel fmt reappears
        chars: list[str] = []
        while True:
            val = struct.unpack("<d", f.read(8))[0]
            if val == fmt:
                break
            chars.append(chr(int(val)))
        dtype_str = "".join(chars)

        dtype_size = {"double": 8, "int32": 4, "int16": 2}.get(dtype_str, 8)

        # sample/time info for fs estimate
        start_samp = struct.unpack("<d", f.read(8))[0]
        start_time_s = struct.unpack("<d", f.read(8))[0]

        f.seek(-16, os.SEEK_END)
        end_samp = struct.unpack("<d", f.read(8))[0]
        end_time_s = struct.unpack("<d", f.read(8))[0]

        fs = int(round((end_samp - start_samp) / (end_time_s - start_time_s)))

        if not loaddata:
            return np.empty((0, n_chan), dtype=np.float32), fs, n_chan, start_time

        # NOTE: These constants are based on the known GUI layout for format -4.
        header_bytes = 208
        footer_bytes = 40
        file_size = filename.stat().st_size
        n_bytes = file_size - header_bytes - footer_bytes
        n_samples = n_bytes // dtype_size

        f.seek(header_bytes)

        if dtype_size == 8:
            raw = np.fromfile(f, dtype=np.float64, count=n_samples)
            data = raw.astype(np.float32, copy=False)
        elif dtype_size == 4:
            data = np.fromfile(f, dtype=np.float32, count=n_samples)
        else:
            data = np.fromfile(f, dtype=np.int16, count=n_samples).astype(np.float32)

        # Reshape to (frames, channels) - data is interleaved
        n_frames = n_samples // n_chan
        data = data[:n_frames * n_chan].reshape((n_frames, n_chan))

        return np.ascontiguousarray(data), fs, n_chan, start_time


class DatFile:
    """Container for a .dat file on disk (acquisition-GUI format -4)

    This class provides the same interface as AudioFile but reads from .dat files.

    Arguments
    ---------
    path : str
        Full path to the .dat file on disk
    """

    def __init__(self, path):
        self._path = path
        self._max_frame = None
        self._data = None

        # Read metadata without loading data
        _, self._sampling_rate, self._channels, self._start_time = load_dat(path, loaddata=False)

        # We need to calculate actual frames from file size
        path_obj = Path(path)
        with path_obj.open("rb") as f:
            fmt = struct.unpack("<d", f.read(8))[0]
            if fmt != -4:
                raise ValueError(f"Only format -4 supported (got {fmt}).")

            # Skip to n_chan position
            f.seek(104)  # Skip format (8) + acq_time (48) + created_time (48)
            n_chan = int(struct.unpack("<d", f.read(8))[0])
            f.read(8 * n_chan)  # channel IDs
            f.read(8 * n_chan)  # scales
            f.read(8 * n_chan)  # offsets

            # dtype string
            chars = []
            while True:
                val = struct.unpack("<d", f.read(8))[0]
                if val == fmt:
                    break
                chars.append(chr(int(val)))
            dtype_str = "".join(chars)
            dtype_size = {"double": 8, "int32": 4, "int16": 2}.get(dtype_str, 8)

        header_bytes = 208
        footer_bytes = 40
        file_size = path_obj.stat().st_size
        n_bytes = file_size - header_bytes - footer_bytes
        n_samples = n_bytes // dtype_size
        self._actual_frames = n_samples // self._channels

    def is_open(self):
        return self._data is not None

    def is_closed(self):
        return self._data is None

    def open(self):
        if not self.is_open():
            self._data, _, _, _ = load_dat(self._path, loaddata=True)

    def close(self):
        if self.is_open():
            self._data = None

    def __repr__(self):
        return "<DatFile: {}; {} Hz; {} Ch; {} frames>".format(
            os.path.basename(self._path),
            self.sampling_rate,
            self.channels,
            self.frames
        )

    def set_max_frame(self, frames):
        """Set the maximum frame to read from the file

        This can be used to force multiple DatFiles to behave as if they have the
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
            raise RuntimeError("Cannot force DatFile to use more frames than on disk")

        self._max_frame = frames

    def clear_max_frame(self):
        self._max_frame = None

    def __eq__(self, other_file) -> bool:
        if isinstance(other_file, DatFile):
            return self.path == other_file.path
        else:
            raise ValueError("Can only compare DatFile equality with other DatFiles")

    def __hash__(self):
        return id(self)

    @property
    def path(self) -> str:
        """str: Full path to dat file"""
        return self._path

    @property
    def sampling_rate(self) -> int:
        """int: Sampling rate of the dat file"""
        return self._sampling_rate

    @property
    def frames(self) -> int:
        """int: Number of readable samples in dat file"""
        return self._max_frame or self._actual_frames

    @property
    def channels(self) -> int:
        """int: Number of channels in dat file"""
        return self._channels

    @property
    def start_time(self) -> dt.datetime:
        """datetime: Acquisition start timestamp"""
        return self._start_time

    def read(self, i0: int, i1: int) -> np.ndarray:
        """Read samples from i0 to i1

        Arguments
        ---------
        i0 : int
            Starting index to read from (inclusive)
        i1 : int
            Ending index to read until (exclusive)

        Returns
        -------
        data : ndarray
            A 2D array of shape (frames: int, channels: int) containing data.
            The first dimension is the sample index, the second dimension is the channel
            axis.
        """
        read_start = i0
        read_stop = min(i1, self.frames)

        if self.is_closed():
            self.open()

        return self._data[read_start:read_stop, :]


class NWBFile:
    """Container for a NWB file on disk with microphone data
    
    This class provides the same interface as AudioFile but reads from NWB files.
    It expects the microphone data to be stored in the acquisition group.
    
    Arguments
    ---------
    path : str
        Full path to the NWB file on disk
    microphone_name : str, optional
        Name of the microphone TimeSeries in the NWB file (default: "microphone")
    """
    
    def __init__(self, path, microphone_name="audio"):
        if not HAS_NWB:
            raise ImportError("pynwb is required to read NWB files. Install with: pip install pynwb")
        
        self._path = path
        self._microphone_name = microphone_name
        self._max_frame = None
        self._io = None
        self._nwbfile = None
        self._microphone_data = None
        
        # Read metadata from the file
        with NWBHDF5IO(path, 'r') as io:
            nwbfile = io.read()
            
            # Try to find microphone data in acquisition
            if microphone_name in nwbfile.acquisition:
                mic_series = nwbfile.acquisition[microphone_name]
            else:
                # Try to find any TimeSeries that might be microphone data
                acquisition_names = list(nwbfile.acquisition.keys())
                if len(acquisition_names) > 0:
                    mic_series = nwbfile.acquisition[acquisition_names[0]]
                    warnings.warn(f"Microphone '{microphone_name}' not found. Using '{acquisition_names[0]}' instead.")
                else:
                    raise ValueError(f"No acquisition data found in NWB file {path}")
            
            self._sampling_rate = int(mic_series.rate) if hasattr(mic_series, 'rate') else int(mic_series.starting_time)
            
            # Get data shape
            data_shape = mic_series.data.shape
            self._actual_frames = data_shape[0]
            
            # Handle channels - NWB data might be 1D (single channel) or 2D (multiple channels)
            if len(data_shape) == 1:
                self._channels = 1
            else:
                self._channels = data_shape[1]
    
    def is_open(self):
        return self._io is not None and self._nwbfile is not None
    
    def is_closed(self):
        return self._io is None or self._nwbfile is None
    
    def open(self):
        if not self.is_open():
            self._io = NWBHDF5IO(self._path, 'r')
            self._nwbfile = self._io.read()
            
            # Get reference to microphone data
            if self._microphone_name in self._nwbfile.acquisition:
                mic_series = self._nwbfile.acquisition[self._microphone_name]
            else:
                acquisition_names = list(self._nwbfile.acquisition.keys())
                mic_series = self._nwbfile.acquisition[acquisition_names[0]]
            
            self._microphone_data = mic_series.data
    
    def close(self):
        if self.is_open():
            self._io.close()
            self._io = None
            self._nwbfile = None
            self._microphone_data = None
    
    def __repr__(self):
        return "<NWBFile: {}; {} Hz; {} Ch; {} frames>".format(
            os.path.basename(self._path),
            self.sampling_rate,
            self.channels,
            self.frames
        )
    
    def set_max_frame(self, frames):
        """Set the maximum frame to read from the file
        
        This can be used to force multiple NWBFiles to behave as if they have the
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
            raise RuntimeError("Cannot force NWBFile to use more frames than on disk")
        
        self._max_frame = frames
    
    def clear_max_frame(self):
        self._max_frame = None
    
    def __eq__(self, other_file) -> bool:
        if isinstance(other_file, NWBFile):
            return self.path == other_file.path
        else:
            raise ValueError("Can only compare NWBFile equality with other NWBFiles")
    
    def __hash__(self):
        return id(self)
    
    @property
    def path(self) -> str:
        """str: Full path to NWB file"""
        return self._path
    
    @property
    def sampling_rate(self) -> int:
        """int: Sampling rate of the microphone data"""
        return self._sampling_rate
    
    @property
    def frames(self) -> int:
        """int: Number of readable samples in NWB file"""
        return self._max_frame or self._actual_frames
    
    @property
    def channels(self) -> int:
        """int: Number of channels in NWB file"""
        return self._channels
    
    def read(self, i0: int, i1: int) -> np.ndarray:
        """Read samples from i0 to i1
        
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
            The first dimension is the sample index, the second dimension is the channel axis.
        """
        read_start = i0
        read_stop = min(i1, self.frames)
        
        if self.is_closed():
            self.open()
        
        # Read data from NWB file
        if self._channels == 1:
            # Single channel - reshape to 2D
            data = self._microphone_data[read_start:read_stop]
            data = data.reshape(-1, 1)
        else:
            # Multiple channels
            data = self._microphone_data[read_start:read_stop, :]
        
        # Convert to float32 if needed
        return data.astype(np.float32)

    @staticmethod
    def read_soundsep_sources(nwb_path: str) -> list:
        """Read soundsep source data from the NWB file's analysis group.

        Arguments
        ---------
        nwb_path : str
            Path to the NWB file

        Returns
        -------
        sources : list
            List of dicts with keys 'SourceName', 'SourceChannel', 'SourceIndex'
            Returns empty list if no soundsep data exists
        """
        if not HAS_NWB:
            raise ImportError("pynwb is required to read NWB files")

        sources = []
        with NWBHDF5IO(nwb_path, 'r') as io:
            nwbfile = io.read()
            
            if 'soundsep_sources' in nwbfile.analysis:
                table = nwbfile.analysis['soundsep_sources']
                
                # Read from DynamicTable
                source_names = table['SourceName'][:]
                source_channels = table['SourceChannel'][:]
                source_indices = table['SourceIndex'][:]
                
                for i in range(len(source_names)):
                    sources.append({
                        'SourceName': str(source_names[i]),
                        'SourceChannel': int(source_channels[i]),
                        'SourceIndex': int(source_indices[i]),
                    })
        
        return sources

    @staticmethod
    def write_soundsep_sources(nwb_path: str, sources: list):
        """Write soundsep source data to the NWB file's analysis group.

        Arguments
        ---------
        nwb_path : str
            Path to the NWB file
        sources : list
            List of dicts with keys 'SourceName', 'SourceChannel', 'SourceIndex'
        """
        if not HAS_NWB:
            raise ImportError("pynwb is required to write to NWB files")

        import tempfile
        import shutil

        # Check if soundsep_sources already exists
        has_existing = False
        with NWBHDF5IO(nwb_path, 'r') as io:
            nwbfile = io.read()
            has_existing = 'soundsep_sources' in nwbfile.analysis

        if has_existing:
            # Export to a new file, excluding the old soundsep_sources
            # This avoids hdmf builder comparison issues with numpy arrays
            temp_fd, temp_path = tempfile.mkstemp(suffix='.nwb')
            os.close(temp_fd)
            try:
                with NWBHDF5IO(nwb_path, 'r') as read_io:
                    nwbfile = read_io.read()

                    # Remove existing soundsep_sources
                    del nwbfile.analysis['soundsep_sources']

                    # Add the new sources table
                    sources_table = NWBFile._create_sources_table(sources)
                    nwbfile.add_analysis(sources_table)

                    # Export to temp file
                    with NWBHDF5IO(temp_path, 'w') as export_io:
                        export_io.export(src_io=read_io, nwbfile=nwbfile)

                # Replace original with temp
                shutil.move(temp_path, nwb_path)
            except Exception:
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise
        else:
            # No existing data, can just append
            with NWBHDF5IO(nwb_path, 'r+') as io:
                nwbfile = io.read()
                sources_table = NWBFile._create_sources_table(sources)
                nwbfile.add_analysis(sources_table)
                io.write(nwbfile)

    @staticmethod
    def _create_sources_table(sources: list):
        """Create a DynamicTable for soundsep sources."""
        from hdmf.common import DynamicTable, VectorData

        if len(sources) > 0:
            source_names = [s['SourceName'] for s in sources]
            source_channels = [s['SourceChannel'] for s in sources]
            source_indices = [s['SourceIndex'] for s in sources]
        else:
            source_names = []
            source_channels = []
            source_indices = []

        return DynamicTable(
            name='soundsep_sources',
            description='Soundsep source definitions',
            columns=[
                VectorData(
                    name='SourceName',
                    description='Name of the source',
                    data=source_names
                ),
                VectorData(
                    name='SourceChannel',
                    description='Channel index of the source',
                    data=source_channels
                ),
                VectorData(
                    name='SourceIndex',
                    description='Index of the source',
                    data=source_indices
                ),
            ]
        )

    @staticmethod
    def has_soundsep_data(nwb_path: str) -> bool:
        """Check if an NWB file contains soundsep data.

        Arguments
        ---------
        nwb_path : str
            Path to the NWB file

        Returns
        -------
        bool
            True if the file contains soundsep source data
        """
        if not HAS_NWB:
            return False

        try:
            with NWBHDF5IO(nwb_path, 'r') as io:
                nwbfile = io.read()
                return 'soundsep_sources' in nwbfile.analysis
        except Exception:
            return False

    @staticmethod
    def read_soundsep_segments(nwb_path: str, sampling_rate: int) -> list:
        """Read soundsep segment data from the NWB file's intervals group.

        Arguments
        ---------
        nwb_path : str
            Path to the NWB file
        sampling_rate : int
            Sampling rate to convert times to sample indices

        Returns
        -------
        segments : list
            List of dicts with keys: 'SourceName', 'SourceChannel', 'StartIndex',
            'StopIndex', 'Tags', 'Coords', 'SegmentID'
            Returns empty list if no segment data exists
        """
        if not HAS_NWB:
            raise ImportError("pynwb/h5py is required to read NWB files")

        segments = []
        with h5py.File(nwb_path, 'r') as f:
            if 'intervals' not in f or 'soundsep_segments' not in f['intervals']:
                return segments

            intervals = f['intervals']['soundsep_segments']

            # Read the data columns
            start_times = intervals['start_time'][:]
            stop_times = intervals['stop_time'][:]
            segment_ids = intervals['id'][:]

            # Read custom columns
            source_names = intervals['source_name'][:] if 'source_name' in intervals else [b''] * len(start_times)
            source_channels = intervals['source_channel'][:] if 'source_channel' in intervals else [0] * len(start_times)
            tags = intervals['tags'][:] if 'tags' in intervals else [b'[]'] * len(start_times)
            coords = intervals['coords'][:] if 'coords' in intervals else [b'null'] * len(start_times)

            for i in range(len(start_times)):
                source_name = source_names[i].decode() if isinstance(source_names[i], bytes) else str(source_names[i])
                tags_str = tags[i].decode() if isinstance(tags[i], bytes) else str(tags[i])
                coords_str = coords[i].decode() if isinstance(coords[i], bytes) else str(coords[i])

                segments.append({
                    'SourceName': source_name,
                    'SourceChannel': int(source_channels[i]),
                    'StartIndex': int(start_times[i] * sampling_rate),
                    'StopIndex': int(stop_times[i] * sampling_rate),
                    'Tags': tags_str,  # JSON string
                    'Coords': coords_str,  # JSON string
                    'SegmentID': int(segment_ids[i]),
                })

        return segments

    @staticmethod
    def write_soundsep_segments(nwb_path: str, segments: list, sampling_rate: int):
        """Write soundsep segment data to the NWB file's intervals group.

        Arguments
        ---------
        nwb_path : str
            Path to the NWB file
        segments : list
            List of dicts with keys: 'SourceName', 'SourceChannel', 'StartIndex',
            'StopIndex', 'Tags', 'Coords', 'SegmentID'
        sampling_rate : int
            Sampling rate to convert sample indices to times
        """
        if not HAS_NWB:
            raise ImportError("pynwb/h5py is required to write to NWB files")

        # Use h5py directly for reliable intervals writing
        with h5py.File(nwb_path, 'a') as f:
            # Create intervals group if it doesn't exist
            if 'intervals' not in f:
                f.create_group('intervals')

            intervals = f['intervals']

            # Remove existing soundsep_segments if present
            if 'soundsep_segments' in intervals:
                del intervals['soundsep_segments']

            # Create the intervals table group
            seg_group = intervals.create_group('soundsep_segments')

            n_segments = len(segments)

            if n_segments > 0:
                # Standard interval columns
                start_times = np.array([s['StartIndex'] / sampling_rate for s in segments], dtype=np.float64)
                stop_times = np.array([s['StopIndex'] / sampling_rate for s in segments], dtype=np.float64)
                segment_ids = np.array([s['SegmentID'] for s in segments], dtype=np.int64)

                # Custom columns
                source_names = np.array([s['SourceName'].encode() if isinstance(s['SourceName'], str) else s['SourceName']
                                        for s in segments], dtype='S256')
                source_channels = np.array([s['SourceChannel'] for s in segments], dtype=np.int32)
                tags = np.array([s['Tags'].encode() if isinstance(s['Tags'], str) else s['Tags']
                                for s in segments], dtype='S1024')
                coords = np.array([s['Coords'].encode() if isinstance(s['Coords'], str) else s['Coords']
                                  for s in segments], dtype='S1024')
            else:
                start_times = np.array([], dtype=np.float64)
                stop_times = np.array([], dtype=np.float64)
                segment_ids = np.array([], dtype=np.int64)
                source_names = np.array([], dtype='S256')
                source_channels = np.array([], dtype=np.int32)
                tags = np.array([], dtype='S1024')
                coords = np.array([], dtype='S1024')

            # Create datasets
            seg_group.create_dataset('start_time', data=start_times)
            seg_group.create_dataset('stop_time', data=stop_times)
            seg_group.create_dataset('id', data=segment_ids)
            seg_group.create_dataset('source_name', data=source_names)
            seg_group.create_dataset('source_channel', data=source_channels)
            seg_group.create_dataset('tags', data=tags)
            seg_group.create_dataset('coords', data=coords)

            # Add attributes to make it identifiable as a soundsep intervals table
            seg_group.attrs['neurodata_type'] = 'TimeIntervals'
            seg_group.attrs['description'] = 'Soundsep segmentation intervals'

    @staticmethod
    def has_soundsep_segments(nwb_path: str) -> bool:
        """Check if an NWB file contains soundsep segment data.

        Arguments
        ---------
        nwb_path : str
            Path to the NWB file

        Returns
        -------
        bool
            True if the file contains soundsep segment data
        """
        if not HAS_NWB:
            return False

        try:
            with h5py.File(nwb_path, 'r') as f:
                return 'intervals' in f and 'soundsep_segments' in f['intervals']
        except Exception:
            return False
        

def load_photo_metadata(project_name):
    """
    Load metadata for a recording saved by ContinuousRecordingThread.

    Parameters:
    -----------
    project_name : str
        Name of the project (corresponding to the filename prefix of the .dat and _metadata.json files)

    Returns:
    --------
    dict : Metadata dictionary containing recording info such as channels, sample_rate, dtype, etc.
    """
    import json
    import os
    # first remove any file extension from project_name to get the base name
    project_name = os.path.splitext(project_name)[0]

    # Load metadata
    metadata_file = project_name + '_metadata.json'
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # lets get number of samples using memmap to avoid loading the whole file
    data_file = project_name + '.dat'
    dtype = np.dtype(metadata['dtype'])
    file_size = os.path.getsize(data_file)
    n_samples = file_size // (dtype.itemsize * metadata['n_channels'])
    metadata['n_samples'] = n_samples

    print(f"Loaded metadata:")
    print(f"  Channels: {metadata['channels']}")
    print(f"  Sample rate: {metadata['sample_rate']} Hz")
    print(f"  Data type: {metadata['dtype']}")

    return metadata

def load_photo_recording(project_name):
    """
    Load a recording saved by ContinuousRecordingThread.

    Parameters:
    -----------
    project_name : str
        Name of the project (corresponding to the filename prefix of the .dat and _metadata.json files)

    Returns:
    --------
    tuple : (data, metadata) where data is a numpy array (n_samples, n_channels)
            and metadata is a dictionary with recording info
    """
    import json
    import os
    # first remove any file extension from project_name to get the base name
    project_name = os.path.splitext(project_name)[0]

    # Load metadata
    metadata_file = project_name + '_metadata.json'
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Load data
    data_file = project_name + '.dat'
    data = np.fromfile(data_file, dtype=metadata['dtype'])
    data_reshaped = data.reshape(-1, metadata['n_channels'])

    print(f"Loaded recording:")
    print(f"  Channels: {metadata['channels']}")
    print(f"  Sample rate: {metadata['sample_rate']} Hz")
    print(f"  Duration: {metadata['n_samples'] / metadata['sample_rate']:.2f}s")
    print(f"  Shape: {data_reshaped.shape}")

    return data_reshaped, metadata

class PhotoProject:
    """Container for a photo project on disk

    This class provides the same interface as AudioFile but reads from a photo
    project file. It expects the photo project to be stored in a .json file with
    a specific structure.

    Arguments
    ---------
    path : str
        Full path to the photo project .json file on disk
    """
    @staticmethod
    def is_photo_project(path):
        """Check if a given path corresponds to a valid photo project

        Arguments
        ---------
        path : str
            Path to check

        Returns
        -------
        bool
            True if the path corresponds to a valid photo project, False otherwise
        """
        try:
            metadata = load_photo_metadata(path)
            return True
        except Exception:
            return False

    def __init__(self, path):
        self._path = path
        self.project_path = os.path.splitext(path)[0]
        self._max_frame = None
        self._file = None
        self.metadata = load_photo_metadata(path)
        self._sampling_rate = self.metadata['sample_rate']
        self._channels = self.metadata['n_channels']
        # TODO could include chanel names
        self._actual_frames = self.metadata['n_samples']

    def is_open(self):
        return self._file is not None and self._file._mmap is not None

    def is_closed(self):
        return self._file is None or self._file._mmap is None

    def open(self):
        if not self.is_open():
            self._file = np.memmap(self.project_path + '.dat', dtype=self.metadata['dtype'], mode='r') # this is 1d interleaved data, we will reshape on read

    def close(self):
        if self.is_open():
            self._file._mmap.close()
            self._file = None
    def __repr__(self):
        return "<PhotoProject: {}; {} Hz; {} Ch; {} frames>".format(
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
        if isinstance(other_file, PhotoProject):
            return self.path == other_file.path
        else:
            raise ValueError("Can only compare PhotoProject equality with other PhotoProjects")
    def __hash__(self):
        return id(self)

    @property
    def path(self) -> str:
        """str: Full path to photo project file"""
        return self._path

    @property
    def sampling_rate(self) -> int:
        """int: Sampling rate of the photo project"""
        return self._sampling_rate

    @property
    def frames(self) -> int:
        """int: Number of readable frames in the photo project"""
        return self._max_frame or self._actual_frames

    @property
    def channels(self) -> int:
        """int: Number of channels in the photo project"""
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

        # Read data from memmap and reshape to (frames, channels)
        data = self._file[read_start * self.channels : read_stop * self.channels]
        data = data.reshape(-1, self.channels)
        return data.astype(np.float32)


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
        self._block_start_frames = []
        frame = 0
        for block in self.blocks:
            self._block_start_frames.append(frame)
            frame += block.frames
        
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

    # TODO we should cache these values
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

    def read_by_blocks(
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
        return self._read_by_project_indices(start, stop, channels, concatenate=False)
    
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
        return self._read_by_project_indices(start, stop, channels, concatenate=True)

    def _read_by_project_indices(
            self,
            start: 'ProjectIndex',
            stop: 'ProjectIndex',
            channels: List[int],
            concatenate: bool = True
        ) -> np.typing.ArrayLike:
        """Reads slice's data from one or more Blocks in project"""
        if not isinstance(start, ProjectIndex) or not isinstance(stop, ProjectIndex):
            raise RuntimeError("_read_by_project_indicies should never be called with anything but ProjectIndex instances")

        out_data = []
        # TODO: This can be preallocated using a numpy array of the correct shape
        # TODO: we can predetermine which blocks we need by using searchsorted on self._block_start_frames
        # start_id = np.searchsorted(self._block_start_frames, start, 'right')
        # stop_id = np.searchsorted(self._block_start_frames, stop, 'right')
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
        
        if concatenate:
            return np.concatenate(out_data)
        else:
            return out_data

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
            try:
                iblock = bisect.bisect_left(self._block_start_frames, int(index)) - 1
                block = self._blocks[iblock]
                return BlockIndex(block, index - self._block_start_frames[iblock])
            except:
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
        """BaseIndex + int -> BaseIndex | BaseIndex + BaseIndex -> int"""
        if isinstance(other, BaseIndex) or not isinstance(other, int):
            if type(self) != type(other):
                raise TypeError("Cannot add {} to {}".format(type(self).__name__, type(other).__name__))
            return super().__add__(other)
        elif isinstance(other, int):
            args = self._args + [super().__add__(other)]
            return self.__class__(*args)
        else:
            raise TypeError("Cannot add {} and {}".format(type(other).__name__, type(self).__name__))

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
