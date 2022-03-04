"""
Read datasets from canary data set up in Soundsep (https://github.com/theunissenlab/soundsep2.git)
"""
import pathlib
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from soundsep import open_project
from soundsep_prediction.stft import (
    StftParameters,
    compute_pad_for_windowing,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"


@dataclass
class DatasetParameters:
    """Define where to sample from in a Soundsep Project

    Attributes
    ----------
    radius : int
        Number of STFT samples in each sample (e.g. # of timepoints)
    from_ : float
        Fraction of Soundsep project to start reading data from. Starts from
        beginning of Project if left unspecified.
    until : float
        Fraction of Soundsep project to stop reading data from. Ends at end
        of Project if left unspecified.
    """
    radius: int = 30
    from_index: int = None
    until_index: int = None

    @property
    def width(self) -> int:
        return 2 * self.radius + 1

    def copy(self) -> 'DatasetParameters':
        """Make a copy of the current parameter object"""
        return DatasetParameters(
            radius=self.radius,
            from_index=self.from_index,
            until_index=self.until_index,
        )


class SoundsepAudioDataset:
    """Provides access to audio data and labels

    FIXME: This is very inefficient when contiguous chunks since
    much of adjcaent datapoints overlap
    """

    def __init__(
            self,
            project_dir: pathlib.Path,
            syllable_table: pd.DataFrame,
            source_names: List[str],
            stft_params: StftParameters = StftParameters(),
            params: DatasetParameters = DatasetParameters(),
        ):
        self.project = open_project(project_dir)
        self.stft_params = stft_params
        self.params = params

        self.source_names = np.array(source_names)
        self.y = torch.zeros((len(self), len(self.source_names)), dtype=torch.float)
        self.syllable_table = syllable_table[
            syllable_table["SourceName"].isin(source_names)
            & (syllable_table["StartIndex"] >= self.start)
            & (syllable_table["StopIndex"] <= self.stop)
        ]

        self.setup_labels()

    @property
    def start(self):
        """The first frame of Project (inclusive) to include in dataset"""
        return self.params.from_index or 0

    @property
    def stop(self):
        """The last frame of Project (non-inclusive) to include in dataset"""
        return self.params.until_index or self.project.frames

    def setup_labels(self):
        """Set up an array of labels for each sample of the STFT

        True if the sample lies in a syllable, and False if not
        """
        for i in range(len(self.syllable_table)):
            row = self.syllable_table.iloc[i]
            source_index = np.where(self.source_names == row["SourceName"])[0]
            first_index = max(0, (row["StartIndex"] - self.start) // self.stft_params.hop)
            last_index = max(0, (row["StopIndex"] - self.start) // self.stft_params.hop)
            self.y[first_index:last_index, source_index] = True
            if last_index > len(self):
                break

    def __getitem__(self, spec_index: int) -> Tuple[torch.Tensor, bool]:
        """Returns a tensor of the spectrogram at the given spec index

        Arguments
        ---------
        spec_index : int
            The index of the spectrogram sample. The max value is the
            number of frames in the project divided by the stft hop

        Returns
        -------
        data : torch.Tensor
            A data array of shape (N, M, 1), where N is the number of
            spectrogram samples, and M is the number of frequencies
        label : torch.Tensor
            A single value boolean tensor with True if the sample is centered on a syllable,
            and False if not
        """
        if spec_index >= len(self):
            raise IndexError

        # Compute spectrograms centered at spec_index
        center = (spec_index * self.stft_params.hop) + self.start
        last = center + (self.params.radius * self.stft_params.hop)
        first = center - (self.params.radius * self.stft_params.hop)

        pad_start, pad_stop, start_index, stop_index = compute_pad_for_windowing(
            array_length=self.project.frames,
            first_window_pos=first,
            last_window_pos=last,
            half_window=self.stft_params.half_window,
        )

        arr = self.project[start_index:stop_index]

        padding = ((pad_start, pad_stop),) + tuple([(0, 0) for _ in range(arr.ndim - 1)])
        padded_arr = np.pad(arr, padding, mode="reflect")

        return (
            torch.tensor(padded_arr.T, dtype=torch.float),
            self.y[spec_index:spec_index + 1]
        )

    def load_range(self, spec_index_0: int, spec_index_1: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load the audio data from a specific range of data-points directly

        Useful for visualizing a model output on a contiguous set of datapoints
        """
        if spec_index_0 >= len(self) or spec_index_1 > len(self):
            raise IndexError

        # Compute spectrograms centered at spec_index
        last = self.start + (spec_index_1 - 1) * self.stft_params.hop
        first = self.start + spec_index_0 * self.stft_params.hop

        pad_start, pad_stop, start_index, stop_index = compute_pad_for_windowing(
            array_length=self.project.frames,
            first_window_pos=first,
            last_window_pos=last,
            half_window=0,
        )
        arr = self.project[start_index:stop_index]

        padding = ((pad_start, pad_stop),) + tuple([(0, 0) for _ in range(arr.ndim - 1)])
        padded_arr = np.pad(arr, padding, mode="reflect")

        return (
            torch.tensor(padded_arr.T, dtype=torch.float),
            self.y[spec_index_0:spec_index_1]
        )

    def __len__(self):
        """Returns the size of the dataset"""
        return (self.stop - self.start) // self.stft_params.hop


class CompositeDataset:
    """A wrapper around multiple datasets

    Pass in time_ranges, a list of tuples of the time (in seconds) of the start and end of each range
    """
    def __init__(
            self,
            project_dir: pathlib.Path,
            syllable_table: pd.DataFrame,
            source_names: List[str],
            time_ranges: List[Tuple[float, float]],
            stft_params: StftParameters = StftParameters(),
            params: DatasetParameters = DatasetParameters(),
            ):
        self.project = open_project(project_dir)
        self.stft_params = stft_params
        self.params = params
        self.source_names = np.array(source_names)

        self.datasets = []
        self.dataset_start_indexes = [0]
        for t0, t1 in time_ranges:
            param_copy = params.copy()
            param_copy.from_index = int(t0 * self.project.sampling_rate)
            param_copy.until_index = int(t1 * self.project.sampling_rate)
            self.datasets.append(SoundsepAudioDataset(
                project_dir,
                syllable_table,
                source_names,
                stft_params,
                param_copy
            ))
            self.dataset_start_indexes.append(self.dataset_start_indexes[-1] + len(self.datasets[-1]))

    def __getitem__(self, stft_index):
        dataset_start_index = np.searchsorted(self.dataset_start_indexes, stft_index, side="right")

        dataset_first_stft_index = self.dataset_start_indexes[dataset_start_index - 1]
        dataset = self.datasets[dataset_start_index - 1]
        return dataset[stft_index - dataset_first_stft_index]

    def __len__(self):
        return self.dataset_start_indexes[-1]

