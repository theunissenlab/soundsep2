from collections.abc import Iterable
from typing import List

import numpy as np

from soundsep.core.io import AudioFile, Block


class ProjectIndex(int):
    """An integer index that is a global index into a specific Project

    The range of values are clamped to the bounds of the Project itself
    """
    def __new__(cls, project, value: int):
        return int.__new__(cls, value)

    def __init__(cls, project, value: int):
        if value < -project.frames:
            value = -project.frames
        elif value > project.frames:
            value = project.frames

        self.project = project

        super().__init__(value)


class BlockIndex(int):
    """An integer index that is local to a specific Block

    The range of values are clamped to the bounds of the Block
    """
    def __new__(cls, block: Block, value: int):
        return int.__new__(cls, value)

    def __init__(cls, block: Block, value: int):
        if value < -block.frames:
            value = -block.frames
        elif value > block.frames:
            value = block.frames

        self.block = block

        super().__init__(value)


class BaseSlice:

    IndexType = None

    def __init__(self, start, stop, step):
        self.validate(start, stop, step)
        self._slice = slice(start, stop, step)

    def _validate(self, start, stop, step):
        if (not isinstance(start, self.IndexType) or 
                not isinstance(stop, self.IndexType) or 
                not isinstance(step, self.IndexType)
                ):
            raise ValueError("Slice can only be instantiated with index of correct type:"
                    " ({}, {})".format(self.__class__, self.IndexType))

    @property
    def start(self):
        return self._slice.start

    @property
    def stop(self):
        return self._slice.stop

    @property
    def step(self):
        return self._slice.step

    def indices(self, length):
        result = self._slice.indices(length)
        return self.__class__(
            self.IndexType(result.start),
            self.IndexType(result.stop),
            self.IndexType(result.step),
        )

    def read(self, channels: List[int]):
        raise NotImplementedError


class BlockSlice(BaseSlice):

    IndexType = BlockIndex

    def _validate(self, start, stop, step):
        super()._validate(start, stop, step)
        if start.block != stop.block or stop.block != step.block:
            raise ValueError("Cannot create BlockSlice with BlockIndexes from different Blocks")

    def read(self, channels: List[int]):
        """Reads slice's data from Block"""
        block = self.start.block
        return block.read(self.start, self.stop, channels)[::self.step]


class ProjectSlice(BaseSlice):

    IndexType = ProjectIndex

    def _validate(self, start, stop, step):
        super()._validate(start, stop, step)
        if start.project != stop.project or stop.project != step.project:
            raise ValueError("Cannot create ProjectSlice with ProjectIndexes from different Projects")

    def read(self, channels: List[int]):
        """Reads slice's data from one or more Blocks in project"""
        project = self.start.project

        out_data = []
        for (i0, i1), block in project.iter_blocks():
            if i1 < self.start:
                continue
            elif i0 > self.stop:
                break
            else:
                block_read_start = max(self.start - i0, 0)
                block_read_stop = min(self.stop - i0, block.frames)
                out_data.append(
                    block.read(block_read_start, block_read_stop, channels=channels)
                )

        return np.concatenate(out_data)[::self.step]
            

class Project:

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

    def __getitem__(self, slices):
        """Slice a section of data

        First parameters to the slice selects the indices, and the second parameter (optional)
        selects the channels. The second parameter can either be an int, Python slice object,
        or iterable.
        """
        if isinstance(slices, tuple):
            if not len(slices, 2):
                raise ValueError("Invalid index into Project of length {}".format(len(slices)))

            index_slice, channel_slice = slices

            if isinstance(channel_slice, int):
                return index_slice.read(channels=[channel_slice])
            elif isinstance(channel_slice, slice):
                channel_slice = channel_slice.indices(self.channels)
                return index_slice.read(
                    channels=list(range(channel_slice.start, channel_slice.stop, channel_slice.step))
                )
            elif isinstance(channel_slice, Iterable):
                return index_slice.read(channels=list(channel_slice))
            else:
                raise ValueError("Invalid index into Project: {}".format(slices))
        elif isinstance(slices, BaseSlice):
            return slices.read(channels=list(range(self.channels)))
        elif isinstance(slices, ProjectIndex):
            index = self.project2block(slices)
            return index.block.read_one(index, list(range(self.channels)))
        elif isinstance(slices, BlockIndex):
            index = slices
            return index.block.read_one(index, list(range(self.channels)))

        raise ValueError("Invalid index into Project {}".format(slices))

    def project2block(self, project_index):
        """Convert a ProjectIndex to a BlockIndex"""
        for (i0, i1), block in project.iter_blocks():
            if i1 > project_index:
                return BlockIndex(block, project_index - i0)
        raise ValueError("Could not find BlockIndex")

    def block2project(self, block_index):
        for (i0, i1), block in project.iter_blocks():
            if block == block_index.block:
                return ProjectIndex(self, i0 + block_index)
        raise ValueError("Could not find BlockIndex")
