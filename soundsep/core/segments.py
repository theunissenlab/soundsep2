"""Models for representing sections of data

ProjectSegments and BlockSegments have different purposes. ProjectSegments are
for seamlessly interacting with data while using the app, while BlockSegments
are useful for writing the data to disk in a way that is indepenent of the Project.

Experiments may contain multiple sources (more than number of microphones/channels).
A source is associated with one primary channel in a project.
"""

import numpy as np

from soundsep.core.models import Block, Project, ProjectIndex, BlockIndex, Source


class OrderableSegment:

    def __gt__(self, other: 'Segment'):
        return self.start.__gt__(other.start)

    def __lt__(self, other: 'Segment'):
        return self.start.__lt__(other.start)

    def __ge__(self, other: 'Segment'):
        return self.start.__ge__(other.start)

    def __le__(self, other: 'Segment'):
        return self.start.__le__(other.start)


class Segment(OrderableSegment):
    """A reference to a window of data corresponding to a given Source

    Arguments
    ---------
    start : ProjectIndex
    stop : ProjectIndex
    source : Source
        The source this segment is associated with. Right now, is not allowed to be None. (Future plan is to
        make it refer abstractly to the specified range (start, stop) if None)
    """
    def __init__(self, start: ProjectIndex, stop: ProjectIndex, source: Source, data: 'Optional[object]' = None):
        if start.project != stop.project:
            raise RuntimeError("ProjectSegment endpoints must be from the same project")

        if data is None:
            data = {}

        self._project = start.project

        if source.project != self._project:
            raise RuntimeError("Source project must match index project")

        self.source = source
        self.start = start
        self.stop = stop
        self.data = data

    @property
    def project(self) -> Project:
        return self._project

    def _to_block_segment(self):
        return BlockSegment(
            self._project.to_block_index(self.start),
            self._project.to_block_index(self.stop),
            source=self.source
        )

    def to_dict(self) -> dict:
        return self._to_block_segment().to_dict()

    def read(self) -> np.ndarray:
        return self.project[self.start:self.stop]


class BlockSegment(OrderableSegment):
    """

    Ideally these segments can live independently of a Project but
    since Sources have to have a project maybe not?

    Arguments
    ---------
    start : BlockIndex
    stop : BlockIndex
    source : Source
        The source this segment is associated with. Right now, is not allowed to be None. (Future plan is to
        make it refer abstractly to the specified range (start, stop) if None)
    """

    def __init__(self, start: BlockIndex, stop: BlockIndex, source: Source, data: None):
        # TODO: Can we support exporting across block boundaries? would have to handle the edge cases where
        # segments span more than 2 blocks!
        if data is None:
            data = {}

        if start.block != stop.block:
            raise RuntimeError("BlockSegment endpoints must be from the same Block")

        self._block = start.block

        if self._block not in source.project.blocks:
            raise RuntimeError("Segment endpoints must refer to a block in the source's project")

        self.source = source
        self.start = start
        self.stop = stop
        self.data = data

    @property
    def block(self) -> Block:
        return self._block

    def to_dict(self):
        """Export the block segment endpoints to a format that is more easily exportable"""
        file_path, channel_idx = self.block.get_channel_info(self.source.channel)

        return {
            "filename": file_path,
            "channel": channel_idx,
            "source": self.source.name,
            "start_frame": int(self.start),
            "stop_frame": int(self.stop),
            "data": self.data
        }

    def read(self) -> np.ndarray:
        return self.project[self.start:self.stop]
