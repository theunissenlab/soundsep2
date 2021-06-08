"""Module for file loading

TODO: Needs more helper functions to help users get set up or debug their filenames

# n channels, k Blocks
ONE_FILE = 1  # n channels per file, 1 wav file -> 1 block
N_FILES = 2   # 1 channel per file, n wav files -> 1 block
K_FILES = 3   # n channels per file, k wav files -> k blocks
NK_FILES = 4  # 1 channels per file, n*k wav files -> k blocks
"""
import collections
import glob
import itertools
import os
import parse
from collections.abc import Iterable
from enum import Enum
from typing import List

import soundfile

from soundsep.core.io import AudioFile, Block
from soundsep.core.project import Project


def load_project(
        directory,
        organization: AudioFileOrganization,
        filename_pattern: str = None,
        block_keys: List[str] = None,
        channel_keys: List[str] = None,
    ):
    filelist = glob.glob(os.path.join(directory, "*.wav"))

    if filename_pattern is None  and len(filelist) != 1:
        raise ValueError("Expected to find one .wav file in {}, found {}".format(directory, len(filelist)))

    return _load_project_by_blocks(filelist, filename_pattern, block_keys, channel_keys)


def _load_project_by_blocks(
        filelist: List[str],
        filename_pattern: str,
        block_keys: List[str],
        channel_keys: List[str],
    ):
    if filename_pattern is None:
        filename_pattern = "{}"

    parsed_wav_files = []
    for path in filelist:
        parse_result = parse.parse(filename_pattern, os.path.basename(path))

        if parse_result is None:
            raise ValueError("Filename was not parse-able under pattern {}: {}".format(
                filename_pattern,
                path
            ))

        try:
            if callable(block_keys):
                block_id = block_keys(parse_result)
            elif isinstance(block_keys, Iterable):
                block_id = tuple([parse_result[k] for k in block_keys])
            else:
                block_id = None

            if callable(channel_keys):
                channel_id = channel_keys(parse_result)
            elif isinstance(channel_keys, Iterable):
                channel_id = tuple([parse_result[k] for k in channel_keys])
            else:
                channel_id = None

            parsed_wav_files.append({
                "wav_file": AudioFile(path),
                "block_id": block_id,
                "channel_id": channel_id,
            })
        except KeyError:
            raise ValueError("Block or channel information could not be parsed from {}\n"
                    "{}\nblock_keys={}\nchannel_keys={}".format(path, parse_result, block_keys, channel_keys))

    parsed_wav_files = sorted(parsed_wav_files, key=lambda x: (x["block_id"], x["channel_id"]))

    blocks = []
    channel_ids = collections.defaultdict(list)

    # Collect the blocks but also make sure every block has the same channel ids defined
    for key, group in itertools.groupby(parsed_wav_files, key=lambda x: x["block_id"]):
        group = list(group)
        new_block = Block([g["wav_file"] for g in group], fix_uneven_frame_counts=False)
        blocks.append(new_block)
        channel_ids[tuple([g["channel_id"] for g in group])].append(new_block)

    if len(channel_ids) != 1:
        raise ValueError("Channel ids were not consistent over read blocks. "
            "Check the filename_pattern, block_keys, and channel_keys;\n"
            "For example:\n{}".format(
                "\n".join(
                    [str(([os.path.basename(f.path) for f in v[0]._files], k))
                        for k, v in channel_ids.items()]
                )
            ))

    return Project(blocks=blocks)
