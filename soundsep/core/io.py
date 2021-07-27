"""Module for reading and writing data to and from the Project/Block/AudioFile organization
"""
import collections
import itertools
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import List

import parse

from soundsep.core.models import AudioFile, Block, Project


def load_project(
        directory: Path,
        filename_pattern: str = None,
        block_keys: List[str] = None,
        channel_keys: List[str] = None,
        recursive: bool = False,
    ) -> Project:
    """Load a single WAV file or a directory of WAV files

    Example
    -------
    To load WAV files from a folder that looks like this::

        ./
          data/
            Red77_01012020_12345_Channel0.wav
            Red77_01012020_12345_Channel1and2.wav
            Red77_01012020_23456_Channel0.wav
            Red77_01012020_23456_Channel1and2.wav

    You might load the data like this

    >>> project = load_project(
    ...     "data/",
    ...     filename_pattern="{subject}_{date}_{time}_Channel{channel}.wav",
    ...     block_keys=["date", "time"],
    ...     channel_keys=["channel"]
    ... )

    Arguments
    ---------
    directory : str
        The directory to search for WAV files in. If the given path
        points to a WAV file, create a project containing a single file.
    filename_pattern : str
        A filename pattern with curly bracket "{}" variables. Each set of
        brackets can contain a variable name that can be used to group files as
        blocks and/or order them as channels (see block_keys and channel_keys).
        **NOTE**: your pattern must resolve ambiguity! For example, if filenames
        are of the form "{subject}_{date}_{time}_{channel}.wav", a pattern of
        "{subject}_{timestamp}_{channel}.wav" would be ambiguous, as it could group
        subject and date together, or date and tiem together. Just be careful.
        TODO: add a function that previews how groupings will be made given
        a filename pattern?
    block_keys : List[str]
        A list of keys in filename_pattern that define a single block. All files
        whose filenames match on all of block_keys are grouped into one Block.
    channel_keys : List[str]
        A list of keys in filename_pattern that define the ordering of channels.
        This is used to enforce a consistent mapping of file channels to Block
        channels.
    recursive : bool
        A flag to indicate if the function should search through all subdirectories
        of directory for wav files.

    Returns
    -------
    project : Project
        A soundsep.core.models.Project instance linking all Blocks found that match
        the filename_pattern provided
    """
    if not directory.is_dir() and directory.suffix == ".wav":
        filelist = [directory]
    else:
        filelist = search_for_wavs(directory, recursive=recursive)

    if filename_pattern is None and len(filelist) != 1:
        raise ValueError("Expected to find one .wav file in {}, found {}".format(directory, len(filelist)))

    return _load_project_by_blocks(directory, filelist, filename_pattern, block_keys, channel_keys)


class LoadProjectError(Exception):
    pass


def search_for_wavs(base_directory: Path, recursive: bool = False) -> Path:
    """Look for WAV files in a directory with option to search all subdirectories

    Arguments
    ---------
    base_directory : pathlib.Path
        top level directory to start search from
    recursive : bool (default False)
        if set, will search for WAV files recursively through the directory structure.
        otherwise, will only look for WAV files directly in base_directory

    Returns
    -------
    filelist : List[pathlib.Path]
        A list of **absolute paths** to wav files relative to base_directory
    """
    if recursive:
        return list(base_directory.rglob("*.wav"))
    else:
        return list(base_directory.glob("*.wav"))


def group_files_by_pattern(
        base_directory: Path,
        filelist: List[Path],
        filename_pattern: str,
        block_keys: List[str],
        channel_keys: List[str],
        ) -> Iterable:
    """Build a generator that yields the files in each block

    Returns
    -------
    block_groups : List[Tuple[str, List[AudioFile]]]
        Yields tuples of the form (str, List[AudioFile]), where the
        first element is the block_id parsed from the list of audio
        files in the second element. The AudioFiles in the second
        element are sorted according to the parsed channel_ids

        These potential blocks have not been validated for consistency
        at this point.
    errors : List[str]
        A list of filenames that failed to be parsed successfully into
        a group.
    """
    if filename_pattern is None:
        filename_pattern = "{}"

    parsed_wav_files = []
    bad_wav_files = []    # List of tuples
    for path in filelist:
        relpath = os.path.relpath(path, base_directory)
        parse_result = parse.parse(filename_pattern, relpath)

        if parse_result is None:
            bad_wav_files.append((relpath, parse_result))
            continue

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

            if block_id is None and channel_id is None:
                block_id = path

            parsed_wav_files.append({
                "wav_file": AudioFile(path),
                "block_id": block_id,
                "channel_id": channel_id,
            })
        except KeyError:
            bad_wav_files.append((relpath, parse_result))

    parsed_wav_files = sorted(parsed_wav_files, key=lambda x: (x["block_id"], x["channel_id"]))

    block_groups = [
        (k, list(v))
        for k, v
        in itertools.groupby(parsed_wav_files, key=lambda x: x["block_id"])
    ]

    return (
        block_groups,
        bad_wav_files
    )


def _load_project_by_blocks(
        base_directory: Path,
        filelist: List[str],
        filename_pattern: str,
        block_keys: List[str],
        channel_keys: List[str],
    ):
    block_groups, errors = group_files_by_pattern(
            base_directory,
            filelist,
            filename_pattern,
            block_keys,
            channel_keys
    )

    if len(errors):
        raise LoadProjectError("Failed to parse {} files with\n"
                "filename_pattern={}, block_keys={}, channel_keys={}\n"
                "({} loaded successfully)\n"
                "Files failed:\n{}".format(
                    len(errors),
                    filename_pattern,
                    block_keys,
                    channel_keys,
                    len(block_groups),
                    ",".join([str(e) for e in errors])
                ))

    blocks = []
    channel_ids = collections.defaultdict(list)

    # Collect the blocks but also make sure every block has the same channel ids defined
    for key, group in block_groups:
        group = list(group)
        new_block = Block([g["wav_file"] for g in group], fix_uneven_frame_counts=False)
        blocks.append(new_block)
        channel_ids[tuple([g["channel_id"] for g in group])].append(new_block)

    if channel_keys is not None and len(channel_ids) != 1:
        raise LoadProjectError("Channel ids were not consistent over read blocks. "
            "Check the filename_pattern, block_keys, and channel_keys;\n"
            "For example:\n{}".format(
                "\n".join(
                    [str(([os.path.basename(f.path) for f in v[0]._files], k))
                        for k, v in channel_ids.items()]
                )
            ))

    return Project(blocks=blocks)


import numpy as np
from itertools import combinations


def common_subsequence(items: 'List[Iterable]'):
    def _contains(x, y):
        """Return True if y contains x"""
        i = 0
        for j in range(len(y)):
            if y[j] == x[i]:
                i += 1
            if i == len(x):
                return True

    items = np.array([np.array(list(x)) for x in items], dtype=object)
    shortest = items[np.argmin([len(x) for x in items])]

    results = []
    for l in range(len(shortest), 0, -1):
        for indexes in combinations(np.arange(len(shortest), dtype=np.int), r=l):
            subset = shortest[np.array(indexes)]
            is_valid = np.all([_contains(subset, x) for x in items])
            if is_valid:
                results.append(subset)
        if len(results):
            return list(sorted(set([tuple(x) for x in results]), key=lambda x: (-len(x), tuple(x))))

    return []


def guess_filename_pattern(base_directory: Path, filelist: List[str]):
    """Guesses a set of block keys, channel keys, and filename pattern for a given set of files

    1. Identifies potential variables by common separators ("-", "_", "/", " ")
    2. Excludes variables that are in common across all files
    3. Potential block keys are those combinations of variables that form even numbered groups when grouped by
        and whose groups have the same number of channels and frames (maximizing group size)
    4. Other keys are channel keys
    """
    separator_regex = "[; ,./\\|\\\\\\-_\\+\\:\\=\\(\\)\\{\\}\\[\\]\\*\\?]"

    separators_detected = []
    relpaths = []
    for path in filelist:
        relpath = os.path.relpath(path, base_directory)
        separators_detected.append(re.findall(separator_regex, relpath))
        relpaths.append(relpath)

    separators = [""] + list(common_subsequence(separators_detected)[0]) + [""]
    initial_guess = "{}".join(separators)
    var_names = ["var{}".format(i) for i in range(len(separators) - 1)]
    var_values = ["{{{}}}".format(v) for v in var_names]

    def current_guess():
        return initial_guess.format(*var_values)

    constants = []
    potential_keys = []

    results = {}
    # Fill in constants and find unique keys as well as keys that split the data well
    for var_idx, var_name in enumerate(var_names):
        extracted = [parse.parse(current_guess(), path)[var_name] for path in relpaths]
        if len(set(extracted)) == 1:
            constants.append(var_name)
            var_values[var_idx] = extracted[0]
        else:
            potential_keys.append(var_name)

    valid_block_keys = []
    filename_pattern = current_guess()
    for l in range(len(potential_keys), 0, -1):
        for indexes in combinations(np.arange(len(potential_keys), dtype=np.int), r=l):
            group_keys = list(np.array(potential_keys)[np.array(indexes)])

            groups, errors = group_files_by_pattern(
                base_directory,
                filelist,
                filename_pattern,
                block_keys=group_keys,
                channel_keys=None
            )

            is_valid = True

            if len(errors):
                is_valid = False

            for k, block_info in groups:
                block_len = block_info[0]["wav_file"].frames
                if not all([b["wav_file"].frames == block_len for b in block_info]):
                    is_valid = False

            if is_valid:
                valid_block_keys.append((group_keys, len(groups)))

    if not len(valid_block_keys):
        best_block_keys_guess = []
    else:
        best_block_keys_guess = list(sorted(valid_block_keys, key=lambda x: x[1]))[0][0]

    return best_block_keys_guess, current_guess()
