"""Module for reading and writing data to and from the Project/Block/AudioFile organization
"""
import collections
import itertools
import os
import re
import yaml
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import parse
from tqdm import tqdm

from soundsep.app.exceptions import BadConfigFormat, ConfigDoesNotExist
from soundsep.core.models import AudioFile, PhotoProject, DatFile, NWBFile, Block, Project


def open_project(path: Path):
    """Loads a project folder from config file or project directory

    Arguments
    ---------
    path : pathlib.Path
        Path to either project folder containing a soundsep.yaml file,
        or a soundsep.yaml file directly

    Returns
    -------
    project : soundsep.core.models.Project
        The Project instance
    """
    path = Path(path)
    if not path.exists() or (path.is_dir() and not (path / "soundsep.yaml").exists()):
        raise ConfigDoesNotExist(f"Config does not exist at {path / 'soundsep.yaml'}")
    
    if not path.is_dir() and path.suffix == ".yaml":
        config_path = path
    elif path.is_dir():
        config_path = path / "soundsep.yaml"

    try:
        with open(config_path, "r") as f:
            local_config = yaml.load(f, Loader=yaml.SafeLoader)
    except:
        raise BadConfigFormat(f"Error reading {config_path}")

    if local_config["audio_directory"]:
        audio_dir = Path(local_config["audio_directory"])
        if not audio_dir.is_absolute():
            local_config["audio_directory"] = str(config_path.parent / audio_dir)

    return load_project(
        Path(local_config["audio_directory"]),
        local_config["filename_pattern"],
        local_config["block_keys"],
        local_config["channel_keys"],
        recursive=local_config["recursive_search"],
    )


def load_project(
        directory: Path,
        filename_pattern: str = None,
        block_keys: List[str] = None,
        channel_keys: List[str] = None,
        recursive: bool = False,
    ) -> Project:
    """Load a single audio file or a directory of audio files (WAV, NWB, or DAT)

    Example
    -------
    To load audio files from a folder that looks like this::

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
        The directory to search for audio files (WAV, NWB, or DAT) in. If the given path
        points to an audio file, create a project containing a single file.
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
        of directory for audio files.

    Returns
    -------
    project : Project
        A soundsep.core.models.Project instance linking all Blocks found that match
        the filename_pattern provided
    """
    if not directory.is_dir() and directory.suffix in [".wav", ".nwb", ".dat"]:
        filelist = [directory]
    else:
        filelist = search_for_audio_files(directory, recursive=recursive)

    if filename_pattern is None and len(filelist) != 1:
        raise ValueError("Expected to find one audio file in {}, found {}".format(directory, len(filelist)))

    return _load_project_by_blocks(
            directory,
            filelist,
            filename_pattern,
            block_keys,
            channel_keys,
            only_include_matching=True
            )


class LoadProjectError(Exception):
    pass


def search_for_audio_files(base_directory: Path, recursive: bool = False) -> List[Path]:
    """Look for audio files (WAV, NWB, and DAT) in a directory with option to search all subdirectories

    Arguments
    ---------
    base_directory : pathlib.Path
        top level directory to start search from
    recursive : bool (default False)
        if set, will search for audio files recursively through the directory structure.
        otherwise, will only look for audio files directly in base_directory

    Returns
    -------
    filelist : List[pathlib.Path]
        A list of **absolute paths** to audio files relative to base_directory
    """
    if recursive:
        wav_files = list(base_directory.rglob("*.wav"))
        nwb_files = list(base_directory.rglob("*.nwb"))
        dat_files = list(base_directory.rglob("*.dat"))
        return sorted(wav_files + nwb_files + dat_files)
    else:
        wav_files = list(base_directory.glob("*.wav"))
        nwb_files = list(base_directory.glob("*.nwb"))
        dat_files = list(base_directory.glob("*.dat"))
        return sorted(wav_files + nwb_files + dat_files)


def search_for_wavs(base_directory: Path, recursive: bool = False) -> List[Path]:
    """Look for WAV files in a directory with option to search all subdirectories
    
    Deprecated: Use search_for_audio_files instead.

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

def load_file(path: Path):
    """Load a single audio file based on its extension"""
    # if path is a string, turn it into a path
    if isinstance(path, str):
        path = Path(path)
    if PhotoProject.is_photo_project(path):
        return PhotoProject(path)
    elif path.suffix.lower() == ".nwb":
        return NWBFile(path)
    elif path.suffix.lower() == ".dat":
        return DatFile(path)
    else:
        return AudioFile(path)

def group_files_by_pattern(
        base_directory: Path,
        filelist: List[Path],
        filename_pattern: str,
        block_keys: List[str],
        channel_keys: List[str],
        load_files: bool = True,
        parallel: bool = True,
        max_workers: int = None
        ) -> Iterable:
    """Build a generator that yields the files in each block

    Returns
    -------
    block_groups : List[Tuple[str, List[Union[AudioFile, NWBFile, DatFile]]]]
        Yields tuples of the form (str, List[AudioFile/NWBFile/DatFile]), where the
        first element is the block_id parsed from the list of audio
        files in the second element. The AudioFiles/NWBFiles/DatFiles in the second
        element are sorted according to the parsed channel_ids

        These potential blocks have not been validated for consistency
        at this point.
    errors : List[str]
        A list of filenames that failed to be parsed successfully into
        a group.
    """
    if filename_pattern is None:
        filename_pattern = "{}"

    # First pass: parse filenames and build metadata (no file loading)
    parsed_entries = []
    bad_wav_files = []
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
                block_id = str(path)

            parsed_entries.append({
                "block_id": block_id,
                "channel_id": channel_id,
                "path": path
            })
        except KeyError:
            bad_wav_files.append((relpath, parse_result))

    # Second pass: load files (optionally in parallel)
    parsed_wav_files = []
    if load_files and parsed_entries:
        paths_to_load = [e["path"] for e in parsed_entries]

        if parallel and len(paths_to_load) > 1:
            # Load files in parallel using threads (I/O bound)
            loaded_files = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_path = {executor.submit(load_file, p): p for p in paths_to_load}
                for future in tqdm(as_completed(future_to_path), total=len(paths_to_load),
                                   desc="Loading audio files", unit="file"):
                    path = future_to_path[future]
                    try:
                        loaded_files[path] = future.result()
                    except Exception as e:
                        relpath = os.path.relpath(path, base_directory)
                        bad_wav_files.append((relpath, str(e)))

            # Build final list with loaded files
            for entry in parsed_entries:
                if entry["path"] in loaded_files:
                    parsed_wav_files.append({
                        "wav_file": loaded_files[entry["path"]],
                        "block_id": entry["block_id"],
                        "channel_id": entry["channel_id"],
                        "path": entry["path"]
                    })
        else:
            # Load files sequentially
            for entry in tqdm(parsed_entries, desc="Loading audio files", unit="file"):
                try:
                    file_obj = load_file(entry["path"])
                    parsed_wav_files.append({
                        "wav_file": file_obj,
                        "block_id": entry["block_id"],
                        "channel_id": entry["channel_id"],
                        "path": entry["path"]
                    })
                except Exception as e:
                    relpath = os.path.relpath(entry["path"], base_directory)
                    bad_wav_files.append((relpath, str(e)))
    else:
        # No file loading needed
        for entry in parsed_entries:
            parsed_wav_files.append({
                "wav_file": None,
                "block_id": entry["block_id"],
                "channel_id": entry["channel_id"],
                "path": entry["path"]
            })

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
        only_include_matching: bool,
    ):
    block_groups, errors = group_files_by_pattern(
            base_directory,
            filelist,
            filename_pattern,
            block_keys,
            channel_keys
    )

    if len(errors) and not only_include_matching:
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

    if len(blocks) == 0 and len(channel_ids) == 0:
        raise LoadProjectError("No data found. Check data path and channel, block keys:\n"
                f"{base_directory} {'EXISTS' if base_directory.exists() else 'DOES NOT EXIST'}\n"
                f"matching pattern: {filename_pattern}\n"
                f"using block keys: {block_keys}\n"
                f"using channel keys: {channel_keys}")

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
        for indexes in combinations(np.arange(len(shortest), dtype=int), r=l):
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
        for indexes in combinations(np.arange(len(potential_keys), dtype=int), r=l):
            group_keys = list(np.array(potential_keys)[np.array(indexes)])

            groups, errors = group_files_by_pattern(
                base_directory,
                filelist,
                filename_pattern,
                block_keys=group_keys,
                channel_keys=None, 
                load_files=False
            )

            is_valid = True

            if len(errors):
                is_valid = False

            # lets test up to 10 random groups
            group_inds_to_test = np.random.choice(len(groups), min(10, len(groups)), replace=False)
            for k, block_info in [groups[i] for i in group_inds_to_test]:
                loaded_file = load_file(block_info[0]["path"])
                block_len = loaded_file.frames
                if not all([load_file(b["path"]).frames == block_len for b in block_info]):
                    is_valid = False

            if is_valid:
                valid_block_keys.append((group_keys, len(groups)))

    if not len(valid_block_keys):
        best_block_keys_guess = []
    else:
        best_block_keys_guess = list(sorted(valid_block_keys, key=lambda x: x[1]))[0][0]

    return best_block_keys_guess, current_guess()
