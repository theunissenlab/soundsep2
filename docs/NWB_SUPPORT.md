# NWB File Support for Soundsep

## Overview

Soundsep now supports reading microphone data from NWB (Neurodata Without Borders) files in addition to WAV files. This allows you to load and analyze audio data stored in the standardized NWB format.

## Installation

To use NWB file support, ensure you have the required dependencies installed:

```bash
pip install pynwb hdmf
```

Or install from the updated requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage

### Using the GUI (ProjectCreator)

The ProjectCreator dialog now supports both WAV and NWB files:

1. Open Soundsep and choose "Create New Project"
2. Browse to select a folder containing WAV and/or NWB files
3. The system will automatically detect both file types
4. Configure filename patterns, block keys, and channel keys as usual
5. Create your project configuration file

The GUI will display both WAV and NWB files in the file tree and validate them during project creation.

### Loading NWB Files Programmatically

NWB files can be loaded just like WAV files using the existing API:

```python
from soundsep import open_project

# Load a single NWB file
project = open_project("path/to/file.nwb")

# Load a directory containing both WAV and NWB files
project = open_project("path/to/directory/")
```

### Project Configuration

You can configure a project that includes NWB files in the `soundsep.yaml` file:

```yaml
audio_directory: "data/"
filename_pattern: "{subject}_{session}_{channel}.nwb"
block_keys: ["subject", "session"]
channel_keys: ["channel"]
recursive_search: false
```

### Mixed Projects

The system supports projects with both WAV and NWB files:

```python
# Directory structure:
# data/
#   recording1_ch0.wav
#   recording1_ch1.wav
#   recording2_mic.nwb
#   recording3_mic.nwb

project = load_project(
    directory="data/",
    filename_pattern="{recording}_*",
    block_keys=["recording"],
    recursive=False
)
```

## NWB File Requirements

The NWB files must contain microphone or audio data in the acquisition group. By default, the system looks for a TimeSeries named "microphone", but it will use the first available acquisition data if "microphone" is not found.

### Expected NWB Structure

```python
# Example NWB file structure:
nwbfile.acquisition['microphone']  # TimeSeries with audio data
  .data  # The actual audio samples (1D or 2D array)
  .rate  # Sampling rate in Hz
```

### Custom Microphone Names

If your NWB file uses a different name for the microphone data, you can specify it when creating an NWBFile directly:

```python
from soundsep.core.models import NWBFile

# Specify custom microphone name
nwb_file = NWBFile("path/to/file.nwb", microphone_name="audio_recording")
```

## API Reference

### NWBFile Class

The `NWBFile` class provides the same interface as `AudioFile`:

- `path`: Full path to the NWB file
- `sampling_rate`: Sampling rate in Hz
- `channels`: Number of audio channels
- `frames`: Total number of samples
- `read(i0, i1)`: Read samples from index i0 to i1
- `open()`: Open the file for reading
- `close()`: Close the file

### Supported Operations

All operations that work with WAV files also work with NWB files:

- Loading projects with mixed file types
- Reading audio data
- Spectrogram computation
- Segmentation and analysis
- Playback (through the GUI)
- Export functionality

### NWB Intervals Plugin

The NWB Intervals Plugin allows you to view and navigate intervals stored in NWB files:

**Features:**
- Automatically detects all NWB files in your project
- Lists all available interval types (from `nwbfile.intervals.keys()`)
- Displays a table of intervals including:
  - Source file
  - Start time
  - Stop time
  - Duration
  - Labels (if available)
- Double-click any interval to jump to its start time in the waveform view

**Usage:**
1. Load a project containing NWB files with interval data
2. Open the "NWB Intervals" panel from the plugin toolbox
3. Select an interval type from the dropdown menu
4. Browse the table of intervals
5. Double-click any row to navigate to that interval's start time

**Note:** The plugin requires NWB files to have intervals stored in the standard `intervals` group.

## Notes

- NWB files are read using the HDF5 backend, which may be slower than WAV files for random access
- The microphone data is automatically converted to float32 format for consistency with WAV files
- Both single-channel (1D) and multi-channel (2D) audio data are supported
- Files are opened lazily and can be closed to free resources

## Troubleshooting

### "pynwb is required to read NWB files"

Install pynwb: `pip install pynwb hdmf`

### "No acquisition data found in NWB file"

Ensure your NWB file contains audio data in the acquisition group. You can inspect the file structure using:

```python
from pynwb import NWBHDF5IO

with NWBHDF5IO('file.nwb', 'r') as io:
    nwbfile = io.read()
    print("Available acquisition data:", list(nwbfile.acquisition.keys()))
```

### "Microphone 'microphone' not found"

The system will automatically use the first available acquisition data with a warning. To use a specific name, create the NWBFile object directly with the `microphone_name` parameter.
