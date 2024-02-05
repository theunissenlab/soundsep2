# soundsep2

Extensible tool for visualizing and labeling WAV file data.

## Get started

If instaling with pip, it is suggested to use a virtual environment. See note at bottom for help installing on M1 Macs (PyQt5 is not compatible and needs to be installed through Rosetta).

```
pip install git+https://github.com/theunissenlab/soundsep2.git@v0.2.0
sep run
```

Replace `v0.2.0` with `main` for the lastest version.

### Preparing files

Prepare a folder that contains your audio data files. For SoundSep to group files recorded simultaneously, they should share common path elements (e.g. be stored in the same folders) or parts of the filename (e.g. a timestamp).

### Loading project

The project directory should contain a config file `soundsep.yaml`. When running the launcher, you should open the directory containing that config file.

To load a project from within a python shell, do the following:

```python
from soundsep import open_project
project = open_project(PATH_TO_SOUNDSEP_YAML_FILE)

# Load data, e.g.
data = project[:44100]
```

### Autodetection with PyTorch

We have a basic auto-detection routine using 2D convolutional neural networks.

The model predicts the presence or absence of a syllable at any given time-point in the spectrogram. It is currently implemented as a 2D convolutional neural network applied in the Spectrogram space, witha  full-connected linear output layer). The output probability timeseries is then thresholded and chunked into segments that can be re-written over the default soundsep save file.

The procedure is typically:

1. Create soundsep project

2. Label some amount of data that will be used as training data (e.g. first 10 minutes, a 10 minute chunk at beginning and 10 minute chunk at end)

3. Note the time ranges you want to use as training data (e.g. 0.0s to 600.0s, 14000.0s to 14600.0s)

4. Run `sep predict train-model` on the training time ranges to train a model. The following example trains on those ranges for 1 epoch (i.e. 1 pass through the data) and loads the model weights from a pretrained model called "pretrained-model.pt". I found that training for at least two epochs is good when a lot (1+ hr of data is labeled), and 5-10 epochs when data size is small. You can update a previously trained model by loading it with `-f`. This might be useful to incrementally increase the data size.

    ```shell
    sep predict train-model \
      -p PROJECT_DIR \
      -r 0.0 600.0 \
      -r 14000.0 14600.0 \
      --save-model saved-model.pt \
      -e 1 \
      --l 1e-3 \
      -f pretrained-model.pt \
      --model MelPredictionNetwork
    ```

5. Run `sep predict apply-model` on the remainder of data or a subset of data to create or overwrite the save file with autogenerated segments. Using 0.0 as the second term in a `-r/--range` option means to go to the end of the project

    ```shell
    sep predict apply-model \
      -p PROJECT_DIR \
      -r 600.0 14000.0 \
      -r 14600.0 0.0 \
      -f saved-model.pt \
      --peak-threshold 0.75 \
      --threshold 0.5 \
      --min-gap-duration 8 \
      --min-segment-duration 8 \
      --tag auto \
      --model MelPredictionNetwork \
      --append-default
    ```

## Installing for development

```
git clone git@github.com:theunissenlab/soundsep2.git
cd soundsep2
pip install -e .
```

## Scripts

See `sep --help` for info. Includes scripts for

* building and opening sphinx auto documentation

* launching Qt Designer

* converting Qt Designer .ui files into .py files

* running unit tests

* creating a template plugin

## Troubleshooting Install on Ubuntu

If you have this error: "qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem."
You may want to try (has worked twice, not sure exactly why):
sudo apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev

## Installation on M1 Mac

The `pyqt6` branch now works without Rosetta on M1 macs but the PyQt libraries are a bit screwy.  THe instructions below worked for M1 and an installation on 09/25/2023

NB: as of 2/5/24 PyQt6.6 does not work properly with pyqtgraph / soundsep. make sure you install PyQt6.5 with the following command
`pip install PyQt5-sip==12.11.0 PyQt6==6.5.2 PyQt6-Qt6==6.5.2 PyQt6-sip==13.5.2 pyqtgraph==0.13.3`

### Instructions:

1. Use the finder (or the find command) to locate the library libqcocoa.dylib and find its full path. In my case it was:
   /Users/frederictheunissen/opt/anaconda3/envs/canary/lib/python3.9/site-packages/PyQt6/Qt6/plugins/platform

   Note that 'canary' is the name of my anaconda environment 

2. QT needs to find this path. For an immedate use make an new environment variable 

`export QT_PLUGIN_PATH=/Users/frederictheunissen/opt/anaconda3/envs/canary/lib/python3.9/site-packages/PyQt6/Qt6/plugins`

Note that you don't specify the platform directory.

3. To make the change permanent, modify the qt.conf (in my case located in /Users/frederictheunissen/opt/anaconda3/envs/canary/bin) and added the plugin directory path to the file, just below [Paths]:

Plugins = /Users/frederictheunissen/opt/anaconda3/envs/canary/lib/python3.9/site-packages/PyQt6/Qt6/plugins
