# soundsep2

Extensible tool for visualizing and labeling WAV file data.

## Install

If instaling with pip, it is suggested to use a virtual environment

```
git clone git@github.com:kevinyu/soundsep2.git
cd soundsep2
pip install -e .
```

## Run Soundsep

```shell
sep run
```

## Develop commands

See `sep --help` for info. Includes scripts for

* building and opening sphinx auto documentation

* launching Qt Designer

* converting Qt Designer .ui files into .py files

* running unit tests

## Planned Features

* Load wav files recorded simultaneously and/or chunked in time

* Visualze and label on temporally aligned spectrograms

* Live view of user selection of frequency bands

* Plugin system for extending functions on data manipulation, storage, and display

