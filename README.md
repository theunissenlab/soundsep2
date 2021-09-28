# soundsep2

Extensible tool for visualizing and labeling WAV file data.

## Get started

If instaling with pip, it is suggested to use a virtual environment

```
pip install git+https://github.com/theunissenlab/soundsep2.git@v0.1.3
sep run
```

Replace `v0.1.3` with `main` for the lastest version.

### Preparing files

Prepare a folder that contains your audio data files. For SoundSep to group files recorded simultaneously, they should share common path elements (e.g. be stored in the same folders) or parts of the filename (e.g. a timestamp).

### Loading project

The project directory should contain a config file `soundsep.yaml`. When running the launcher, you should open the directory containing that config file.

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
