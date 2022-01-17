# soundsep2

Extensible tool for visualizing and labeling WAV file data.

## Get started

If instaling with pip, it is suggested to use a virtual environment. See note at bottom for help installing on M1 Macs (PyQt5 is not compatible and needs to be installed through Rosetta).

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

## Installation on M1 Mac

Installation on a M1 Mac is more complicated because PyQt5 is incompatible and needs to be installed via Rosetta. The instructions here are based on [this stackoverflow answer](https://stackoverflow.com/a/68038451):

1. First create a duplicate Terminal that opens in Rosetta (duplicate the terminal in your `Applications/Utilities` folder, rename it, right-click > Get Info, and check the Rosetta box)
2. Open the rosetta Terminal and double check that it is running in Rosetta (type `arch` and make sure it says `i386` instead of `arm`.
3. Create your virtual environment using the system python, i.e. `/usr/bin/python3 -m venv env`.
4. Then activate the environment and upgrade pip and install PyQt5:
      ```
      source env/bin/activate
      pip install --upgrade pip
      pip install PyQt5
      ```
5. Finally, you can go back into a normal, non-Rosetta Terminal, activate the environment, and install the rest: `pip install -e .`
