#!/usr/bin/env python
import inspect
import os
from pathlib import Path

import click


__location__ = os.path.join(os.getcwd(), os.path.dirname(
        inspect.getfile(inspect.currentframe())))


@click.group()
def cli():
    pass


@click.command(help="Run SoundSep GUI")
@click.option("-d", "--debug", help="Run with log level DEBUG", is_flag=True)
def run(debug):
    from soundsep.app.launcher import Launcher
    from soundsep.app.start import run_app
    run_app(MainWindow=Launcher, debug=debug)


@click.command(help="Get project info")
@click.option("-d", "--dir", "_dir", required=True, type=click.Path(exists=True))
def project_info(_dir):
    from soundsep.app.app import SoundsepApp
    from soundsep.core.io import load_project
    from soundsep.core.utils import hhmmss

    if not os.path.basename(_dir) == "soundsep.yaml":
        _dir = os.path.join(_dir, "soundsep.yaml")

    config = SoundsepApp.read_config(_dir)
    project = load_project(
        Path(config["audio_directory"]),
        config["filename_pattern"],
        config["block_keys"],
        config["channel_keys"],
        recursive=config["recursive_search"]
    )

    click.echo("Soundsep project with config {}".format(config))
    click.echo("Channels: {}".format(project.channels))
    click.echo("Blocks: {}".format(len(project.blocks)))
    click.echo("Sampling rate: {}".format(project.sampling_rate))
    click.echo("Frames: {}".format(project.frames))
    click.echo("Duration: {}".format(hhmmss(project.frames / project.sampling_rate, dec=4)))


@click.command(help="Get wav file info")
@click.option("-p", "--path", "path", required=True, type=click.Path(exists=True))
def wav_info(path):
    import soundfile
    from soundsep.core.utils import hhmmss
    with soundfile.SoundFile(path) as f:
        click.echo("WAV file {}".format(path))
        click.echo("Channels: {}".format(f.channels))
        click.echo("Sampling rate: {}".format(f.samplerate))
        click.echo("Frames: {}".format(f.frames))
        click.echo("Duration: {}".format(hhmmss(f.frames / f.samplerate, dec=4)))


@click.command(help="Open sphinx documentation in browser")
def open_doc():
    import webbrowser
    webbrowser.open("file://" + os.path.realpath(os.path.join(__location__, "docs", "_build", "html", "index.html")), new=2)


@click.command("pyuic", help="Run pyuic for QtDesigner .ui -> .py conversion")
def build_ui():
    import glob
    import subprocess

    ui_dir = os.path.join(__location__, "soundsep", "ui")

    '''
    for ui_file in glob.glob(os.path.join(ui_dir, "*.qrc")):
        basename = os.path.splitext(os.path.basename(ui_file))[0]
        p = subprocess.Popen([
            "pyrcc6",
            os.path.join(ui_dir, "{}.qrc".format(basename)),
            "-o",
            os.path.join(ui_dir, "{}_rc.py".format(basename)),
        ])
    '''

    '''
    for ui_file in glob.glob(os.path.join(ui_dir, "*.ui")):
        basename = os.path.splitext(os.path.basename(ui_file))[0]
        p = subprocess.Popen([
            "pyuic6",
            os.path.join(ui_dir, "{}.ui".format(basename)),
            "-o",
            os.path.join(ui_dir, "{}.py".format(basename)),
            "--import-from=soundsep.ui",
            "--resource-suffix=_rc",
        ])
    '''

    for ui_file in glob.glob(os.path.join(ui_dir, "*.ui")):
        basename = os.path.splitext(os.path.basename(ui_file))[0]
        p = subprocess.Popen([
            "pyuic6",
            os.path.join(ui_dir, "{}.ui".format(basename)),
            "-o",
            os.path.join(ui_dir, "{}.py".format(basename)),
        ])

    p.communicate()


@click.command(help="Build sphinx documentation")
def build_doc():
    import subprocess
    p = subprocess.Popen(["make", "html"], cwd=os.path.join(__location__, "docs"))
    p.communicate()


@click.command(help="Run unittests")
@click.option("-d", "--dir", "_dir", type=str, default="soundsep/test")
@click.option("-v", "--verbose", type=int, default=1)
@click.option("-c", "--coverage", "_coverage", help="Save coverage report", is_flag=True)
def unittest(_dir, verbose, _coverage):
    import unittest

    if _coverage:
        from coverage import Coverage
        cov = Coverage()
        cov.start()

    if os.path.isdir(_dir):
        testsuite = unittest.TestLoader().discover(_dir)
    else:
        testsuite = unittest.TestLoader().loadTestsFromName(_dir)
    unittest.TextTestRunner(verbosity=verbose).run(testsuite)

    if _coverage:
        import webbrowser
        cov.stop()
        cov.html_report(directory=os.path.join(__location__, "coverage_html"))
        webbrowser.open("file://" + os.path.realpath(os.path.join(__location__, "coverage_html", "index.html")), new=2)


@click.command(help="Create a new plugin from template")
@click.option("-n", "--name", type=str, required=True, help="New plugin name in snake case, e.g. new_plugin")
def create_plugin(name):
    def _to_camel(s):
        return "".join([part.capitalize() for part in s.split("_")])

    if name.endswith(".py"):
        name = name[:-3]
    camel_name = _to_camel(name)

    target_location = os.path.join(__location__, "soundsep", "plugins", name)
    if os.path.exists(target_location) or os.path.exists(target_location + ".py"):
        click.echo("File or directory already exists at {}. Choose a different --name or move the existing plugin.".format(target_location))
        return

    with open(os.path.join(__location__, "soundsep", "develop", "template_plugin.py.txt"), "r") as f:
        contents = f.read()

    with open(os.path.join(target_location + ".py"), "w+") as f:
        f.write(contents.format(PluginName=camel_name))
    click.echo("Wrote new plugin {} at {}".format(camel_name, target_location + ".py"))
    

cli.add_command(run)
cli.add_command(project_info)
cli.add_command(wav_info)
cli.add_command(unittest)
cli.add_command(build_doc)
cli.add_command(open_doc)
cli.add_command(build_ui)
cli.add_command(create_plugin)


if __name__ == "__main__":
    cli()

