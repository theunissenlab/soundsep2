#!/usr/bin/env python
import inspect
import os

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


@click.command(help="Open sphinx documentation in browser")
def open_doc():
    import webbrowser
    webbrowser.open("file://" + os.path.realpath(os.path.join(__location__, "docs", "_build", "html", "index.html")), new=2)


@click.command("pyuic", help="Run pyuic for QtDesigner .ui -> .py conversion")
def build_ui():
    import glob
    import subprocess

    ui_dir = os.path.join(__location__, "soundsep", "ui")

    for ui_file in glob.glob(os.path.join(ui_dir, "*.qrc")):
        basename = os.path.splitext(os.path.basename(ui_file))[0]
        p = subprocess.Popen([
            "pyrcc5",
            os.path.join(ui_dir, "{}.qrc".format(basename)),
            "-o",
            os.path.join(ui_dir, "{}_rc.py".format(basename)),
        ])

    for ui_file in glob.glob(os.path.join(ui_dir, "*.ui")):
        basename = os.path.splitext(os.path.basename(ui_file))[0]
        p = subprocess.Popen([
            "pyuic5",
            os.path.join(ui_dir, "{}.ui".format(basename)),
            "-o",
            os.path.join(ui_dir, "{}.py".format(basename)),
            "--import-from=soundsep.ui",
            "--resource-suffix=_rc",
        ])

    p.communicate()


@click.command(help="Build sphinx documentation")
def build_doc():
    import subprocess
    p = subprocess.Popen(["make", "html"], cwd=os.path.join(__location__, "docs"))
    p.communicate()


@click.command(help="Run unittests")
@click.option("-d", "--dir", "_dir", type=str, default=".")
@click.option("-v", "--verbose", type=int, default=1)
@click.option("-c", "--coverage", "_coverage", help="Save coverage report", is_flag=True)
def unittest(_dir, verbose, _coverage):
    import unittest

    if _coverage:
        from coverage import Coverage
        cov = Coverage()
        cov.start()

    if os.path.isdir(_dir):
        testsuite = unittest.TestLoader().discover(".")
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

    with open(os.path.join(__location__, "soundsep", "develop", "template_plugin.py"), "r") as f:
        contents = f.read()

    with open(os.path.join(target_location + ".py"), "w+") as f:
        f.write(contents.format(PluginName=camel_name))
    click.echo("Wrote new plugin {} at {}".format(camel_name, target_location + ".py"))
    

cli.add_command(run)
cli.add_command(unittest)
cli.add_command(build_doc)
cli.add_command(open_doc)
cli.add_command(build_ui)
cli.add_command(create_plugin)


if __name__ == "__main__":
    cli()

