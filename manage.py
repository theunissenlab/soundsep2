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
def run():
    from soundsep.app.main import SoundsepApp, run_app
    from soundsep.app.app import SoundsepController

    run_app(SoundsepController(), None, MainWindow=SoundsepApp)


@click.command(help="Open sphinx documentation in browser")
def open_doc():
    import webbrowser
    webbrowser.open("file://" + os.path.realpath(os.path.join(__location__, "docs", "_build", "html", "index.html")), new=2)


@click.command("pyuic", help="Run pyuic for QtDesigner .ui -> .py conversion")
def build_ui():
    import subprocess
    p = subprocess.Popen([
        "pyuic5",
        os.path.join(__location__, "soundsep", "gui", "ui", "main_window.ui"),
        "-o",
        os.path.join(__location__, "soundsep", "gui", "ui", "main_window.py"),
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
    

cli.add_command(run)
cli.add_command(unittest)
cli.add_command(build_doc)
cli.add_command(open_doc)
cli.add_command(build_ui)


if __name__ == "__main__":
    cli()

