import click

from soundsep.core.io import load_project
from soundsep.gui.main import run_app
from soundsep.app import MainApp


@click.group()
def cli():
    pass


@click.command(help="Run SoundSep GUI")
@click.option("-d", "--dir", "_dir", type=str, default="data")
def run(_dir):
    project = load_project(_dir, filename_pattern="ch{channel}.wav", channel_keys=["channel"])
    run_app(MainApp, project)


@click.command(help="Open sphinx documentation in browser")
def doc():
    import webbrowser
    webbrowser.open('docs/_build/html/index.html', new=2)


@click.command(help="Build sphinx documentation")
def build_doc():
    import os
    import inspect
    import subprocess

    __location__ = os.path.join(os.getcwd(), os.path.dirname(
        inspect.getfile(inspect.currentframe())))
    p = subprocess.Popen(["make", "html"], cwd=os.path.join(__location__, "docs"))
    p.communicate()


@click.command(help="Run unittests")
@click.option("-d", "--dir", "_dir", type=str, default=".")
@click.option("-v", "--verbose", type=int, default=1)
@click.option("-c", "--coverage", "_coverage", help="Save coverage report", is_flag=True)
def unittest(_dir, verbose, _coverage):
    import os
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
        cov.stop()
        cov.html_report(directory="coverage_html")
        click.echo("Open coverage_html/index.html")
    

cli.add_command(run)
cli.add_command(unittest)
cli.add_command(build_doc)
cli.add_command(doc)


if __name__ == "__main__":
    cli()

