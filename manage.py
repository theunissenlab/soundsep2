#!/usr/bin/env python
import inspect
import os
from pathlib import Path
from typing import Tuple

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


@click.group()
def predict():
    """Commands for fitting and running predefined models"""
    pass


@click.command()
def check_cuda():
    """Check CUDA version and if pytorch can see it"""
    import torch
    if torch.cuda.is_available():
        click.echo("CUDA is available")
    else:
        click.echo("CUDA is NOT available")
    click.echo(f"Torch is using CUDA version {torch.version.cuda}")


def clean_range_input(range_input: Tuple[float, float], project: 'soundsep.core.models.Project'):
    x0, x1 = range_input
    if x1 == 0.0:
        x1 = project.frames / project.sampling_rate

    if x1 < x0:
        raise ValueError(f"Cannot use a range where second value ({x1}) is greater than first ({x0})")

    return x0, x1


@click.command()
@click.option("-p", "--project", "project_dir", required=True, type=click.Path(exists=True))
@click.option("-r", "--ranges", help="Ranges in seconds to include in training data, set second number 0.0 to predict to end", type=(float, float), multiple=True)
@click.option("-f", "--model-file", type=click.Path(exists=False))
@click.option("-s", "--save-model", type=click.Path(exists=False))
@click.option("-d", "--device", type=click.Choice(["cuda", "cpu"]))
@click.option("-e", "--epochs", default=1, type=int)
@click.option("-b", "--batch-size", default=64, type=int)
@click.option("-l", "--lr", default=1e-2, type=float)
@click.option("-m", "--model", "model_name", default="MelPredictionNetwork", type=str)
@click.option("--shuffle", help="Shuffle training data", is_flag=True)
def train_model(
        project_dir,
        ranges,
        model_file,
        save_model,
        device,
        batch_size,
        epochs,
        lr,
        shuffle,
        model_name,
        ):
    """Train a Pytorch model to predict Sources in given project
    """
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from soundsep import open_project
    from soundsep_prediction.dataset import CompositeDataset
    from soundsep_prediction.fit import partial_fit
    from soundsep_prediction import models

    PredictionNetwork = getattr(models, model_name)

    project_dir = Path(project_dir)
    project = open_project(project_dir)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if model_file:
        model = PredictionNetwork.from_file(model_file, 4, output_channels=3)
    else:
        model = PredictionNetwork(4, output_channels=3)
    model.to(device)

    segments = pd.read_csv(
        project_dir / "_appdata" / "save" / "segments.csv",
        converters={"Tags": str},
        index_col=0,
    )
    source_names = np.sort(segments.SourceName.unique())

    if not ranges:
        ranges = [(
            segments.iloc[0]["StartIndex"] / project.sampling_rate,
            segments.iloc[-1]["StopIndex"] / project.sampling_rate
        )]
    else:
        ranges = [clean_range_input(range_in, project) for range_in in ranges]

    ds = CompositeDataset(
        project_dir=project_dir,
        syllable_table=segments,
        source_names=source_names,
        time_ranges=ranges,
    )
    dl = DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=True)

    loss = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def on_epoch_complete(epoch, model, avg_loss):
        print(f"Epoch {epoch}; Loss={avg_loss:.4f}")
        if save_model:
            torch.save(model.state_dict(), save_model)

    partial_fit(epochs, model, loss, opt, dl, device=device, on_epoch_complete=on_epoch_complete)


@click.command()
@click.option("-p", "--project", "project_dir", required=True, type=click.Path(exists=True))
@click.option("-d", "--device", type=click.Choice(["cuda", "cpu"]))
@click.option("-b", "--batch-size", default=128, type=int)
@click.option("-f", "--model-file", type=click.Path(exists=False))
@click.option("-r", "--ranges", help="Ranges in seconds to predict", type=(float, float), multiple=True)
@click.option("-t", "--threshold", help="Probability to threshold syllable at", default=0.5, type=float)
@click.option("--min-gap-duration", help="How log of a period to ignore if probability dips below threshold (in ms)", type=float, default=8.0)
@click.option("--min-segment-duration", help="How log a segment must be (in ms)", default=8.0)
@click.option("--peak-threshold", help="A segment must have a peak probability of peak-threshold to be counted", type=float, default=0.0)
@click.option("-a", "--append-to", help="Soundsep save file (_appdata/save/segments.csv) to add predicted segments to", type=click.Path(exists=True))
@click.option("--append-default", help="Append and delete segments from default save file (_appdata/save/segments.csv)", is_flag=True)
@click.option("-m", "--model", "model_name", default="MelPredictionNetwork", type=str)
@click.option("--tag", help="Add a tag to autogenerated segments", type=str)
@click.option("--eval", "eval_", help="Evaluate model on data with true labels", is_flag=True)
def apply_model(
        project_dir,
        device,
        batch_size,
        model_file,
        ranges,
        threshold,
        min_gap_duration,
        min_segment_duration,
        peak_threshold,
        append_to,
        append_default,
        model_name,
        tag,
        eval_,
        write_savefile=False
        ):
    """Apply a trained model to predict syllable boundaries

    Create a table in the form of a soundsep save file (with columns SourceName,
    """
    if eval_ and append_to:
        raise ValueError("Cannot evaluate and update a segment table simultaneously")
    if append_to and append_default:
        raise ValueError("Cannot specify -a/--append_to and --append-default at the same time. "
                "--append-default should update the save file in place.")

    import json
    import secrets
    import shutil

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from soundsep import open_project
    from soundsep_prediction import models
    from soundsep_prediction.dataset import CompositeDataset
    from soundsep_prediction.fit import partial_predict, partial_test, to_segments_table

    PredictionNetwork = getattr(models, model_name)

    project_dir = Path(project_dir)
    project = open_project(project_dir)

    if append_default:
        append_to = project_dir / "_appdata" / "save" / "segments.csv"

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = PredictionNetwork.from_file(model_file, 4, output_channels=3)
    model.to(device)
    model.eval()

    segments = pd.read_csv(
        project_dir / "_appdata" / "save" / "segments.csv",
        converters={"Tags": str},
        index_col=0,
    )
    source_names = np.sort(segments.SourceName.unique())
    source_channels = [
        segments[segments.SourceName == source_name].iloc[0]["SourceChannel"]
        for source_name in source_names
    ]

    if not ranges:
        ranges = [(
            segments.iloc[-1]["StopIndex"] / project.sampling_rate,
            project.frames / project.sampling_rate
        )]
    else:
        ranges = [clean_range_input(range_in, project) for range_in in ranges]

    # Get sorted source names
    ds = CompositeDataset(
        project_dir=project_dir,
        syllable_table=segments,
        source_names=source_names,
        time_ranges=ranges,
    )
    dl = DataLoader(ds, batch_size=batch_size, num_workers=4)

    if eval_:
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = partial_test(model, loss_fn, dl, device=device)
        click.echo(f"Loss: {loss:.5f}")
        return

    p = partial_predict(model, dl, return_labels=False, device=device) 
    p = p.numpy()

    # Chunk p by time range, since each time range will have its own offset time
    segments_table = []
    current_offset = 0
    for _ds in ds.datasets:
        next_ds_table = to_segments_table(
            p[current_offset:current_offset + len(_ds)],
            threshold,
            source_names,
            source_channels,
            ds.stft_params.hop,
            min_gap_size=int(
                ((min_gap_duration / 1000) * project.sampling_rate) / ds.stft_params.hop
            ),
            min_segment_size=int(
                ((min_segment_duration / 1000) * project.sampling_rate) / ds.stft_params.hop
            ),
            min_p_max=peak_threshold
        )
        next_ds_table["StartIndex"] += _ds.start
        next_ds_table["StopIndex"] += _ds.start
        current_offset += len(_ds)
        segments_table.append(next_ds_table)
    segments_table = pd.concat(segments_table, ignore_index=True)

    if tag:
        segments_table["Tags"] = json.dumps([tag])

    if append_to:
        original_segments = pd.read_csv(
            Path(append_to),
            converters={"Tags": str},
            index_col=0,
        )
        # Assume file exists because we used click.Path(exists=True)
        backup_file = f"{append_to}.backup.{secrets.token_hex()[:6]}"
        click.echo(f"Backing up {append_to} to {backup_file}")
        shutil.copy(append_to, backup_file)

        # Clear out segments that overlap this time range
        overlapping_selector = np.zeros(len(original_segments)).astype(bool)
        for t0, t1 in ranges:
            i0 = int(t0 * project.sampling_rate)
            i1 = int(t1 * project.sampling_rate)
            overlapping_selector = overlapping_selector | (
                ((original_segments["StopIndex"] >= i0) & (original_segments["StopIndex"] <= i1))
                | ((original_segments["StartIndex"] >= i0) & (original_segments["StartIndex"] <= i1))
                | ((original_segments["StartIndex"] <= i0) & (original_segments["StopIndex"] >= i1))
            )
        original_segments = original_segments[~overlapping_selector]
        n_deleted = int(np.sum(overlapping_selector))
        new_table = pd.concat([original_segments, segments_table], ignore_index=True).sort_values("StartIndex").reindex()
        new_table.to_csv(append_to)
        click.echo(f"Updated {append_to}; Deleted {n_deleted}; Created {len(segments_table)}")
    else:
        click.echo(segments_table.to_csv())


cli.add_command(run)
cli.add_command(project_info)
cli.add_command(wav_info)
cli.add_command(unittest)
cli.add_command(build_doc)
cli.add_command(open_doc)
cli.add_command(build_ui)
cli.add_command(create_plugin)

predict.add_command(check_cuda)
predict.add_command(train_model)
predict.add_command(apply_model)

cli.add_command(predict)

if __name__ == "__main__":
    cli()
