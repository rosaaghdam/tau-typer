"""App tab 3 - one folder per SuStaIn run, and starting pySuStaIn in its own process.

A run folder holds: settings.yaml, zscores.csv, zscore_reference.csv (if any), log.txt (all
that pySuStaIn prints), status.txt (finished / failed / cancelled) and pySuStaIn's own files.
"""
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent          # project folder


def prepare_run_folder(folder, settings, z, reference=None):
    """Make a NEW run folder and save what the run needs. Fails if the folder already exists."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)          # never reuse: pySuStaIn would load old results
    with open(folder / "settings.yaml", "w") as f:
        yaml.safe_dump(settings, f, sort_keys=False)
    z.to_csv(folder / "zscores.csv", index_label="subject")
    if reference is not None:                           # needed later to z-score new people the same way
        reference.T.to_csv(folder / "zscore_reference.csv", index_label="region")


def start_run(folder):
    """Start src/run_job.py for this folder as a separate process (so it can run for hours and be stopped)."""
    log = open(Path(folder) / "log.txt", "w")
    env = dict(os.environ, MPLBACKEND="Agg")             # figures go to files only, no windows
    return subprocess.Popen([sys.executable, "-u", "-m", "src.run_job", str(folder)],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
