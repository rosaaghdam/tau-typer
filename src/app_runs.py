"""App tab 3 - one folder per SuStaIn run, and starting pySuStaIn in its own process.

A run folder holds: settings.yaml, zscores.csv, zscore_reference.csv (if any), covariates.csv
(the other columns: age, sex, diagnosis, ... - used by tab 5), log.txt (all
that pySuStaIn prints), status.txt (finished / failed / cancelled) and pySuStaIn's own files.
"""
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent          # project folder


def prepare_run_folder(folder, settings, z, reference=None, covariates=None):
    """Make a NEW run folder and save what the run needs. Fails if the folder already exists."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)          # never reuse: pySuStaIn would load old results
    with open(folder / "settings.yaml", "w") as f:
        yaml.safe_dump(settings, f, sort_keys=False)
    z.to_csv(folder / "zscores.csv", index_label="subject")
    if reference is not None:                           # needed later to z-score new people the same way
        reference.T.to_csv(folder / "zscore_reference.csv", index_label="region")
    if covariates is not None:                          # same people and IDs as zscores.csv
        covariates.to_csv(folder / "covariates.csv", index_label="subject")


def start_run(folder):
    """Start src/run_job.py for this folder as a separate process (so it can run for hours and be stopped)."""
    log = open(Path(folder) / "log.txt", "w")
    env = dict(os.environ, MPLBACKEND="Agg")             # figures go to files only, no windows
    return subprocess.Popen([sys.executable, "-u", "-m", "src.run_job", str(folder)],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)


def list_runs(runs_folder):
    """One row per run folder, newest first: status, start time and the main settings."""
    rows = []
    for settings_file in Path(runs_folder).glob("*/settings.yaml"):
        folder, settings = settings_file.parent, yaml.safe_load(open(settings_file))
        status = folder / "status.txt"
        rows.append({"run": folder.name,
                     "status": status.read_text().strip() if status.exists() else "running (or stopped)",
                     "started": datetime.fromtimestamp(settings_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                     "people": settings["n_people"], "max subtypes": settings["n_subtypes_max"],
                     "start points": settings["n_startpoints"], "MCMC": settings["n_iterations_mcmc"],
                     "folds": settings["n_folds"] or "-"})
    columns = ["run", "status", "started", "people", "max subtypes", "start points", "MCMC", "folds"]
    return pd.DataFrame(rows, columns=columns).sort_values("started", ascending=False)
