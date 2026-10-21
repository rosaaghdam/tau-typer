"""App tab 3 - one folder per SuStaIn run, and starting pySuStaIn in its own process.

A run folder holds: settings.yaml, zscores.csv, zscore_reference.csv (if any), covariates.csv
(the other columns: age, sex, diagnosis, ... - used by tab 5), log.txt (all
that pySuStaIn prints), pid.txt (the process number while it runs), status.txt (finished / failed /
cancelled) and pySuStaIn's own files. A run folder can be downloaded as a ZIP and uploaded again.
"""
import io
import os
import re
import subprocess
import sys
import zipfile
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
    process = subprocess.Popen([sys.executable, "-u", "-m", "src.run_job", str(folder)],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    (Path(folder) / "pid.txt").write_text(f"{process.pid}\n")
    return process


def run_status(folder):
    """finished / failed / cancelled (from status.txt), else running - or stopped if its process is
    gone (e.g. the app was closed during the run)."""
    status, pid = folder / "status.txt", folder / "pid.txt"
    if status.exists():
        return status.read_text().strip()
    if pid.exists():
        try:
            os.kill(int(pid.read_text()), 0)            # signal 0 = only check that the process exists
            return "running"
        except (OSError, ValueError):
            pass
    return "stopped"


def list_runs(runs_folder):
    """One row per run folder, newest first: status, start time and the main settings."""
    rows = []
    for settings_file in Path(runs_folder).glob("*/settings.yaml"):
        folder, settings = settings_file.parent, yaml.safe_load(open(settings_file))
        rows.append({"run": folder.name, "status": run_status(folder),
                     "started": datetime.fromtimestamp(settings_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                     "people": settings["n_people"], "max subtypes": settings["n_subtypes_max"],
                     "start points": settings["n_startpoints"], "MCMC": settings["n_iterations_mcmc"],
                     "folds": settings["n_folds"] or "-"})
    columns = ["run", "status", "started", "people", "max subtypes", "start points", "MCMC", "folds"]
    runs = pd.DataFrame(rows, columns=columns).sort_values("started", ascending=False)
    unfinished = ~runs["status"].isin(["finished", "running"])          # failed, cancelled, stopped: at the end
    return pd.concat([runs[~unfinished], runs[unfinished]])


def zip_run(folder):
    """The whole run folder as a ZIP file (bytes), to keep it outside the app."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for path in sorted(Path(folder).rglob("*")):
            if path.is_file() and path.name != "pid.txt":
                zip_file.write(path, path.relative_to(folder))
    return buffer.getvalue()


def unzip_run(zip_path, runs_folder, name):
    """Make a new run folder from a downloaded run ZIP. Returns the folder.
    Checks the ZIP first: it must have settings.yaml and only plain file names (nothing outside the folder)."""
    with zipfile.ZipFile(zip_path) as zip_file:
        names = zip_file.namelist()
        if "settings.yaml" not in names:
            raise ValueError("This is not a Tau-typer run: settings.yaml is missing.")
        if any(n.startswith(("/", "\\")) or ".." in Path(n).parts for n in names):
            raise ValueError("The ZIP has file names outside its folder.")
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", name).strip("._") or "uploaded_run"
        folder, n = Path(runs_folder) / name, 2
        while folder.exists():                         # never overwrite a run
            folder, n = Path(runs_folder) / f"{name}_{n}", n + 1
        zip_file.extractall(folder)
    return folder
