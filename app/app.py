"""Tau-typer app - tau-PET subtyping with SuStaIn (Vogel et al. 2021), step by step.

Run from the project folder (environment active):   shiny run app/app.py
This file only puts the tabs together: each tab is in its own file (tab1_data.py, ...), shared
settings and texts are in common.py, and all calculations live in src/.
"""
import secrets
import sys
from pathlib import Path

from shiny import App, ui

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so the tab files can be imported
import tab1_data
import tab2_normalization
import tab3_run
import tab4_figures
import tab5_analysis
import tab_about
from common import CONFIG, RUNS_FOLDER

TABS = [tab1_data, tab2_normalization, tab3_run, tab4_figures, tab5_analysis, tab_about]

app_ui = ui.page_navbar(*[tab.panel for tab in TABS], title="Tau-typer")


def server(input, output, session):
    # Where this visit's runs are saved. On a shared server (config runs.private_per_visit: true) every
    # visit gets its own folder, sees only its own runs, and the folder is deleted when the visit ends.
    private = CONFIG["runs"].get("private_per_visit", False)
    runs_folder = RUNS_FOLDER / f"visit_{secrets.token_hex(8)}" if private else RUNS_FOLDER
    shared = {"private": private, "runs_folder": runs_folder}   # each tab adds what the later tabs need
    for tab in TABS:
        tab.server(input, output, session, shared)


app = App(app_ui, server)
