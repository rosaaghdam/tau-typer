"""Tau-typer app - tau-PET subtyping with SuStaIn (Vogel et al. 2021), step by step.

Run from the project folder (environment active):   shiny run app/app.py
This file is only the user interface; all calculations live in src/.
"""
from pathlib import Path

import pandas as pd
from shiny import App, reactive, render, req, ui

ROOT = Path(__file__).resolve().parent.parent          # project folder (one level above app/)
SAMPLE_DATA = ROOT / "data" / "toy" / "toy_tau.csv"    # simulated toy data, safe to share

app_ui = ui.page_navbar(
    ui.nav_panel("1. Data & QC",
                 ui.layout_sidebar(
                     ui.sidebar(
                         ui.input_file("data_file", "Upload tau-PET data (CSV)", accept=[".csv"]),
                         ui.download_link("download_sample", "Sample Data"),
                         ui.p("Download simulated toy data (500 people) to see the expected format.",
                              class_="text-muted small"),
                         width=320, open="always"),          # always visible, like CMiNet
                     ui.navset_tab(
                         ui.nav_panel("Data Head", ui.output_ui("data_status"),
                                      ui.output_data_frame("data_head"))))),
    ui.nav_panel("2. Normalization",
                 ui.h3("Turn SUVR into z-scores (Gaussian mixture model)"),
                 ui.p("Coming: GMM per region, reference group, cut-points, Z_max.")),
    ui.nav_panel("3. Run SuStaIn",
                 ui.h3("Fit the SuStaIn model (original pySuStaIn)"),
                 ui.p("Coming: hyperparameters with the paper's defaults, background runs, saved runs.")),
    ui.nav_panel("4. Figures",
                 ui.h3("Summary figures"),
                 ui.p("Coming: PVD, stage progression, subtype probabilities, cross-validation.")),
    ui.nav_panel("5. Subtype analysis",
                 ui.h3("Compare the subtypes"),
                 ui.p("Coming: demographics, cognition, regional contrasts, export.")),
    title="Tau-typer",
)


def server(input, output, session):
    # ---------- Tab 1: data upload ----------
    @reactive.calc
    def uploaded_data():
        """The uploaded CSV as a table (re-read automatically when a new file is uploaded)."""
        file = input.data_file()
        if not file:
            return None
        return pd.read_csv(file[0]["datapath"])

    @render.ui
    def data_status():
        data = uploaded_data()
        if data is None:
            return ui.p("No data uploaded yet.")
        return ui.p(f"{data.shape[0]} rows and {data.shape[1]} columns. First 20 rows:")

    @render.data_frame
    def data_head():
        data = uploaded_data()
        req(data is not None)            # show nothing until a file is uploaded
        return data.head(20)

    @render.download_link(filename="tau_typer_sample_data.csv")     # pairs with ui.download_link
    def download_sample():
        return str(SAMPLE_DATA)


app = App(app_ui, server)
