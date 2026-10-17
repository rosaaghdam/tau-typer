"""Tau-typer app - tau-PET subtyping with SuStaIn (Vogel et al. 2021), step by step.

Run from the project folder (environment active):   shiny run app/app.py
This file is only the user interface; all calculations live in src/.
"""
import sys
from pathlib import Path

import pandas as pd
import yaml
from shiny import App, reactive, render, req, ui

ROOT = Path(__file__).resolve().parent.parent          # project folder (one level above app/)
sys.path.insert(0, str(ROOT))                           # so Python can find src/
from src.qc import apply_inclusion_rule, demographics_table, guess_columns, qc_summary
from src.figures import plot_region_histograms

CONFIG = yaml.safe_load(open(ROOT / "config.yaml"))    # all settings live in config.yaml

SAMPLE_DATA = ROOT / "data" / "toy" / "toy_tau.csv"    # simulated toy data, safe to share
ROLE_LABELS = {"id": "Subject ID", "age": "Age", "sex": "Sex",
               "diagnosis": "Diagnosis", "amyloid": "Amyloid status"}

app_ui = ui.page_navbar(
    ui.nav_panel("1. Data & QC",
                 ui.layout_sidebar(
                     ui.sidebar(
                         ui.input_file("data_file", "Upload tau-PET data (CSV)", accept=[".csv"]),
                         ui.download_link("download_sample", "Sample Data"),
                         ui.p("Download simulated toy data (500 people) to see the expected format.",
                              class_="text-muted small"),
                         ui.output_ui("column_matching"),           # filled in after an upload
                         width=320, open="always"),          # always visible, like CMiNet
                     ui.navset_tab(
                         ui.nav_panel("Data Head", ui.output_ui("data_status"),
                                      ui.output_data_frame("data_head")),
                         ui.nav_panel("QC summary", ui.output_ui("qc_status"),
                                      ui.output_data_frame("qc_table"),
                                      ui.h6("Inclusion rule", class_="mt-4"),
                                      ui.output_ui("inclusion_status"),
                                      ui.output_data_frame("inclusion_table")),
                         ui.nav_panel("Region distributions",
                                      ui.p("Raw values of each selected region column, before normalization.",
                                           class_="text-muted small"),
                                      ui.output_plot("region_hist", height="520px")),
                         ui.nav_panel("Demographics",
                                      ui.p("Everyone, and per diagnosis group.", class_="text-muted small"),
                                      ui.output_data_frame("demographics"))))),
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
    def column_matching():
        """Dropdowns to say which column is what, pre-filled with guess_columns()."""
        data = uploaded_data()
        if data is None:
            return ui.p("Upload a file to match its columns.", class_="text-muted small")
        columns = list(data.columns)
        guess = guess_columns(columns)
        choices = {"": "(not in file)", **{col: col for col in columns}}
        selects = [ui.input_select(f"col_{role}", label, choices=choices, selected=guess[role] or "")
                   for role, label in ROLE_LABELS.items()]
        regions = ui.input_selectize("col_regions", "Region columns (the SuStaIn features)",
                                     choices=columns, selected=guess["regions"], multiple=True)
        rule = ui.input_checkbox("apply_rule", "Apply the paper's inclusion rule "
                                 "(CN older than 40, or MCI/AD amyloid-positive)", value=False)
        return ui.TagList(ui.hr(), ui.h6("Match the columns"), *selects, regions, ui.hr(), rule)

    @reactive.calc
    def chosen_columns():
        """The user's dropdown choices: role -> column ("" = not in file)."""
        return {role: input[f"col_{role}"]() for role in ROLE_LABELS}

    @reactive.calc
    def included():
        """(people used from here on, reasons table or None). The rule runs only if the box is ticked."""
        data = uploaded_data()
        req(data is not None)
        if not input.apply_rule():
            return data, None
        rule = CONFIG["inclusion"]
        return apply_inclusion_rule(data, chosen_columns(), rule["min_age_cn"],
                                    rule["cn_labels"], rule["impaired_labels"])

    @reactive.calc
    def analysis_data():
        """The people every later step uses (all, or only those kept by the inclusion rule)."""
        return included()[0]

    @render.ui
    def inclusion_status():
        reasons = included()[1]
        if reasons is None:
            return ui.p("Inclusion rule not applied (tick the box in the sidebar to apply it).",
                        class_="text-muted small")
        return ui.p(f"Inclusion rule: {len(analysis_data())} of {len(uploaded_data())} people kept.",
                    class_="fw-bold")

    @render.data_frame
    def inclusion_table():
        reasons = included()[1]
        req(reasons is not None)
        return reasons

    @reactive.calc
    def qc_results():
        data = analysis_data()
        qc = CONFIG["qc"]
        return qc_summary(data, chosen_columns(), input.col_regions() or [], qc["suvr_min"], qc["suvr_max"])

    @render.ui
    def qc_status():
        n_warnings = int((qc_results()["Status"] == "Warning").sum())
        if n_warnings == 0:
            return ui.p("All checks OK.", class_="text-success fw-bold")
        return ui.p(f"{n_warnings} warning(s): please look at the rows marked in red.",
                    class_="text-danger fw-bold")

    @render.data_frame
    def qc_table():
        table = qc_results()
        warning_rows = [i for i, status in enumerate(table["Status"]) if status == "Warning"]
        return render.DataGrid(table, styles=[{"rows": warning_rows, "style": {"background-color": "#fde2e1"}}])

    @render.data_frame
    def demographics():
        return demographics_table(analysis_data(), chosen_columns())

    @render.plot
    def region_hist():
        req(input.col_regions())                         # wait for at least 1 region
        return plot_region_histograms(analysis_data(), input.col_regions())

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
