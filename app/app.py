"""Tau-typer app - tau-PET subtyping with SuStaIn (Vogel et al. 2021), step by step.

Run from the project folder (environment active):   shiny run app/app.py
This file is only the user interface; all calculations live in src/.
"""
import sys
from pathlib import Path

import pandas as pd
import yaml
from shiny import App, reactive, render, req, ui
from shiny.types import SafeException

ROOT = Path(__file__).resolve().parent.parent          # project folder (one level above app/)
sys.path.insert(0, str(ROOT))                           # so Python can find src/
from src.qc import apply_inclusion_rule, demographics_table, guess_columns, qc_summary
from src.figures import plot_gmm_fit, plot_region_histograms
from src.sustain_model import count_people_above, parse_cut_points, z_max_table
from src.zscore import (amyloid_negative_cn, fit_gmm, reference_group_normal, zscore_regions,
                        zscore_with_reference)

NORM_METHODS = {"gmm": "GMM on everyone (paper)", "reference": "Reference group: amyloid-negative CN",
                "zscored": "My data is already z-scored"}

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
                 ui.layout_sidebar(
                     ui.sidebar(
                         ui.input_radio_buttons("norm_method", "Where 'normal' comes from",
                                                choices=NORM_METHODS, selected="gmm"),
                         ui.panel_conditional(
                             "input.norm_method === 'gmm'",
                             ui.input_numeric("gmm_components", "Number of GMM components",
                                              value=CONFIG["pipeline"]["gmm_components"], min=2, max=5),
                             ui.p("Per region, a Gaussian mixture model splits the values into bell curves; "
                                  "the one with the lowest mean is 'normal'. Paper: 2.", class_="text-muted small")),
                         ui.input_text("cut_points", "Cut-points (the SuStaIn events)",
                                       value=", ".join(f"{z:g}" for z in CONFIG["sustain"]["z_events"])),
                         ui.p("z-scores at which a region counts as 'reaching an event'. Paper: 2, 5, 10.",
                              class_="text-muted small"),
                         ui.input_radio_buttons("z_max_method", "Z_max (the z-score each region reaches at the very end)",
                                                choices={"fixed": "Fixed number", "percentile": "Percentile per region"},
                                                selected=CONFIG["normalization"]["z_max_method"]),
                         ui.panel_conditional("input.z_max_method === 'fixed'",
                                              ui.input_numeric("z_max_fixed", "Fixed Z_max", value=CONFIG["sustain"]["z_max"])),
                         ui.panel_conditional("input.z_max_method === 'percentile'",
                                              ui.input_numeric("z_max_percentile", "Percentile (among people with z >= first cut-point)",
                                                               value=CONFIG["normalization"]["z_max_percentile"], min=50, max=100)),
                         ui.p("Not reported in the paper. Must be above the last cut-point.", class_="text-muted small"),
                         ui.input_action_button("normalize", "Normalize", class_="btn-primary"),
                         width=320, open="always"),
                     ui.navset_tab(
                         ui.nav_panel("Normal reference", ui.output_ui("norm_status"),
                                      ui.output_data_frame("norm_reference")),
                         ui.nav_panel("GMM fit per region", ui.output_ui("gmm_region_choice"),
                                      ui.output_plot("gmm_fit_plot", height="420px")),
                         ui.nav_panel("Z-scores",
                                      ui.p("z = (value - normal mean) / normal SD. First 20 people:",
                                           class_="text-muted small"),
                                      ui.output_data_frame("z_head"),
                                      ui.output_plot("z_hist", height="520px")),
                         ui.nav_panel("Events",
                                      ui.p("Number of people reaching each cut-point in each region.",
                                           class_="text-muted small"),
                                      ui.output_ui("events_status"),
                                      ui.output_data_frame("events_table")),
                         ui.nav_panel("Z_max", ui.output_ui("zmax_status"),
                                      ui.output_data_frame("zmax_table"))))),
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

    # ---------- Tab 2: normalization ----------
    @reactive.calc
    @reactive.event(input.normalize)                    # runs only when Normalize is clicked
    def normalized():
        """z-scores of the kept people with the chosen method ("gmm", "reference" or "zscored").
        Returns a dict: z-scores, normal reference (None if already z-scored), values used,
        method, settings used, reference-group size and how many people were left out."""
        regions = list(input.col_regions() or [])
        req(regions)
        data = analysis_data()
        values = data[regions].apply(pd.to_numeric, errors="coerce")
        complete = values.dropna()                       # people with every region value
        method, n_components, n_reference = input.norm_method(), input.gmm_components(), None
        if method == "gmm":
            z, reference = zscore_regions(complete, regions, n_components, CONFIG["pipeline"]["seed"])
        elif method == "reference":
            columns = chosen_columns()
            if not (columns.get("diagnosis") and columns.get("amyloid")):
                raise SafeException("The reference group needs the diagnosis and amyloid columns (tab 1).")
            in_reference = amyloid_negative_cn(data.loc[complete.index], columns["diagnosis"],
                                               columns["amyloid"], CONFIG["inclusion"]["cn_labels"])
            n_reference = int(in_reference.sum())
            if n_reference < 2:
                raise SafeException("No amyloid-negative CN people found for the reference group.")
            reference = reference_group_normal(complete, regions, in_reference)
            z = zscore_with_reference(complete, regions, reference)
        else:                                            # the uploaded values already are z-scores
            z, reference = complete.copy(), None
        id_col = chosen_columns().get("id")
        if id_col:                                       # label rows with subject IDs
            ids = data.loc[complete.index, id_col].values
            z.index, complete.index = ids, ids
        return {"z": z, "reference": reference, "values": complete, "method": method,
                "n_components": n_components, "n_reference": n_reference,
                "n_left_out": len(values) - len(complete)}

    @render.ui
    def norm_status():
        if input.normalize() == 0:
            return ui.p("Choose the settings and click Normalize.", class_="text-muted")
        result = normalized()
        lines = [ui.p(f"{NORM_METHODS[result['method']]}: {len(result['z'])} people in {result['z'].shape[1]} "
                      f"regions ({result['n_left_out']} left out because of missing region values).",
                      class_="fw-bold")]
        if result["method"] == "reference":
            lines.append(ui.p(f"Reference group: {result['n_reference']} amyloid-negative CN people."))
        if result["method"] == "zscored":
            median = float(result["z"].median().median())
            lines.append(ui.p("No normal reference is computed: the uploaded values are used as z-scores."))
            if abs(median) > 1:
                lines.append(ui.p(f"Warning: the median value is {median:.2f}, far from 0. Are these really "
                                  "z-scores (not raw SUVR)?", class_="text-danger fw-bold"))
        return ui.TagList(*lines)

    @render.data_frame
    def norm_reference():
        reference = normalized()["reference"]
        req(reference is not None)                       # nothing to show for already z-scored data
        return reference.T.round(3).reset_index(names="Region")   # one row per region

    @render.ui
    def gmm_region_choice():
        if input.normalize() == 0:
            return ui.p("Click Normalize first.", class_="text-muted")
        if normalized()["method"] != "gmm":
            return ui.p("The GMM figure is only used for 'GMM on everyone'.", class_="text-muted")
        regions = list(normalized()["values"].columns)
        return ui.input_select("gmm_region", "Region", choices=regions, selected=regions[0])

    @render.plot
    def gmm_fit_plot():
        result = normalized()
        req(result["method"] == "gmm")
        region = input.gmm_region()
        req(region in result["values"].columns)
        values = result["values"][region]
        gmm = fit_gmm(values, result["n_components"], CONFIG["pipeline"]["seed"])   # same as Normalize
        reference = result["reference"]
        return plot_gmm_fit(values, gmm, region, reference.loc["normal_mean", region],
                            reference.loc["normal_sd", region], cut_points())

    @render.data_frame
    def z_head():
        return normalized()["z"].head(20).round(2).reset_index(names="Subject")

    @render.plot
    def z_hist():
        z = normalized()["z"]
        return plot_region_histograms(z, list(z.columns), xlabel="z-score")

    @reactive.calc
    def cut_points():
        """The typed cut-points as numbers; a clear message instead of an error if the text is invalid."""
        try:
            return parse_cut_points(input.cut_points())
        except ValueError as error:
            raise SafeException(str(error))              # shown to the user in the outputs

    @reactive.calc
    def event_counts():
        """People reaching each cut-point (rows = regions, columns = cut-points)."""
        z = normalized()["z"]
        return count_people_above(z, list(z.columns), cut_points())

    @render.ui
    def events_status():
        if input.normalize() == 0:
            return ui.p("Click Normalize first.", class_="text-muted")
        minimum = CONFIG["normalization"]["min_people_per_event"]
        n_rare = int((event_counts() < minimum).sum().sum())
        if n_rare == 0:
            return ui.p(f"All events are reached by at least {minimum} people.", class_="text-success fw-bold")
        return ui.p(f"{n_rare} event(s) are reached by fewer than {minimum} people (rows in red).",
                    class_="text-danger fw-bold")

    @render.data_frame
    def events_table():
        counts = event_counts()
        minimum = CONFIG["normalization"]["min_people_per_event"]
        rare_rows = [i for i, region in enumerate(counts.index) if (counts.loc[region] < minimum).any()]
        return render.DataGrid(counts.reset_index(names="Region"),
                               styles=[{"rows": rare_rows, "style": {"background-color": "#fde2e1"}}])

    @reactive.calc
    def z_max_values():
        """Z_max per region with the user's method, plus max/p99/p95 to help choose."""
        return z_max_table(normalized()["z"], input.z_max_method(), input.z_max_fixed(),
                           input.z_max_percentile(), cut_points()[0])

    @render.ui
    def zmax_status():
        if input.normalize() == 0:
            return ui.p("Click Normalize first.", class_="text-muted")
        last_cut = cut_points()[-1]
        too_low = z_max_values().index[z_max_values()["Z_max used"] <= last_cut].tolist()
        if not too_low:
            return ui.p(f"Z_max is above the last cut-point ({last_cut:g}) in every region.",
                        class_="text-success fw-bold")
        return ui.p(f"Z_max must be above the last cut-point ({last_cut:g}). Too low in: {', '.join(too_low)}",
                    class_="text-danger fw-bold")

    @render.data_frame
    def zmax_table():
        table = z_max_values().reset_index(names="Region")
        table["people with z >= min"] = table["people with z >= min"].astype(int)   # 130, not 130.0
        return table

    @render.download_link(filename="tau_typer_sample_data.csv")     # pairs with ui.download_link
    def download_sample():
        return str(SAMPLE_DATA)


app = App(app_ui, server)
