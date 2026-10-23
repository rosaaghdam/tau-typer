"""Tab 1 - Data & QC: upload the CSV, match the columns, check the data, optional inclusion rule."""
import numpy as np
import pandas as pd
from shiny import reactive, render, req, ui
from shiny.types import SafeException

from src.figures import (group_colors, plot_category_bars, plot_correlations, plot_covariate_boxplots,
                         plot_region_histograms)
from src.qc import (apply_inclusion_rule, covariate_summary, demographics_table, guess_columns,
                    number_and_category_columns, qc_summary)
from common import CONFIG, RULE_ROLES, ROLE_LABELS

panel = ui.nav_panel("1. Data & QC",
             ui.layout_sidebar(
                 ui.sidebar(
                     ui.input_file("data_file", "Upload tau-PET data (CSV)", accept=[".csv"]),
                     ui.download_link("download_sample", "Sample Data"),
                     ui.p("Download simulated toy data (500 people, 10 regions, 10 clinical variables) to see "
                          "the expected format.",
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
                                  ui.output_data_frame("demographics"),
                                  ui.output_ui("covariates_view")))))


def server(input, output, session, shared):
    """Outputs and reactive steps of this tab. shared = what earlier tabs made (and what this tab
    adds for the later ones)."""
    @reactive.calc
    def uploaded_data():
        """The uploaded CSV as a table (re-read automatically when a new file is uploaded)."""
        file = input.data_file()
        if not file:
            return None
        limit = CONFIG["runs"]["max_upload_csv_mb"]
        if file[0]["size"] > limit * 1e6:
            raise SafeException(f"The file is too big ({file[0]['size'] / 1e6:.0f} MB; the limit is {limit} MB, "
                                "set in config.yaml).")
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
        selects = [ui.input_select(f"col_{role}", f"{label} (optional)", choices=choices, selected=guess[role] or "")
                   for role, label in ROLE_LABELS.items()]
        regions = ui.input_selectize("col_regions", "Region columns (the SuStaIn features)",
                                     choices=columns, selected=guess["regions"], multiple=True)
        return ui.TagList(ui.hr(), ui.h6("Match the columns"), *selects, regions, ui.hr(),
                          ui.output_ui("inclusion_rule_ui"))

    def missing_roles(roles):
        """Names of the needed columns that are set to "(not in file)" in tab 1."""
        columns = chosen_columns()
        return [ROLE_LABELS[role] for role in roles if not columns.get(role)]

    @render.ui
    def inclusion_rule_ui():
        """The inclusion-rule checkbox - only when Age, Diagnosis and Amyloid columns are chosen."""
        missing = missing_roles(RULE_ROLES)
        title = ui.h6("Filtering rule")
        if missing:
            return ui.TagList(title, ui.p(f"Inclusion rule not available: it needs {', '.join(missing)}.",
                                          class_="text-muted small"))
        return ui.TagList(title, ui.input_checkbox("apply_rule", "Apply the inclusion rule "
                                                   "(CN/SCD older than 40, or MCI/AD amyloid-positive)", value=False))

    @reactive.calc
    def chosen_columns():
        """The user's dropdown choices: role -> column ("" = not in file)."""
        return {role: input[f"col_{role}"]() for role in ROLE_LABELS}

    @reactive.calc
    def included():
        """(people used from here on, reasons table or None). The rule runs only if the box is ticked."""
        data = uploaded_data()
        req(data is not None)
        if missing_roles(RULE_ROLES) or not input.apply_rule():
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
        return render.DataGrid(table, width="100%",
                               styles=[{"rows": warning_rows, "style": {"background-color": "#fde2e1"}},
                                       {"cols": [0], "style": {"min-width": "260px"}}])   # wide "Check" column

    @render.data_frame
    def demographics():
        table = demographics_table(analysis_data(), chosen_columns())
        group_col, groups = diagnosis_groups()
        if group_col:                                       # same order as the figures: CN, SCD, MCI, AD
            table = table.set_index("Group").reindex(["All"] + groups).reset_index()
        colors = group_colors([g for g in table["Group"] if g != "All"])
        styles = [{"rows": [i], "cols": [0], "style": {"color": colors[g], "font-weight": "bold"}}
                  for i, g in enumerate(table["Group"]) if g in colors]
        return render.DataGrid(table, styles=styles)

    @reactive.calc
    def other_columns():
        """All columns that are not regions and not the ID: age, sex, MMSE, ..."""
        skip = set(input.col_regions() or []) | {chosen_columns().get("id")}
        return [col for col in analysis_data().columns if col not in skip]

    @reactive.calc
    def diagnosis_groups():
        """(diagnosis column, its values in a sensible order: CN, SCD, ..., MCI, AD) or (None, None)."""
        col = chosen_columns().get("diagnosis")
        if not col:
            return None, None
        present = list(analysis_data()[col].dropna().astype(str).unique())
        known = CONFIG["inclusion"]["cn_labels"] + CONFIG["inclusion"]["impaired_labels"]
        order = [g for g in known if g in present] + sorted(g for g in present if g not in known)
        return col, order

    def plot_height(n_panels, per_row=4, row_px=240):
        """Height of a figure with n_panels small plots, per_row in each row."""
        n_rows = max(1, int(np.ceil(n_panels / per_row)))
        return f"{n_rows * row_px}px"

    @render.ui
    def covariates_view():
        columns = [c for c in other_columns() if c != diagnosis_groups()[0]]
        if not columns:
            return None
        numbers, categories = number_and_category_columns(analysis_data(), columns)
        return ui.TagList(
            ui.h6("All other columns", class_="mt-4"),
            ui.p("Numbers: mean (SD). Categories: % of each value (the 4 most common).", class_="text-muted small"),
            ui.output_data_frame("covariate_table"),
            ui.h6("Number columns by diagnosis", class_="mt-4"),
            ui.output_plot("covariate_boxplots", height=plot_height(len(numbers))),
            ui.h6("Category columns by diagnosis", class_="mt-4"),
            ui.output_plot("category_bars", height=plot_height(len(categories))),
            ui.h6("Correlations between number columns (and the mean of the regions)", class_="mt-4"),
            ui.output_plot("correlation_plot", height=f"{35 * len(numbers) + 300}px", width="70%"))

    @render.data_frame
    def covariate_table():
        group_col, groups = diagnosis_groups()
        columns = [c for c in other_columns() if c != group_col]
        table = covariate_summary(analysis_data(), columns, group_col, groups)
        colors = group_colors(groups) if group_col else {}
        styles = [{"cols": [list(table.columns).index(g)], "style": {"color": colors[g], "font-weight": "bold"}}
                  for g in colors]
        return render.DataGrid(table, width="100%", styles=styles)

    @render.plot
    def covariate_boxplots():
        group_col, groups = diagnosis_groups()
        numbers, _ = number_and_category_columns(analysis_data(), [c for c in other_columns() if c != group_col])
        req(numbers)
        return plot_covariate_boxplots(analysis_data(), numbers, group_col, groups)

    @render.plot
    def category_bars():
        group_col, groups = diagnosis_groups()
        _, categories = number_and_category_columns(analysis_data(), [c for c in other_columns() if c != group_col])
        req(categories)
        return plot_category_bars(analysis_data(), categories, group_col, groups)

    @render.plot
    def correlation_plot():
        numbers, _ = number_and_category_columns(analysis_data(), other_columns())
        regions = list(input.col_regions() or [])
        data = analysis_data().copy()
        if regions:
            data["mean of regions"] = data[regions].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            numbers = numbers + ["mean of regions"]
        req(len(numbers) > 1)
        return plot_correlations(data, numbers)

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

    # what the later tabs use
    shared.update(analysis_data=analysis_data, chosen_columns=chosen_columns, missing_roles=missing_roles,
                  other_columns=other_columns, uploaded_data=uploaded_data)
