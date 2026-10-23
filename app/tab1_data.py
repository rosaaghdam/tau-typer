"""Tab 1 - Data & QC: upload the CSV, match the columns, check the data, optional inclusion rule."""
import numpy as np
import pandas as pd
from shiny import reactive, render, req, ui
from shiny.types import SafeException, SilentException

from src.figures import (OTHER_COLORS, colour_list, colour_map, group_colors, override_colors, plot_category_bars,
                         plot_correlations, plot_covariate_boxplots, plot_region_eda, plot_region_histograms)
from src.qc import (apply_inclusion_rule, covariate_summary, demographics_table, guess_columns,
                    number_and_category_columns, qc_summary, region_trend_tests)
from common import (CONFIG, DATA_HELP, ROLE_LABELS, RULE_ROLES, add_figure, add_table, as_picture, figure_card,
                    figure_tools, step_status, step_title, table_tools)

GROUP_COLOURS = "Colours (hex codes, in legend order; empty = automatic)"

panel = ui.nav_panel(step_title(1, "Data & QC", "step1_status"),
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
                     ui.nav_panel("Region distributions", figure_card(
                         "Explore the regional values", DATA_HELP["regions"],
                         ui.layout_columns(
                             ui.input_select("eda_x", "Plot the regions against", choices={"": "(nothing)"}),
                             ui.input_select("eda_split", "Split by (colour)", choices={"": "(no split)"}),
                             col_widths=[4, 4]),
                         ui.output_plot("region_hist", height="560px"),
                         figure_tools("eda", "", in_report=False, colors_label=GROUP_COLOURS),
                         ui.output_ui("eda_tests_view"))),
                     ui.nav_panel("Demographics",
                                  ui.p("Everyone, and per diagnosis group.", class_="text-muted small"),
                                  ui.output_data_frame("demographics"),
                                  table_tools("demographics"),
                                  ui.output_ui("covariates_view")))))


def server(input, output, session, shared):
    """Outputs and reactive steps of this tab. shared = what earlier tabs made (and what this tab
    adds for the later ones)."""
    restore = shared["restore"]
    report_items = shared["report_items"]

    @reactive.effect
    @reactive.event(input.data_file)
    def _new_upload():
        restore.set(None)                                 # a new file replaces a loaded run's data

    def read_upload():
        """The uploaded CSV as a table, or None."""
        file = input.data_file()
        if not file:
            return None
        limit = CONFIG["runs"]["max_upload_csv_mb"]
        if file[0]["size"] > limit * 1e6:
            raise SafeException(f"The file is too big ({file[0]['size'] / 1e6:.0f} MB; the limit is {limit} MB, "
                                "set in config.yaml).")
        return pd.read_csv(file[0]["datapath"])

    @reactive.calc
    def old_run_from_file():
        """A loaded run made before the app saved its input data (no regional values), when the file
        uploaded here has the run's ID and region columns: the file's rows of the run's people. Else None."""
        saved = restore()
        if saved is None or saved["complete"]:
            return None
        data, id_col = read_upload(), saved["columns"].get("id")
        if data is None or not id_col or id_col not in data or not set(saved["regions"]) <= set(data.columns):
            return None
        people = set(saved["data"][id_col].astype(str))
        rows = data[data[id_col].astype(str).isin(people)]
        return rows if len(rows) else None

    @reactive.calc
    def uploaded_data():
        """The uploaded CSV as a table (re-read automatically when a new file is uploaded), or the data
        of a run loaded in tab 3."""
        if restore() is not None:
            from_file = old_run_from_file()
            return restore()["data"] if from_file is None else from_file
        return read_upload()

    @render.ui
    def column_matching():
        """Dropdowns to say which column is what, pre-filled with guess_columns()."""
        data = uploaded_data()
        if data is None:
            return ui.p("Upload a file to match its columns.", class_="text-muted small")
        columns = list(data.columns)
        guess = guess_columns(columns)
        if restore() is not None:                         # a loaded run: its own column choices
            saved = restore()["columns"]
            guess = {**{role: saved.get(role) for role in ROLE_LABELS},
                     "regions": [r for r in restore()["regions"] if r in columns]}
        choices = {"": "(not in file)", **{col: col for col in columns}}
        selects = [ui.input_select(f"col_{role}", f"{label} (optional)", choices=choices, selected=guess[role] or "")
                   for role, label in ROLE_LABELS.items()]
        regions = ui.input_selectize("col_regions", "Region columns (the SuStaIn features)",
                                     choices=columns, selected=guess["regions"], multiple=True)
        if restore() is not None and not guess["regions"]:          # an older run without its regional values
            regions = ui.TagList(ui.p("Region columns", class_="fw-bold small mb-1"),
                                 ui.p(f"This run used {', '.join(restore()['regions'])}, but their values were not "
                                      "saved with it. Upload the same file above to see them here.",
                                      class_="text-muted small"),
                                 ui.div(regions, class_="d-none"))          # kept (empty) for the other steps
        return ui.TagList(ui.hr(), ui.h6("Match the columns"), *selects, regions, ui.hr(),
                          ui.output_ui("inclusion_rule_ui"))

    def rule_ticked():
        """The inclusion-rule box; False until it exists (it is drawn when tab 1 is first opened)."""
        try:
            return bool(input.apply_rule())
        except SilentException:
            return False

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
                                                   "(CN/SCD older than 40, or MCI/AD amyloid-positive)", value=False),
                          ui.output_ui("rule_note"))

    @render.ui
    def rule_note():
        """Right under the checkbox: how many people the rule keeps."""
        req(input.apply_rule())
        kept, total = len(analysis_data()), len(uploaded_data())
        return ui.p(f"Inclusion rule: {kept} of {total} people kept ({100 * kept / total:.0f}%).",
                    class_="fw-bold small", style="color: var(--tt-accent);")

    @reactive.calc
    def chosen_columns():
        """The user's dropdown choices: role -> column ("" = not in file). Before tab 1 has been opened
        after loading a run, the boxes do not exist yet: then the run's own choices are used."""
        try:
            return {role: input[f"col_{role}"]() for role in ROLE_LABELS}
        except SilentException:
            req(restore() is not None)
            return {role: restore()["columns"].get(role) or "" for role in ROLE_LABELS}

    @reactive.calc
    def included():
        """(people used from here on, reasons table or None). The rule runs only if the box is ticked."""
        data = uploaded_data()
        req(data is not None)
        if missing_roles(RULE_ROLES) or not rule_ticked():
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

    def make_demographics():
        table = demographics_table(analysis_data(), chosen_columns())
        group_col, groups = diagnosis_groups()
        if group_col:                                       # same order as the figures: CN, SCD, MCI, AD
            table = table.set_index("Group").reindex(["All"] + groups).reset_index()
        return table

    @render.data_frame
    def demographics():
        table = make_demographics()
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
            table_tools("covariates"),
            figure_card("Number columns by diagnosis", DATA_HELP["covariates"],
                        ui.output_plot("covariate_boxplots", height=plot_height(len(numbers))),
                        figure_tools("boxplots", "", in_report=False, colors_label=GROUP_COLOURS)),
            figure_card("Category columns by diagnosis", DATA_HELP["covariates"],
                        ui.output_plot("category_bars", height=plot_height(len(categories))),
                        figure_tools("bars", "", in_report=False, colors_label=GROUP_COLOURS)),
            figure_card("Correlations between number columns (and the mean of the regions)", DATA_HELP["correlations"],
                        ui.output_image("correlation_plot", height="auto"),
                        figure_tools("correlations", "RdBu", in_report=False,
                                     colors_label="Colours: a map name (RdBu, PuOr, ...) or hex codes from -1 to +1")))

    def make_covariate_table():
        group_col, groups = diagnosis_groups()
        columns = [c for c in other_columns() if c != group_col]
        return covariate_summary(analysis_data(), columns, group_col, groups)

    @render.data_frame
    def covariate_table():
        group_col, groups = diagnosis_groups()
        table = make_covariate_table()
        colors = group_colors(groups) if group_col else {}
        styles = [{"cols": [list(table.columns).index(g)], "style": {"color": colors[g], "font-weight": "bold"}}
                  for g in colors]
        return render.DataGrid(table, width="100%", styles=styles)

    def diagnosis_colors(typed_text):
        """{diagnosis: colour} with the user's typed colours first (or {"All": colour} without diagnosis)."""
        group_col, groups = diagnosis_groups()
        return override_colors(group_colors(groups) if group_col else {"All": "#11A0D9"}, colour_list(typed_text))

    def make_boxplots():
        group_col, groups = diagnosis_groups()
        numbers, _ = number_and_category_columns(analysis_data(), [c for c in other_columns() if c != group_col])
        req(numbers)
        return plot_covariate_boxplots(analysis_data(), numbers, group_col, groups,
                                       colors=diagnosis_colors(input.boxplots_colors()))

    @render.plot
    def covariate_boxplots():
        return make_boxplots()

    def make_bars():
        group_col, groups = diagnosis_groups()
        _, categories = number_and_category_columns(analysis_data(), [c for c in other_columns() if c != group_col])
        req(categories)
        return plot_category_bars(analysis_data(), categories, group_col, groups,
                                  colors=diagnosis_colors(input.bars_colors()))

    @render.plot
    def category_bars():
        return make_bars()

    @render.image(delete_file=True)
    def correlation_plot():
        return as_picture(make_correlations(), width="70%")

    def make_correlations():
        numbers, _ = number_and_category_columns(analysis_data(), other_columns())
        regions = list(input.col_regions() or [])
        data = analysis_data().copy()
        if regions:
            data["mean of regions"] = data[regions].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            numbers = numbers + ["mean of regions"]
        req(len(numbers) > 1)
        return plot_correlations(data, numbers, cmap=colour_map(input.correlations_colors(), "RdBu"))

    def make_region_figure():
        req(input.col_regions())                         # wait for at least 1 region
        x, groups, colors = eda_choice()
        typed = colour_list(input.eda_colors())
        if x is None and groups is None:
            return plot_region_histograms(analysis_data(), input.col_regions(), color=(typed or ["#4C72B0"])[0])
        return plot_region_eda(analysis_data(), list(input.col_regions()), x, groups, colors,
                               single_color=(typed or ["#52606D"])[0])

    @render.plot
    def region_hist():
        return make_region_figure()

    @reactive.effect
    def _eda_choices():
        """'Plot against' and 'Split by' offer the other columns of the file (age, sex, MMSE, ...)."""
        columns = other_columns()
        with reactive.isolate():
            keep_x, keep_split = input.eda_x(), input.eda_split()
        ui.update_select("eda_x", choices={"": "(nothing)", **{c: c for c in columns}},
                         selected=keep_x if keep_x in columns else "")
        ui.update_select("eda_split", choices={"": "(no split)", **{c: c for c in columns}},
                         selected=keep_split if keep_split in columns else "")

    def eda_choice():
        """(x column or None, group of each person or None, {group: colour}). A number column used to split
        is cut into thirds (low / middle / high); the diagnosis keeps its own colours."""
        data = analysis_data()
        x = input.eda_x() or None
        split = input.eda_split() or None
        if x is not None and x not in data.columns or split is not None and split not in data.columns:
            return None, None, None
        return (x, *split_groups(data, split, colour_list(input.eda_colors())))

    def split_groups(data, split, typed=()):
        """(group of each person, {group: colour}) for 'Split by', or (None, None) for no split. A number column
        is cut into thirds; the diagnosis keeps its own colours. typed: the user's colours, used first.
        Used by tab 1 (values) and tab 2 (z-scores)."""
        if split is None or split not in data.columns:
            return None, None
        numbers, _ = number_and_category_columns(data, [split])
        if numbers:
            groups, names = number_groups(pd.to_numeric(data[split], errors="coerce"))
        else:
            groups = data[split].astype(str).where(data[split].notna())
            names = sorted(groups.dropna().unique())
        if split == chosen_columns().get("diagnosis"):
            names = diagnosis_groups()[1]
            colors = group_colors(names)
        else:
            colors = {g: OTHER_COLORS[i % len(OTHER_COLORS)] for i, g in enumerate(names)}
        return groups, override_colors(colors, list(typed))

    def number_groups(values):
        """A number column cut into about equal thirds, named by their value range (e.g. "18-27", ">27-30").
        Fewer groups when many people share a value (e.g. MMSE = 30), so no group is empty."""
        bins, edges = pd.qcut(values, 3, retbins=True, duplicates="drop")
        edges[0] = values.min()                            # qcut moves the lowest edge a tiny bit down
        names = [f"{'>' if i else ''}{low:.3g}-{high:.3g}"      # e.g. "18-27", ">27-30" (27 is in the first)
                 for i, (low, high) in enumerate(zip(edges[:-1], edges[1:]))]
        groups = bins.cat.rename_categories(names).astype(str).where(bins.notna())
        return groups, names

    @render.ui
    def eda_tests_view():
        x, groups, _ = eda_choice()
        if x is None:
            return None
        return tests_view(x, input.eda_split() if groups is not None else None, "eda_tests")

    def tests_view(x, split, output_id, value="region value"):
        """Heading + how to read the per-region tests + the table (output_id)."""
        return ui.TagList(
            ui.h6("Tests per region", class_="mt-3"),
            ui.p(f"Linear model: {value} ~ {x}" + (f" x {split}" if split else "") +
                 f". 'p: x' = does it change with {x}" + (
                     "; 'p: group' = do the groups differ; 'p: x by group' = does the trend differ between the "
                     "groups (interaction)" if split else "") +
                 ". p-values are not corrected for testing many regions.", class_="text-muted small"),
            ui.output_data_frame(output_id),
            table_tools(output_id))

    def tests_grid(table):
        """The per-region tests as a table; p below 0.05 in bold."""
        shown = table.copy()
        for col in shown.columns:
            shown[col] = shown[col].map(lambda v: f"{v:.3g}")
        bold = [{"rows": [i for i, v in enumerate(table[col]) if v < 0.05], "cols": [j + 1],
                 "style": {"font-weight": "bold"}} for j, col in enumerate(table.columns) if col.startswith("p:")]
        return render.DataGrid(shown.reset_index(names="region"), styles=bold, width="100%")

    def make_eda_tests():
        x, groups, _ = eda_choice()
        req(x is not None)
        return region_trend_tests(analysis_data(), list(input.col_regions()), x, groups)

    @render.data_frame
    def eda_tests():
        return tests_grid(make_eda_tests())

    @render.ui
    def data_status():
        data = uploaded_data()
        if data is None:
            return ui.p("No data uploaded yet.")
        text = f"{data.shape[0]} rows and {data.shape[1]} columns. First 20 rows:"
        if restore() is None:
            return ui.p(text)
        if restore()["complete"]:
            note = f"The data of run {restore()['run']} (the people that run used)."
        elif old_run_from_file() is not None:
            note = (f"Run {restore()['run']} did not save its regional values, so they are taken from the file "
                    "uploaded here (the rows of the run's people, matched by ID).")
        else:
            note = (f"Run {restore()['run']} was made before the app saved the input data, so its regional values "
                    "are missing here: only the other columns are shown. To see them, upload the same CSV file "
                    "(sidebar) and load the run again in tab 3 - or just upload the file to start fresh.")
        return ui.TagList(ui.div(note, class_="alert alert-info py-2"), ui.p(text))

    @render.data_frame
    def data_head():
        data = uploaded_data()
        req(data is not None)            # show nothing until a file is uploaded
        return data.head(20)

    @render.ui
    def step1_status():
        if uploaded_data() is None:
            return step_status("Upload a CSV", "attention")
        if restore() is not None:
            return step_status(f"{len(analysis_data())} people (from the run)", "done")
        rule = " (rule applied)" if included()[1] is not None else ""
        return step_status(f"{len(analysis_data())} people{rule}", "done")

    # Save and "Include in report" for the figures and tables of this tab
    add_table(report_items, "demographics", "Demographics", lambda: make_demographics().set_index("Group"))
    add_figure(report_items, "eda", "Regional values", make_region_figure)
    add_table(report_items, "eda_tests", "Regional values: tests per region", make_eda_tests)
    add_table(report_items, "covariates", "Other columns per diagnosis", make_covariate_table)
    add_figure(report_items, "boxplots", "Number columns by diagnosis", make_boxplots)
    add_figure(report_items, "bars", "Category columns by diagnosis", make_bars)
    add_figure(report_items, "correlations", "Correlations between number columns", make_correlations)

    # what the later tabs use
    shared.update(analysis_data=analysis_data, chosen_columns=chosen_columns, missing_roles=missing_roles,
                  other_columns=other_columns, split_groups=split_groups, tests_grid=tests_grid,
                  tests_view=tests_view, uploaded_data=uploaded_data)
