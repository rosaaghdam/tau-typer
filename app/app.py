"""Tau-typer app - tau-PET subtyping with SuStaIn (Vogel et al. 2021), step by step.

Run from the project folder (environment active):   shiny run app/app.py
This file is only the user interface; all calculations live in src/.
"""
import re
import secrets
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from shiny import App, reactive, render, req, ui
from shiny.types import SafeException

ROOT = Path(__file__).resolve().parent.parent          # project folder (one level above app/)
sys.path.insert(0, str(ROOT))                           # so Python can find src/
from src.qc import (apply_inclusion_rule, covariate_summary, demographics_table, guess_columns,
                    number_and_category_columns, qc_summary)
from src.app_runs import list_runs, prepare_run_folder, start_run, unzip_run, zip_run
from src.report import pysustain_info, report_zip
from src.results import (event_orders, load_sustain_result, mcmc_traces, model_likelihoods, pvd_order, size_order,
                         subject_table, subtype_summary)
from src.figures import (group_colors, plot_asymmetry, plot_category_bars, plot_correlations,
                         plot_covariate_boxplots, plot_gmm_fit, plot_model_choice, plot_person, plot_pvd,
                         plot_region_histograms, plot_regional_contrasts, plot_stage_progression, plot_stage_vs,
                         plot_stages, plot_subtype_patterns, plot_subtype_probabilities, plot_subtype_tree)
from src.sustain_model import (count_people_above, cuts_per_region, make_z_vals, parse_cut_points, parse_z_max,
                               sustain_inputs, z_max_table)
from src.subtype_analysis import asymmetry_index, compare_runs, group_table, regional_contrasts, stage_association
from src.zscore import (amyloid_negative_cn, covariate_adjustment, covariate_matrix, fit_gmm, reference_group_normal, zscore_regions,
                        zscore_with_reference)

NORM_METHODS = {"gmm": "GMM on everyone", "reference": "Reference group: amyloid-negative CN",
                "zscored": "The data is already normalized"}

CONFIG = yaml.safe_load(open(ROOT / "config.yaml"))    # all settings live in config.yaml

RUNS_FOLDER = ROOT / CONFIG["runs"]["folder"]          # one sub-folder per SuStaIn run
SAMPLE_DATA = ROOT / "data" / "toy" / "toy_tau_clinical.csv"   # simulated toy data + 10 clinical variables + SCD
FIG_HELP = {    # short help text above each tab-4 figure
    "pvd": "Positional variance diagram, drawn by pySuStaIn. Rows = regions, columns = stages. Dark = SuStaIn "
           "is sure the event happens at that stage, pale = unsure.",
    "patterns": "Top: how much tau each subtype has in each region - the average z-score of its tau+ people "
                "(number in each cell). Bottom: the same, split into stage groups (early stages on the left, late "
                "on the right), so you can see where tau spreads as the stage rises. n = number of people.",
    "probs": "How clearly each tau+ person belongs to one subtype. Each dot is a person, each "
             "corner a subtype: close to a corner = a clear member of that subtype; towards the middle = a mix "
             "of subtypes. The dashed box marks the middle; the legend gives the % of people inside and outside "
             "it.",
    "stages": "Left: how many tau+ people are at each SuStaIn stage, coloured by subtype; "
              "people at stage 0 (S0, no abnormal tau) are only counted in the legend. Middle: how sure SuStaIn is "
              "about each person's stage (SD of their stage probabilities) - low = sure, high = the stage could "
              "be one of several (often higher in the middle stages, where events overlap - the pale squares "
              "of the PVD); 'Show a person from' a subtype = that subtype in colour, the others grey; ringed dot = the person chosen above. Right: the same per subtype. "
              "Dashed red line = your limit; the people above it are listed below.",
    "choice": "Top: log-likelihood (rises with every extra subtype) and CVIC after cross-validation (lowest = "
              "best). Bottom: subtype tree - where people go when one more subtype is allowed.",
    "person": "Bars: this person's z-scores. Dots: average of people in the same subtype and stage group. "
              "Middle: subtype probabilities. Right: stage probabilities.",
}

AN_HELP = {     # short help text in tab 5
    "table1": "Numbers: mean (SD); categories: %. p (subtypes): Kruskal-Wallis or chi-square "
              "between the subtypes. Adjusted: linear model variable ~ subtype + stage + the covariates chosen "
              "in 'Adjust tests for'. * = this subtype differs from the other tau+ people (FDR p < 0.05).",
    "regions": "For each subtype and region: this subtype vs all other tau+ people, at the same "
               "stage (linear model z ~ subtype + stage + 'Adjust tests for'). Red = more tau, blue = less; "
               "* = FDR p < 0.05.",
    "stage": "Spearman correlation between SuStaIn stage and the chosen variable (tau+ people), "
             "overall and per subtype; one line per subtype in the figure.",
    "asymmetry": "Asymmetry = mean over left/right region pairs of (L - R) / (|L| + |R|); "
                 "above 0 = more tau on the left.",
    "compare": "How well do two runs agree on the same people (e.g. other settings, Z_max or normalization)? "
               "Cross-table of the groups, adjusted Rand index (1 = same subtypes) and stage correlation.",
}

RULE_ROLES = ("age", "diagnosis", "amyloid")      # columns the inclusion rule needs
REFERENCE_ROLES = ("diagnosis", "amyloid")       # columns the reference-group normalization needs
REFERENCE_GROUPS = {"cn_scd": "Amyloid-negative CN + SCD", "cn": "Amyloid-negative CN only"}
DISEASE_WORDS = ["mmse", "moca", "cdr", "adas", "cognit", "memory", "executive", "language", "visuo",
                 "tau", "amyloid", "abeta", "diagnos", "dx"]   # covariate names that look like disease measures
ROLE_LABELS = {"id": "Subject ID", "age": "Age", "sex": "Sex",
               "diagnosis": "Diagnosis", "amyloid": "Amyloid status"}

app_ui = ui.page_navbar(
    ui.nav_panel("1. Data & QC",
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
                                      ui.output_ui("covariates_view"))))),
    ui.nav_panel("2. Normalization",
                 ui.layout_sidebar(
                     ui.sidebar(
                         ui.input_radio_buttons("norm_method", "Where 'normal' comes from",
                                                choices=NORM_METHODS, selected="gmm"),
                         ui.output_ui("norm_method_note"),
                         ui.input_checkbox("adjust_cov", "Adjust for covariates first (W-score style)", value=False),
                         ui.panel_conditional(
                             "input.adjust_cov",
                             ui.input_selectize("adjust_candidates", "Covariates to test", choices=[], multiple=True),
                             ui.output_ui("adjust_warning"),
                             ui.input_checkbox("adjust_only_sig", "Remove only where significant", value=True),
                             ui.panel_conditional(
                                 "input.adjust_only_sig",
                                 ui.input_numeric("adjust_alpha", "p-value limit",
                                                  value=CONFIG["normalization"]["covariate_alpha"],
                                                  min=0.001, max=0.5, step=0.01))),
                         ui.panel_conditional(
                             "input.norm_method === 'gmm'",
                             ui.input_numeric("gmm_components", "Number of GMM components",
                                              value=CONFIG["pipeline"]["gmm_components"], min=2, max=5),
                             ui.p("Per region, a Gaussian mixture model splits the values into bell curves; "
                                  "the one with the lowest mean is 'normal'.", class_="text-muted small")),
                         ui.input_checkbox("per_region", "Different cut-points / Z_max per region", value=False),
                         ui.panel_conditional(                  # one set of values for all regions
                             "!input.per_region",
                             ui.input_text("cut_points", "Cut-points (the SuStaIn events)",
                                           value=", ".join(f"{z:g}" for z in CONFIG["sustain"]["z_events"])),
                             ui.input_radio_buttons("z_max_method", "Z_max (the z-score each region reaches at the very end)",
                                                    choices={"manual": "One value for all regions",
                                                             "percentile": "Percentile per region"},
                                                    selected=CONFIG["normalization"]["z_max_method"]),
                             ui.panel_conditional("input.z_max_method === 'manual'",
                                                  ui.input_text("z_max_manual", "Z_max",
                                                                value=f"{CONFIG['sustain']['z_max']:g}")),
                             ui.panel_conditional("input.z_max_method === 'percentile'",
                                                  ui.input_numeric("z_max_percentile", "Percentile (among people with z >= first cut-point)",
                                                                   value=CONFIG["normalization"]["z_max_percentile"], min=50, max=100)),
                             ),
                         ui.panel_conditional(                  # values per region: in the main area
                             "input.per_region",
                             ui.p("Set the cut-points and Z_max of each region in the 'Per-region values' tab "
                                  "(on the right).", class_="text-primary small")),
                         ui.input_action_button("normalize", "Normalize", class_="btn-primary"),
                         width=320, open="always"),
                     ui.navset_tab(
                         ui.nav_panel("Normal reference", ui.output_ui("norm_status"),
                                      ui.output_ui("slopes_view"),
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
                         ui.nav_panel("Per-region values", ui.output_ui("per_region_ui"),
                                      ui.panel_conditional("input.per_region",
                                                           ui.input_action_button("pr_reset", "Reset to sidebar values",
                                                                                  class_="mt-2"))),
                         ui.nav_panel("Z_max", ui.output_ui("zmax_status"),
                                      ui.output_data_frame("zmax_table")),
                         ui.nav_panel("SuStaIn input", ui.output_ui("sustain_input_summary"),
                                      ui.download_button("download_z", "Download z-scores (CSV)"),
                                      ui.download_button("download_reference", "Download normal reference (CSV)"),
                                      ui.p("Keep the normal reference: new people must be z-scored with "
                                           "these same numbers.", class_="text-muted small mt-2")),
                         id="norm_tabs"))),
    ui.nav_panel("3. Run SuStaIn",
                 ui.layout_sidebar(
                     ui.sidebar(
                         ui.input_radio_buttons("run_preset", "Preset", selected="paper",
                                                choices={"paper": "Full run (recommended)", "fast": "Fast test (rough results)"}),
                         ui.input_numeric("n_subtypes_max", "Max number of subtypes",
                                          value=CONFIG["sustain"]["n_subtypes_max"], min=1, max=8),
                         ui.input_numeric("n_startpoints", "Start points", value=CONFIG["sustain"]["n_startpoints"], min=1),
                         ui.input_numeric("n_mcmc", "MCMC iterations", value=CONFIG["sustain"]["n_iterations_mcmc"], min=100),
                         ui.input_numeric("run_seed", "Random seed", value=CONFIG["pipeline"]["seed"]),
                         ui.input_checkbox("use_parallel", "Parallel start points (faster on big computers)",
                                           value=CONFIG["sustain"]["use_parallel_startpoints"]),
                         ui.input_checkbox("do_cv", "Cross-validation (to choose the number of subtypes)"),
                         ui.panel_conditional("input.do_cv",
                                              ui.input_numeric("n_folds", "Number of folds",
                                                               value=CONFIG["cross_validation"]["n_folds"], min=2)),
                         ui.input_text("run_name", "Run name (new folder)",
                                       value=datetime.now().strftime("run_%Y-%m-%d_%H%M")),
                         ui.input_action_button("run_button", "Run SuStaIn", class_="btn-primary"),
                         ui.input_action_button("cancel_run", "Cancel run", class_="btn-outline-danger mt-2"),
                         width=320, open="always"),
                     ui.navset_tab(
                         ui.nav_panel("Run", ui.output_ui("run_settings_view"),
                                      ui.output_ui("run_checks_view"),
                                      ui.output_ui("run_progress_view")),
                         ui.nav_panel("Model choice", ui.output_ui("model_choice_status"),
                                      ui.output_data_frame("model_likelihood_table"),
                                      ui.output_plot("model_choice_plot", height="330px"),
                                      ui.output_image("cv_folds_plot", height="auto")),
                         ui.nav_panel("Quick result", ui.output_ui("quick_result_status"),
                                      ui.input_select("result_model", "Model to look at", choices=[]),
                                      ui.output_data_frame("subtype_summary_table"),
                                      ui.download_button("download_subjects", "Download subtype + stage per person (CSV)"),
                                      ui.output_ui("event_order_view")),
                         ui.nav_panel("Saved runs",
                                      ui.p("Every run has its own folder. Load a finished run to see it in "
                                           "Model choice and Quick result (and later in tabs 4-5). Stopped = "
                                           "the app was closed while it ran.", class_="text-muted small"),
                                      ui.output_data_frame("saved_runs_table"),
                                      ui.input_select("saved_run", "Run", choices=[]),
                                      ui.div(ui.input_action_button("load_run", "Load this run"),
                                             ui.download_button("download_run", "Download this run (ZIP)"),
                                             ui.input_action_button("delete_run", "Delete this run",
                                                                    class_="btn-outline-danger"),
                                             class_="d-flex gap-2 mb-3"),
                                      ui.input_file("upload_run", "Upload a saved run (ZIP)", accept=[".zip"]),
                                      ui.p("On a shared server your runs are deleted when you close the page: "
                                           "download a run to keep it, and upload it here later to see it again.",
                                           class_="text-muted small"))))),
    ui.nav_panel("4. Figures",
                 ui.layout_sidebar(
                     ui.sidebar(
                         ui.output_ui("fig_run_info"),
                         ui.input_select("fig_model", "Model (number of subtypes)", choices=[]),
                         ui.input_text("subtype_names", "Subtype names (optional)",
                                       placeholder="e.g. limbic, posterior, ..."),
                         ui.p("Comma-separated, Subtype 1 first. Subtype 1 = the subtype with the most "
                              "tau+ people, Subtype 2 the next, ... (the same numbers in every figure and table).",
                              class_="text-muted small"),
                         ui.input_numeric("fig_vmax", "Colour scale maximum (z)",
                                          value=CONFIG["figures"]["heatmap_vmax"], min=1),
                         width=320, open="always"),
                     ui.navset_tab(
                         ui.nav_panel("PVD", ui.p(FIG_HELP["pvd"], class_="text-muted small"),
                                      ui.output_image("fig_pvd", height="auto")),
                         ui.nav_panel("Tau patterns", ui.p(FIG_HELP["patterns"], class_="text-muted small"),
                                      ui.output_plot("fig_patterns", height="520px"),
                                      ui.output_plot("fig_progression", height="460px")),
                         ui.nav_panel("Subtype probabilities", ui.p(FIG_HELP["probs"], class_="text-muted small"),
                                      ui.output_plot("fig_probs", height="560px")),
                         ui.nav_panel("Stages", ui.p(FIG_HELP["stages"], class_="text-muted small"),
                                      ui.layout_columns(
                                          ui.input_numeric("sd_limit", "Uncertain stage: SD above (stages)",
                                                           value=CONFIG["results"]["stage_sd_limit"], min=0, step=0.1),
                                          ui.input_select("stages_subtype", "Show a person from", choices=[]),
                                          ui.input_selectize("stages_person", "Person (circled)", choices=[]),
                                          col_widths=[3, 3, 3]),
                                      ui.output_plot("fig_stages", height="440px"),
                                      ui.output_ui("uncertain_status"),
                                      ui.output_data_frame("uncertain_people")),
                         ui.nav_panel("Choosing k", ui.p(FIG_HELP["choice"], class_="text-muted small"),
                                      ui.output_plot("fig_choice", height="380px"),
                                      ui.output_plot("fig_tree", height="480px")),
                         ui.nav_panel("Person", ui.input_selectize("fig_person", "Person", choices=[]),
                                      ui.p(FIG_HELP["person"], class_="text-muted small"),
                                      ui.output_plot("fig_person_plot", height="420px"))))),
    ui.nav_panel("5. Subtype analysis",
                 ui.layout_sidebar(
                     ui.sidebar(
                         ui.output_ui("an_run_info"),
                         ui.input_select("an_model", "Model (number of subtypes)", choices=[]),
                         ui.input_selectize("an_variables", "Variables to compare (Table 1)", choices=[],
                                            multiple=True),
                         ui.input_select("an_stage_variable", "Variable against stage", choices=[]),
                         ui.input_selectize("an_adjust", "Adjust tests for (e.g. age, sex, education)",
                                            choices=[], multiple=True),
                         ui.p("Variables come from the other columns of your file (age, sex, MMSE, APOE4, ...). "
                              "Only tau+ people (stage > 0) are compared between subtypes.",
                              class_="text-muted small"),
                         width=320, open="always"),
                     ui.navset_tab(
                         ui.nav_panel("Table 1", ui.output_ui("an_status"), ui.output_data_frame("an_table1"),
                                      ui.p(AN_HELP["table1"], class_="text-muted small mt-2")),
                         ui.nav_panel("Regional differences", ui.p(AN_HELP["regions"], class_="text-muted small"),
                                      ui.output_plot("an_regions_plot", height="520px")),
                         ui.nav_panel("Stage vs variable", ui.p(AN_HELP["stage"], class_="text-muted small"),
                                      ui.output_data_frame("an_stage_table"),
                                      ui.output_plot("an_stage_plot", height="460px")),
                         ui.nav_panel("Left-right asymmetry", ui.p(AN_HELP["asymmetry"], class_="text-muted small"),
                                      ui.output_ui("an_asymmetry_status"),
                                      ui.output_plot("an_asymmetry_plot", height="440px")),
                         ui.nav_panel("Compare runs", ui.p(AN_HELP["compare"], class_="text-muted small"),
                                      ui.layout_columns(
                                          ui.input_select("an_other_run", "Other finished run", choices=[]),
                                          ui.input_select("an_other_k", "Its model (number of subtypes)", choices=[]),
                                          ui.input_file("compare_upload", "... or upload a run to compare (ZIP)",
                                                        accept=[".zip"]),
                                          col_widths=[4, 3, 5]),
                                      ui.output_ui("an_compare_summary"),
                                      ui.output_data_frame("an_compare_table")),
                         ui.nav_panel("Download",
                                      ui.p("One row per person: subtype, stage, probabilities and the other "
                                           "columns of your file.", class_="text-muted small"),
                                      ui.download_button("an_download", "Download results per person (CSV)"),
                                      ui.p("Report: every figure of tabs 4-5 (PNG 300 dpi + PDF), every table "
                                           "(CSV) and the run's settings, for the model chosen in the sidebar.",
                                           class_="text-muted small mt-3"),
                                      ui.download_button("an_report", "Download report (ZIP)"))))),
    ui.nav_panel("About", ui.output_ui("about")),
    title="Tau-typer",
)


def server(input, output, session):
    # Where this visit's runs are saved. On a shared server (config runs.private_per_visit: true) every
    # visit gets its own folder, sees only its own runs, and the folder is deleted when the visit ends.
    private = CONFIG["runs"].get("private_per_visit", False)
    runs_folder = RUNS_FOLDER / f"visit_{secrets.token_hex(8)}" if private else RUNS_FOLDER

    def _end_visit():
        """Private visits only: stop a running SuStaIn and delete this visit's folder."""
        with reactive.isolate():
            j = job()
        if j is not None and j["process"].poll() is None:
            j["process"].terminate()
        shutil.rmtree(runs_folder, ignore_errors=True)

    if private:
        session.on_ended(_end_visit)

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

    @reactive.effect
    def _adjust_candidate_choices():
        """'Covariates to test' offers every non-region, non-ID column except diagnosis and amyloid
        (they define the reference group and are the disease itself); age and sex are pre-selected."""
        roles = chosen_columns()
        columns = [c for c in other_columns() if c not in (roles.get("diagnosis"), roles.get("amyloid"))]
        default = [roles[r] for r in ("age", "sex") if roles.get(r) in columns]
        ui.update_selectize("adjust_candidates", choices=columns, selected=default)

    @render.ui
    def adjust_warning():
        """Red note if a chosen covariate looks like a disease measure (MMSE, CDR, p-tau, ...)."""
        risky = [c for c in (input.adjust_candidates() or []) if any(w in c.lower() for w in DISEASE_WORDS)]
        if risky:
            verb = "looks like a disease measure" if len(risky) == 1 else "look like disease measures"
            return ui.p(f"Careful: {', '.join(risky)} {verb}. Removing it would remove the disease signal "
                        "SuStaIn should find.", class_="text-danger small fw-bold")

    def reference_labels():
        """Diagnosis labels of the healthy reference group (always used together with amyloid = 0)."""
        labels = list(CONFIG["normalization"]["reference_labels_cn"])
        if CONFIG["normalization"]["reference_group"] == "cn_scd":
            labels += CONFIG["normalization"]["reference_labels_scd"]
        return labels

    @reactive.effect
    def _norm_method_choices():
        """Tab 2 offers 'Reference group' only when Diagnosis and Amyloid columns are chosen."""
        missing = missing_roles(REFERENCE_ROLES)
        choices = {key: label for key, label in NORM_METHODS.items() if not (missing and key == "reference")}
        with reactive.isolate():
            current = input.norm_method()
        ui.update_radio_buttons("norm_method", choices=choices, selected=current if current in choices else "gmm")

    @render.ui
    def norm_method_note():
        missing = missing_roles(REFERENCE_ROLES)
        if missing:
            return ui.p(f"'Reference group' needs {', '.join(missing)} (tab 1).", class_="text-muted small")

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
        complete, adjustment = adjust_for_covariates(data, complete, regions, method)
        if method == "gmm":
            z, reference = zscore_regions(complete, regions, n_components, CONFIG["pipeline"]["seed"])
        elif method == "reference":
            columns = chosen_columns()
            if not (columns.get("diagnosis") and columns.get("amyloid")):
                raise SafeException("The reference group needs the diagnosis and amyloid columns (tab 1).")
            in_reference = amyloid_negative_cn(data.loc[complete.index], columns["diagnosis"],
                                               columns["amyloid"], reference_labels())
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
                "n_left_out": len(values) - len(complete), "inputs": norm_inputs(),
                "adjustment": adjustment}

    def adjust_for_covariates(data, complete, regions, method):
        """If the box is ticked: test the chosen covariates in the healthy reference group and remove
        them (only the significant ones, if that box is ticked) from everyone's SUVR.
        Returns (values, None) or (adjusted values, dict with p-values, slopes and details)."""
        names = list(input.adjust_candidates() or [])
        if not input.adjust_cov() or method == "zscored":
            return complete, None
        if not names:
            raise SafeException("Choose at least one covariate to test (or untick 'Adjust for covariates').")
        columns = chosen_columns()
        if not (columns.get("diagnosis") and columns.get("amyloid")):
            raise SafeException("The covariate adjustment needs the diagnosis and amyloid columns (tab 1).")
        covariates = covariate_matrix(data.loc[complete.index], names).dropna()   # known values only
        complete = complete.loc[covariates.index]
        in_reference = amyloid_negative_cn(data.loc[complete.index], columns["diagnosis"], columns["amyloid"],
                                           reference_labels())
        if in_reference.sum() < 10:
            raise SafeException("The covariate adjustment needs at least 10 people in the healthy reference group.")
        alpha = input.adjust_alpha() or CONFIG["normalization"]["covariate_alpha"]
        try:
            adjusted, p_values, estimates, slopes = covariate_adjustment(complete, regions, covariates,
                                                                         in_reference, input.adjust_only_sig(), alpha)
        except ValueError as error:
            raise SafeException(str(error))
        skipped = [c for c in covariates.columns if c not in p_values.columns]
        return adjusted, {"p_values": p_values, "estimates": estimates, "slopes": slopes,
                          "n_reference": int(in_reference.sum()),
                          "skipped": skipped,
                          "only_significant": input.adjust_only_sig(), "alpha": alpha,
                          "group": REFERENCE_GROUPS[CONFIG["normalization"]["reference_group"]]}

    def norm_inputs():
        """Everything Normalize depends on: method, GMM components, regions and the people (a hash)."""
        method = input.norm_method()
        return (method, input.gmm_components() if method == "gmm" else None, tuple(input.col_regions() or []),
                input.adjust_cov(), tuple(input.adjust_candidates() or []), input.adjust_only_sig(),
                input.adjust_alpha(),
                int(pd.util.hash_pandas_object(analysis_data(), index=True).sum()))

    @reactive.calc
    def norm_outdated():
        """True if a setting or the data changed after the last click on Normalize."""
        if input.normalize() == 0:
            return False
        try:
            return normalized()["inputs"] != norm_inputs()
        except Exception:                                   # e.g. the data was removed
            return True

    def outdated_warning():
        return ui.p("Settings or data changed after Normalize - click Normalize again to update the z-scores.",
                    class_="text-danger fw-bold")

    @render.ui
    def norm_status():
        if input.normalize() == 0:
            return ui.p("Choose the settings and click Normalize.", class_="text-muted")
        result = normalized()
        lines = [outdated_warning()] if norm_outdated() else []
        lines += [ui.p(f"{NORM_METHODS[result['method']]}: {len(result['z'])} people in {result['z'].shape[1]} "
                      f"regions ({result['n_left_out']} left out because of missing region"
                       f"{' or covariate' if result['adjustment'] is not None else ''} values).",
                      class_="fw-bold")]
        if result["method"] == "reference":
            lines.append(ui.p(f"Reference group: {result['n_reference']} people "
                              f"({REFERENCE_GROUPS[CONFIG['normalization']['reference_group']]})."))
        adjustment = result["adjustment"]
        if adjustment is not None:
            used = (adjustment["slopes"] != 0).sum()
            lines.append(ui.p(f"Covariates tested in {adjustment['n_reference']} people "
                              f"({adjustment['group']}); removed where "
                              f"{'p < ' + format(adjustment['alpha'], 'g') if adjustment['only_significant'] else 'chosen'}"
                              f": {', '.join(f'{c} in {n} region(s)' for c, n in used.items())}. "
                              "Details: table at the bottom.", class_="text-primary"))
            if adjustment["skipped"]:
                lines.append(ui.p(f"Skipped (the same for everyone in the reference group): "
                                  f"{', '.join(adjustment['skipped'])}.", class_="text-muted small"))
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
                            reference.loc["normal_sd", region], region_cuts()[region])

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

    @reactive.effect
    @reactive.event(input.per_region, ignore_init=True)
    def _open_per_region_tab():
        """Ticking 'Different cut-points / Z_max per region' opens the tab with the region rows."""
        if input.per_region():
            ui.update_navset("norm_tabs", selected="Per-region values")

    @render.ui
    def per_region_ui():
        """One row per region with its own cut-points and Z_max, pre-filled with the sidebar values.
        Built again (back to the sidebar values) when the regions change or Reset is clicked."""
        input.pr_reset()
        regions = list(input.col_regions() or [])
        if not input.per_region():
            return ui.p("Tick 'Different values per region' in the sidebar to set values for each region.",
                        class_="text-muted")
        if not regions:
            return ui.p("Choose the region columns in tab 1 first.", class_="text-muted")
        with reactive.isolate():
            cuts_text = input.cut_points()
            try:
                z_max = sidebar_z_max_values()["Z_max used"]
            except Exception:                                  # not normalized yet
                z_max = pd.Series(CONFIG["sustain"]["z_max"], index=regions)
        header = ui.row(ui.column(3, ui.strong("Region")), ui.column(5, ui.strong("Cut-points")),
                        ui.column(4, ui.strong("Z_max")))
        rows = [header]
        for i, region in enumerate(regions):
            rows.append(ui.row(
                ui.column(3, ui.p(region, class_="mt-2")),
                ui.column(5, ui.input_text(f"pr_cuts_{i}", None, value=cuts_text)),
                ui.column(4, ui.input_numeric(f"pr_zmax_{i}", None, value=round(float(z_max.get(region, 15)), 1)))))
        return ui.TagList(
            ui.p("Each region can have its own cut-points (e.g. 2, 5 or 2, 4, 6, 8) and Z_max. The checks in "
                 "the Events, Z_max and SuStaIn input tabs use these values.", class_="text-muted small"),
            *rows)

    @reactive.calc
    def region_cuts():
        """{region: cut-points} - the sidebar cut-points for every region, or the per-region values."""
        regions = list(normalized()["z"].columns)
        if not input.per_region():
            return cuts_per_region(regions, cut_points())
        cuts = {}
        for i, region in enumerate(regions):
            try:
                cuts[region] = parse_cut_points(input[f"pr_cuts_{i}"]())
            except ValueError as error:
                raise SafeException(f"{region}: {error}")
        return cuts

    @reactive.calc
    def event_counts():
        """People reaching each cut-point (rows = regions, columns = cut-points)."""
        z = normalized()["z"]
        return count_people_above(z, list(z.columns), region_cuts())

    @render.ui
    def events_status():
        if input.normalize() == 0:
            return ui.p("Click Normalize first.", class_="text-muted")
        minimum = CONFIG["normalization"]["min_people_per_event"]
        n_rare = int((event_counts() < minimum).sum().sum())          # empty cells are not counted
        if n_rare == 0:
            return ui.p(f"All events are reached by at least {minimum} people.", class_="text-success fw-bold")
        return ui.p(f"{n_rare} event(s) are reached by fewer than {minimum} people (rows in red).",
                    class_="text-danger fw-bold")

    @render.data_frame
    def events_table():
        counts = event_counts()
        minimum = CONFIG["normalization"]["min_people_per_event"]
        rare_rows = [i for i, region in enumerate(counts.index) if (counts.loc[region] < minimum).any()]
        if input.per_region():                                   # show each region's own cut-points
            counts.insert(0, "Cut-points", [", ".join(f"{c:g}" for c in region_cuts()[r]) for r in counts.index])
        return render.DataGrid(counts.reset_index(names="Region"),
                               styles=[{"rows": rare_rows, "style": {"background-color": "#fde2e1"}}])

    @reactive.calc
    def sidebar_z_max_values():
        """Z_max per region with the sidebar method, plus max/p99/p95 to help choose."""
        z = normalized()["z"]
        manual = None
        if input.z_max_method() == "manual":
            try:
                manual = parse_z_max(input.z_max_manual(), list(z.columns))
            except ValueError as error:
                raise SafeException(str(error))
        first_cuts = {region: cuts[0] for region, cuts in region_cuts().items()}
        return z_max_table(z, input.z_max_method(), manual, input.z_max_percentile(), first_cuts)

    @reactive.calc
    def z_max_values():
        """The Z_max used: the sidebar method, or the numbers typed per region."""
        if not input.per_region():
            return sidebar_z_max_values()
        z = normalized()["z"]
        manual = {}
        for i, region in enumerate(z.columns):
            value = input[f"pr_zmax_{i}"]()
            if value is None:
                raise SafeException(f"{region}: type a Z_max in the 'Per-region values' tab.")
            manual[region] = float(value)
        first_cuts = {region: cuts[0] for region, cuts in region_cuts().items()}
        return z_max_table(z, "manual", manual, input.z_max_percentile(), first_cuts)

    @render.ui
    def zmax_status():
        if input.normalize() == 0:
            return ui.p("Click Normalize first.", class_="text-muted")
        values = z_max_values()["Z_max used"]
        too_low = [region for region, cuts in region_cuts().items() if values[region] <= cuts[-1]]
        if not too_low:
            return ui.p("Z_max is above the last cut-point in every region.", class_="text-success fw-bold")
        return ui.p(f"Z_max must be above the region's last cut-point. Too low in: {', '.join(too_low)}",
                    class_="text-danger fw-bold")

    @render.data_frame
    def zmax_table():
        table = z_max_values().reset_index(names="Region")
        table["people with z >= min"] = table["people with z >= min"].astype(int)   # 130, not 130.0
        return table

    @reactive.calc
    def sustain_input():
        """Exactly what tab 3 gives to pySuStaIn (data, Z_vals, Z_max) + names, IDs and checks."""
        z, cuts = normalized()["z"], region_cuts()
        regions = list(z.columns)
        z_max = z_max_values().loc[regions, "Z_max used"].to_numpy()
        data, Z_vals, Z_max = sustain_inputs(z, regions, cuts, z_max)
        minimum = CONFIG["normalization"]["min_people_per_event"]
        last_cuts = np.array([cuts[region][-1] for region in regions])
        checks = [("At least 2 regions", len(regions) >= 2),
                  ("Z_max above the last cut-point in every region", bool((Z_max > last_cuts).all())),
                  (f"Every event reached by at least {minimum} people",
                   bool((event_counts().fillna(minimum) >= minimum).all().all()))]
        return {"data": data, "Z_vals": Z_vals, "Z_max": Z_max, "regions": regions,
                "subjects": list(z.index), "cut_points": cuts, "checks": checks}

    def cuts_text(cuts):
        """'2, 5, 10' if every region has the same cut-points, else 'different per region'."""
        first = next(iter(cuts.values()))
        if all(c == first for c in cuts.values()):
            return ", ".join(f"{c:g}" for c in first)
        return "different per region"

    @render.ui
    def sustain_input_summary():
        if input.normalize() == 0:
            return ui.p("Click Normalize first.", class_="text-muted")
        s = sustain_input()
        n_events = int((s["Z_vals"] > 0).sum())
        z_max_text = (f"{s['Z_max'][0]:g} in every region" if len(set(s["Z_max"])) == 1
                      else f"{s['Z_max'].min():.2f} - {s['Z_max'].max():.2f} (per region)")
        facts = [f"Normalization: {NORM_METHODS[normalized()['method']]}",
                 f"People x regions: {s['data'].shape[0]} x {s['data'].shape[1]}",
                 f"Cut-points: {cuts_text(s['cut_points'])} -> {n_events} events, SuStaIn stages 0-{n_events}",
                 f"Z_max: {z_max_text}"]
        n_levels = len(np.unique(s["Z_vals"][s["Z_vals"] > 0]))
        if n_levels > 6:
            facts.append(f"Note: {n_levels} different cut-point values - SuStaIn runs, but its PVD figure can "
                         "only show 6.")
        checks = [ui.tags.li(("✓ " if ok else "✗ ") + text, class_="text-success" if ok else "text-danger")
                  for text, ok in s["checks"]]
        ready = all(ok for _, ok in s["checks"])
        verdict = (ui.p("Ready for tab 3 (Run SuStaIn).", class_="text-success fw-bold") if ready else
                   ui.p("Not ready: fix the items marked ✗.", class_="text-danger fw-bold"))
        return ui.TagList(outdated_warning() if norm_outdated() else None,
                          ui.h6("What goes into SuStaIn"), ui.tags.ul(*[ui.tags.li(f) for f in facts]),
                          ui.h6("Checks"), ui.tags.ul(*checks), verdict)

    @render.download_button(filename="tau_typer_zscores.csv")       # pairs with ui.download_button
    def download_z():
        yield normalized()["z"].round(4).to_csv(index_label="subject")

    @render.download_button(filename="tau_typer_normal_reference.csv")
    def download_reference():
        result = normalized()
        if result["reference"] is None:
            yield "note\nNo normal reference: the uploaded data was already normalized.\n"
        else:
            table = result["reference"].T.round(4)
            table["method"] = NORM_METHODS[result["method"]]
            yield table.to_csv(index_label="region")

    @render.download_link(filename="tau_typer_sample_data.csv")     # pairs with ui.download_link
    def download_sample():
        return str(SAMPLE_DATA)

    # ---------------- Tab 3: Run SuStaIn ----------------
    @reactive.effect
    @reactive.event(input.run_preset, ignore_init=True)
    def _apply_preset():
        """Fill in start points and MCMC iterations for the chosen preset (numbers from config.yaml)."""
        settings = CONFIG["sustain"]["fast_test"] if input.run_preset() == "fast" else CONFIG["sustain"]
        ui.update_numeric("n_startpoints", value=settings["n_startpoints"])
        ui.update_numeric("n_mcmc", value=settings["n_iterations_mcmc"])

    @reactive.effect
    @reactive.event(input.do_cv, ignore_init=True)
    def _cv_max_subtypes():
        """Cross-validation tries more subtypes (6) to choose the number; a normal run uses 4."""
        section = "cross_validation" if input.do_cv() else "sustain"
        ui.update_numeric("n_subtypes_max", value=CONFIG[section]["n_subtypes_max"])

    @reactive.effect
    def _fresh_run_name():
        """Each new visit gets a run name with the current date and time."""
        ui.update_text("run_name", value=datetime.now().strftime("run_%Y-%m-%d_%H%M"))

    @reactive.calc
    def run_checks():
        """Checks before a run, as (text, ok) - same style as tab 2."""
        try:
            tab2_ready = all(ok for _, ok in sustain_input()["checks"])
        except Exception:                                   # no data yet, or not normalized
            tab2_ready = False
        name = input.run_name().strip()
        name_ok = re.fullmatch(r"[A-Za-z0-9_-]+", name) is not None and not (runs_folder / name).exists()
        return [("Tab 2 says 'Ready for tab 3'", tab2_ready),
                ("Normalization is up to date (no changes after Normalize)", not norm_outdated()),
                ("Run name is new and uses only letters, digits, - and _", name_ok),
                ("Max number of subtypes is at least 1", (input.n_subtypes_max() or 0) >= 1),
                ("No other run is going on", run_state() != "running")]

    @reactive.calc
    def run_settings():
        """Everything one run needs, in one dictionary (saved as settings.yaml in the run folder)."""
        s = sustain_input()
        return {"n_people": len(s["subjects"]), "regions": s["regions"],
                "normalization": normalized()["method"], "cut_points": {r: [float(c) for c in cuts] for r, cuts in s["cut_points"].items()},
                "covariate_adjustment": (None if normalized()["adjustment"] is None else  # slopes per region
                                         normalized()["adjustment"]["slopes"].round(6).to_dict(orient="index")),
                "reference_group": REFERENCE_GROUPS[CONFIG["normalization"]["reference_group"]],
                "z_max": [round(float(v), 4) for v in s["Z_max"]],
                "n_subtypes_max": input.n_subtypes_max(), "n_startpoints": input.n_startpoints(),
                "n_iterations_mcmc": input.n_mcmc(), "seed": input.run_seed(),
                "use_parallel_startpoints": input.use_parallel(),
                "n_folds": input.n_folds() if input.do_cv() else None,       # None = no cross-validation
                "dataset_name": CONFIG["runs"]["dataset_name"],
                "columns": {role: col for role, col in chosen_columns().items() if col},   # for tab 5
                "pysustain": pysustain_info()}                                          # what was run

    @render.ui
    def run_settings_view():
        try:
            st = run_settings()
        except Exception:                                   # no data yet, or not normalized
            return ui.p("Upload data (tab 1) and click Normalize (tab 2) first.", class_="text-muted")
        n_events = int((make_z_vals(st["regions"], st["cut_points"]) > 0).sum())
        facts = [f"Data: {st['n_people']} people x {len(st['regions'])} regions, {n_events} events "
                 f"(stages 0-{n_events}); normalization: {NORM_METHODS[st['normalization']]}",
                 "Models: " + ("1 subtype only" if st["n_subtypes_max"] == 1
                               else f"1, 2, ... {st['n_subtypes_max']} subtypes (one model each)"),
                 f"Start points: {st['n_startpoints']}, MCMC iterations: {st['n_iterations_mcmc']:,}, "
                 f"seed: {st['seed']}, parallel: {'yes' if st['use_parallel_startpoints'] else 'no'}",
                 "Cross-validation: " + (f"{st['n_folds']} folds" if st['n_folds'] else "off"),
                 f"Results folder: {(runs_folder / input.run_name().strip()).relative_to(ROOT)}"]
        return ui.TagList(ui.h6("This run"), ui.tags.ul(*[ui.tags.li(f) for f in facts]))

    @render.ui
    def run_checks_view():
        items = [ui.tags.li(("✓ " if ok else "✗ ") + text, class_="text-success" if ok else "text-danger")
                 for text, ok in run_checks()]
        return ui.TagList(ui.h6("Checks before a run"), ui.tags.ul(*items))

    @reactive.effect
    def _run_button_state():
        """The Run button only works when every check passes."""
        ui.update_action_button("run_button", disabled=not all(ok for _, ok in run_checks()))

    @reactive.calc
    def run_covariates():
        """All non-region columns (age, sex, diagnosis, ...) of the people in the run, same IDs as z."""
        data, z = analysis_data(), normalized()["z"]
        others = data.drop(columns=list(input.col_regions()))
        id_col = chosen_columns().get("id")
        if id_col:
            others = others.set_index(id_col)
        return others.loc[z.index]

    job = reactive.value(None)          # the run started in this visit: process, folder, start time

    @reactive.effect
    @reactive.event(input.run_button)
    def _start_run():
        """Make the new run folder, then start pySuStaIn in its own process."""
        folder = runs_folder / input.run_name().strip()
        prepare_run_folder(folder, run_settings(), normalized()["z"], normalized()["reference"],
                           run_covariates())
        job.set({"process": start_run(folder), "folder": folder, "started": datetime.now()})

    @reactive.calc
    def run_state():
        """'none', 'running', 'finished', 'failed' or 'cancelled' - rechecked every 2 s while running."""
        j = job()
        if j is None:
            return "none"
        if j["process"].poll() is None:                     # poll() is None = still running
            reactive.invalidate_later(2)
            return "running"
        status = j["folder"] / "status.txt"
        return status.read_text().strip() if status.exists() else "failed"

    @render.ui
    def run_progress_view():
        state = run_state()
        if state == "none":
            return None
        j = job()
        log = (j["folder"] / "log.txt").read_text(errors="replace")
        lines = [line for line in re.split(r"[\r\n]+", log) if line.strip()][-12:]   # progress bars use \r
        elapsed = str(datetime.now() - j["started"]).split(".")[0]
        color = {"running": "text-primary", "finished": "text-success"}.get(state, "text-danger")
        return ui.TagList(ui.h6(f"Run {j['folder'].name}: {state} ({elapsed})", class_=color),
                          ui.tags.pre("\n".join(lines), style="font-size: 0.75rem; white-space: pre-wrap"))

    @reactive.effect
    @reactive.event(input.cancel_run)
    def _cancel_run():
        """Stop the running pySuStaIn process; what it saved so far stays in the folder."""
        j = job()
        if j is not None and j["process"].poll() is None:
            j["process"].terminate()
            (j["folder"] / "status.txt").write_text("cancelled\n")

    @reactive.effect
    def _cancel_button_state():
        ui.update_action_button("cancel_run", disabled=run_state() != "running")

    shown_run = reactive.value(None)    # folder of the run whose results are shown

    @reactive.effect
    def _show_finished_run():
        if run_state() == "finished":
            shown_run.set(job()["folder"])

    @reactive.calc
    def shown_settings():
        return yaml.safe_load(open(req(shown_run()) / "settings.yaml"))

    def no_run_yet():
        return ui.p("No finished run yet: run SuStaIn (tab Run) first.", class_="text-muted")

    @render.ui
    def model_choice_status():
        if shown_run() is None:
            return no_run_yet()
        cv_file = shown_run() / "cross_validation.csv"
        if cv_file.exists():
            best = pd.read_csv(cv_file, index_col="subtypes")["CVIC"].idxmin()
            return ui.p(f"Run {shown_run().name}. Cross-validation: the lowest CVIC is for {best} subtype(s) - "
                        "that is the number the data supports best. Second figure: "
                        "pySuStaIn's test log-likelihood per fold.", class_="text-success small")
        return ui.p(f"Run {shown_run().name}. Higher log-likelihood = fits this data better, but it rises "
                    "with every extra subtype - use cross-validation (CVIC) to choose the number. "
                    "Figure: pySuStaIn's own MCMC likelihood plot.", class_="text-muted small")

    @render.data_frame
    def model_likelihood_table():
        table = model_likelihoods(req(shown_run()), shown_settings()["dataset_name"])
        cv_file = shown_run() / "cross_validation.csv"
        if cv_file.exists():                                # only after a run with cross-validation
            table = table.join(pd.read_csv(cv_file, index_col="subtypes"))
        return render.DataGrid(table.round(1).reset_index())

    @render.plot
    def model_choice_plot():
        """Small figure: fit per number of subtypes (+ CVIC) and the MCMC line of each model."""
        folder, dataset = req(shown_run()), shown_settings()["dataset_name"]
        cv_file = folder / "cross_validation.csv"
        cv = pd.read_csv(cv_file, index_col="subtypes") if cv_file.exists() else None
        return plot_model_choice(model_likelihoods(folder, dataset), cv, mcmc_traces(folder, dataset))

    @render.image
    def cv_folds_plot():
        path = req(shown_run()) / "Log_likelihoods_cv_folds.png"   # made by pySuStaIn's cross-validation
        req(path.exists())
        return {"src": str(path), "width": "100%"}

    @reactive.effect
    def _result_model_choices():
        k = shown_settings()["n_subtypes_max"]
        ui.update_select("result_model", choices={str(n): f"{n} subtype(s)" for n in range(1, k + 1)},
                         selected=str(min(k, CONFIG["results"]["n_subtypes"])))       # paper: 4

    @reactive.calc
    def shown_table():
        folder, settings = req(shown_run()), shown_settings()
        k = int(req(input.result_model()))
        req(k <= settings["n_subtypes_max"])               # wait until the model list fits a newly shown run
        result = load_sustain_result(folder, settings["dataset_name"], k)
        subjects = pd.read_csv(folder / "zscores.csv", index_col="subject").index
        return subject_table(result, subjects, CONFIG["results"]["poor_fit_threshold"], size_order(result))

    @render.ui
    def quick_result_status():
        if shown_run() is None:
            return no_run_yet()
        return ui.p(f"Run {shown_run().name}. S0 = no abnormal tau (stage 0, no subtype). Poor fit = subtype "
                    f"probability below {CONFIG['results']['poor_fit_threshold']}. Stage [Q1, median, Q3]: a quarter "
                    "of the group is at or below Q1, half at or below the median, three quarters at or below Q3. "
                    "Last event = the last event reached at that stage (MCMC order below). All figures are in "
                    "tab 4.", class_="text-muted small")

    @render.data_frame
    def subtype_summary_table():
        table = shown_table()                               # first: it waits until a run and model are chosen
        folder, settings = shown_run(), shown_settings()
        result = load_sustain_result(folder, settings["dataset_name"], int(input.result_model()))
        Z_vals = make_z_vals(settings["regions"], settings["cut_points"])
        orders = event_orders(result, Z_vals, settings["regions"], size_order(result))
        mcmc = {f"Subtype {n}": mcmc_order for n, (_, mcmc_order, _) in enumerate(orders, start=1)}
        return render.DataGrid(subtype_summary(table, mcmc).reset_index(), width="100%")

    @render.ui
    def event_order_view():
        """EM order and MCMC order of each subtype; events at a different place are red."""
        shown_table()                                       # waits until a run and model are chosen
        folder, settings = shown_run(), shown_settings()
        result = load_sustain_result(folder, settings["dataset_name"], int(input.result_model()))
        Z_vals = make_z_vals(settings["regions"], settings["cut_points"])
        orders = event_orders(result, Z_vals, settings["regions"], size_order(result))

        def show(events, other, ranges=None):
            """Events joined by arrows; red where the other order has a different event at that place;
            grey (9-11) after an event = the stages it takes in the MCMC samples."""
            parts = []
            for i, event in enumerate(events):
                if i > 0:
                    parts.append(" → ")
                parts.append(ui.tags.span(event, style="color: #C62828; font-weight: bold;")
                             if event != other[i] else event)
                if ranges and ranges[i]:
                    parts.append(ui.tags.span(f" ({ranges[i]})", class_="text-muted"))
            return ui.TagList(*parts)

        rows = []
        for n, (em, mcmc, ranges) in enumerate(orders, start=1):
            n_diff = sum(a != b for a, b in zip(em, mcmc))
            note = "same order" if n_diff == 0 else f"{n_diff} of {len(em)} places differ"
            rows.append(ui.tags.tr(ui.tags.td(ui.strong(f"Subtype {n}"), ui.br(), ui.tags.small(note), rowspan=2),
                                   ui.tags.td("EM"), ui.tags.td(show(em, mcmc))))
            rows.append(ui.tags.tr(ui.tags.td("MCMC"), ui.tags.td(show(mcmc, em, ranges))))
        return ui.TagList(
            ui.h6("Event order of each subtype", class_="mt-4"),
            ui.p("EM = the single best order SuStaIn found (the MCMC starts from it). MCMC = events sorted by "
                 "their average place over all MCMC samples - what the PVD shows. Red = a different event at "
                 "that place in the other order. (9-11) = the event is at stage 9 to 11 in the MCMC samples "
                 "(stages in at least 5% of samples); no range = always the same stage.",
                 class_="text-muted small"),
            ui.tags.table(ui.tags.thead(ui.tags.tr(ui.tags.th("subtype"), ui.tags.th("from"),
                                                   ui.tags.th("event order (first → last)"))),
                          ui.tags.tbody(*rows), class_="table table-sm table-bordered small"))

    @render.download_button(filename=lambda: f"{shown_run().name}_subtypes_stages.csv")
    def download_subjects():
        yield shown_table().to_csv(index_label="subject")

    runs_changed = reactive.value(0)    # +1 after a delete or upload, so the run list refreshes at once

    @reactive.calc
    def saved_runs():
        run_state()                                         # refresh when a run starts or ends ...
        runs_changed()
        reactive.invalidate_later(10)                       # ... and every 10 s (runs from other visits)
        return list_runs(runs_folder).set_index("run", drop=False)

    @reactive.calc
    def finished_runs():
        runs = saved_runs()
        return list(runs.loc[runs["status"] == "finished", "run"])

    @render.data_frame
    def saved_runs_table():
        return render.DataGrid(saved_runs())

    @reactive.effect
    def _saved_run_choices():
        runs = saved_runs()
        with reactive.isolate():                            # keep the user's choice when the list refreshes
            keep = input.saved_run()
        ui.update_select("saved_run", choices={run: f"{run} ({status})" for run, status in runs["status"].items()},
                         selected=keep if keep in runs.index else None)

    def chosen_saved_run():
        """Name of the run chosen in the list (only real run folders), and its status."""
        name, runs = input.saved_run(), saved_runs()
        req(name in runs.index)
        return name, runs.loc[name, "status"]

    @reactive.effect
    @reactive.event(input.load_run)
    def _load_saved_run():
        name, status = chosen_saved_run()
        if status != "finished":
            ui.notification_show(f"Run {name} is {status}: only finished runs can be loaded.", type="warning")
            return
        shown_run.set(runs_folder / name)
        ui.notification_show(f"Showing run {name} in Model choice and Quick result.")

    @render.download_button(filename=lambda: f"{input.saved_run()}.zip")
    def download_run():
        name, _ = chosen_saved_run()
        yield zip_run(runs_folder / name)

    @reactive.effect
    @reactive.event(input.delete_run)
    def _ask_delete_run():
        name, status = chosen_saved_run()
        if status == "running":
            ui.notification_show("This run is still running: cancel it first.", type="warning")
            return
        ui.modal_show(ui.modal(f"Delete run {name}? Its folder and all its results are removed for good.",
                               title="Delete run",
                               footer=ui.TagList(ui.input_action_button("confirm_delete", "Delete",
                                                                        class_="btn-danger"),
                                                 ui.modal_button("Keep it"))))

    @reactive.effect
    @reactive.event(input.confirm_delete)
    def _delete_run():
        name, _ = chosen_saved_run()
        ui.modal_remove()
        if shown_run() == runs_folder / name:
            shown_run.set(None)
        shutil.rmtree(runs_folder / name)
        runs_changed.set(runs_changed() + 1)
        ui.notification_show(f"Run {name} deleted.")

    @reactive.effect
    @reactive.event(input.upload_run)
    def _upload_run():
        folder = save_uploaded_run(input.upload_run()[0])
        if folder is not None:
            ui.notification_show(f"Run uploaded as {folder.name}. Choose it in the list and load it.")

    def save_uploaded_run(file):
        """Save an uploaded run ZIP as a new saved run; returns its folder (None if the file is not a run)."""
        try:
            folder = unzip_run(file["datapath"], runs_folder, Path(file["name"]).stem)
        except Exception as error:                          # not a ZIP, or not a Tau-typer run
            ui.notification_show(f"Could not use this file: {error}", type="error", duration=10)
            return None
        runs_changed.set(runs_changed() + 1)
        return folder

    # ---------------- Tab 4: Figures ----------------
    @render.ui
    def fig_run_info():
        if shown_run() is None:
            return ui.p("No finished run yet: run SuStaIn or load a saved run in tab 3.", class_="text-danger small")
        return ui.p(f"Run shown: {shown_run().name}", class_="fw-bold small")

    @reactive.effect
    def _fig_model_choices():
        k = shown_settings()["n_subtypes_max"]
        for select_id in ["fig_model", "an_model"]:                                    # tabs 4 and 5
            ui.update_select(select_id, choices={str(n): f"{n} subtype(s)" for n in range(1, k + 1)},
                             selected=str(min(k, CONFIG["results"]["n_subtypes"])))   # paper: 4

    def model_view(k):
        """One model (k subtypes) of the shown run, as tabs 4 and 5 need it; subtypes numbered by size."""
        folder, settings = req(shown_run()), shown_settings()
        req(k <= settings["n_subtypes_max"])               # wait until the model list fits a newly shown run
        result = load_sustain_result(folder, settings["dataset_name"], k)
        order = size_order(result)
        z = pd.read_csv(folder / "zscores.csv", index_col="subject", dtype={"subject": str})   # IDs as text
        table = subject_table(result, z.index, CONFIG["results"]["poor_fit_threshold"], order)
        Z_vals = make_z_vals(settings["regions"], settings["cut_points"])   # list (old runs) or per region
        n_events = int((Z_vals > 0).sum())
        edges = np.unique(np.linspace(1, n_events + 1, CONFIG["figures"]["n_stage_groups"] + 1).round().astype(int))
        return {"result": result, "order": order, "z": z, "table": table, "k": k, "edges": list(edges),
                "regions": settings["regions"], "Z_vals": Z_vals}

    def subtype_titles(k):
        """The subtype names typed in tab 4, or Subtype 1, 2, ... if the number does not match k."""
        names = [name.strip() for name in input.subtype_names().split(",") if name.strip()]
        return names if len(names) == k else [f"Subtype {n}" for n in range(1, k + 1)]

    @reactive.calc
    def fig_data():
        return model_view(int(req(input.fig_model())))

    @reactive.calc
    def fig_titles():
        return subtype_titles(fig_data()["k"])

    @render.image(delete_file=True)
    def fig_pvd():
        """Saved as a picture at its own size (a resized plot squeezes the labels together)."""
        d = fig_data()
        n_levels = len(np.unique(d["Z_vals"][d["Z_vals"] > 0]))
        if n_levels > 6:                                         # pySuStaIn has colours for 6 values
            raise SafeException(f"pySuStaIn's PVD can show at most 6 different cut-point values; this run uses "
                                f"{n_levels}. The other figures still work.")
        path = Path(tempfile.gettempdir()) / f"tau_typer_pvd_{secrets.token_hex(8)}"   # deleted after showing
        fig = plot_pvd(d["result"], d["Z_vals"], d["regions"], fig_titles(), str(path),
                       subtype_order=pvd_order(d["result"], d["order"]))
        plt.close(fig)
        return {"src": f"{path}.png", "width": "100%"}

    @render.plot
    def fig_patterns():
        d = fig_data()
        return plot_subtype_patterns(d["z"], d["table"], d["regions"], fig_titles(), input.fig_vmax() or 12)

    @render.plot
    def fig_progression():
        d = fig_data()
        return plot_stage_progression(d["z"], d["table"], d["regions"], fig_titles(), d["edges"], None,
                                      vmax=input.fig_vmax() or 12)

    @render.plot
    def fig_probs():
        d = fig_data()
        return plot_subtype_probabilities(d["result"]["prob_subtype"][:, d["order"]], d["table"], fig_titles())

    @render.plot
    def fig_stages():
        d = fig_data()
        chosen = input.stages_subtype()
        subtype = int(chosen) if chosen and chosen != "all" else None
        return plot_stages(d["table"], fig_titles(), input.sd_limit(), input.stages_person() or None, subtype)

    @reactive.effect
    def _stages_subtype_choices():
        titles = fig_titles()
        ui.update_select("stages_subtype", choices={"all": "All tau+ people",
                                                    **{str(n): t for n, t in enumerate(titles, start=1)}})

    @reactive.effect
    def _stages_person_choices():
        """People of the chosen subtype (tau+ only), most uncertain stage first; none circled at the start."""
        table, chosen = fig_data()["table"], input.stages_subtype()
        people = table[table["ml_stage"] > 0]
        if chosen and chosen != "all":
            people = people[people["ml_subtype"] == int(chosen)]
        people = people.sort_values("stage_sd", ascending=False)
        choices = {"": "(none)", **{p: f"{p} (stage {r.ml_stage}, SD {r.stage_sd:.2f})" for p, r in people.iterrows()}}
        ui.update_selectize("stages_person", choices=choices, selected="")

    def uncertain_table():
        """Tau+ people whose stage SD is above the limit, most uncertain first."""
        table, limit = fig_data()["table"], req(input.sd_limit())
        above = table[(table["ml_stage"] > 0) & (table["stage_sd"] > limit)]
        return above[["subtype", "ml_stage", "stage_sd", "prob_stage"]].sort_values("stage_sd", ascending=False)

    @render.ui
    def uncertain_status():
        table, above = fig_data()["table"], uncertain_table()
        n_tau = int((table["ml_stage"] > 0).sum())
        return ui.p(f"{len(above)} of {n_tau} tau+ people ({100 * len(above) / max(n_tau, 1):.0f}%) have a stage "
                    f"SD above {input.sd_limit():g}. Their stage is uncertain - check them, or leave them out of "
                    "stage analyses.", class_="small mt-2")

    @render.data_frame
    def uncertain_people():
        above = uncertain_table().rename(columns={"ml_stage": "stage", "stage_sd": "stage SD",
                                                  "prob_stage": "probability of this stage"})
        return render.DataGrid(above.round(3).reset_index(names="person"))

    @render.plot
    def fig_choice():
        folder = req(shown_run())
        cv_file = folder / "cross_validation.csv"
        cv = pd.read_csv(cv_file, index_col="subtypes") if cv_file.exists() else None
        return plot_model_choice(model_likelihoods(folder, shown_settings()["dataset_name"]), cv)

    @render.plot
    def fig_tree():
        return plot_subtype_tree(subtypes_by_k(fig_data()["z"].index))

    def subtypes_by_k(people):
        """Subtype of every person in each model 1, 2, ... (0 = S0), all numbered by size."""
        folder, settings = req(shown_run()), shown_settings()
        by_k = {}
        for k in range(1, settings["n_subtypes_max"] + 1):
            result = load_sustain_result(folder, settings["dataset_name"], k)
            table = subject_table(result, people, CONFIG["results"]["poor_fit_threshold"], size_order(result))
            by_k[k] = table["ml_subtype"].where(table["ml_stage"] > 0, 0)
        return by_k

    @reactive.effect
    def _fig_person_choices():
        table = fig_data()["table"]
        tau_pos = list(table.index[table["ml_stage"] > 0])
        ui.update_selectize("fig_person", choices=list(table.index), selected=tau_pos[0] if tau_pos else None)

    @render.plot
    def fig_person_plot():
        d = fig_data()
        person = req(input.fig_person())
        req(person in d["table"].index)
        row, table = d["table"].loc[person], d["table"]
        group = pd.cut(table["ml_stage"], d["edges"], right=False)
        same = (table["ml_subtype"] == row["ml_subtype"]) & (group == group[person])
        expected = d["z"][same].mean() if row["ml_stage"] > 0 else None
        i = d["z"].index.get_loc(person)
        return plot_person(person, d["z"].loc[person], d["regions"], expected,
                           d["result"]["prob_subtype"][i][d["order"]], d["result"]["prob_stage"][i], fig_titles())

    # ---------------- Tab 5: Subtype analysis ----------------
    @render.ui
    def an_run_info():
        if shown_run() is None:
            return ui.p("No finished run yet: run SuStaIn or load a saved run in tab 3.", class_="text-danger small")
        return ui.p(f"Run shown: {shown_run().name}", class_="fw-bold small")

    @reactive.calc
    def an_data():
        return model_view(int(req(input.an_model())))

    @reactive.calc
    def an_covariates():
        """(other columns of the run's people, where they come from). From covariates.csv in the run
        folder, or else from the file in tab 1 matched by subject ID; (None, reason) if neither works."""
        folder, people = req(shown_run()), an_data()["z"].index
        if (folder / "covariates.csv").exists():
            covariates = pd.read_csv(folder / "covariates.csv", index_col="subject", dtype={"subject": str})
            return covariates, "saved with the run"
        data = uploaded_data()
        id_col = chosen_columns().get("id") if data is not None else None
        if not id_col:
            return None, "this run has no saved covariates - upload the same file in tab 1 (with its ID column)"
        others = data.drop(columns=list(input.col_regions() or []), errors="ignore")
        others.index = others.pop(id_col).astype(str)
        if not people.isin(others.index).any():
            return None, "the IDs in tab 1 do not match the people of this run"
        return others.reindex(people), "from the file in tab 1, matched by ID"

    def adjust_for():
        """The covariates chosen in "Adjust tests for" (none if the run has no covariates)."""
        if an_covariates()[0] is None:
            return []
        return [c for c in (input.an_adjust() or []) if c in an_covariates()[0].columns]

    def covariates_or_stop():
        """The covariates table; stops the output quietly if there is none (req cannot test a table)."""
        covariates = an_covariates()[0]
        req(covariates is not None)
        return covariates

    @reactive.effect
    def _an_variable_choices():
        covariates = an_covariates()[0]
        columns = [] if covariates is None else list(covariates.columns)
        numeric = [c for c in columns if pd.to_numeric(covariates[c], errors="coerce").notna().mean() > 0.9]
        ui.update_selectize("an_variables", choices=columns, selected=columns)
        ui.update_select("an_stage_variable", choices=numeric)
        roles = shown_settings().get("columns") or {}                     # saved with the run
        default = [roles[r] for r in ("age", "sex") if roles.get(r) in columns]   # age and sex if present
        ui.update_selectize("an_adjust", choices=columns, selected=default)

    @render.ui
    def an_status():
        if shown_run() is None:
            return no_run_yet()
        covariates, source = an_covariates()
        return ui.p(f"Run {shown_run().name}, {an_data()['k']} subtype(s). Variables: {source}.",
                    class_="small " + ("text-danger" if covariates is None else "text-muted"))

    @render.data_frame
    def an_table1():
        covariates = covariates_or_stop()
        return render.DataGrid(group_table(an_data()["table"], covariates, list(input.an_variables() or []),
                                           adjust_for()))

    @render.plot
    def an_regions_plot():
        d = an_data()
        t_values, p_fdr = regional_contrasts(d["z"], d["table"], d["regions"], an_covariates()[0], adjust_for())
        return plot_regional_contrasts(t_values, p_fdr, subtype_titles(d["k"]))

    @render.data_frame
    def an_stage_table():
        variable = req(input.an_stage_variable())
        return render.DataGrid(stage_association(an_data()["table"], covariates_or_stop()[variable]))

    @render.plot
    def an_stage_plot():
        d, variable = an_data(), req(input.an_stage_variable())
        return plot_stage_vs(d["table"], covariates_or_stop()[variable], variable, subtype_titles(d["k"]))

    @render.ui
    def an_asymmetry_status():
        if asymmetry_index(an_data()["z"], an_data()["regions"]) is None:
            return ui.p("No left/right region pairs found in the region names (e.g. L_MTL / R_MTL).",
                        class_="text-danger small")

    @render.plot
    def an_asymmetry_plot():
        d = an_data()
        asymmetry = asymmetry_index(d["z"], d["regions"])
        req(asymmetry is not None)
        return plot_asymmetry(d["table"], asymmetry, subtype_titles(d["k"]))

    compare_pick = reactive.value(None)  # a run just uploaded in Compare runs: select it in the list

    @reactive.effect
    def _an_other_runs():
        others = [run for run in finished_runs() if shown_run() is None or run != shown_run().name]
        with reactive.isolate():
            keep = compare_pick() or input.an_other_run()
            compare_pick.set(None)
        ui.update_select("an_other_run", choices=others, selected=keep if keep in others else None)

    @reactive.effect
    @reactive.event(input.compare_upload)
    def _compare_upload():
        """Upload a downloaded run here; it is saved like in tab 3 (Saved runs) and chosen as run B."""
        folder = save_uploaded_run(input.compare_upload()[0])
        if folder is None:
            return
        if (folder / "status.txt").exists() and (folder / "status.txt").read_text().strip() == "finished":
            compare_pick.set(folder.name)
            ui.notification_show(f"Run uploaded as {folder.name} and chosen for the comparison.")
        else:
            ui.notification_show(f"Run uploaded as {folder.name}, but it is not finished - it cannot be compared.",
                                 type="warning")

    @reactive.effect
    def _an_other_k_choices():
        """The other run's models only (1 ... its max number of subtypes)."""
        name = req(input.an_other_run())
        req(name in finished_runs())
        k_max = yaml.safe_load(open(runs_folder / name / "settings.yaml"))["n_subtypes_max"]
        ui.update_select("an_other_k", choices={str(k): f"{k} subtype(s)" for k in range(1, k_max + 1)},
                         selected=str(min(k_max, CONFIG["results"]["n_subtypes"])))

    @reactive.calc
    def an_comparison():
        """(cross-table, summary, k used) for the shown run (A) against another finished run (B)."""
        name = req(input.an_other_run())
        req(name in finished_runs())                        # only real, finished run folders
        folder = runs_folder / name
        settings = yaml.safe_load(open(folder / "settings.yaml"))
        k = int(req(input.an_other_k()))
        req(k <= settings["n_subtypes_max"])               # wait until the model list fits the chosen run
        other = load_sustain_result(folder, settings["dataset_name"], k)
        ids = pd.read_csv(folder / "zscores.csv", index_col="subject", dtype={"subject": str}).index
        other_table = subject_table(other, ids, CONFIG["results"]["poor_fit_threshold"], size_order(other))
        crosstab, summary = compare_runs(an_data()["table"], other_table)
        return crosstab, summary, k

    @render.ui
    def an_compare_summary():
        _, summary, k = an_comparison()
        facts = [f"Run A = {shown_run().name} ({an_data()['k']} subtypes), run B = {input.an_other_run()} ({k} subtypes)"]
        facts += [f"{key}: {value}" for key, value in summary.items()]
        return ui.tags.ul(*[ui.tags.li(f) for f in facts])

    @render.data_frame
    def an_compare_table():
        return render.DataGrid(an_comparison()[0].reset_index())

    def per_person_table():
        """One row per person: subtype, stage, probabilities and the other columns of the file."""
        d = an_data()
        probabilities = pd.DataFrame(d["result"]["prob_subtype"][:, d["order"]], index=d["table"].index,
                                     columns=[f"prob {t}" for t in subtype_titles(d["k"])])
        parts = [d["table"], probabilities]
        if an_covariates()[0] is not None:
            parts.append(an_covariates()[0])
        return pd.concat(parts, axis=1).rename_axis("subject")

    @render.download_button(filename=lambda: f"{shown_run().name}_{an_data()['k']}subtypes_per_person.csv")
    def an_download():
        yield per_person_table().to_csv()

    @render.download_button(filename=lambda: f"{shown_run().name}_{an_data()['k']}subtypes_report.zip")
    def an_report():
        """All figures (tabs 4-5) and tables of the shown run and model in one ZIP."""
        d, folder, settings = an_data(), shown_run(), shown_settings()
        titles, vmax = subtype_titles(d["k"]), input.fig_vmax() or CONFIG["figures"]["heatmap_vmax"]
        cv_file = folder / "cross_validation.csv"
        likelihoods = model_likelihoods(folder, settings["dataset_name"])
        if cv_file.exists():
            likelihoods = likelihoods.join(pd.read_csv(cv_file, index_col="subtypes"))
        t_values, p_fdr = regional_contrasts(d["z"], d["table"], d["regions"], an_covariates()[0], adjust_for())
        figures = {
            "01_pvd": plot_pvd(d["result"], d["Z_vals"], d["regions"], titles, None,
                               subtype_order=pvd_order(d["result"], d["order"])),
            "02_tau_patterns": plot_subtype_patterns(d["z"], d["table"], d["regions"], titles, vmax),
            "03_stage_progression": plot_stage_progression(d["z"], d["table"], d["regions"], titles, d["edges"],
                                                           None, vmax=vmax),
            "04_subtype_probabilities": plot_subtype_probabilities(d["result"]["prob_subtype"][:, d["order"]],
                                                                   d["table"], titles),
            "05_stages": plot_stages(d["table"], titles, input.sd_limit()),
            "06_model_choice": plot_model_choice(likelihoods[["mean log-likelihood"]],
                                                 likelihoods[["CVIC"]] if "CVIC" in likelihoods else None),
            "07_subtype_tree": plot_subtype_tree(subtypes_by_k(d["z"].index)),
            "08_regional_differences": plot_regional_contrasts(t_values, p_fdr, titles),
        }
        asymmetry = asymmetry_index(d["z"], d["regions"])
        if asymmetry is not None:
            figures["09_left_right_asymmetry"] = plot_asymmetry(d["table"], asymmetry, titles)
        tables = {"per_person": per_person_table(), "model_choice": likelihoods,
                  "regional_differences_t": t_values, "regional_differences_p_fdr": p_fdr}
        covariates = an_covariates()[0]
        if covariates is not None:
            tables["table1"] = group_table(d["table"], covariates, list(input.an_variables() or []),
                                           adjust_for()).set_index("variable")
            stage_variable = input.an_stage_variable()
            if stage_variable:
                tables[f"stage_vs_{stage_variable}"] = stage_association(d["table"], covariates[stage_variable])
        readme = (f"Tau-typer report - run {folder.name}, model with {d['k']} subtype(s)\n"
                  f"Made {datetime.now():%Y-%m-%d %H:%M}. Subtypes are numbered by number of tau+ people "
                  f"(Subtype 1 = most). Subtype names: {', '.join(titles)}.\n"
                  "figures/: PNG (300 dpi) and PDF. tables/: CSV. settings.yaml: everything the run used.\n")
        yield report_zip(figures, tables, {"README.txt": readme,
                                           "settings.yaml": (folder / "settings.yaml").read_text()})

    # ---------------- About ----------------
    @render.ui
    def about():
        info = pysustain_info()
        commit = info["commit"] or "unknown (not installed from git)"
        steps = ["1. Data & QC - upload a CSV (one row per person), match the columns, check the data, "
                 "optionally apply the inclusion rule.",
                 "2. Normalization - turn regional tau values into z-scores (GMM, an "
                 "amyloid-negative CN reference group, or data already normalized); choose cut-points and Z_max.",
                 "3. Run SuStaIn - fit the original pySuStaIn (ZscoreSustain) in the background, with "
                 "cross-validation to choose the number of subtypes; every run is saved in its own folder.",
                 "4. Figures - PVD, tau patterns, subtype probabilities, stages, choosing k, one person.",
                 "5. Subtype analysis - Table 1, regional differences, stage vs age / cognition, "
                 "left-right asymmetry, comparing runs, downloads and a full report."]
        papers = ["Vogel JW, Young AL, Oxtoby NP, et al. Four distinct trajectories of tau deposition identified "
                  "in Alzheimer's disease. Nature Medicine 27, 871-881 (2021).",
                  "Young AL, Marinescu RV, Oxtoby NP, et al. Uncovering the heterogeneity and temporal complexity "
                  "of neurodegenerative diseases with Subtype and Stage Inference. Nature Communications 9, "
                  "4273 (2018).",
                  "Aksman LM, Wijeratne PA, Oxtoby NP, et al. pySuStaIn: a Python implementation of the Subtype "
                  "and Stage Inference algorithm. SoftwareX 16, 100811 (2021)."]
        return ui.TagList(
            ui.h3("Tau-typer"),
            ui.p("Automated tau-PET subtyping with SuStaIn, following Vogel et al. 2021. The app only calls "
                 "the original pySuStaIn package - its code is never changed or copied."),
            ui.h5("The five steps"), ui.tags.ul(*[ui.tags.li(step) for step in steps]),
            ui.h5("pySuStaIn used by this app"),
            ui.tags.ul(ui.tags.li(f"version: {info['version']}"), ui.tags.li(f"git commit: {commit}"),
                       ui.tags.li(f"source: {info['source'] or 'unknown'}")),
            ui.p("Every run saves this information in its settings.yaml.", class_="text-muted small"),
            ui.h5("Please cite"), ui.tags.ul(*[ui.tags.li(paper) for paper in papers]),
            ui.h5("Your data"),
            ui.p("Uploaded data and results stay on the computer that runs this app (in results/app_runs/). "
                 "Nothing is sent anywhere else."),
            ui.p("Private runs: " + ("ON - this visit has its own folder, only you see its runs, and it is "
                                     "deleted when you close the app (download what you need first)."
                                     if private else "off - everyone using this app sees all saved runs."),
                 class_="fw-bold" if private else "text-muted"),
        )


    @render.ui
    def slopes_view():
        if input.normalize() == 0 or normalized()["adjustment"] is None:
            return None
        return ui.TagList(ui.h6("Covariate adjustment per region: slope (p-value)"),
                          ui.p("Slope = change in SUVR per unit of the covariate in the healthy reference group "
                               "(per year of age, per year of education, or vs the first category). "
                               "Bold with ✓ = removed from everyone's SUVR.", class_="text-muted small"),
                          ui.output_data_frame("slopes_table"),
                          ui.download_button("download_slopes", "Download covariate adjustment (CSV)"),
                          ui.p("Keep this file too: new people must be adjusted with the same slopes.",
                               class_="text-muted small"))

    def adjustment_table():
        """For the download: per region and covariate the estimated slope, its p-value and removed (yes/no)."""
        adjustment = req(normalized()["adjustment"])
        table = pd.DataFrame(index=adjustment["p_values"].index)
        for col in adjustment["p_values"].columns:
            table[f"{col}: slope"] = adjustment["estimates"][col]
            table[f"{col}: p"] = adjustment["p_values"][col]
            table[f"{col}: removed"] = adjustment["slopes"][col] != 0
        return table

    @render.data_frame
    def slopes_table():
        """One column per covariate: 'slope (p = ...)', bold with ✓ where it was removed."""
        adjustment = req(normalized()["adjustment"])        # only after an adjusted normalization
        removed = adjustment["slopes"] != 0
        shown = pd.DataFrame(index=adjustment["p_values"].index)
        styles = []
        for j, col in enumerate(adjustment["p_values"].columns):
            shown[col] = [f"{adjustment['estimates'].loc[r, col]:+.4f} (p = {adjustment['p_values'].loc[r, col]:.2g})"
                          + (" ✓ removed" if removed.loc[r, col] else "") for r in shown.index]
            rows = [i for i, region in enumerate(shown.index) if removed.loc[region, col]]
            if rows:
                styles.append({"rows": rows, "cols": [1 + j], "style": {"font-weight": "bold",
                                                                      "background-color": "#e3f2fd"}})
        return render.DataGrid(shown.reset_index(names="Region"), styles=styles, width="100%")

    @render.download_button(filename="tau_typer_covariate_adjustment.csv")
    def download_slopes():
        yield adjustment_table().to_csv(index_label="region")



app = App(app_ui, server)
