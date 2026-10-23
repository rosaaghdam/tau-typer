"""Tab 5 - Subtype analysis: Table 1, regional differences, stage vs variable, asymmetry, compare runs."""
from datetime import datetime

import pandas as pd
import yaml
from shiny import reactive, render, req, ui

from src.figures import plot_asymmetry, plot_regional_contrasts, plot_stage_vs
from src.results import event_orders, load_sustain_result, model_likelihoods, size_order, subject_table
from src.subtype_analysis import asymmetry_index, compare_runs, group_table, regional_contrasts, stage_association
from common import CONFIG, AN_HELP, figure_card, no_run_banner, step_status, step_title

panel = ui.nav_panel(step_title(5, "Subtype analysis", "step5_status"),
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
                 ui.output_ui("an_empty"),
                 ui.navset_tab(
                     ui.nav_panel("Table 1", figure_card("Table 1: the subtypes compared", AN_HELP["table1"],
                                                         ui.output_ui("an_status"),
                                                         ui.output_data_frame("an_table1"))),
                     ui.nav_panel("Regional differences", figure_card(
                         "Where does each subtype differ?", AN_HELP["regions"],
                         ui.output_plot("an_regions_plot", height="520px"))),
                     ui.nav_panel("Stage vs variable", figure_card("Stage against a variable", AN_HELP["stage"],
                                                                   ui.output_data_frame("an_stage_table"),
                                                                   ui.output_plot("an_stage_plot", height="460px"))),
                     ui.nav_panel("Left-right asymmetry", figure_card(
                         "Left-right asymmetry", AN_HELP["asymmetry"], ui.output_ui("an_asymmetry_status"),
                         ui.output_plot("an_asymmetry_plot", height="440px"))),
                     ui.nav_panel("Compare runs", ui.p(AN_HELP["compare"], class_="text-muted small"),
                                  ui.layout_columns(
                                      ui.input_select("an_other_run", "Other finished run", choices=[]),
                                      ui.input_select("an_other_k", "Its model (number of subtypes)", choices=[]),
                                      ui.input_file("compare_upload", "... or upload a run to compare (ZIP)",
                                                    accept=[".zip"]),
                                      col_widths=[4, 3, 5]),
                                  ui.output_ui("an_compare_summary"),
                                  ui.output_data_frame("an_compare_table")))))


def server(input, output, session, shared):
    """Outputs and reactive steps of this tab. shared = what earlier tabs made (and what this tab
    adds for the later ones)."""
    chosen_columns = shared["chosen_columns"]
    uploaded_data = shared["uploaded_data"]
    finished_runs = shared["finished_runs"]
    no_run_yet = shared["no_run_yet"]
    save_uploaded_run = shared["save_uploaded_run"]
    shown_run = shared["shown_run"]
    shown_settings = shared["shown_settings"]
    model_view = shared["model_view"]
    subtype_titles = shared["subtype_titles"]
    subtypes_by_k = shared["subtypes_by_k"]
    runs_folder = shared["runs_folder"]

    @render.ui
    def step5_status():
        if shown_run() is None:
            return step_status("Waiting for a run")
        return step_status(f"{input.an_model() or '?'} subtypes", "done")

    @render.ui
    def an_empty():
        return no_run_banner() if shown_run() is None else None

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
        t_values, p_fdr = an_contrasts()
        return plot_regional_contrasts(t_values, p_fdr, subtype_titles(an_data()["k"]))

    @reactive.calc
    def an_contrasts():
        """Regional differences (t-values, FDR p) - computed once, used by the figure and the report."""
        d = an_data()
        return regional_contrasts(d["z"], d["table"], d["regions"], an_covariates()[0], adjust_for())

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

    def report_inputs():
        """Everything the report needs, as plain data (it is made in a separate process)."""
        d, folder, settings = an_data(), shown_run(), shown_settings()
        titles = subtype_titles(d["k"])
        cv_file = folder / "cross_validation.csv"
        likelihoods = model_likelihoods(folder, settings["dataset_name"])
        if cv_file.exists():
            likelihoods = likelihoods.join(pd.read_csv(cv_file, index_col="subtypes"))
        t_values, p_fdr = an_contrasts()
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
        orders = event_orders(d["result"], d["Z_vals"], d["regions"], d["order"])
        return {"run": folder.name, "k": d["k"], "settings": settings,
                "mcmc_orders": [mcmc for _, mcmc, _ in orders],
                "result": d["result"], "Z_vals": d["Z_vals"], "regions": d["regions"], "order": d["order"],
                "z": d["z"], "table": d["table"], "edges": d["edges"], "titles": titles,
                "vmax": input.fig_vmax() or CONFIG["figures"]["heatmap_vmax"], "sd_limit": input.sd_limit(),
                "likelihoods": likelihoods, "subtypes_by_k": subtypes_by_k(d["z"].index),
                "t_values": t_values, "p_fdr": p_fdr, "tables": tables,
                "texts": {"README.txt": readme, "settings.yaml": (folder / "settings.yaml").read_text()},
                "file_name": f"{folder.name}_{d['k']}subtypes_report.zip"}

    # what the Report step (tab 6) uses
    shared.update(an_data=an_data, report_inputs=report_inputs, per_person_table=per_person_table)
