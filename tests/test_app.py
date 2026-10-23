"""The app's server, run in memory (no browser): no output may fail in the main steps.
Run from the project folder:   ./env/bin/python -m pytest tests -q
"""
from shiny.testserver import test_server

from conftest import APP, DATA_INPUTS, FIGURE_INPUTS, NORM_INPUTS, errors, finished_runs, toy_upload


def load(ts, run, k="2"):
    """Load a saved run in tab 3 and choose the model in tabs 3-5. The model boxes are set twice: the
    app also sets them when a run is loaded, and the test server applies that one step later."""
    models = {"result_model": k, "fig_model": k, "an_model": k}
    ts.set_inputs(saved_run=run, load_run=1, **models)
    ts.set_inputs(**models)


def test_upload_and_normalize():
    """Tabs 1-2 with the toy data: upload, match columns, normalize (GMM)."""
    with test_server(APP, timeout_secs=120) as ts:
        ts.set_inputs(data_file=toy_upload(), **DATA_INPUTS, **NORM_INPUTS)
        assert errors(ts) == {}
        assert ts.get_output("qc_table").status == "ok"
        ts.set_inputs(normalize=1)
        assert errors(ts) == {}
        assert ts.get_output("z_head").status == "ok"            # z-scores were made
        assert ts.get_output("events_table").status == "ok"


def test_normalize_with_reference_group_and_covariates():
    """Reference-group normalization with age adjusted first."""
    with test_server(APP, timeout_secs=120) as ts:
        ts.set_inputs(data_file=toy_upload(), **DATA_INPUTS, **NORM_INPUTS)
        ts.set_inputs(norm_method="reference", adjust_cov=True, adjust_candidates=["age"], normalize=1)
        assert errors(ts) == {}
        assert ts.get_output("slopes_table").status == "ok"


def test_show_saved_run(one_run):
    """Tabs 3-5: load a finished run; every figure and table renders without an error."""
    others = [run for run in finished_runs() if run != one_run]
    with test_server(APP, timeout_secs=300) as ts:
        ts.set_inputs(saved_run=one_run, an_other_run=others[0] if others else "", an_other_k="2",
                      **FIGURE_INPUTS)
        load(ts, one_run)
        assert errors(ts) == {}
        for name in ["subtype_summary_table", "event_order_view", "fig_pvd", "fig_probs", "fig_stages",
                     "fig_tree", "an_table1", "an_regions_plot"]:
            assert ts.get_output(name).status == "ok", name


def test_s0_people_have_no_subtype(one_run):
    """Stage 0 = S0 in the summary table, as in the paper (pySuStaIn itself still gives them a subtype)."""
    with test_server(APP, timeout_secs=300) as ts:
        ts.set_inputs(saved_run=one_run, **FIGURE_INPUTS)
        load(ts, one_run)
        rows = ts.get_output("subtype_summary_table").value["payload"]["data"]
        assert rows[0][0] == "S0"


def test_report_is_made_in_the_background(one_run):
    """Tab 6: 'Make report (ZIP)' and 'Make PDF report' build the files in a separate process; then the
    download buttons appear. (Both in one test: background jobs of two test sessions clash.)"""
    import time
    with test_server(APP, timeout_secs=300) as ts:
        ts.set_inputs(saved_run=one_run, **FIGURE_INPUTS)
        load(ts, one_run)
        ts.set_inputs(make_report=1)
        for _ in range(120):                              # wait up to 2 minutes
            if ts.get_output("report_ready").status == "ok" and ts.get_output("report_ready").value:
                break
            time.sleep(1)
            ts.set_inputs(fig_vmax=12)                    # lets the app check whether the job is done
        assert errors(ts) == {}
        assert "Download" in str(ts.get_output("report_ready").value)
        ts.set_inputs(make_pdf=1)
        wait_for(ts, "pdf_ready")
        assert errors(ts) == {}
        assert "Download" in str(ts.get_output("pdf_ready").value)


def test_inclusion_rule_note_and_region_exploration():
    """Tab 1: the note under the rule checkbox, and regions plotted against age, split by sex, with tests."""
    with test_server(APP, timeout_secs=120) as ts:
        ts.set_inputs(data_file=toy_upload(), **DATA_INPUTS, **NORM_INPUTS, eda_x="", eda_split="")
        ts.set_inputs(apply_rule=True)
        ts.set_inputs(apply_rule=True)                    # the note sits in a box the app adds itself
        assert "people kept" in str(ts.get_output("rule_note").value)
        ts.set_inputs(eda_x="age", eda_split="sex")
        assert errors(ts) == {}
        assert ts.get_output("region_hist").status == "ok"
        assert "interaction" in str(ts.get_output("eda_tests").value)


def wait_for(ts, output, seconds=120):
    """Wait until a background job has filled 'output' (lets the app check every second)."""
    import time
    for _ in range(seconds):
        if ts.get_output(output).status == "ok" and ts.get_output(output).value:
            return
        time.sleep(1)
        ts.set_inputs(fig_vmax=12)


def test_report_summary(one_run):
    """Tab 6: the written summary of the run."""
    with test_server(APP, timeout_secs=300) as ts:
        ts.set_inputs(saved_run=one_run, **FIGURE_INPUTS)
        load(ts, one_run)
        assert "have abnormal tau" in str(ts.get_output("report_summary").value)


def test_loading_a_run_fills_tabs_1_and_2(one_run, tmp_path):
    """A run saved with its input data: loading it shows the data in tab 1 and its z-scores in tab 2."""
    import shutil
    import pandas as pd
    from conftest import ROOT, TOY
    runs = ROOT / "results" / "app_runs"
    copy = runs / "_test_restore"
    shutil.rmtree(copy, ignore_errors=True)
    shutil.copytree(runs / one_run, copy)
    try:
        z = pd.read_csv(copy / "zscores.csv", index_col="subject")
        data = pd.read_csv(TOY)
        data[data["subject_id"].isin(z.index)].to_csv(copy / "input_data.csv", index=False)
        data.set_index("subject_id").loc[z.index, list(z.columns)].to_csv(copy / "input_values.csv",
                                                                        index_label="subject")
        with test_server(APP, timeout_secs=300) as ts:
            ts.set_inputs(saved_run="_test_restore", **FIGURE_INPUTS)
            load(ts, "_test_restore")
            assert "_test_restore" in str(ts.get_output("data_status").value)
            assert ts.get_output("data_head").status == "ok"
            assert "as saved with the run" in str(ts.get_output("norm_status").value)
            assert ts.get_output("z_head").status == "ok"
            assert errors(ts) == {}
    finally:
        shutil.rmtree(copy, ignore_errors=True)
