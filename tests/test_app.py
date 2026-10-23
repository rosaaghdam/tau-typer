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
