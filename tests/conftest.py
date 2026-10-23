"""Shared test settings: the project folder on the path, and the inputs a browser would fill in."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
APP = ROOT / "app" / "app.py"
TOY = ROOT / "data" / "toy" / "toy_tau_clinical.csv"
REGIONS = ["L_MTL", "R_MTL", "L_temporal", "R_temporal", "L_parietal", "R_parietal",
           "L_occipital", "R_occipital", "L_frontal", "R_frontal"]

# test_server has no browser, so boxes that the browser fills in (defaults, or choices the app sends)
# are set by hand - the same values the app starts with
BUTTONS = {name: 0 for name in ["normalize", "pr_reset", "run_button", "cancel_run", "load_run", "delete_run"]}
DATA_INPUTS = {**BUTTONS, "col_id": "subject_id", "col_age": "age", "col_sex": "sex", "col_diagnosis": "diagnosis",
               "col_amyloid": "amyloid_positive", "col_regions": REGIONS, "apply_rule": False}
NORM_INPUTS = {"norm_method": "gmm", "adjust_cov": False, "adjust_candidates": [], "adjust_only_sig": True,
               "adjust_alpha": 0.05, "gmm_components": 2, "per_region": False, "cut_points": "2, 5, 10",
               "z_max_method": "manual", "z_max_manual": "15", "z_max_percentile": 95, "gmm_region": "L_MTL"}
FIGURE_INPUTS = {**BUTTONS, "subtype_names": "", "fig_vmax": 12, "sd_limit": 1.0, "stages_subtype": "all",
                 "stages_person": "", "fig_person": "TOY_0001", "an_variables": ["age", "sex", "MMSE"],
                 "an_stage_variable": "age", "an_adjust": ["age", "sex"]}


def toy_upload():
    """The value input_file gives the server after a CSV upload."""
    return [{"name": TOY.name, "size": TOY.stat().st_size, "type": "text/csv", "datapath": str(TOY)}]


def finished_runs():
    """Finished runs in results/app_runs (not in git - the run tests are skipped without one)."""
    from src.app_runs import list_runs
    runs = list_runs(ROOT / "results" / "app_runs")
    return list(runs.loc[runs["status"] == "finished", "run"]) if len(runs) else []


def errors(ts):
    """Outputs that failed (an empty output is fine: it waits for something)."""
    return {name: value.error for name, value in ts.to_values().outputs.items() if value.status == "error"}


@pytest.fixture
def one_run():
    runs = finished_runs()
    if not runs:
        pytest.skip("no finished run in results/app_runs")
    return runs[0]
