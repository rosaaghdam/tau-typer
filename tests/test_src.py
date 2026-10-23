"""Small checks of the calculations in src/ (no app, no pySuStaIn run)."""
import io
import pickle
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.app_runs import unzip_run, zip_run
from src.results import NumbersOnlyUnpickler, event_names, stage_quartiles, stage_sd
from src.sustain_model import make_z_vals


def test_stage_sd_sure_and_unsure():
    sure = np.array([[0, 0, 1.0, 0]])                     # always stage 2
    split = np.array([[0, 0.5, 0.5, 0]])                  # stage 1 or 2
    assert stage_sd(sure)[0] == 0
    assert stage_sd(split)[0] == pytest.approx(0.5)


def test_stage_quartiles_are_real_stages():
    assert stage_quartiles(pd.Series([12, 13])) == [12, 12, 13]   # no 12.5


def test_event_names_follow_pysustain_order():
    Z_vals = make_z_vals(["A", "B"], {"A": [2, 5], "B": [2]})       # B has only one event
    assert event_names(Z_vals, ["A", "B"]) == ["A z2", "B z2", "A z5"]


def test_result_files_with_code_are_refused():
    class Evil:
        def __reduce__(self):
            import os
            return (os.system, ("echo hacked",))
    with pytest.raises(pickle.UnpicklingError):
        NumbersOnlyUnpickler(io.BytesIO(pickle.dumps(Evil()))).load()
    assert NumbersOnlyUnpickler(io.BytesIO(pickle.dumps({"x": np.arange(3)}))).load()["x"].sum() == 3


def test_run_zip_round_trip(tmp_path):
    run = tmp_path / "runs" / "my_run"
    run.mkdir(parents=True)
    (run / "settings.yaml").write_text("n_people: 1\n")
    (run / "pid.txt").write_text("123\n")                # never in the ZIP
    (tmp_path / "run.zip").write_bytes(zip_run(run))
    copy = unzip_run(tmp_path / "run.zip", tmp_path / "runs", "my_run")
    assert copy.name == "my_run_2"                        # never overwrites the original
    assert (copy / "settings.yaml").exists() and not (copy / "pid.txt").exists()


def test_zip_with_paths_outside_is_refused(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("settings.yaml", "x: 1")
        z.writestr("../escape.txt", "x")
    (tmp_path / "bad.zip").write_bytes(buffer.getvalue())
    with pytest.raises(ValueError):
        unzip_run(tmp_path / "bad.zip", tmp_path / "runs", "bad")


def test_region_trend_tests_find_an_interaction():
    """A region that rises with age only in one group: the interaction p-value is small."""
    from src.qc import region_trend_tests
    rng = np.random.default_rng(0)
    age = rng.uniform(50, 90, 200)
    sex = np.where(np.arange(200) % 2 == 0, "F", "M")
    region = 1 + np.where(sex == "F", 0.02 * age, 0) + rng.normal(0, 0.05, 200)
    data = pd.DataFrame({"age": age, "sex": sex, "R1": region})
    table = region_trend_tests(data, ["R1"], "age", data["sex"])
    assert table.loc["R1", "p: x by group (interaction)"] < 0.001
