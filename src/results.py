"""Paper steps 9-11 - read SuStaIn's saved results and turn them into readable tables and figures."""
import os
import pickle

import numpy as np
import pandas as pd


def load_sustain_result(output_folder, dataset_name, n_subtypes):
    """Load pySuStaIn's saved result for the model with n_subtypes subtypes.

    pySuStaIn counts from 0: the 4-subtype model is saved as <dataset_name>_subtype3.pickle.
    """
    path = os.path.join(output_folder, "pickle_files", f"{dataset_name}_subtype{n_subtypes - 1}.pickle")
    with open(path, "rb") as f:
        return pickle.load(f)


def subject_table(result, subject_ids, poor_fit_threshold):
    """One row per person: subtype, stage and their probabilities.

    Stage 0 = no abnormal tau -> "S0" (no subtype), as in the paper.
    poor_fit = tau+ but subtype probability below the threshold (paper: 0.5).
    """
    table = pd.DataFrame({
        "ml_subtype": result["ml_subtype"].ravel().astype(int) + 1,   # 1..k, like the figure titles
        "prob_subtype": result["prob_ml_subtype"].ravel(),
        "ml_stage": result["ml_stage"].ravel().astype(int),
        "prob_stage": result["prob_ml_stage"].ravel(),
    }, index=subject_ids)
    table["subtype"] = np.where(table["ml_stage"] == 0, "S0", "Subtype " + table["ml_subtype"].astype(str))
    table["poor_fit"] = (table["ml_stage"] > 0) & (table["prob_subtype"] < poor_fit_threshold)
    return table
