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


def model_likelihoods(output_folder, dataset_name):
    """Log-likelihood of every saved model (1, 2, ... subtypes), from pySuStaIn's MCMC samples.

    Higher = fits this data better. It (almost) always rises with more subtypes, so it cannot
    choose the number of subtypes on its own - cross-validation (CVIC) does that.
    """
    rows, n = {}, 1
    while os.path.exists(os.path.join(output_folder, "pickle_files", f"{dataset_name}_subtype{n - 1}.pickle")):
        likelihood = load_sustain_result(output_folder, dataset_name, n)["samples_likelihood"].ravel()
        rows[n] = {"mean log-likelihood": likelihood.mean(), "best log-likelihood": likelihood.max()}
        n += 1
    return pd.DataFrame(rows).T.rename_axis("subtypes")


def subtype_summary(table):
    """People per group (S0, Subtype 1, ...): number, %, how many fit poorly, median stage."""
    groups = table.groupby("subtype")
    summary = pd.DataFrame({"people": groups.size(), "poor fit": groups["poor_fit"].sum(),
                            "median stage": groups["ml_stage"].median()})
    summary.insert(1, "%", (100 * summary["people"] / len(table)).round(1))
    return summary
