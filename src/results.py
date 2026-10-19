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


def size_order(result):
    """pySuStaIn's own subtype numbers, the subtype with the most tau+ people first.

    (pySuStaIn's PVD sorts by its estimated subtype proportions, which can differ from the number of
    people assigned; the app passes this order to the PVD too, so all figures and tables agree.)
    """
    k = result["samples_f"].shape[0]
    tau_pos = result["ml_stage"].ravel() > 0
    counts = np.bincount(result["ml_subtype"].ravel()[tau_pos].astype(int), minlength=k)
    return np.argsort(-counts, kind="stable")


def pvd_order(result, order):
    """The subtype_order to give pySuStaIn's PVD so its panels follow `order` (from size_order).

    pySuStaIn's ml_subtype and prob_subtype number the subtypes AFTER sorting them by estimated
    proportion f (in subtype_and_stage_individuals), but samples_sequence - drawn by the PVD - keeps
    the unsorted numbers. So we translate back, sorting exactly as pySuStaIn does.
    """
    by_f = np.argsort(np.mean(result["samples_f"], axis=1))[::-1]
    return by_f[np.asarray(order)]


def subject_table(result, subject_ids, poor_fit_threshold, order=None):
    """One row per person: subtype, stage and their probabilities.

    Stage 0 = no abnormal tau -> "S0" (no subtype), as in the paper.
    poor_fit = tau+ but subtype probability below the threshold (paper: 0.5).
    order (from size_order): renumber subtypes so Subtype 1 = largest, as in the PVD.
    Without order, pySuStaIn's own numbers are kept.
    """
    raw = result["ml_subtype"].ravel().astype(int)
    if order is not None:
        raw = np.argsort(order)[raw]                    # position of each subtype in the size order
    table = pd.DataFrame({
        "ml_subtype": raw + 1,                                         # 1..k, like the figure titles
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


def mcmc_traces(output_folder, dataset_name):
    """{number of subtypes: log-likelihood of every MCMC sample} for every saved model."""
    traces, n = {}, 1
    while os.path.exists(os.path.join(output_folder, "pickle_files", f"{dataset_name}_subtype{n - 1}.pickle")):
        traces[n] = load_sustain_result(output_folder, dataset_name, n)["samples_likelihood"].ravel()
        n += 1
    return traces


def subtype_summary(table):
    """People per group (S0, Subtype 1, ...): number, %, how many fit poorly, median stage."""
    groups = table.groupby("subtype")
    summary = pd.DataFrame({"people": groups.size(), "poor fit": groups["poor_fit"].sum(),
                            "median stage": groups["ml_stage"].median()})
    summary.insert(1, "%", (100 * summary["people"] / len(table)).round(1))
    return summary
