"""Paper steps 6-9 - prepare the inputs for pySuStaIn's ZscoreSustain and run it.

We only WRAP pySuStaIn here: all the modelling is done by the original package.
"""
import os

import numpy as np
import pandas as pd
from pySuStaIn import ZscoreSustain
from sklearn.model_selection import KFold


def sustain_inputs(z, regions, z_events, z_max):
    """Turn our z-score table + settings into the 3 inputs ZscoreSustain needs.

    Returns:
        data  - people x regions array of z-scores (plain numbers, no names)
        Z_vals - regions x cut-points table, every row = z_events (e.g. [2, 5, 10])
        Z_max  - the z-score each region reaches at the very end (one per region)
    z_max can be one number (same for every region) or one number per region.
    """
    data = z[regions].to_numpy()
    Z_vals = np.array([z_events] * len(regions), dtype=float)
    Z_max = np.broadcast_to(np.asarray(z_max, dtype=float), (len(regions),)).copy()
    return data, Z_vals, Z_max


def count_people_above(z, regions, z_events):
    """How many people reach each cut-point in each region (rarely reached events are hard to place)."""
    return pd.DataFrame({f"z>={cut:g}": (z[regions] >= cut).sum() for cut in z_events})   # :g -> "2", not "2.0"


def parse_cut_points(text):
    """Turn typed cut-points like "2, 5, 10" into sorted numbers [2.0, 5.0, 10.0].

    Raises ValueError (with a message for the user) if the text is empty, not numbers, or <= 0.
    """
    try:
        values = sorted(float(part) for part in text.replace(";", ",").split(",") if part.strip())
    except ValueError:
        raise ValueError("Cut-points must be numbers separated by commas, e.g. 2, 5, 10")
    if not values or values[0] <= 0:
        raise ValueError("Cut-points must be positive numbers, e.g. 2, 5, 10")
    return values


def z_max_table(z, method, fixed_value, percentile, min_z):
    """Z_max per region, and the numbers that help choose it.

    Per region, max / p99 / p95 of z among people with z >= min_z (clearly abnormal there - tau
    status is unknown in real data). method "fixed": Z_max = fixed_value for every region;
    method "percentile": Z_max = that percentile (e.g. 95) of the same people.
    """
    rows = {}
    for region in z.columns:
        high = z[region][z[region] >= min_z]
        rows[region] = {"people with z >= min": len(high), "max": high.max(),
                        "p99": high.quantile(0.99), "p95": high.quantile(0.95),
                        "Z_max used": fixed_value if method == "fixed" else high.quantile(percentile / 100)}
    return pd.DataFrame(rows).T.round(2)


def run_sustain(data, Z_vals, Z_max, regions, settings, output_folder, seed, plot=True):
    """Fit pySuStaIn's ZscoreSustain with 1, 2, ... n_subtypes_max subtypes (paper steps 7-9).

    pySuStaIn saves one result file per number of subtypes in <output_folder>/pickle_files/
    and (plot=True) the positional variance diagrams as PNG files in <output_folder>.
    NOTE: if those result files already exist, pySuStaIn LOADS them instead of fitting again,
    so use a new output_folder whenever you change a setting.
    Returns the model and its outputs for the largest number of subtypes.
    """
    os.makedirs(output_folder, exist_ok=True)
    model = ZscoreSustain(data, Z_vals, Z_max, regions,
                          settings["n_startpoints"], settings["n_subtypes_max"],
                          settings["n_iterations_mcmc"], output_folder,
                          settings["dataset_name"], settings["use_parallel_startpoints"], seed)
    outputs = model.run_sustain_algorithm(plot=plot)
    return model, outputs


def cross_validate(model, n_people, n_folds, seed):
    """pySuStaIn's own cross-validation (paper: 10 folds), for every model 1, 2, ... subtypes.

    We only make the folds: scikit-learn's KFold, shuffled with the seed (so the same folds every time).
    pySuStaIn fits each model on 9/10 of the people and scores it on the other 1/10, for each fold.
    Returns CVIC per model (lower = better; the paper chose the number of subtypes this way) and the
    test log-likelihood per fold and model. plot=True: pySuStaIn saves Log_likelihoods_cv_folds.png.
    """
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    test_idxs = [test for _, test in folds.split(np.zeros(n_people))]
    return model.cross_validate_sustain_model(test_idxs, plot=True)
