"""Paper steps 6-9 - prepare the inputs for pySuStaIn's ZscoreSustain and run it.

We only WRAP pySuStaIn here: all the modelling is done by the original package.
"""
import os

import numpy as np
import pandas as pd
from pySuStaIn import ZscoreSustain


def sustain_inputs(z, regions, z_events, z_max):
    """Turn our z-score table + settings into the 3 inputs ZscoreSustain needs.

    Returns:
        data  - people x regions array of z-scores (plain numbers, no names)
        Z_vals - regions x cut-points table, every row = z_events (e.g. [2, 5, 10])
        Z_max  - the z-score each region reaches at the very end (one per region)
    """
    data = z[regions].to_numpy()
    Z_vals = np.array([z_events] * len(regions), dtype=float)
    Z_max = np.full(len(regions), float(z_max))
    return data, Z_vals, Z_max


def count_people_above(z, regions, z_events):
    """How many people reach each cut-point in each region (rarely reached events are hard to place)."""
    return pd.DataFrame({f"z>={cut}": (z[regions] >= cut).sum() for cut in z_events})


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
