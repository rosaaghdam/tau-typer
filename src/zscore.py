"""Paper step 5 - turn regional SUVR into tau z-scores with a Gaussian mixture model (GMM).

For each region, a GMM separates a "normal" bell curve from an "abnormal" one.
z = (SUVR - normal mean) / normal SD, so z = 2 means "2 SD above normal".
"""
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def fit_gmm(values, n_components, seed):
    """Fit a Gaussian mixture model to one region's values and return the fitted model."""
    gmm = GaussianMixture(n_components=n_components, random_state=seed)
    gmm.fit(np.asarray(values).reshape(-1, 1))   # scikit-learn wants one column: people x 1
    return gmm


def fit_normal_distribution(values, n_components, seed):
    """Fit a GMM to one region's values; return the mean and SD of the "normal" component.

    The normal component is the one with the LOWEST mean (no tau = lowest signal).
    """
    gmm = fit_gmm(values, n_components, seed)
    means = gmm.means_.ravel()
    sds = np.sqrt(gmm.covariances_.ravel())      # covariance = SD squared
    normal = np.argmin(means)
    return means[normal], sds[normal]


def zscore_regions(data, regions, n_components, seed):
    """z-score every region. Returns (z table, reference = normal mean and SD per region)."""
    z = pd.DataFrame(index=data.index)
    reference = pd.DataFrame(index=["normal_mean", "normal_sd"], columns=regions, dtype=float)
    for region in regions:
        mean, sd = fit_normal_distribution(data[region].to_numpy(), n_components, seed)
        reference[region] = [mean, sd]
        z[region] = (data[region] - mean) / sd
    return z, reference
