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


# ---------- Alternative: a reference group instead of a GMM (Young et al. 2018 style) ----------
def amyloid_negative_cn(data, diagnosis_col, amyloid_col, cn_labels):
    """True for people who are cognitively normal (label in cn_labels, any case) AND amyloid = 0."""
    diagnosis = data[diagnosis_col].astype(str).str.strip().str.upper()
    amyloid = pd.to_numeric(data[amyloid_col], errors="coerce")
    return diagnosis.isin([label.upper() for label in cn_labels]) & (amyloid == 0)


def reference_group_normal(data, regions, in_reference):
    """Normal mean and SD per region = plain mean and SD of the reference group.
    Same table format as zscore_regions' reference (rows normal_mean, normal_sd)."""
    group = data.loc[in_reference, regions]
    return pd.DataFrame({"normal_mean": group.mean(), "normal_sd": group.std()}).T


def zscore_with_reference(data, regions, reference):
    """z = (value - normal mean) / normal SD, with a given reference table (any method)."""
    return (data[regions] - reference.loc["normal_mean"]) / reference.loc["normal_sd"]


# ---------- Option: remove normal ageing (and sex) before normalization (W-score style) ----------

def covariate_adjustment(data, regions, covariates, in_reference):
    """Remove the effect of age (and sex) from each region, learned in the reference group only.

    Per region: fit SUVR = a + b1*age (+ b2*sex) in the reference people (amyloid-negative CN,
    i.e. normal ageing). Then for everyone: adjusted SUVR = SUVR - b * (covariate - reference mean),
    so everyone is moved to the reference group's average age (and sex). The values stay on the
    SUVR scale, so the GMM (or reference group) step after it works as before.
    covariates: numbers only (sex as 1 = female, 0 = male), same rows as data, no empty cells.
    Returns (adjusted values, table of slopes: one row per region, one column per covariate).
    """
    reference_x = covariates[in_reference]
    centered = covariates - reference_x.mean()
    design = np.column_stack([np.ones(len(reference_x)), reference_x.to_numpy()])
    adjusted = data[regions].copy()
    slopes = pd.DataFrame(index=regions, columns=covariates.columns, dtype=float)
    for region in regions:
        coefficients = np.linalg.lstsq(design, data.loc[in_reference, region].to_numpy(), rcond=None)[0]
        slopes.loc[region] = coefficients[1:]                     # [0] is the intercept
        adjusted[region] = data[region] - centered.to_numpy() @ coefficients[1:]
    return adjusted, slopes
