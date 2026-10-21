"""Paper step 5 - turn regional SUVR into tau z-scores with a Gaussian mixture model (GMM).

For each region, a GMM separates a "normal" bell curve from an "abnormal" one.
z = (SUVR - normal mean) / normal SD, so z = 2 means "2 SD above normal".
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.mixture import GaussianMixture

from src.qc import number_and_category_columns


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


# ---------- Option: remove covariate effects (age, sex, ...) before normalization (W-score style) ----------

def covariate_matrix(data, names):
    """The chosen covariates as numbers: number columns as they are, category columns (sex, APOE4,
    site, ...) as 0/1 columns, one per value except the first (e.g. "sex=M")."""
    numbers, _ = number_and_category_columns(data, names)
    parts = []
    for name in names:
        if name in numbers:
            parts.append(pd.to_numeric(data[name], errors="coerce").rename(name))
        else:
            values = data[name].map(lambda v: v if pd.isna(v) else (f"{v:g}" if isinstance(v, float) else str(v)))
            dummies = pd.get_dummies(values, prefix=name, prefix_sep="=", drop_first=True, dtype=float)
            dummies.loc[values.isna()] = np.nan                  # unknown stays unknown
            parts.append(dummies)
    return pd.concat(parts, axis=1)


def covariate_adjustment(data, regions, covariates, in_reference, only_significant=True, alpha=0.05):
    """Remove covariate effects (age, sex, ...) from each region, learned in the reference group only.

    Per region:
    1. In the reference people (amyloid-negative, cognitively normal = normal ageing), fit
       SUVR = a + b1*cov1 + b2*cov2 + ... and keep the p-value of every covariate.
    2. Keep the covariates with p < alpha (or all of them if only_significant is False) and fit
       again with only those.
    3. For everyone: adjusted SUVR = SUVR - sum of b * (covariate - reference mean), so everyone is
       moved to the reference group's average. The values stay on the SUVR scale.
    covariates: from covariate_matrix, same rows as data, no empty cells.
    Covariates that do not vary in the reference group are skipped (not in the tables).
    Returns (adjusted values, p-value table, slope table) - regions x covariates; slope 0 = not used.
    """
    reference_x = covariates[in_reference]
    varies = reference_x.std() > 0                                # a covariate the same for everyone
    if not varies.all():                                          # in the reference group cannot be tested
        covariates, reference_x = covariates.loc[:, varies], reference_x.loc[:, varies]
    if covariates.shape[1] == 0:
        raise ValueError("None of the chosen covariates varies in the healthy reference group.")
    if np.linalg.matrix_rank(sm.add_constant(reference_x, has_constant="add")) < reference_x.shape[1] + 1:
        raise ValueError("Some chosen covariates are fully determined by each other in the reference group "
                         "(e.g. two codings of the same thing) - remove one of them.")
    centered = covariates - reference_x.mean()
    adjusted = data[regions].copy()
    p_values = pd.DataFrame(index=regions, columns=covariates.columns, dtype=float)
    slopes = pd.DataFrame(0.0, index=regions, columns=covariates.columns)
    for region in regions:
        y = data.loc[in_reference, region]
        fit = sm.OLS(y, sm.add_constant(reference_x, has_constant="add")).fit()
        p_values.loc[region] = fit.pvalues[covariates.columns]
        used = [c for c in covariates.columns if fit.pvalues[c] < alpha] if only_significant else list(covariates.columns)
        if not used:
            continue                                              # nothing to remove in this region
        fit = sm.OLS(y, sm.add_constant(reference_x[used], has_constant="add")).fit()
        slopes.loc[region, used] = fit.params[used]
        adjusted[region] = data[region] - centered[used].to_numpy() @ fit.params[used].to_numpy()
    return adjusted, p_values, slopes
