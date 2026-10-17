"""Figures of SuStaIn results. Drawing is done by pySuStaIn's own plot functions where possible."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pySuStaIn import ZscoreSustain
from scipy import stats


def plot_pvd(result, Z_vals, regions, subtype_titles, save_path, stage_interval=5):
    """Clearer positional variance diagram (PVD), drawn by pySuStaIn's plot_positional_var.

    Rows = regions, columns = SuStaIn stages. Red = region reaches z 2, magenta = z 5,
    blue = z 10. Dark square = SuStaIn is sure the event happens at that stage; pale = unsure.
    save_path is given without ".png"; pySuStaIn adds "_all-subtypes.png" itself.
    """
    figs, _ = ZscoreSustain.plot_positional_var(
        result["samples_sequence"], result["samples_f"], np.inf, Z_vals,
        biomarker_labels=regions, subtype_titles=subtype_titles,
        figsize=(14, 8), title_font_size=14, label_font_size=11, stage_font_size=10,
        stage_interval=stage_interval, save_path=save_path,
        save_kwargs={"dpi": 150, "bbox_inches": "tight"})
    return figs[0]


def plot_stage_progression(z, table, regions, subtype_titles, stage_edges, save_path, vmax=12):
    """Average z-score per region for real people, grouped by subtype and stage (paper Fig. 1e idea).

    One panel per subtype: rows = regions, columns = stage groups (e.g. 1-6, 7-12, ...).
    Darker red = more tau on average. Reading left to right shows tau spreading with stage.
    """
    labels = [f"{a}-{b - 1}" for a, b in zip(stage_edges[:-1], stage_edges[1:])]
    stage_group = pd.cut(table["ml_stage"], stage_edges, right=False, labels=labels)
    fig, axes = plt.subplots(1, len(subtype_titles), figsize=(4 * len(subtype_titles), 5), sharey=True)
    for k, (ax, title) in enumerate(zip(axes, subtype_titles), start=1):
        people = (table["ml_subtype"] == k) & (table["ml_stage"] > 0)      # this subtype, tau+ only
        means = z[people].groupby(stage_group[people], observed=False)[regions].mean().T  # regions x groups
        counts = stage_group[people].value_counts().reindex(labels)
        image = ax.imshow(means, cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(labels)), [f"{lab}\nn={n}" for lab, n in zip(labels, counts)], fontsize=9)
        ax.set_yticks(range(len(regions)), regions)
        ax.set_title(title)
        ax.set_xlabel("SuStaIn stage")
    fig.colorbar(image, ax=axes, label="average z-score", shrink=0.8)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_region_histograms(data, regions, n_cols=5, xlabel="value (e.g. SUVR)"):
    """One small histogram per region column - raw values in tab 1, z-scores in tab 2.

    Shows the "normal" bump and the tau+ tail of each region, and makes outliers or odd extra
    bumps easy to spot. xlabel is the label under each panel.
    """
    values = data[list(regions)].apply(pd.to_numeric, errors="coerce")   # text becomes missing
    n_rows = max(1, int(np.ceil(len(regions) / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.4 * n_rows), squeeze=False)
    for ax, region in zip(axes.flat, regions):
        ax.hist(values[region].dropna(), bins=40, color="#4C72B0")
        ax.set_title(region, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
    for ax in axes.flat[len(regions):]:
        ax.set_visible(False)                                             # hide unused panels
    fig.tight_layout()
    return fig


def plot_gmm_fit(values, gmm, region, normal_mean, normal_sd, z_events):
    """One region: histogram of the values + every GMM bell curve (normal one in green), and the
    cut-points drawn on the value axis (SUVR = normal mean + z * normal SD) - the tab-2 check."""
    values = np.asarray(values)
    cut_values = [normal_mean + z * normal_sd for z in z_events]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(values, bins=60, density=True, color="lightgray", label="people")
    x = np.linspace(values.min(), max(values.max(), max(cut_values)), 500)
    means, weights = gmm.means_.ravel(), gmm.weights_
    sds = np.sqrt(gmm.covariances_.ravel())
    normal = np.argmin(means)
    for k in np.argsort(means):                          # draw from lowest to highest mean
        is_normal = k == normal
        ax.plot(x, weights[k] * stats.norm.pdf(x, means[k], sds[k]), lw=2.5 if is_normal else 1.5,
                color="#2E7D32" if is_normal else "#C62828", label="normal" if is_normal else "abnormal")
    for z, cut in zip(z_events, cut_values):
        ax.axvline(cut, ls="--", color="black", lw=1)
        ax.text(cut, ax.get_ylim()[1] * 0.95, f" z={z}", fontsize=9, va="top")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))                    # one legend entry per label
    ax.legend(unique.values(), unique.keys())
    ax.set_title(f"{region}: normal mean {normal_mean:.3f}, normal SD {normal_sd:.3f}")
    ax.set_xlabel("value (e.g. SUVR)")
    ax.set_ylabel("density")
    fig.tight_layout()
    return fig
