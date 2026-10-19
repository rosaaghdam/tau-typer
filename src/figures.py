"""Figures of SuStaIn results. Drawing is done by pySuStaIn's own plot functions where possible."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pySuStaIn import ZscoreSustain
from scipy import stats

from src.subtype_analysis import region_pairs


def plot_pvd(result, Z_vals, regions, subtype_titles, save_path, stage_interval=5, subtype_order=None):
    """Clearer positional variance diagram (PVD), drawn by pySuStaIn's plot_positional_var.

    Rows = regions, columns = SuStaIn stages. Red = region reaches z 2, magenta = z 5,
    blue = z 10. Dark square = SuStaIn is sure the event happens at that stage; pale = unsure.
    save_path is given without ".png"; pySuStaIn adds "_all-subtypes.png" itself (None = do not save).
    subtype_order: which of pySuStaIn's subtypes to draw first, second, ... (None = pySuStaIn's choice).
    """
    figs, _ = ZscoreSustain.plot_positional_var(
        result["samples_sequence"], result["samples_f"], np.inf, Z_vals,
        biomarker_labels=regions, subtype_titles=subtype_titles,
        figsize=(14, 8), title_font_size=14, label_font_size=11, stage_font_size=10,
        stage_interval=stage_interval, save_path=save_path, subtype_order=subtype_order,
        save_kwargs={"dpi": 150, "bbox_inches": "tight"})
    return figs[0]


def plot_stage_progression(z, table, regions, subtype_titles, stage_edges, save_path, vmax=12):
    """Average z-score per region for real people, grouped by subtype and stage (paper Fig. 1e idea).

    One panel per subtype: rows = regions, columns = stage groups (e.g. 1-6, 7-12, ...).
    Darker red = more tau on average. Reading left to right shows tau spreading with stage.
    """
    labels = [f"{a}-{b - 1}" for a, b in zip(stage_edges[:-1], stage_edges[1:])]
    stage_group = pd.cut(table["ml_stage"], stage_edges, right=False, labels=labels)
    n = len(subtype_titles)                               # one panel per subtype + a thin colour-bar column
    fig, axes = plt.subplots(1, n + 1, figsize=(4 * n + 1, 5), gridspec_kw={"width_ratios": [1] * n + [0.08]})
    colorbar_ax, axes = axes[-1], axes[:-1]
    for k, (ax, title) in enumerate(zip(axes, subtype_titles), start=1):
        people = (table["ml_subtype"] == k) & (table["ml_stage"] > 0)      # this subtype, tau+ only
        means = z[people].groupby(stage_group[people], observed=False)[regions].mean().T  # regions x groups
        counts = stage_group[people].value_counts().reindex(labels)
        image = ax.imshow(means, cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(labels)), [f"{lab}\nn={n}" for lab, n in zip(labels, counts)], fontsize=9)
        ax.set_yticks(range(len(regions)), regions if k == 1 else [])   # region names only on the left
        ax.set_title(title)
        ax.set_xlabel("SuStaIn stage")
    fig.colorbar(image, cax=colorbar_ax, label="average z-score")
    fig.tight_layout()
    if save_path:                                         # the app shows the figure without saving
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def left_right_layout(regions, max_cols=6):
    """Panel order for region figures: a row of left regions with their right partners in the row
    right below (same column), at most max_cols pairs per row; regions without a partner come last.
    Returns (list of region names, None = empty panel; number of columns)."""
    pairs = list(region_pairs(regions).values())
    paired = {region for pair in pairs for region in pair}
    others = [region for region in regions if region not in paired]
    n_cols = min(len(pairs), max_cols) if pairs else min(len(regions), 5)
    cells = []
    for start in range(0, len(pairs), n_cols):
        chunk = pairs[start:start + n_cols]
        empty = [None] * (n_cols - len(chunk))
        cells += [left for left, _ in chunk] + empty + [right for _, right in chunk] + empty
    return cells + others, n_cols


def plot_region_histograms(data, regions, xlabel="value (e.g. SUVR)"):
    """One small histogram per region column - raw values in tab 1, z-scores in tab 2.

    Left regions on top, each right region right below its left partner (same x-axis), so
    left/right differences are easy to see. Shows the "normal" bump and the tau+ tail of each
    region, and makes outliers or odd extra bumps easy to spot. xlabel is the label under each panel.
    """
    values = data[list(regions)].apply(pd.to_numeric, errors="coerce")   # text becomes missing
    cells, n_cols = left_right_layout(list(regions))
    n_rows = max(1, int(np.ceil(len(cells) / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.4 * n_rows), squeeze=False)
    pair_rows = 2 * int(np.ceil(len(region_pairs(list(regions))) / n_cols))
    for col in range(n_cols):                             # left and right partner: same x-axis
        for row in range(1, pair_rows, 2):
            axes[row, col].sharex(axes[row - 1, col])
    for ax, region in zip(axes.flat, cells + [None] * (axes.size - len(cells))):
        if region is None:
            ax.set_visible(False)                         # empty panel
            continue
        ax.hist(values[region].dropna(), bins=40, color="#4C72B0")
        ax.set_title(region, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
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
        ax.text(cut, ax.get_ylim()[1] * 0.95, f" z={z:g}", fontsize=9, va="top")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))                    # one legend entry per label
    ax.legend(unique.values(), unique.keys())
    ax.set_title(f"{region}: normal mean {normal_mean:.3f}, normal SD {normal_sd:.3f}")
    ax.set_xlabel("value (e.g. SUVR)")
    ax.set_ylabel("density")
    fig.tight_layout()
    return fig


def plot_subtype_patterns(z, table, regions, subtype_titles, vmax=12):
    """Average z-score per region in each subtype, tau+ people only (paper Fig. 1a/c, as a heatmap).

    Columns = subtypes (n = number of tau+ people), rows = regions, number in each cell = average z.
    """
    tau_pos = table["ml_stage"] > 0
    means = z[tau_pos].groupby(table.loc[tau_pos, "ml_subtype"])[regions].mean().T   # regions x subtypes
    counts = table.loc[tau_pos, "ml_subtype"].value_counts()
    fig, ax = plt.subplots(figsize=(1.8 * len(means.columns) + 3, 0.45 * len(regions) + 1.8))
    image = ax.imshow(means, cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(means.columns)), [f"{subtype_titles[k - 1]}\nn={counts[k]}" for k in means.columns])
    ax.set_yticks(range(len(regions)), regions)
    for (row, col), value in np.ndenumerate(means.to_numpy()):
        ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=8,
                color="white" if value > 0.6 * vmax else "black")
    fig.colorbar(image, ax=ax, label="average z-score")
    ax.set_title("Average tau (z-score) per subtype - tau+ people (stage > 0)")
    fig.tight_layout()
    return fig


def subtype_colors(k):
    """One fixed colour per subtype (Subtype 1 = first colour), used by all tab-4 figures."""
    return plt.cm.tab10(np.arange(k) % 10)


def plot_subtype_probabilities(prob_subtype, table, subtype_titles, threshold):
    """Left: every tau+ person inside a polygon with one subtype at each corner (paper Fig. 1b).
    Close to a corner = surely that subtype; near the middle = a mix. Right: histogram of each
    person's highest subtype probability; left of the dashed line = poor fit (paper Ext. Fig. 2).

    prob_subtype: people x subtypes, columns already in the app's order (Subtype 1 first).
    """
    k = prob_subtype.shape[1]
    tau_pos = (table["ml_stage"] > 0).to_numpy()
    colors = subtype_colors(k)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    angles = np.pi / 2 + 2 * np.pi * np.arange(k) / k
    corners = np.column_stack([np.cos(angles), np.sin(angles)])
    xy = prob_subtype[tau_pos] @ corners                  # weighted average of the corners
    ax1.scatter(xy[:, 0], xy[:, 1], c=colors[table["ml_subtype"].to_numpy()[tau_pos] - 1], s=14, alpha=0.6)
    ax1.plot(*np.vstack([corners, corners[:1]]).T, color="gray", lw=1)
    for (x, y), title in zip(corners, subtype_titles):
        ax1.text(1.15 * x, 1.15 * y, title, ha="center", va="center", fontsize=11)
    ax1.set_xlim(-1.4, 1.4)
    ax1.set_ylim(-1.4, 1.4)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title("Subtype probabilities of tau+ people")
    best = table.loc[tau_pos, "prob_subtype"]
    ax2.hist(best, bins=np.linspace(0, 1, 41), color="#4C72B0")
    ax2.axvline(threshold, ls="--", color="black")
    ax2.set_xlabel("highest subtype probability")
    ax2.set_ylabel("people (tau+)")
    ax2.set_title(f"Poor fit (< {threshold:g}): {(best < threshold).sum()} of {len(best)} tau+ people")
    fig.tight_layout()
    return fig


def plot_stages(table, prob_stage, subtype_titles):
    """Left: how many people are at each SuStaIn stage, per subtype (stage 0 = S0, paper Ext. Fig. 2).
    Right: how sure SuStaIn is about each person's stage - the spread (SD) of the stage
    probabilities against the most likely stage. Low = sure; high = the stage could be several values.
    """
    k = len(subtype_titles)
    colors = subtype_colors(k)
    stages = np.arange(prob_stage.shape[1])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    bottom = np.zeros(len(stages))
    s0 = (table["ml_stage"] == 0).sum()
    ax1.bar(0, s0, color="lightgray", label=f"S0 (n={s0})")
    for n, title in enumerate(subtype_titles, start=1):
        counts = table.loc[(table["ml_subtype"] == n) & (table["ml_stage"] > 0), "ml_stage"]
        heights = counts.value_counts().reindex(stages, fill_value=0).to_numpy()
        ax1.bar(stages, heights, bottom=bottom, color=colors[n - 1], label=f"{title} (n={len(counts)})")
        bottom += heights
    ax1.set_xlabel("most likely SuStaIn stage")
    ax1.set_ylabel("people")
    ax1.legend(fontsize=8)
    prob_stage = prob_stage / prob_stage.sum(axis=1, keepdims=True)      # each row sums to 1
    mean = prob_stage @ stages
    spread = np.sqrt(prob_stage @ stages ** 2 - mean ** 2)
    tau_pos = (table["ml_stage"] > 0).to_numpy()
    ax2.scatter(table["ml_stage"][tau_pos], spread[tau_pos], s=12, alpha=0.6,
                c=colors[table["ml_subtype"].to_numpy()[tau_pos] - 1])
    ax2.set_xlabel("most likely SuStaIn stage")
    ax2.set_ylabel("uncertainty (SD of stage, in stages)")
    ax2.set_title("How sure is the stage? (tau+ people)")
    fig.tight_layout()
    return fig


def plot_model_choice(likelihoods, cross_validation=None):
    """How many subtypes? Log-likelihood (rises with every extra subtype) and, after cross-validation,
    CVIC (lower = better; the paper chose the number of subtypes by CVIC) - paper Ext. Fig. 1c-e."""
    n_panels = 1 if cross_validation is None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4), squeeze=False)
    ax = axes[0, 0]
    ax.plot(likelihoods.index, likelihoods["mean log-likelihood"], "o-")
    ax.set_xlabel("number of subtypes")
    ax.set_ylabel("mean log-likelihood (MCMC)")
    ax.set_xticks(likelihoods.index)
    if cross_validation is not None:
        ax = axes[0, 1]
        cvic = cross_validation["CVIC"]
        ax.plot(cvic.index, cvic, "o-", color="#C62828")
        ax.plot(cvic.idxmin(), cvic.min(), "*", ms=16, color="#C62828", label=f"lowest: {cvic.idxmin()}")
        ax.set_xlabel("number of subtypes")
        ax.set_ylabel("CVIC (lower = better)")
        ax.set_xticks(cvic.index)
        ax.legend()
    fig.tight_layout()
    return fig


def plot_subtype_tree(subtypes_by_k):
    """How people move between subtypes when one more subtype is allowed (paper Ext. Fig. 1f).

    subtypes_by_k: {k: Series of subtype numbers per person (0 = S0)}, same people for every k.
    Each column = one model; boxes = subtypes (height = people); bands = people moving from a subtype
    in model k to a subtype in model k+1. S0 (no tau) is left out.
    """
    ks = sorted(subtypes_by_k)
    fig, ax = plt.subplots(figsize=(2.6 * len(ks) + 1, 5.5))
    n_people = max((s > 0).sum() for s in subtypes_by_k.values())
    gap = 0.03 * n_people
    tops = {}                                             # (k, subtype) -> y of the box top
    for x, k in enumerate(ks):
        counts = subtypes_by_k[k][subtypes_by_k[k] > 0].value_counts().sort_index()
        y = 0
        for subtype, count in counts.items():
            ax.add_patch(plt.Rectangle((x - 0.08, y), 0.16, count, color=subtype_colors(k)[subtype - 1]))
            ax.text(x, y + count / 2, str(subtype), ha="center", va="center", fontsize=9, color="white")
            tops[(k, subtype)] = y
            y += count + gap
    for x, (k1, k2) in enumerate(zip(ks[:-1], ks[1:])):
        both = (subtypes_by_k[k1] > 0) & (subtypes_by_k[k2] > 0)
        flows = pd.crosstab(subtypes_by_k[k1][both], subtypes_by_k[k2][both])
        left = {s: tops[(k1, s)] for s in flows.index}
        right = {s: tops[(k2, s)] for s in flows.columns}
        for a in flows.index:
            for b in flows.columns:
                n = flows.loc[a, b]
                if n == 0:
                    continue
                ax.fill_between([x + 0.08, x + 0.92], [left[a], right[b]], [left[a] + n, right[b] + n],
                                color=subtype_colors(k1)[a - 1], alpha=0.35, lw=0)
                left[a] += n
                right[b] += n
    ax.set_xticks(range(len(ks)), [f"{k} subtype(s)" for k in ks])
    ax.set_xlim(-0.5, len(ks) - 0.5)
    ax.set_ylim(0, None)
    ax.set_yticks([])
    ax.set_ylabel("tau+ people")
    ax.set_title("Subtype tree: where do people go when one more subtype is allowed?")
    fig.tight_layout()
    return fig


def plot_person(person, z_person, regions, expected, prob_subtype, prob_stage, subtype_titles):
    """One person's "fingerprint": their z-scores per region (bars) next to the average of the people
    in the same subtype and stage group (dots), plus their subtype and stage probabilities."""
    k = len(subtype_titles)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5), gridspec_kw={"width_ratios": [2.2, 1, 1.6]})
    ax1.bar(range(len(regions)), z_person[regions], color="#4C72B0", label=person)
    if expected is not None:
        ax1.plot(range(len(regions)), expected[regions], "o", color="black", label="average of subtype + stage group")
    ax1.set_xticks(range(len(regions)), regions, rotation=45, ha="right")
    ax1.set_ylabel("z-score")
    ax1.legend(fontsize=8)
    ax2.bar(range(k), prob_subtype, color=subtype_colors(k))
    ax2.set_xticks(range(k), subtype_titles, rotation=45, ha="right")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("subtype probability")
    ax3.bar(range(len(prob_stage)), prob_stage / prob_stage.sum(), color="gray")
    ax3.set_xlabel("SuStaIn stage")
    ax3.set_ylabel("stage probability")
    fig.tight_layout()
    return fig


def plot_regional_contrasts(t_values, p_fdr, subtype_titles):
    """Paper Fig. 1d: t-value per region and subtype (this subtype vs all other tau+ people, same
    stage). Red = more tau than the others, blue = less. * = FDR-corrected p < 0.05."""
    limit = max(np.nanmax(np.abs(t_values.to_numpy())), 1)
    fig, ax = plt.subplots(figsize=(1.8 * t_values.shape[1] + 3, 0.45 * t_values.shape[0] + 1.8))
    image = ax.imshow(t_values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(t_values.shape[1]), [subtype_titles[k - 1] for k in t_values.columns])
    ax.set_yticks(range(t_values.shape[0]), t_values.index)
    for (row, col), t in np.ndenumerate(t_values.to_numpy()):
        star = "*" if p_fdr.iloc[row, col] < 0.05 else ""
        ax.text(col, row, f"{t:.1f}{star}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="t (this subtype vs other tau+, adjusted for stage)")
    ax.set_title("Where does each subtype differ? (* = FDR p < 0.05)")
    fig.tight_layout()
    return fig


def plot_stage_vs(table, values, label, subtype_titles):
    """Paper Fig. 3a/b: SuStaIn stage against a number (age, MMSE, ...) for tau+ people,
    coloured by subtype, with one straight line per subtype."""
    numeric = pd.to_numeric(values, errors="coerce")
    colors = subtype_colors(len(subtype_titles))
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, title in enumerate(subtype_titles, start=1):
        people = (table["ml_subtype"] == k) & (table["ml_stage"] > 0) & numeric.notna()
        x, y = table.loc[people, "ml_stage"], numeric[people]
        ax.scatter(x, y, s=14, alpha=0.6, color=colors[k - 1], label=f"{title} (n={people.sum()})")
        if people.sum() > 2 and x.nunique() > 1:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.array([x.min(), x.max()])
            ax.plot(xs, intercept + slope * xs, color=colors[k - 1], lw=2)
    ax.set_xlabel("SuStaIn stage")
    ax.set_ylabel(label)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_asymmetry(table, asymmetry, subtype_titles):
    """Paper Ext. Fig. 9: left-right asymmetry of tau per subtype (box plot) and against stage.
    Above 0 = more tau on the left, below 0 = more on the right."""
    colors = subtype_colors(len(subtype_titles))
    tau_pos = table["ml_stage"] > 0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    data = [asymmetry[tau_pos & (table["ml_subtype"] == k)] for k in range(1, len(subtype_titles) + 1)]
    boxes = ax1.boxplot(data, patch_artist=True)
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
    ax1.set_xticks(range(1, len(subtype_titles) + 1), subtype_titles, rotation=30, ha="right")
    ax1.axhline(0, color="gray", lw=1)
    ax1.set_ylabel("asymmetry (left - right)")
    ax2.scatter(table.loc[tau_pos, "ml_stage"], asymmetry[tau_pos], s=12, alpha=0.6,
                c=colors[table.loc[tau_pos, "ml_subtype"].to_numpy() - 1])
    ax2.axhline(0, color="gray", lw=1)
    ax2.set_xlabel("SuStaIn stage")
    ax2.set_ylabel("asymmetry (left - right)")
    fig.tight_layout()
    return fig
