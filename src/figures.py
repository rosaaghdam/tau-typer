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
    save_path is given without ".png" (None = do not save).
    subtype_order: which of pySuStaIn's subtypes to draw first, second, ... (None = pySuStaIn's choice).
    """
    k = len(subtype_titles)
    n_rows, n_cols = (1, k) if k < 3 else (2, int(np.ceil(k / 2)))     # pySuStaIn's own grid (up to 6)
    n_events = int((Z_vals > 0).sum())
    panel_height = 7 / n_cols * len(regions) / n_events + 1.2          # cells are square in the PVD
    figs, _ = ZscoreSustain.plot_positional_var(
        result["samples_sequence"], result["samples_f"], np.inf, Z_vals,
        biomarker_labels=regions, subtype_titles=subtype_titles,
        figsize=(14, n_rows * panel_height), title_font_size=13, label_font_size=10, stage_font_size=9,
        stage_interval=stage_interval, subtype_order=subtype_order)
    fig = figs[0]
    for ax in fig.axes:
        ax.set_xlabel("")                                       # columns = stages (said in the app's help text)
    fig.subplots_adjust(hspace=0.35, wspace=0.08)                     # room for titles and stage labels
    # one legend for the colours (pySuStaIn's colours: 1st cut-point red, 2nd magenta, 3rd blue, ...)
    colours = ["red", "magenta", "blue", "#8000ff", "cyan", "#00ff80"]
    levels = np.unique(Z_vals[Z_vals > 0])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours[:len(levels)]]
    fig.legend(handles, [f"z-score {z:g}" for z in levels], loc="lower center", ncol=len(levels),
               bbox_to_anchor=(0.5, -0.1 / fig.get_figheight()), frameon=False, fontsize=11)
    if save_path is not None:
        fig.savefig(f"{save_path}.png", dpi=150, bbox_inches="tight")
    return fig


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
    fig.tight_layout()
    return fig


def subtype_colors(k):
    """One fixed colour per subtype (Subtype 1 = first colour), used by all tab-4 figures."""
    return plt.cm.tab10(np.arange(k) % 10)


def plot_subtype_probabilities(prob_subtype, table, subtype_titles, seed=0):
    """Every tau+ person inside a polygon with one subtype at each corner (paper Fig. 1b).
    Close to a corner = surely that subtype; towards the middle = a mix of subtypes.
    Corners: number (and %) of tau+ people in each subtype. Dashed box (as in the paper): the middle
    of the polygon, half its size; the legend gives the % of people inside and outside it.
    A little random jitter spreads the many people who sit exactly on a corner.

    prob_subtype: people x subtypes, columns already in the app's order (Subtype 1 first).
    """
    k = prob_subtype.shape[1]
    tau_pos = (table["ml_stage"] > 0).to_numpy()
    n_tau = tau_pos.sum()
    subtype = table["ml_subtype"].to_numpy()[tau_pos]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    angles = np.pi / 2 + 2 * np.pi * np.arange(k) / k
    corners = np.column_stack([np.cos(angles), np.sin(angles)])
    xy = prob_subtype[tau_pos] @ corners                  # weighted average of the corners
    inside = (np.abs(xy) <= 0.5).all(axis=1)             # inside the dashed box (before the jitter)
    xy = xy + np.random.default_rng(seed).normal(0, 0.02, xy.shape)
    ax.plot(*np.vstack([corners, corners[:1]]).T, color="gray", lw=2, zorder=1)
    ax.scatter(xy[:, 0], xy[:, 1], c=subtype_colors(k)[subtype - 1], s=30, edgecolors="black", linewidths=0.5,
               zorder=2)
    box = ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, fill=False, ls="--", lw=2, color="black", zorder=3))
    for n, ((x, y), title) in enumerate(zip(corners, subtype_titles), start=1):
        count = (subtype == n).sum()
        side = np.sign(round(x, 6)) * 0.3                   # side corners: label away from the dots
        ax.text(1.12 * x + side, 1.17 * y, f"{title}\nn = {count} ({100 * count / n_tau:.0f}%)",
                ha="center", va="center", fontsize=10, color=subtype_colors(k)[n - 1], fontweight="bold")
    ax.legend([box], [f"Inside: {100 * inside.mean():.0f}%\nOutside: {100 * (1 - inside.mean()):.0f}%"],
              loc="upper right", bbox_to_anchor=(1, 0.92), frameon=False, handlelength=1.5, handleheight=2.5)
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_stages(table, prob_stage, subtype_titles):
    """Left: how many tau+ people are at each SuStaIn stage, per subtype (paper Ext. Fig. 2); stage 0
    (S0) is only counted in the legend, so its large bar does not squash the others.
    Right: how sure SuStaIn is about each person's stage - the spread (SD) of the stage
    probabilities against the most likely stage. Low = sure; high = the stage could be several values.
    """
    k = len(subtype_titles)
    colors = subtype_colors(k)
    stages = np.arange(prob_stage.shape[1])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    bottom = np.zeros(len(stages))
    s0 = (table["ml_stage"] == 0).sum()
    for n, title in enumerate(subtype_titles, start=1):
        counts = table.loc[(table["ml_subtype"] == n) & (table["ml_stage"] > 0), "ml_stage"]
        heights = counts.value_counts().reindex(stages, fill_value=0).to_numpy()
        ax1.bar(stages, heights, bottom=bottom, color=colors[n - 1], label=f"{title} (n={len(counts)})")
        bottom += heights
    ax1.set_xlabel("most likely SuStaIn stage")
    ax1.set_ylabel("people (tau+)")
    ax1.set_xlim(0.4, stages[-1] + 0.6)
    ax1.bar(0, 0, color="white", label=f"S0 (no abnormal tau): {s0} ({100 * s0 / len(table):.0f}%), not shown")
    ax1.legend(fontsize=8)
    prob_stage = prob_stage / prob_stage.sum(axis=1, keepdims=True)      # each row sums to 1
    mean = prob_stage @ stages
    spread = np.sqrt(prob_stage @ stages ** 2 - mean ** 2)
    tau_pos = (table["ml_stage"] > 0).to_numpy()
    ax2.scatter(table["ml_stage"][tau_pos], spread[tau_pos], s=12, alpha=0.6,
                c=colors[table["ml_subtype"].to_numpy()[tau_pos] - 1])
    ax2.set_xlabel("most likely SuStaIn stage")
    ax2.set_ylabel("uncertainty (SD of stage, in stages)")
    fig.tight_layout()
    return fig


def plot_model_choice(likelihoods, cross_validation=None, traces=None):
    """How many subtypes? Log-likelihood (rises with every extra subtype) and, after cross-validation,
    CVIC (lower = better; the paper chose the number of subtypes by CVIC) - paper Ext. Fig. 1c-e.
    traces {k: MCMC log-likelihoods}: extra panel, one line per model; flat = the MCMC is stable."""
    n_panels = 1 + (cross_validation is not None) + (traces is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 3.2), squeeze=False)
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
    if traces is not None:
        ax = axes[0, -1]
        for k, trace in traces.items():
            ax.plot(trace, lw=1, label=f"{k} subtype{'s' if k > 1 else ''}")
        ax.set_xlabel("MCMC sample")
        ax.set_ylabel("log-likelihood")
        ax.set_title("MCMC per model (flat = stable)", fontsize=10)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_subtype_tree(subtypes_by_k):
    """How people move between subtypes when one more subtype is allowed (paper Ext. Fig. 1f).

    subtypes_by_k: {k: Series of subtype numbers per person (0 = S0)}, same people for every k.
    Each column = one model; boxes = subtypes (height = people); bands = people moving from a subtype
    in model k to a subtype in model k+1, labelled with the % of the subtype that goes there.
    S0 (no tau) is left out.
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
            ax.add_patch(plt.Rectangle((x - 0.13, y), 0.26, count, color=subtype_colors(k)[subtype - 1]))
            ax.text(x, y + count / 2, f"{subtype}\n{count} ({100 * count / counts.sum():.0f}%)",
                    ha="center", va="center", fontsize=8, color="white")
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
                ax.fill_between([x + 0.13, x + 0.87], [left[a], right[b]], [left[a] + n, right[b] + n],
                                color=subtype_colors(k1)[a - 1], alpha=0.35, lw=0)
                # % of subtype a (model k1) that goes to subtype b (model k2), at the start of the band
                ax.text(x + 0.15, left[a] + n / 2, f"→ {b}: {100 * n / flows.loc[a].sum():.0f}%",
                        ha="left", va="center", fontsize=8)
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


DIAGNOSIS_COLORS = {"CN": "#11A0D9", "CU": "#11A0D9", "HC": "#11A0D9", "control": "#11A0D9",
                    "SCD": "#9491D9", "MCI": "#F2B705", "AD": "#F24405", "dementia": "#F24405"}
OTHER_COLORS = ["#3F8C61", "#8C2E62", "#332288", "#44AA99", "#DDCC77"]


def group_colors(groups):
    """One fixed color per diagnosis: blue for healthy, red for AD; other labels get the next free color."""
    colors = {}
    extra = 0
    for g in groups:
        if g in DIAGNOSIS_COLORS:
            colors[g] = DIAGNOSIS_COLORS[g]
        else:
            colors[g] = OTHER_COLORS[extra % len(OTHER_COLORS)]
            extra += 1
    return colors


def plot_covariate_boxplots(data, numbers, group_col=None, groups=None, n_cols=4):
    """One box plot per number column (MMSE, age, ...), one colored box per group, with every person as a dot."""
    n_rows = int(np.ceil(len(numbers) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 2.8 * n_rows), squeeze=False)
    labels = groups if group_col else ["All"]
    colors = group_colors(labels) if group_col else {"All": "#11A0D9"}
    rng = np.random.default_rng(0)                                 # same dot positions every time
    for ax, col in zip(axes.flat, numbers):
        values = pd.to_numeric(data[col], errors="coerce")
        for i, g in enumerate(labels, start=1):
            in_group = data[group_col].astype(str) == g if group_col else values.notna()
            group_values = values[in_group].dropna()
            if len(group_values) == 0:
                continue
            box = ax.boxplot([group_values], positions=[i], widths=0.6, patch_artist=True, showfliers=False)
            box["boxes"][0].set(facecolor=colors[g], alpha=0.35)
            box["medians"][0].set(color="black")
            x = i + rng.uniform(-0.2, 0.2, len(group_values))
            ax.scatter(x, group_values, s=6, color=colors[g], alpha=0.7)
        ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=8)
        ax.set_title(col, fontsize=10)
    for ax in axes.flat[len(numbers):]:
        ax.set_visible(False)
    if group_col:                                                  # one legend for all panels
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[g]) for g in labels]
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    return fig


def plot_category_bars(data, categories, group_col=None, groups=None, n_cols=4):
    """One bar chart per category column (sex, APOE4, ...): % of each value, one bar per group
    (same colors as the box plots, so no legend here)."""
    n_rows = int(np.ceil(len(categories) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 2.8 * n_rows), squeeze=False)
    for ax, col in zip(axes.flat, categories):
        values = data[col].dropna()
        values = values.map(lambda v: f"{v:g}" if isinstance(v, float) else str(v))
        if group_col:
            group = data.loc[values.index, group_col].astype(str)
            table = pd.crosstab(values, group, normalize="columns")[groups] * 100
            colors = [group_colors(groups)[g] for g in groups]
        else:
            table = values.value_counts(normalize=True).to_frame("All") * 100
            colors = ["#11A0D9"]
        table.plot.bar(ax=ax, legend=False, rot=0, color=colors)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("%")
    for ax in axes.flat[len(categories):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_correlations(data, numbers):
    """Spearman correlation between all number columns. Red = rise together, blue = one falls."""
    values = data[numbers].apply(pd.to_numeric, errors="coerce")
    corr = values.corr(method="spearman")
    size = 0.45 * len(numbers) + 3
    fig, ax = plt.subplots(figsize=(size + 1, size))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numbers)), numbers, rotation=90, fontsize=8)
    ax.set_yticks(range(len(numbers)), numbers, fontsize=8)
    for (row, col), r in np.ndenumerate(corr.to_numpy()):
        ax.text(col, row, f"{r:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="Spearman r")
    fig.tight_layout()
    return fig
