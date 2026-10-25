"""A schematic brain map: tau per subtype drawn on the lobes of a simple side view of the brain.

Only a drawing to see the pattern at a glance - not anatomical. Regions are matched to lobes by their
names (e.g. L_frontal, R_MTL): frontal, parietal, occipital, temporal and MTL (medial temporal lobe:
hippocampus, entorhinal, amygdala), left or right by the start of the name (L_, R_, lh_, rh_, left_, right_).
"""
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Polygon
from scipy.interpolate import splev, splprep

LOBE_WORDS = {"mtl": ["mtl", "medial_temporal", "medialtemporal", "hippocamp", "entorhinal", "amygdala",
                      "parahippocamp"],
              "temporal": ["temporal"], "parietal": ["parietal"], "occipital": ["occipital"],
              "frontal": ["frontal"]}

# left hemisphere seen from the side, front (frontal pole) on the left; x 0-1, y 0-0.72
OUTLINE = [(0.02, 0.35), (0.06, 0.52), (0.18, 0.64), (0.36, 0.71), (0.56, 0.71), (0.74, 0.65), (0.88, 0.55),
           (0.97, 0.43), (0.99, 0.31), (0.94, 0.20), (0.84, 0.15), (0.72, 0.12), (0.60, 0.05), (0.45, 0.02),
           (0.32, 0.05), (0.24, 0.12), (0.19, 0.20), (0.10, 0.22), (0.04, 0.27)]
LOBES = {"frontal": [(-0.1, 0.20), (0.17, 0.22), (0.24, 0.31), (0.44, 0.37), (0.52, 0.8), (-0.1, 0.8)],
         "parietal": [(0.44, 0.37), (0.52, 0.8), (0.82, 0.8), (0.80, 0.40), (0.64, 0.37)],
         "occipital": [(0.82, 0.8), (1.1, 0.8), (1.1, -0.1), (0.78, -0.1), (0.78, 0.25), (0.80, 0.40)],
         "temporal": [(0.17, 0.22), (0.24, 0.31), (0.64, 0.37), (0.80, 0.40), (0.78, 0.25), (0.78, -0.1),
                      (0.1, -0.1), (0.1, 0.2)]}
TEXT_AT = {"frontal": (0.22, 0.50), "parietal": (0.64, 0.56), "occipital": (0.89, 0.37), "temporal": (0.62, 0.24)}
MTL_AT, MTL_SIZE = (0.40, 0.15), (0.24, 0.075)


def lobe_of(region):
    """(side "L" / "R" / None, lobe) of a region name, or None if it is not a lobe the map knows."""
    name = region.lower()
    side = None
    match = re.match(r"^(l|lh|left|r|rh|right)[_\-.]", name)
    if match:
        side = "L" if match.group(1).startswith("l") else "R"
    for lobe, words in LOBE_WORDS.items():               # MTL first: "medial_temporal" is not "temporal"
        if any(word in name for word in words):
            return side, lobe
    return None


def lobe_values(means):
    """{(side, lobe): average} from {region: value}; a region without side counts for both sides."""
    found = {}
    for region, value in means.items():
        place = lobe_of(region)
        if place is None:
            continue
        side, lobe = place
        for s in ([side] if side else ["L", "R"]):
            found.setdefault((s, lobe), []).append(value)
    return {key: float(np.mean(values)) for key, values in found.items()}


def smooth_outline():
    """The brain outline as a smooth closed curve (points x, y)."""
    points = np.array(OUTLINE + OUTLINE[:1])
    tck, _ = splprep([points[:, 0], points[:, 1]], s=0, per=True)
    return np.column_stack(splev(np.linspace(0, 1, 200), tck))


def draw_hemisphere(ax, values, side, x0, mirror, cmap, vmax):
    """One hemisphere at x offset x0 (mirror = facing right), lobes coloured by values[(side, lobe)]."""
    flip = (lambda pts: [(x0 + 1 - x, y) for x, y in pts]) if mirror else (lambda pts: [(x0 + x, y) for x, y in pts])
    outline = Polygon(flip(smooth_outline()), closed=True, facecolor="none", edgecolor="#52606D", lw=1.2, zorder=4)
    ax.add_patch(outline)
    colours = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    for lobe, corners in LOBES.items():
        value = values.get((side, lobe))
        face = "#E4E7EB" if value is None else colours(min(max(value / vmax, 0), 1))
        patch = Polygon(flip(corners), closed=True, facecolor=face, edgecolor="white", lw=1.5, zorder=2,
                        hatch="///" if value is None else None)
        ax.add_patch(patch)
        patch.set_clip_path(outline)
        if value is not None:
            (x, y), = flip([TEXT_AT[lobe]])
            ax.text(x, y, f"{value:.1f}", ha="center", va="center", fontsize=7, zorder=6,
                    color="white" if value > 0.55 * vmax else "#1F2933")
    value = values.get((side, "mtl"))
    (x, y), = flip([MTL_AT])
    ax.add_patch(Ellipse((x, y), *MTL_SIZE, facecolor="#E4E7EB" if value is None else
                         colours(min(max(value / vmax, 0), 1)), edgecolor="#1F2933", lw=1, ls="--", zorder=5))
    ax.text(x, y, "MTL" if value is None else f"MTL {value:.1f}", ha="center", va="center", fontsize=6.5, zorder=6,
            color="white" if value is not None and value > 0.55 * vmax else "#1F2933")
    ax.text(x0 + 0.5, -0.06, "left" if side == "L" else "right", ha="center", va="top", fontsize=8, color="#52606D")


def plot_brain_map(z, table, regions, subtype_titles, vmax=12, cmap="Reds"):
    """Average z-score of each subtype's tau+ people on a schematic side view of both hemispheres (left
    one facing left, right one facing right). The MTL is on the inner side of the brain, so it is drawn as
    the dashed oval. Grey hatched = no region of the data in that lobe. Returns None if no region name
    matches a lobe."""
    tau_pos = table["ml_stage"] > 0
    means = z[tau_pos].groupby(table.loc[tau_pos, "ml_subtype"])[regions].mean()   # subtypes x regions
    if not any(lobe_of(r) for r in regions):
        return None
    k = len(subtype_titles)
    n_cols = min(k, 2)
    n_rows = int(np.ceil(k / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols + 1, 2.3 * n_rows + 0.4), squeeze=False)
    counts = table.loc[tau_pos, "ml_subtype"].value_counts()
    for ax, (n, title) in zip(axes.flat, enumerate(subtype_titles, start=1)):
        values = lobe_values(means.loc[n]) if n in means.index else {}
        draw_hemisphere(ax, values, "L", 0, False, cmap, vmax)
        draw_hemisphere(ax, values, "R", 1.15, True, cmap, vmax)
        ax.set_title(f"{title} (n = {counts.get(n, 0)})")
        ax.set_xlim(-0.05, 2.2)
        ax.set_ylim(-0.12, 0.76)
        ax.set_aspect("equal")
        ax.axis("off")
    for ax in axes.flat[k:]:
        ax.set_visible(False)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, vmax), cmap=cmap), ax=axes.ravel().tolist(),
                       fraction=0.025, pad=0.02, shrink=0.8)
    bar.outline.set_visible(False)
    bar.set_label("average z-score (tau+ people)")
    return fig
