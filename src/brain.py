"""Brain figures: tau per subtype (and per stage group) drawn on the cortex, with the outlines of the
Desikan-Killiany atlas from the ggseg package (python-ggseg, the Python port of the ggseg brain plots used
in many papers). Lateral and medial views of both hemispheres.

Our regions are matched by name. A region named after a lobe (e.g. L_frontal, R_MTL) colours every atlas
region of that lobe; a region named after an atlas region (e.g. L_entorhinal) colours just that one.
Left or right from the start of the name (L_, R_, lh_, rh_, left_, right_).
"""
import functools
import os.path as op
import re

import ggseg
import matplotlib.pyplot as plt
import numpy as np
from ggseg import _svg_parse_
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

ATLAS = op.join(op.dirname(ggseg.__file__), "data", "dk")
LOBES = {    # Desikan-Killiany regions of each lobe (cingulate and insula belong to none)
    "frontal": ["superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal", "parsopercularis",
                "parstriangularis", "parsorbitalis", "lateralorbitofrontal", "medialorbitofrontal", "precentral",
                "paracentral"],
    "parietal": ["superiorparietal", "inferiorparietal", "supramarginal", "postcentral", "precuneus"],
    "temporal": ["superiortemporal", "middletemporal", "inferiortemporal", "bankssts", "fusiform",
                 "transversetemporal"],
    "occipital": ["lateraloccipital", "lingual", "cuneus", "pericalcarine"],
    "mtl": ["entorhinal", "parahippocampal"],
}
LOBE_WORDS = {"mtl": ["mtl", "medial_temporal", "medialtemporal", "hippocamp"], "temporal": ["temporal"],
              "parietal": ["parietal"], "occipital": ["occipital"], "frontal": ["frontal"]}
ATLAS_REGIONS = sorted({r for regions in LOBES.values() for r in regions} |
                       {"caudalanteriorcingulate", "rostralanteriorcingulate", "posteriorcingulate",
                        "isthmuscingulate", "insula"})
NO_DATA = "#E4E7EB"


def side_of(region):
    """"L", "R" or None (both) from the start of a region name."""
    match = re.match(r"^(l|lh|left|r|rh|right)[_\-.]", region.lower())
    return None if not match else ("L" if match.group(1).startswith("l") else "R")


def atlas_regions_of(region):
    """The atlas regions one of our regions stands for: itself if it is named after one (entorhinal,
    precuneus, ...), else every region of its lobe; [] if the name matches nothing."""
    name = region.lower().replace("-", "").replace(" ", "")
    for atlas_region in sorted(ATLAS_REGIONS, key=len, reverse=True):     # longest name first
        if atlas_region in name:
            return [atlas_region]
    for lobe, words in LOBE_WORDS.items():                                # MTL before temporal
        if any(word in name for word in words):
            return LOBES[lobe]
    return []


def lobe_of(region):
    """True-ish if the brain figure can place this region (kept for the app's check)."""
    return bool(atlas_regions_of(region))


def atlas_values(means):
    """{(side, atlas region): value} from {our region: value}; a region without a side counts for both."""
    found = {}
    for region, value in means.items():
        side = side_of(region)
        for atlas_region in atlas_regions_of(region):
            for s in ([side] if side else ["L", "R"]):
                found.setdefault((s, atlas_region), []).append(value)
    return {key: float(np.mean(v)) for key, v in found.items()}


@functools.lru_cache(maxsize=None)
def outline(name):
    """The matplotlib Path of one atlas file (a region, or a whole view such as lateral_left)."""
    codes, verts = _svg_parse_(open(op.join(ATLAS, name)).read())
    return Path(verts, codes)


@functools.lru_cache(maxsize=None)
def view_box(view):
    """(x0, y0, x1, y1) of one view (lateral_left, medial_left, lateral_right, medial_right) in atlas units."""
    path = outline(view)
    points = path.vertices[path.codes != Path.CLOSEPOLY]
    return (*points.min(axis=0), *points.max(axis=0))


def draw_view(ax, view, values, x, y, cmap, vmax):
    """One view of one hemisphere with its top-left corner at (x, y) in the axes (y grows down), each atlas
    region coloured by values[(side, region)]. Returns the view's width and height."""
    side = "L" if view.endswith("left") else "R"
    x0, y0, x1, y1 = view_box(view)
    move = Affine2D().translate(x - x0, y - y0) + ax.transData
    clip = Rectangle((x0 - 2, y0 - 2), x1 - x0 + 4, y1 - y0 + 4, transform=move, visible=False)
    ax.add_patch(clip)
    colours = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    background = PathPatch(outline(view), facecolor="#C9CED6", edgecolor="white", lw=0.4, transform=move)
    ax.add_patch(background)
    background.set_clip_path(clip)
    hemisphere = "left" if side == "L" else "right"
    for region in ATLAS_REGIONS:
        value = values.get((side, region))
        face = NO_DATA if value is None else colours(float(np.clip(value / vmax, 0, 1)))
        patch = PathPatch(outline(f"{region}_{hemisphere}"), facecolor=face, edgecolor="white", lw=0.4,
                          transform=move)
        ax.add_patch(patch)
        patch.set_clip_path(clip)
    return x1 - x0, y1 - y0


def draw_brain(ax, values, views, cmap, vmax, gap=40):
    """Several views side by side in one axes (e.g. lateral and medial of both hemispheres)."""
    x, height = 0, 0
    for view in views:
        width, h = draw_view(ax, view, values, x, 0, cmap, vmax)
        x, height = x + width + gap, max(height, h)
    ax.set_xlim(-gap / 2, x - gap / 2)
    ax.set_ylim(height + gap / 2, -gap / 2)                     # atlas y grows downwards
    ax.set_aspect("equal")
    ax.axis("off")


ALL_VIEWS = ["lateral_left", "medial_left", "medial_right", "lateral_right"]


def colour_bar(fig, axes, cmap, vmax, label):
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, vmax), cmap=cmap), ax=axes, fraction=0.025,
                       pad=0.02, shrink=0.8)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0)
    bar.set_label(label)


def plot_brain_map(z, table, regions, subtype_titles, vmax=12, cmap="Reds"):
    """Average z-score of each subtype's tau+ people on the cortex: one row per subtype, lateral and medial
    views of the left and right hemisphere. Grey = no region of the data there. None if no region name
    matches the atlas."""
    if not any(atlas_regions_of(r) for r in regions):
        return None
    tau_pos = table["ml_stage"] > 0
    means = z[tau_pos].groupby(table.loc[tau_pos, "ml_subtype"])[regions].mean()   # subtypes x regions
    counts = table.loc[tau_pos, "ml_subtype"].value_counts()
    k = len(subtype_titles)
    fig, axes = plt.subplots(k, 1, figsize=(8.5, 1.45 * k + 0.5), squeeze=False)
    for row, (ax, title) in enumerate(zip(axes[:, 0], subtype_titles), start=1):
        draw_brain(ax, atlas_values(means.loc[row]) if row in means.index else {}, ALL_VIEWS, cmap, vmax)
        ax.set_title(f"{title} (n = {counts.get(row, 0)})", loc="left", fontsize=9)
    for position, text in [(0.125, "left, lateral"), (0.375, "left, medial"), (0.625, "right, medial"),
                           (0.875, "right, lateral")]:                # under the last row
        axes[-1, 0].text(position, -0.08, text, transform=axes[-1, 0].transAxes, ha="center", fontsize=8,
                         color="#52606D")
    colour_bar(fig, axes.ravel().tolist(), cmap, vmax, "average z-score (tau+ people)")
    return fig


def plot_brain_stages(z, table, regions, subtype_titles, stage_edges, vmax=12, cmap="Reds"):
    """How tau spreads with stage, on the brain: one row per subtype, one column per stage group (e.g.
    1-6, 7-12, ...), each cell the average z-score of the tau+ people of that subtype and stage group on
    the left hemisphere (lateral above, medial below). n = people in the cell; empty cell = nobody."""
    import pandas as pd
    if not any(atlas_regions_of(r) for r in regions):
        return None
    labels = [f"{a}-{b - 1}" for a, b in zip(stage_edges[:-1], stage_edges[1:])]
    group = pd.cut(table["ml_stage"], stage_edges, right=False, labels=labels)
    k, g = len(subtype_titles), len(labels)
    fig, axes = plt.subplots(k, g, figsize=(1.75 * g + 1.6, 1.6 * k + 0.6), squeeze=False)
    for row, title in enumerate(subtype_titles, start=1):
        people = (table["ml_subtype"] == row) & (table["ml_stage"] > 0)
        for col, label in enumerate(labels):
            ax = axes[row - 1, col]
            members = people & (group == label)
            if members.sum() == 0:
                ax.axis("off")
                ax.text(0.5, 0.5, "nobody", ha="center", va="center", fontsize=8, color="#9AA5B1",
                        transform=ax.transAxes)
            else:
                values = atlas_values(z[members][regions].mean())
                width, height = draw_view(ax, "lateral_left", values, 0, 0, cmap, vmax)
                draw_view(ax, "medial_left", values, 0, height + 30, cmap, vmax)
                medial = view_box("medial_left")
                ax.set_xlim(-20, max(width, medial[2] - medial[0]) + 20)
                ax.set_ylim(height + 30 + medial[3] - medial[1] + 20, -20)
                ax.set_aspect("equal")
                ax.axis("off")
                ax.text(0.5, -0.04, f"n={int(members.sum())}", transform=ax.transAxes, ha="center", va="top",
                        fontsize=7, color="#52606D")
            if row == 1:
                ax.set_title(f"stages {label}", fontsize=9)
        axes[row - 1, 0].text(-0.12, 0.5, title, transform=axes[row - 1, 0].transAxes, ha="right", va="center",
                              fontsize=9, fontweight="bold")
    colour_bar(fig, axes.ravel().tolist(), cmap, vmax, "average z-score")
    fig.text(0.5, 0.0, "left hemisphere: lateral view above, medial view below", ha="center", fontsize=8,
             color="#52606D")
    return fig
