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
from matplotlib.patches import FancyArrowPatch, PathPatch, Rectangle
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


def region_points(atlas_regions, hemisphere, view):
    """The outline points of some atlas regions that lie in one view (to place a label on them)."""
    x0, y0, x1, y1 = view_box(view)
    points = [outline(f"{r}_{hemisphere}").vertices[outline(f"{r}_{hemisphere}").codes != Path.CLOSEPOLY]
              for r in atlas_regions]
    points = np.concatenate(points) if points else np.zeros((0, 2))
    inside = (points[:, 0] >= x0) & (points[:, 0] <= x1) & (points[:, 1] >= y0) & (points[:, 1] <= y1)
    return points[inside]


def draw_view(ax, view, values, x, y, cmap, vmax, labels=()):
    """One view of one hemisphere with its top-left corner at (x, y) in the axes (y grows down). Each atlas
    region is filled with values[(side, region)]; its edge has the same colour, so the atlas regions of one of
    our regions (a lobe) merge into one clear area. labels: (text, atlas regions) written on that area.
    Returns the view's width and height."""
    side = "L" if view.endswith("left") else "R"
    hemisphere = "left" if side == "L" else "right"
    x0, y0, x1, y1 = view_box(view)
    move = Affine2D().translate(x - x0, y - y0) + ax.transData
    clip = Rectangle((x0 - 2, y0 - 2), x1 - x0 + 4, y1 - y0 + 4, transform=move, visible=False)
    ax.add_patch(clip)
    colours = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    shape = PathPatch(outline(view), facecolor="#C9CED6", edgecolor="#9AA5B1", lw=2.5, transform=move)
    ax.add_patch(shape)                                      # the whole view: its outline and the medial wall
    shape.set_clip_path(clip)
    for region in ATLAS_REGIONS:
        value = values.get((side, region))
        face = NO_DATA if value is None else colours(float(np.clip(value / vmax, 0, 1)))
        patch = PathPatch(outline(f"{region}_{hemisphere}"), facecolor=face, edgecolor=face, lw=1.5,
                          transform=move)
        ax.add_patch(patch)
        patch.set_clip_path(clip)
    for text, atlas_regions, value in labels:
        points = region_points(atlas_regions, hemisphere, view)
        if len(points) <= 3:
            continue
        cx, cy = points.mean(axis=0)
        if set(atlas_regions) <= set(LOBES["mtl"]):          # small area: its value just under the view
            ax.text(cx - x0 + x, y1 - y0 + y + 28, f"MTL {text}", ha="center", va="top", fontsize=8,
                    fontweight="bold", color="#1F2933")
        else:
            ax.text(cx - x0 + x, cy - y0 + y, text, ha="center", va="center", fontsize=8, fontweight="bold",
                    color="white" if value > 0.55 * vmax else "#1F2933")
    return x1 - x0, y1 - y0


def lobe_labels(means, regions, side, view):
    """Labels (value, atlas regions, value) for our regions of one side that are best seen in this view."""
    hemisphere = "left" if side == "L" else "right"
    other = view.replace("lateral", "medial") if view.startswith("lateral") else view.replace("medial", "lateral")
    labels = []
    for region in regions:
        atlas_regions = atlas_regions_of(region)
        if side_of(region) not in (side, None) or not atlas_regions or np.isnan(means[region]):
            continue
        here = len(region_points(atlas_regions, hemisphere, view))
        if here >= len(region_points(atlas_regions, hemisphere, other)):    # most of it shows in this view
            labels.append((f"{means[region]:.1f}", atlas_regions, means[region]))
    return labels


def colour_bar_on_top(fig, cmap, vmax, label):
    """A slim horizontal colour bar above the figure."""
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, vmax), cmap=cmap),
                       cax=fig.add_axes([0.3, 0.97, 0.4, 0.016]), orientation="horizontal")
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=8)
    bar.ax.xaxis.set_ticks_position("top")
    bar.ax.xaxis.set_label_position("top")
    bar.set_label(label, fontsize=9)


def plot_brain_map(z, table, regions, subtype_titles, vmax=12, cmap="Reds"):
    """Average z-score of each subtype's tau+ people on the brain: one row per subtype with the left and
    the right hemisphere, each seen from the side (lateral) and from the middle (medial: the MTL is only
    visible there). Each of the data's regions is one coloured area with its value written on it.
    Grey = no region of the data there. None if no region name matches the atlas."""
    if not any(atlas_regions_of(r) for r in regions):
        return None
    tau_pos = table["ml_stage"] > 0
    means = z[tau_pos].groupby(table.loc[tau_pos, "ml_subtype"])[regions].mean()   # subtypes x regions
    counts = table.loc[tau_pos, "ml_subtype"].value_counts()
    k = len(subtype_titles)
    fig, axes = plt.subplots(k, 2, figsize=(10, 1.95 * k), squeeze=False, gridspec_kw={"wspace": 0.08,
                                                                                         "hspace": 0.45})
    for row, title in enumerate(subtype_titles, start=1):
        row_means = means.loc[row] if row in means.index else None
        values = atlas_values(row_means) if row_means is not None else {}
        for col, (side, views) in enumerate([("L", ["lateral_left", "medial_left"]),
                                             ("R", ["medial_right", "lateral_right"])]):
            ax = axes[row - 1, col]
            x = 0
            for view in views:
                labels = lobe_labels(row_means, regions, side, view) if row_means is not None else []
                width, _ = draw_view(ax, view, values, x, 0, cmap, vmax, labels)
                x += width + 30
            ax.set_xlim(-15, x - 15)
            ax.set_ylim(620, -15)                             # room for the MTL value under the views
            ax.set_aspect("equal")
            ax.axis("off")
            if row == 1:
                ax.text(0.5, 1.32, "left hemisphere" if side == "L" else "right hemisphere", ha="center",
                        transform=ax.transAxes, fontsize=9, color="#52606D")
        axes[row - 1, 0].set_title(f"{title} (n = {counts.get(row, 0)})", loc="left", fontsize=10,
                                   fontweight="bold")
    colour_bar_on_top(fig, cmap, vmax, "average z-score (tau+ people)")
    return fig


def plot_brain_stages(z, table, regions, subtype_titles, stage_edges, vmax=12, cmap="Reds"):
    """How tau spreads with stage, on the brain: one row per subtype, one column per stage group (e.g.
    1-6, 7-12, ...), each cell the average z-score of the tau+ people of that subtype and stage group on
    the left hemisphere (lateral above, medial below). n = people in the cell. The colours are those of the
    figure above."""
    import pandas as pd
    if not any(atlas_regions_of(r) for r in regions):
        return None
    labels = [f"{a}-{b - 1}" for a, b in zip(stage_edges[:-1], stage_edges[1:])]
    group = pd.cut(table["ml_stage"], stage_edges, right=False, labels=labels)
    k, g = len(subtype_titles), len(labels)
    fig, axes = plt.subplots(k, g, figsize=(2.3 * g + 1.4, 2.9 * k + 0.6), squeeze=False,
                             gridspec_kw={"hspace": 0.55, "wspace": 0.12})
    medial = view_box("medial_left")
    for row, title in enumerate(subtype_titles, start=1):
        people = (table["ml_subtype"] == row) & (table["ml_stage"] > 0)
        for col, label in enumerate(labels):
            ax = axes[row - 1, col]
            members = people & (group == label)
            ax.axis("off")
            if members.sum() == 0:
                ax.text(0.5, 0.5, "nobody", ha="center", va="center", fontsize=8, color="#9AA5B1",
                        transform=ax.transAxes)
            else:
                values = atlas_values(z[members][regions].mean())
                width, height = draw_view(ax, "lateral_left", values, 0, 0, cmap, vmax)
                draw_view(ax, "medial_left", values, 0, height + 30, cmap, vmax)
                ax.set_xlim(-20, max(width, medial[2] - medial[0]) + 20)
                ax.set_ylim(height + 30 + medial[3] - medial[1] + 20, -20)
                ax.set_aspect("equal")
                ax.text(0.5, -0.02, f"n = {int(members.sum())}", transform=ax.transAxes, ha="center", va="top",
                        fontsize=8, color="#52606D")
            if row == 1:
                ax.set_title(f"stages {label}", fontsize=9, pad=4)
        axes[row - 1, 0].text(-0.08, 0.5, title, transform=axes[row - 1, 0].transAxes, ha="right", va="center",
                              fontsize=10, fontweight="bold")
    first, last = axes[0, 0].get_position(), axes[0, -1].get_position()     # the arrow of the stages
    y = first.y1 + 0.045
    fig.patches.append(FancyArrowPatch(
        (first.x0, y), (last.x1, y), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=18, lw=1.6,
        color="#52606D"))
    fig.text((first.x0 + last.x1) / 2, y + 0.012, "stage (from early to late)", ha="center", va="bottom",
             fontsize=9, color="#52606D")
    return fig
