"""Put the figures and tables of one run into one ZIP file (tab 5, "Download report")."""
import io
import json
import zipfile
from importlib import metadata

import matplotlib.pyplot as plt


def pysustain_info():
    """Which pySuStaIn is installed: version, and the git commit it was installed from (if known)."""
    dist = metadata.distribution("pySuStaIn")
    info = {"version": dist.version, "commit": None, "source": None}
    direct_url = dist.read_text("direct_url.json")      # written by pip when installed from git
    if direct_url:
        data = json.loads(direct_url)
        info["source"] = data.get("url")
        info["commit"] = data.get("vcs_info", {}).get("commit_id")
    return info


def report_zip(figures, tables, texts):
    """figures {name: matplotlib figure} -> figures/name.png (300 dpi) + .pdf, tables {name: DataFrame}
    -> tables/name.csv, texts {file name: text}. Returns the ZIP file as bytes."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for name, fig in figures.items():
            for fmt, options in [("png", {"dpi": 300}), ("pdf", {})]:
                image = io.BytesIO()
                fig.savefig(image, format=fmt, bbox_inches="tight", **options)
                zip_file.writestr(f"figures/{name}.{fmt}", image.getvalue())
            plt.close(fig)
        for name, table in tables.items():
            zip_file.writestr(f"tables/{name}.csv", table.to_csv())
        for name, text in texts.items():
            zip_file.writestr(name, text)
    return buffer.getvalue()


def build_report(d):
    """Make every figure and put them, with the tables and texts, in one ZIP (bytes).
    d: plain data only (numbers, tables, texts) - so this can run in a separate process while the app
    keeps working. Made by the app's tab 5 (report_inputs)."""
    from src.figures import (plot_asymmetry, plot_model_choice, plot_pvd, plot_regional_contrasts,
                             plot_stage_progression, plot_stages, plot_subtype_patterns,
                             plot_subtype_probabilities, plot_subtype_tree)
    from src.results import pvd_order
    from src.subtype_analysis import asymmetry_index

    titles, likelihoods = d["titles"], d["likelihoods"]
    figures = {
        "01_pvd": plot_pvd(d["result"], d["Z_vals"], d["regions"], titles, None,
                           subtype_order=pvd_order(d["result"], d["order"])),
        "02_tau_patterns": plot_subtype_patterns(d["z"], d["table"], d["regions"], titles, d["vmax"]),
        "03_stage_progression": plot_stage_progression(d["z"], d["table"], d["regions"], titles, d["edges"],
                                                       None, vmax=d["vmax"]),
        "04_subtype_probabilities": plot_subtype_probabilities(d["result"]["prob_subtype"][:, d["order"]],
                                                               d["table"], titles),
        "05_stages": plot_stages(d["table"], titles, d["sd_limit"]),
        "06_model_choice": plot_model_choice(likelihoods[["mean log-likelihood"]],
                                             likelihoods[["CVIC"]] if "CVIC" in likelihoods else None),
        "07_subtype_tree": plot_subtype_tree(d["subtypes_by_k"]),
        "08_regional_differences": plot_regional_contrasts(d["t_values"], d["p_fdr"], titles),
    }
    asymmetry = asymmetry_index(d["z"], d["regions"])
    if asymmetry is not None:
        figures["09_left_right_asymmetry"] = plot_asymmetry(d["table"], asymmetry, titles)
    return report_zip(figures, d["tables"], d["texts"])
