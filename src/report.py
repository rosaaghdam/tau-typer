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
