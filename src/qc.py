"""Tab 1 - checks (QC) for an uploaded tau-PET table."""
import re

import pandas as pd

# Words that often appear in each column's name (lower case), most specific first.
COLUMN_HINTS = {
    "id": ["subject_id", "subject", "rid", "id"],
    "age": ["age"],
    "sex": ["sex", "gender"],
    "diagnosis": ["diagnosis", "diag", "dx", "group"],
    "amyloid": ["amyloid", "abeta", "ab_status"],
}
# Region columns start with a hemisphere: L_, R_, lh_, rh_, left_, right_ (or - / . instead of _)
REGION_PATTERN = re.compile(r"^(l|r|lh|rh|left|right)[_\-.]", re.IGNORECASE)


def guess_columns(columns):
    """First guess of which column plays which role. The user checks it in the app.

    For each role: an exact name match first, otherwise the first column containing a hint.
    Returns a dict: role -> column name (None if nothing fits), and "regions" -> list of names.
    """
    by_lower_name = {col.lower(): col for col in columns}
    guesses = {}
    for role, hints in COLUMN_HINTS.items():
        exact = [by_lower_name[h] for h in hints if h in by_lower_name]
        partial = [col for h in hints for col in columns if h in col.lower()]
        guesses[role] = (exact + partial + [None])[0]
    guesses["regions"] = [col for col in columns if REGION_PATTERN.match(col)]
    return guesses


def qc_summary(data, columns, regions, suvr_min, suvr_max):
    """Basic quality checks of the uploaded table. Returns a table: Check, Result, Status.

    columns: dict role -> chosen column ("" or None = not in file), from the tab-1 dropdowns.
    regions: chosen region columns. suvr_min / suvr_max: plausible SUVR range (config.yaml).
    """
    rows = []
    def add(check, result, ok):
        rows.append({"Check": check, "Result": result, "Status": "OK" if ok else "Warning"})

    add("People (rows)", len(data), len(data) > 0)
    if columns.get("id"):
        n_duplicates = int(data[columns["id"]].duplicated().sum())
        add("Duplicate subject IDs", n_duplicates, n_duplicates == 0)
    add("Region columns selected", len(regions), len(regions) >= 2)
    values = data[list(regions)].apply(pd.to_numeric, errors="coerce")   # text becomes missing
    n_missing = int(values.isna().sum().sum())
    add("Missing or non-numeric region values", n_missing, n_missing == 0)
    n_outside = int(((values < suvr_min) | (values > suvr_max)).sum().sum())
    add(f"Region values outside {suvr_min}-{suvr_max} (implausible SUVR)", n_outside, n_outside == 0)
    for role in ["age", "sex", "diagnosis", "amyloid"]:
        if columns.get(role):
            n_missing_role = int(data[columns[role]].isna().sum())
            add(f"Missing {role} values", n_missing_role, n_missing_role == 0)
        else:
            add(f"{role.capitalize()} column", "not in file", False)
    return pd.DataFrame(rows)
