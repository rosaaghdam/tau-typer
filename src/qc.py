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


def demographics_table(data, columns):
    """Who is in the data: N, age, % female and % amyloid+, for everyone and per diagnosis.

    columns: dict role -> chosen column ("" = not in file); roles not in the file are left out.
    Assumes sex values starting with "F" mean female, and amyloid coded 1 = positive, 0 = negative.
    """
    parts = [("All", data)]
    if columns.get("diagnosis"):
        parts += list(data.groupby(data[columns["diagnosis"]].astype(str)))
    rows = []
    for group, part in parts:
        row = {"Group": group, "N": len(part)}
        if columns.get("age"):
            age = pd.to_numeric(part[columns["age"]], errors="coerce")
            row["Age, mean (SD)"] = f"{age.mean():.1f} ({age.std():.1f})"
        if columns.get("sex"):
            first_letter = part[columns["sex"]].astype(str).str.strip().str.upper().str[0]
            row["Female, %"] = round(100 * (first_letter == "F").mean(), 1)
        if columns.get("amyloid"):
            amyloid = pd.to_numeric(part[columns["amyloid"]], errors="coerce")
            row["Amyloid+, %"] = round(100 * amyloid.mean(), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def apply_inclusion_rule(data, columns, min_age_cn, cn_labels, impaired_labels):
    """The paper's inclusion rule (Vogel 2021): keep cognitively normal people older than
    min_age_cn, and MCI/AD people who are amyloid-positive (coded 1). Remove everyone else.

    Labels are compared ignoring upper/lower case. Returns (kept data, table of reasons with counts).
    If the diagnosis, age or amyloid column is missing, nothing is removed and the table says why.
    """
    if not all(columns.get(role) for role in ["diagnosis", "age", "amyloid"]):
        return data, pd.DataFrame([{"Reason": "Rule NOT applied: needs diagnosis, age and amyloid columns",
                                    "N": len(data)}])
    diagnosis = data[columns["diagnosis"]].astype(str).str.strip().str.upper()
    age = pd.to_numeric(data[columns["age"]], errors="coerce")
    amyloid = pd.to_numeric(data[columns["amyloid"]], errors="coerce")
    is_cn = diagnosis.isin([label.upper() for label in cn_labels])
    is_impaired = diagnosis.isin([label.upper() for label in impaired_labels])
    keep_cn = is_cn & (age > min_age_cn)
    keep_impaired = is_impaired & (amyloid == 1)
    reasons = pd.DataFrame([
        (f"Kept: cognitively normal, older than {min_age_cn}", keep_cn.sum()),
        ("Kept: MCI/AD, amyloid-positive", keep_impaired.sum()),
        (f"Removed: cognitively normal, aged {min_age_cn} or younger (or age unknown)", (is_cn & ~keep_cn).sum()),
        ("Removed: MCI/AD, not amyloid-positive (negative or unknown)", (is_impaired & ~keep_impaired).sum()),
        ("Removed: other or missing diagnosis", (~is_cn & ~is_impaired).sum()),
    ], columns=["Reason", "N"])
    return data[keep_cn | keep_impaired], reasons


def number_and_category_columns(data, columns):
    """Split columns into numbers (e.g. MMSE, age) and categories (e.g. sex, APOE4 0/1).
    A number column has almost only numbers and more than 2 different values."""
    numbers, categories = [], []
    for col in columns:
        values = pd.to_numeric(data[col], errors="coerce")
        if values.notna().mean() > 0.9 and data[col].nunique() > 2:
            numbers.append(col)
        else:
            categories.append(col)
    return numbers, categories


def as_text(values):
    """Values as text without empty cells; 1.0 becomes "1"."""
    return values.dropna().map(lambda v: f"{v:g}" if isinstance(v, float) else str(v))


def covariate_summary(data, columns, group_col=None, groups=None):
    """One row per column: type, missing values, and mean (SD) for numbers or % per value for
    categories - for everyone and (if group_col is given) for each group, e.g. each diagnosis."""
    numbers, _ = number_and_category_columns(data, columns)
    parts = [("All", data)]
    if group_col:
        parts += [(g, data[data[group_col].astype(str) == g]) for g in groups]
    rows = []
    for col in columns:
        missing = data[col].isna().sum()
        row = {"Column": col, "Type": "number" if col in numbers else "category",
               "Missing": f"{missing} ({100 * missing / len(data):.0f}%)"}
        for name, part in parts:
            if col in numbers:
                values = pd.to_numeric(part[col], errors="coerce")
                row[name] = f"{values.mean():.1f} ({values.std():.1f})"
            else:
                shares = as_text(part[col]).value_counts(normalize=True)
                row[name] = ", ".join(f"{value}: {100 * share:.0f}%" for value, share in shares.head(4).items())
        rows.append(row)
    return pd.DataFrame(rows)
