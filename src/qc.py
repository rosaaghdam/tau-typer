"""Tab 1 - checks (QC) for an uploaded tau-PET table."""
import re

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
