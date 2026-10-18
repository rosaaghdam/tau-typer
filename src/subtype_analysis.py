"""App tab 5 - compare the subtypes (Vogel et al. 2021: Table 1, Fig. 1d, Fig. 3a/b, Ext. Fig. 9).

All functions take the per-person table from src/results.py subject_table() (subtypes numbered
by size, "S0" = stage 0) plus z-scores or other columns of the same people.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import adjusted_rand_score
from statsmodels.stats.multitest import multipletests


def group_table(table, covariates, variables):
    """Table 1: each variable per group (S0, Subtype 1, ...) and a test between the subtypes.

    Numbers: mean (SD), test = Kruskal-Wallis between the subtypes (tau+ people only), and a
    linear model "variable ~ subtype + stage" (p for subtype, i.e. adjusted for stage).
    Text / categories: % for every value, test = chi-square between the subtypes.
    """
    groups = table["subtype"]
    order = ["S0"] + sorted(g for g in groups.unique() if g != "S0")
    tau_pos = table["ml_stage"] > 0
    rows = [{"variable": "N", **{g: int((groups == g).sum()) for g in order}}]
    for var in variables:
        values = covariates[var]
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().mean() > 0.9 and values.nunique() > 2:                  # a number
            row = {"variable": var, **{g: f"{numeric[groups == g].mean():.1f} ({numeric[groups == g].std():.1f})"
                                       for g in order}}
            samples = [numeric[(groups == g) & tau_pos].dropna() for g in order[1:]]
            row["p (subtypes)"] = stats.kruskal(*samples).pvalue if len(samples) > 1 else np.nan
            data = pd.DataFrame({"y": numeric, "subtype": groups, "stage": table["ml_stage"]})[tau_pos].dropna()
            if len(samples) > 1:                         # ANOVA table of the linear model -> p for subtype
                fit = smf.ols("y ~ C(subtype) + stage", data=data).fit()
                row["p (subtypes, adjusted for stage)"] = sm.stats.anova_lm(fit, typ=2).loc["C(subtype)", "PR(>F)"]
            rows.append(row)
        else:                                                                     # categories
            counts = pd.crosstab(values.astype(str), groups)[order]
            percent = 100 * counts / counts.sum()
            subtype_counts = pd.crosstab(values[tau_pos].astype(str), groups[tau_pos])
            p = stats.chi2_contingency(subtype_counts).pvalue if subtype_counts.shape[1] > 1 and subtype_counts.shape[0] > 1 else np.nan
            for level in counts.index:
                rows.append({"variable": f"{var} = {level}, %", **{g: f"{percent.loc[level, g]:.1f}" for g in order},
                             "p (subtypes)": p if level == counts.index[0] else np.nan})
    result = pd.DataFrame(rows)
    for col in ["p (subtypes)", "p (subtypes, adjusted for stage)"]:
        if col in result:
            result[col] = result[col].map(lambda p: "" if pd.isna(p) else f"{p:.3g}")
    return result


def regional_contrasts(z, table, regions):
    """Paper Fig. 1d: where does each subtype have more (or less) tau than all other tau+ people?

    Per subtype and region a linear model "z ~ this_subtype + stage" (tau+ people only), so the
    comparison is at the same disease stage. Returns t-values and FDR-corrected p-values
    (Benjamini-Hochberg over all subtypes x regions), each as a regions x subtypes table.
    """
    tau_pos = table["ml_stage"] > 0
    subtypes = sorted(table.loc[tau_pos, "ml_subtype"].unique())
    t_values = pd.DataFrame(index=regions, columns=subtypes, dtype=float)
    p_values = t_values.copy()
    for k in subtypes:
        for region in regions:
            data = pd.DataFrame({"z": z.loc[tau_pos, region], "this": (table.loc[tau_pos, "ml_subtype"] == k).astype(int),
                                 "stage": table.loc[tau_pos, "ml_stage"]})
            fit = smf.ols("z ~ this + stage", data=data).fit()
            t_values.loc[region, k], p_values.loc[region, k] = fit.tvalues["this"], fit.pvalues["this"]
    corrected = multipletests(p_values.to_numpy().ravel(), method="fdr_bh")[1].reshape(p_values.shape)
    return t_values, pd.DataFrame(corrected, index=regions, columns=subtypes)


def stage_association(table, values):
    """Paper Fig. 3a/b: Spearman correlation between SuStaIn stage and a number (e.g. age, MMSE),
    tau+ people only - overall and per subtype. Returns a table: group, N, rho, p."""
    numeric = pd.to_numeric(values, errors="coerce")
    tau_pos = (table["ml_stage"] > 0) & numeric.notna()
    rows = []
    for name, people in [("All tau+", tau_pos)] + [(f"Subtype {k}", tau_pos & (table["ml_subtype"] == k))
                                                    for k in sorted(table.loc[tau_pos, "ml_subtype"].unique())]:
        if people.sum() > 2:
            rho, p = stats.spearmanr(table.loc[people, "ml_stage"], numeric[people])
            rows.append({"group": name, "N": int(people.sum()), "rho": round(rho, 3), "p": f"{p:.3g}"})
    return pd.DataFrame(rows)


SIDES = [("L_", "R_", "start"), ("Left-", "Right-", "start"), ("Left_", "Right_", "start"),
         ("lh_", "rh_", "start"), ("_L", "_R", "end"), ("_lh", "_rh", "end")]


def region_pairs(regions):
    """Left/right pairs from names like L_MTL / R_MTL, Left-X / Right-X, lh_X / rh_X, X_L / X_R, X_lh / X_rh.
    Returns {name: (left region, right region)}."""
    pairs = {}
    for region in regions:
        for left, right, where in SIDES:
            if where == "start" and region.startswith(left):
                name = region[len(left):]
                partner = right + name
            elif where == "end" and region.endswith(left):
                name = region[:-len(left)]
                partner = name + right
            else:
                continue
            if partner in regions:
                pairs[name] = (region, partner)
    return pairs


def asymmetry_index(z, regions):
    """Paper Ext. Fig. 9: left-right asymmetry per person = mean over region pairs of
    (left - right) / (|left| + |right|). Positive = more tau on the left. None if no pairs found."""
    pairs = region_pairs(regions)
    if not pairs:
        return None
    parts = [(z[left] - z[right]) / (z[left].abs() + z[right].abs() + 1e-6) for left, right in pairs.values()]
    return pd.concat(parts, axis=1).mean(axis=1).rename("asymmetry")


def compare_runs(table_a, table_b):
    """Agreement between two runs on the same people (paper Fig. 2b idea / sensitivity check):
    cross-table of the groups, adjusted Rand index of the subtypes (tau+ in both runs) and the
    correlation of the stages."""
    common = table_a.index.intersection(table_b.index)
    a, b = table_a.loc[common], table_b.loc[common]
    crosstab = pd.crosstab(a["subtype"], b["subtype"], rownames=["run A"], colnames=["run B"])
    both = (a["ml_stage"] > 0) & (b["ml_stage"] > 0)
    ari = adjusted_rand_score(a.loc[both, "ml_subtype"], b.loc[both, "ml_subtype"]) if both.sum() > 1 else np.nan
    rho = stats.spearmanr(a["ml_stage"], b["ml_stage"])[0] if len(common) > 2 else np.nan
    return crosstab, {"people in both runs": len(common), "tau+ in both": int(both.sum()),
                      "adjusted Rand index (subtypes)": round(ari, 3), "stage correlation (Spearman)": round(rho, 3)}
