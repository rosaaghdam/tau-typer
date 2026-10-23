"""Settings and texts shared by all tabs of the app (everything else is in the tab files)."""
import sys
from pathlib import Path

import yaml
from shiny import ui

ROOT = Path(__file__).resolve().parent.parent          # project folder (one level above app/)
sys.path.insert(0, str(ROOT))                           # so Python can find src/

NORM_METHODS = {"gmm": "GMM on everyone", "reference": "Reference group: amyloid-negative CN",
                "zscored": "The data is already normalized"}

CONFIG = yaml.safe_load(open(ROOT / "config.yaml"))    # all settings live in config.yaml

RUNS_FOLDER = ROOT / CONFIG["runs"]["folder"]          # one sub-folder per SuStaIn run
SAMPLE_DATA = ROOT / "data" / "toy" / "toy_tau_clinical.csv"   # simulated toy data + 10 clinical variables + SCD
FIG_HELP = {    # short help text above each tab-4 figure
    "pvd": "Positional variance diagram, drawn by pySuStaIn. Rows = regions, columns = stages. Dark = SuStaIn "
           "is sure the event happens at that stage, pale = unsure.",
    "patterns": "Top: how much tau each subtype has in each region - the average z-score of its tau+ people "
                "(number in each cell). Bottom: the same, split into stage groups (early stages on the left, late "
                "on the right), so you can see where tau spreads as the stage rises. n = number of people.",
    "probs": "How clearly each tau+ person belongs to one subtype. Each dot is a person, each "
             "corner a subtype: close to a corner = a clear member of that subtype; towards the middle = a mix "
             "of subtypes. The dashed box marks the middle; the legend gives the % of people inside and outside "
             "it.",
    "stages": "Left: how many tau+ people are at each SuStaIn stage, coloured by subtype; "
              "people at stage 0 (S0, no abnormal tau) are only counted in the legend. Middle: how sure SuStaIn is "
              "about each person's stage (SD of their stage probabilities) - low = sure, high = the stage could "
              "be one of several (often higher in the middle stages, where events overlap - the pale squares "
              "of the PVD); 'Show a person from' a subtype = that subtype in colour, the others grey; ringed dot = the person chosen above. Right: the same per subtype. "
              "Dashed red line = your limit; the people above it are listed below.",
    "choice": "Top: log-likelihood (rises with every extra subtype) and CVIC after cross-validation (lowest = "
              "best). Bottom: subtype tree - where people go when one more subtype is allowed.",
    "person": "Bars: this person's z-scores. Dots: average of people in the same subtype and stage group. "
              "Middle: subtype probabilities. Right: stage probabilities.",
}

AN_HELP = {     # short help text in tab 5
    "table1": "Numbers: mean (SD); categories: %. p (subtypes): Kruskal-Wallis or chi-square "
              "between the subtypes. Adjusted: linear model variable ~ subtype + stage + the covariates chosen "
              "in 'Adjust tests for'. * = this subtype differs from the other tau+ people (FDR p < 0.05).",
    "regions": "For each subtype and region: this subtype vs all other tau+ people, at the same "
               "stage (linear model z ~ subtype + stage + 'Adjust tests for'). Red = more tau, blue = less; "
               "* = FDR p < 0.05.",
    "stage": "Spearman correlation between SuStaIn stage and the chosen variable (tau+ people), "
             "overall and per subtype; one line per subtype in the figure.",
    "asymmetry": "Asymmetry = mean over left/right region pairs of (L - R) / (|L| + |R|); "
                 "above 0 = more tau on the left.",
    "compare": "How well do two runs agree on the same people (e.g. other settings, Z_max or normalization)? "
               "Cross-table of the groups, adjusted Rand index (1 = same subtypes) and stage correlation.",
}

RULE_ROLES = ("age", "diagnosis", "amyloid")      # columns the inclusion rule needs
REFERENCE_ROLES = ("diagnosis", "amyloid")       # columns the reference-group normalization needs
REFERENCE_GROUPS = {"cn_scd": "Amyloid-negative CN + SCD", "cn": "Amyloid-negative CN only"}
DISEASE_WORDS = ["mmse", "moca", "cdr", "adas", "cognit", "memory", "executive", "language", "visuo",
                 "tau", "amyloid", "abeta", "diagnos", "dx"]   # covariate names that look like disease measures
ROLE_LABELS = {"id": "Subject ID", "age": "Age", "sex": "Sex",
               "diagnosis": "Diagnosis", "amyloid": "Amyloid status"}


def figure_card(title, help_text, *content):
    """A card with a title, an (i) button that shows the help text, and a button to show it full screen."""
    info = ui.popover(ui.tags.button("ⓘ", type="button", class_="btn btn-link btn-sm p-0 ms-1",
                                     **{"aria-label": f"About: {title}"}),
                      help_text, title=title, placement="right")
    return ui.card(ui.card_header(title, info), *content, full_screen=True)


def no_run_banner():
    """Shown above tabs 4 and 5 until a run is loaded."""
    return ui.div("No run shown yet: run SuStaIn, or load a saved run, in tab 3 (Run SuStaIn).",
                  class_="alert alert-info")
