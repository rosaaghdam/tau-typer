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
    "stages": "Top: how many tau+ people are at each SuStaIn stage, coloured by subtype; people at stage 0 "
              "(S0, no abnormal tau) are only counted in the legend. Bottom left: how sure SuStaIn is about each "
              "person's stage (SD of their stage probabilities) - low = sure, high = the stage could be one of "
              "several (often higher in the middle stages, where events overlap - the pale squares of the PVD). "
              "Bottom right: the same per subtype. Choosing a subtype above shows it in colour and the others "
              "grey; the person chosen above is circled. Dashed red line = your limit; the people above it are "
              "listed below.",
    "choice": "Top: log-likelihood (rises with every extra subtype) and CVIC after cross-validation (lowest = "
              "best). Bottom: subtype tree - where people go when one more subtype is allowed.",
    "person": "Bars: this person's z-scores. Dots: average of people in the same subtype and stage group. "
              "Middle: subtype probabilities. Right: stage probabilities.",
}

DATA_HELP = {   # help (i) of the figures in tabs 1-3
    "regions": "Explore the raw regional values (before normalization). With nothing chosen: the distribution "
               "of each region. Plot against a number (e.g. age): one dot per person and a straight trend "
               "line; against a category (e.g. sex): a box per value. 'Split by' colours the people by a "
               "group (a number column is split into low / middle / high thirds) and draws one trend line "
               "per group: lines that are not parallel mean the trend differs between the groups "
               "(an interaction, e.g. age x sex). The table below tests this per region.",
    "covariates": "The other columns of the file per diagnosis group: numbers as box plots (one dot per "
                  "person), categories as bars (% per group), and the Spearman correlations between the "
                  "number columns and the mean of the regions.",
    "gmm": "One region: histogram of the values and the bell curves of the Gaussian mixture model; the "
           "lowest-mean curve is 'normal'. The cut-points are drawn on the value axis "
           "(value = normal mean + z x normal SD).",
    "zhist": "The z-scores of each region. Most people should sit around 0; the tau+ tail goes to the right.",
    "model": "Fit per number of subtypes (higher = fits this data better, but it always rises with more "
             "subtypes), CVIC after cross-validation (lower = better) and the MCMC trace of each model "
             "(flat = stable).",
    "cv": "pySuStaIn's own figure: test log-likelihood of each cross-validation fold per number of subtypes.",
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


def step_title(number, label, status_id):
    """Title of one step in the step rail: a numbered circle, the step's name and its status
    (status_id = a render.ui in the tab's server that returns step_status(...))."""
    return ui.span(ui.span(str(number), class_="tt-step-num"),
                   ui.span(ui.span(label, class_="tt-step-label"), ui.output_ui(status_id, inline=True),
                           class_="tt-step-text"),
                   class_="d-flex align-items-center gap-2")


def step_status(text, state="waiting"):
    """The small status under a step's name. state: "done", "attention" (needs your action) or "waiting"."""
    return ui.span(text, class_=f"tt-step-status tt-{state}", title=text)   # full text on hover
