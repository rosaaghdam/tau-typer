# Tau-typer

Automated tau-PET subtyping with **SuStaIn**, following Vogel et al. 2021 (*Nature Medicine*).
A Shiny for Python app takes a table of regional tau-PET values, normalizes it, runs the
**original pySuStaIn** (never changed or copied) and shows the subtypes, stages, figures and
analyses of the paper.

## Install (once)

The project uses one conda environment inside the project folder (`./env`), with pySuStaIn
pinned to commit `708fa22`:

```bash
bash code/01a_create_environment.sh
```

## Run the app

From the project folder (see `steps/S10_run_app.txt`):

```bash
conda activate ./env
```

```bash
shiny run app/app.py
```

Then open the address it prints (usually http://127.0.0.1:8000).

| Tab | What you do there |
|---|---|
| 1. Data & QC | Upload a CSV (one row per person), match the columns, check the data, optional inclusion rule |
| 2. Normalization | z-scores (GMM as in the paper, amyloid-negative CN reference, or already z-scored), cut-points, Z_max |
| 3. Run SuStaIn | Run pySuStaIn in the background (fast test or paper settings), cross-validation, saved runs |
| 4. Figures | PVD, tau patterns, subtype probabilities, stages, choosing k, one person |
| 5. Subtype analysis | Table 1, regional differences, stage vs age/cognition, asymmetry, compare runs, report (ZIP) |
| About | pySuStaIn version + commit, papers to cite, privacy |

Try it with the simulated toy data: tab 1 → "Sample Data" → upload that file.

## Folders

| Folder | Contents |
|---|---|
| `app/` | the Shiny app (user interface only) |
| `src/` | all calculations (they only *call* pySuStaIn) |
| `code/`, `steps/` | the step-by-step scripts used to build and check the pipeline |
| `config.yaml` | every setting in one place |
| `notes/` | plans, progress log, explanations (`SuStaIn.md`), review checklist |
| `data/toy/` | simulated toy data (safe to share) |
| `results/` | outputs and app runs - **not in git** |

## Data privacy

Real patient data never goes into git (`data/` and `results/` are ignored). The app keeps uploads and
results on the computer that runs it. On a shared server set `runs.private_per_visit: true` in
`config.yaml`: every visit then gets its own runs folder, which is deleted when the visit ends.

## Cite

- Vogel JW et al. Four distinct trajectories of tau deposition identified in Alzheimer's disease.
  *Nature Medicine* 27, 871-881 (2021).
- Young AL et al. Uncovering the heterogeneity and temporal complexity of neurodegenerative diseases
  with Subtype and Stage Inference. *Nature Communications* 9, 4273 (2018).
- Aksman LM et al. pySuStaIn: a Python implementation of the Subtype and Stage Inference algorithm.
  *SoftwareX* 16, 100811 (2021).

License: to be decided with the lab.
