# code/ – one script per step

Every step we run is recorded here as a script, numbered in the order we did it,
so it can be reviewed and redone. Reusable Python functions live in `src/`;
the scripts here call them.

**How to run a step** (from the project folder; no need to activate the environment):

    bash run.sh 01c        # runs code/01c_...
    bash run.sh            # lists all steps

| Script | Step | What it does | Status |
|---|---|---|---|
| `00a_extract_pdf_text.sh` | 0a | Turns the PDFs in papers/ into text files in papers/text/ | done |
| `00b_compare_pysustain_with_github.sh` | 0b | Checks Github/pySuStaIn-master is identical to the official repo | done |
| `00c_check_installed_pysustain.sh` | 0 | Checks the pySuStaIn installed in ./env is the unchanged original (pinned commit) | done |
| `01a_create_environment.sh` | 1 | Builds the conda environment in ./env from environment.yml and checks pySuStaIn | done |
| `01b_check_event_sequences.py` | 1 | Checks the 30-event "wave" sequence of each subtype | done |
| `01c_check_generator_inputs.py` | 1 | Checks Z_vals, Z_max and the 4 sequences for pySuStaIn's generator | done |
| `01d_check_subtypes_stages.py` | 1 | Checks the true subtype + stage of the 500 people | done |
| `01e_check_suvr.py` | 1 | Checks the simulated z-scores and SUVR values | done |
| `01f_check_covariates.py` | 1 | Checks age, sex, diagnosis, amyloid | done |
| `01g_make_toy_data.py` | 1 | Writes data/toy/toy_tau.csv and data/toy/toy_truth.csv | planned |

Steps 0c (tables) and the notes were writing only – see notes/.
