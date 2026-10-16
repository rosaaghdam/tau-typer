# Tau-typer – instructions for Claude

Automated tau-PET subtyping with SuStaIn, based on Vogel et al., Nature Medicine 2021
(https://www.nature.com/articles/s41591-021-01309-6).
Rosa is new to the field and wants to work SLOWLY and learn as we go.

## WORKING RULES
- Do one small step at a time. After each step, stop and wait for "OK, continue".
- Never write or change more than ~20 lines of code at once.
- Before writing code: explain in plain language what it will do and why.
- After writing code: walk through it line by line (what each line does,
  what goes in, what comes out).
- Do not run scripts, install packages, create environments, or delete/overwrite
  files without asking first.
- Prefer simple, readable code over clever code: clear variable names, short functions,
  comments that explain the "why", no unnecessary abstractions.
- If something is uncertain (e.g. a detail from the paper), say so and ask – don't guess.
- Keep a running file notes/progress.md with what we did, decisions made, and open questions.

## USING SUSTAIN – IMPORTANT
- Use the ORIGINAL pySuStaIn package (https://github.com/ucl-pond/pySuStaIn) exactly
  as published. Never copy, modify, or re-implement its internal code.
- Install it as a package into a dedicated project environment (conda or venv),
  pinned to a specific commit, and record the exact version in the project's
  environment/requirements file. Ask before creating the environment or installing.
  Candidate commit: 708fa22d89c9692dbeafcc2c31a1c8460ced5640 (2026-03-30) – this is
  identical to the copy in Github/pySuStaIn-master.
- Our own code in src/ only wraps SuStaIn: data preparation, choosing parameters
  (z-score thresholds, number of subtypes, start points, MCMC iterations, etc.),
  calling SuStaIn, and saving/plotting/interpreting results.
- Keep all tunable settings in ONE config file (e.g. config.yaml), so changing a
  threshold or parameter never means editing code.

## PROJECT PLAN
1. Step 0 – understand the papers and repos (done 2026-10-15; see notes/).
2. Step 1 – toy data in data/toy/ with known simulated subtypes.
3. Step 2 – clean re-implementation of the paper's steps in src/, calling pySuStaIn.
4. Later – refit on the lab's large tau-PET library; pipeline that applies the
   fitted model to new data (subtype + stage); package (tests, docs) and paper.

## FOLDER LAYOUT
- papers/  – Vogel 2021 (+ supplement), original SuStaIn paper
- Github/  – reference repos (capital G): tau-postmortem-subtypes-main, pySuStaIn-master.
  Reference only – never import from or edit these.
- data/    – data (toy now, real later). NEVER commit real data to git.
- src/     – our new clean code
- notes/   – progress.md, step0_tables.md, SuStaIn.md (plain-language explainer)
- If new files appear in the folder, tell Rosa.

## PRACTICAL NOTES FOR CLAUDE
- The Bash/Read/Write tools may get "Operation not permitted" on this Desktop folder
  (macOS privacy). Workaround: write files in the scratchpad, then copy them with
  the Terminal panel (run_in_terminal) using `cp -n` (never overwrite silently).

## ENVIRONMENT
- One conda environment for the whole project, inside the project folder: `./env`
  (created with `conda env create -f environment.yml -p ./env`; ignored by git).
- Run Python with `./env/bin/python` (or `conda activate ./env`).
- New packages: add to environment.yml first, then ask Rosa before updating the env.

## code/ FOLDER (Rosa's request, 2026-10-15)
- Every step we run gets a script in code/, numbered in order (00a_, 00b_, 01a_, ...),
  so it can be reviewed and redone. Update code/README.md (the index) each time.
- code/ = what we ran, in order. src/ = reusable pipeline functions that code/ scripts call.

## WHO RUNS THE STEPS (Rosa's request, 2026-10-15)
- Rosa runs every step herself on her Mac. For each new step: write the script in code/,
  give her the exact command(s) and the expected output, and let her report back.
- Start of a session:  cd to the project folder, then `conda activate ./env`.
- Every step (including each "check" of a new function) is a script in code/, runnable with
  `bash run.sh <step>` (e.g. `bash run.sh 01e`). Give Rosa only plain shell commands in
  ```bash``` blocks – never `>>>` Python-prompt snippets (she copies the `>>>` too).
- PREFERRED RUN FORMAT (Rosa, 2026-10-15): simple copy-paste files in steps/:
  steps/S0_activate_environment.txt (cd + conda activate ./env + python), then one file per
  step (S1_..., S2_...) with plain Python lines (no >>>) plus the expected output.
  Test the lines before giving them. Keep code/ scripts as the permanent record.
