# Tau-typer – progress log

## 2026-10-15 – Step 0a: read the papers (no code)

**What we did**
- Read Vogel et al. 2021 (main text + Methods + Extended Data captions),
  its Supplementary Information, and Young et al. 2018 (original SuStaIn paper).
- Wrote a numbered summary of the Vogel 2021 method (given in chat).

**Observations about the folder**
- Repos are in `Github/` (capital G), not `github/`.
- The second repo is `Github/pySuStaIn-master` = a downloaded copy of the
  pySuStaIn package itself. Reference only – we will NOT import or edit it;
  we install pySuStaIn from GitHub at a pinned commit later.
- `data/`, `src/`, `notes/` did not exist yet (notes/ created for this file).

**Decisions**
- (none yet)

**Open questions (details the paper does not state)**
1. SuStaIn class: paper describes the linear z-score model -> most likely
   `ZscoreSustain`, but the class name is not written in the paper.
2. Z_max (the maximum z-score reached at the last stage) - not reported.
3. Number of start points for the optimisation - not reported.
4. Exactly which Desikan-Killiany regions make up each of the 10 lobar ROIs
   (especially "MTL") - not listed.
5. Choroid plexus regression: only mentioned in the Extended Data Fig. 1b
   caption; how it was done (which covariate, fitted on whom) is not described.
6. Gaussian mixture model for z-scoring: fitted on all 1,143 people or a subset?
7. Stage count: text says "1 of 30 stages", but elsewhere "stage 5 (of 31)"
   and bins up to 27-31. (30 events -> stages 0-30 = 31 values; to confirm.)
8. Noise level (sigma) given to SuStaIn - not reported (would be 1 if z-scored
   against the "normal" distribution, but not stated).
9. MCMC: paper says 10,000 iterations; the original SuStaIn paper used 1,000,000.

## 2026-10-15 – Step 0b: read the two repos (no code)

**What we did**
- Read all of `Github/tau-postmortem-subtypes-main/src` and skimmed its 3 notebooks.
- Checked `Github/pySuStaIn-master` against the official GitHub repo: all
  `pySuStaIn/*.py` files are IDENTICAL to upstream master, commit
  `708fa22d89c9692dbeafcc2c31a1c8460ced5640` (2026-03-30). Candidate commit to pin.

**Key findings**
- The postmortem repo runs on Newcastle brain-bank histology (AT8 % area per
  tissue punch), not tau-PET. One hemisphere, no left/right.
- It does NOT follow the paper's normalization: log1p -> impute -> tau-/tau+ split
  by a 2-component GMM on mean tau -> age+sex regression on tau- cases ->
  "w-scores" with SD inflated by 0.3*SD(biomarker).
- Cut-points are data-driven (GMM, 1-8 components, chosen by BIC), similar to the
  paper's *replication* analysis, not the main z = 2, 5, 10.
- Z_max = 1.2 x the largest cut-point per region.
- `run_sustain_model.py` comment says it imported a locally MODIFIED pySuStaIn
  ("the one I made changes" – not in our folder). We will not do that.
- No post-hoc T- correction, no choroid plexus step, no "apply model to new data" step.

**Open questions (added)**
10. What were the local changes the postmortem author made to pySuStaIn, and do
    they matter for us?
11. Does the Newcastle mapping produce a "Subcortical" column that went into SuStaIn?
12. Cross-validation branch of run_sustain_model.py looks broken
    (uses undefined `tau_df`; unpacks 6 values from a function returning 7) –
    was the CV actually run from this code?

## 2026-10-15 – Step 0c: comparison tables

**What we did**
- Wrote notes/step0_tables.md: (1) paper step -> code -> missing, (2) SuStaIn parameters.

**Answered**
- Q8 (sigma): `ZscoreSustain` hard-codes noise SD = 1 per biomarker, so input
  must be z-scores in SD units of the "normal" distribution.
- Q7 (stages): 30 events -> stages 0..30 (31 possible values incl. stage 0).

**Next**
- Save WORKING RULES + SUSTAIN rules to CLAUDE.md (end of Step 0).

## 2026-10-15 – End of Step 0

**What we did**
- Saved WORKING RULES + SUSTAIN rules (+ project plan, folder layout) to CLAUDE.md.
- Wrote notes/SuStaIn.md: plain-language explanation of the algorithm, all steps,
  inputs/outputs, and the full Vogel 2021 recipe with thresholds.

**Decisions**
- Rosa will learn the details of the tables step by step while we run things on data.

**Next**
- Step 1: toy data in data/toy/ (explain the simulation before writing it).

## 2026-10-15 – Step 1: toy data – design decisions

**Decisions (Rosa)**
- Expected z-scores come from pySuStaIn's own generator `ZscoreSustain.generate_data`
  (we only call it; we never copy it). That means the environment must be created first.
- 500 people (~60% tau-negative -> ~200 tau+).
- No extra effects for now (no choroid plexus, no false positives).

**Design**
- `data/toy/toy_tau.csv`: subject_id, age, sex, diagnosis, amyloid_positive,
  10 SUVR columns (L/R × MTL, temporal, parietal, occipital, frontal).
- `data/toy/toy_truth.csv`: subject_id, true_subtype, true_stage (answer key, pipeline never reads it).
- Simulate z-scores, then SUVR = normal_mean + z × normal_sd.
- `generate_data` adds noise with SD 1 (z units) and uses numpy's global random seed.
- Event numbering in pySuStaIn: event = level_index × n_biomarkers + biomarker_index
  (all z=2 events first, then all z=5, then all z=10).

**Open**
- Z_max for the toy data: our own choice (paper doesn't report it).

## 2026-10-15 – Project environment created

**Decisions (Rosa)**
- ONE environment for the whole project, stored INSIDE the project folder: `./env`
  (not synced to iCloud; ignored by git via .gitignore).
- `environment.yml` is the single record of all packages; add new ones there.

**What we did**
- Wrote `environment.yml` and `.gitignore`.
- Ran `conda env create -f environment.yml -p ./env` -> success (891 MB).
- Verified: `import pySuStaIn` and `ZscoreSustain` work.
  Installed: pySuStaIn @ 708fa22d89c9692dbeafcc2c31a1c8460ced5640, python 3.11,
  numpy 2.3.5, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.17.1.

**How to use**
- `conda activate ./env` (from the project folder), or run `./env/bin/python ...`
- If the folder is moved: delete `env/` and recreate from environment.yml.

**Later**
- Freeze exact versions (`conda env export`) once the pipeline is stable.

## 2026-10-15 – code/ folder

**Decision (Rosa)**: a `code/` folder with one script per step, to review and redo everything.
- code/ = scripts we run, numbered; src/ = reusable functions.
- environment.yml stays the single list of packages; the script only builds from it.

**What we did**
- code/README.md (index of steps) and code/01a_create_environment.sh.

**Next**
- code/00a_extract_pdf_text.sh and code/00b_compare_pysustain_with_github.sh.

## 2026-10-15 – Step 1: config.yaml part 1 (toy-data general settings)

**What we did**
- Created config.yaml (project root) with a `toy_data:` section: seed 42, 500 subjects,
  60% tau-negative, paper's subtype fractions, z_events [2, 5, 10], z_max 15,
  invented "normal" SUVR mean/SD per lobe.

**Decisions**
- z_max = 15 for toy data (our choice; paper does not report Z_max).
- Normal SUVR values are invented (MTL 1.20/0.10, temporal 1.15/0.08,
  parietal 1.10/0.08, occipital 1.12/0.08, frontal 1.05/0.07), same for L and R.

**Next**
- config.yaml part 2: region list + event order of each subtype.

## 2026-10-15 – Step 1: config.yaml part 2 (subtype event orders)

**What we did**
- Added `regions` (10, fixed column order) and `subtype_region_order` (4 subtypes) to config.yaml.
- Checked: every order contains each of the 10 regions exactly once; subtype names match
  `subtype_fractions`.

**Decision**
- "Wave" rule to turn a region order into 30 events: each region starts one step after the
  previous one while earlier regions keep climbing (2 -> 5 -> 10). Chosen over the
  "layered" rule so subtypes differ clearly at mid stages. Rule lives in the toy-data code.

## 2026-10-15 – Step 1: src/toy_data.py – wave_event_sequence()

**What we did**
- Created src/toy_data.py with its first function `wave_event_sequence(region_order, regions, n_levels)`:
  region order -> 30 event numbers (pySuStaIn numbering: level_index * n_regions + region_index).
- Checked: all 4 subtypes give 30 events, each exactly once; S1 starts
  L_MTL->2, L_MTL->5, R_MTL->2, L_MTL->10, R_MTL->5, L_temporal->2 (matches the wave table).

**Next**
- Function that builds everything pySuStaIn's generator needs (Z_vals, Z_max, sequences,
  true subtype + stage per person).
- Later: turn quick checks like this into proper tests (pytest).

## 2026-10-15 – Step 1: src/toy_data.py – sustain_generator_inputs()

**What we did**
- Added `import numpy as np` and `sustain_generator_inputs(cfg)` -> Z_vals (10x3, [2,5,10]),
  Z_max (10, all 15), sequences (4x30, wave rule).
- Checked: shapes correct; S1 sequence starts 0, 10, 1, 20, 11, 2.
- Rosa now runs steps herself (see CLAUDE.md "WHO RUNS THE STEPS").

**Next**
- Function that gives each of the 500 people a true subtype and true stage.

## 2026-10-15 – Step 1: src/toy_data.py – assign_true_subtypes_and_stages()

**What we did**
- Added `assign_true_subtypes_and_stages(cfg, rng)`: tau-negative -> subtype -1, stage 0;
  tau+ -> subtype drawn with config fractions, stage uniform 1..30; then shuffled.
- Checked with seed 42: counts {-1: 300, S1: 69, S2: 36, S3: 62, S4: 33}, identical on
  two runs; tau+ stages 1-30; all subtype -1 have stage 0.

**Decision**
- Subtype -1 = "no subtype" (S0) in the answer key. Subtype numbers follow config order
  (0 = S1 limbic ... 3 = S4 lateral temporal).

**Next**
- Call pySuStaIn's ZscoreSustain.generate_data to get noisy z-scores, then convert to SUVR.

## 2026-10-15 – Step 1: src/toy_data.py – simulate_regional_suvr()

**What we did**
- Added `simulate_regional_suvr(cfg, subtypes, stages)`: calls pySuStaIn's
  `ZscoreSustain.generate_data` (noise SD 1), then SUVR = normal mean + z * normal SD per lobe.
- numpy's global seed is set from config (generate_data uses it) -> identical output each run.
- Tau-negative people (-1) are passed to the generator as subtype 0 (stage 0 is the same for all).

**Checks (seed 42)**
- S1 at stage 6 without noise: L_MTL 10.4, R_MTL 7.5, L_temporal 2.0, R_temporal 1.3
  -> stage k = "event k just happened"; regions between cut-points are part-way (linear model).
- Tau-negative: z mean -0.01, SD 0.98; mean SUVR = the config's normal means.
- Late stages (>=15) show the intended patterns: S1 MTL highest/occipital lowest; S2 MTL lowest;
  S3 occipital high/frontal low; S4 left >> right.

## 2026-10-15 – Step 1: covariates (config + simulate_covariates)

**What we did**
- config.yaml: `toy_data.covariates` (age 75 - 0.3*stage + N(0,6); 55% female;
  diagnosis cut-offs MCI >= 8, AD >= 19; 25% amyloid+ among tau-negative; tau+ always amyloid+).
- src/toy_data.py: `simulate_covariates(cfg, subtypes, stages, rng)`.

**Checks (seed 42)**
- Diagnosis: CN 350, MCI 69, AD 81. All MCI and AD are amyloid+ (paper's inclusion rule holds).
- 57% female. Mean age CN 74.9, MCI 71.1, AD 66.7. corr(stage, age) in tau+ = -0.47.

**Next**
- code/01c_make_toy_data.py: run all functions, write data/toy/toy_tau.csv + toy_truth.csv.

## 2026-10-15 – Step 2 begins: S3 z-scoring with a GMM (paper step 5)

**What we did**
- Rosa ran S2: data/toy/toy_tau.csv and toy_truth.csv exist.
- config.yaml: new `pipeline:` section (input_csv, id_column, regions, seed, gmm_components 2,
  output_folder results/toy).
- src/zscore.py: `fit_normal_distribution` (2-component GMM, normal = lowest-mean component)
  and `zscore_regions` (z table + reference normal mean/SD per region).
- steps/S3_zscore.txt (saves results/toy/zscores.csv + zscore_reference.csv).

**Finding (toy data)**
- GMM "normal" is slightly too high and too wide: e.g. L_MTL 1.227/0.119 vs true 1.20/0.10.
  Early-stage tau+ people look nearly normal and get absorbed into the normal component.
  Effect: tau-negative z mean -0.24, SD 0.84 (ideal 0 and 1) -> z-scores a bit compressed.
- Likely also true for real data. Open question: does the paper's choroid-plexus regression or
  fitting choice reduce this? Possible later options: fit GMM only on amyloid-negative CN,
  or more components. Keep the paper's method as the default.

**Note**: SuStaIn (pySuStaIn) does NOT normalize; input must already be z-scores.

## 2026-10-15 – S4: SuStaIn inputs (paper step 6)

**What we did**
- config.yaml: new `sustain:` section (z_events [2,5,10], z_max 15, n_subtypes_max 4,
  n_startpoints 25, n_iterations_mcmc 10000, use_parallel_startpoints false, dataset_name toy).
- src/sustain_model.py: `sustain_inputs` (data, Z_vals, Z_max) and `count_people_above`.
- steps/S4_sustain_inputs.txt.

**Checks**: data (500, 10), Z_vals (10, 3), Z_max (10,). Every event reached by >= 23 people.

**Decisions**
- z_max 15 and n_startpoints 25 are our choices (paper does not report them).

## 2026-10-15 – OPEN TOPIC (later): how to choose Z_max, and how much SuStaIn depends on it

**Why it matters**
- Z_max = the noise-free z-score a region reaches at the very last stage (must be > last cut-point).
- Vogel 2021 does not report it. Original SuStaIn: cut-points 1,2,3 -> Z_max 5.
  Postmortem repo: 1.2 x last cut-point. Our toy data was built with 15.

**Toy-data finding (tau+ people, our GMM z units)**

| Region | max | p99 | p95 | true end value |
|---|---|---|---|---|
| L_MTL | 13.5 | 13.3 | 12.3 | 12.3 |
| R_temporal | 14.1 | 13.9 | 12.7 | 12.9 |
| L_frontal | 14.9 | 13.6 | 12.7 | 13.3 |
| R_frontal | 13.1 | 11.8 | 11.0 | 12.1 |
(full table: run scratch check again or recompute)

- The maximum overshoots (noise + one extreme person; real data may have artifacts).
- The 95th percentile is closest on average (sometimes a bit low).
- The true end value in OUR z units is ~12-13, not 15, because the GMM z-scoring stretched
  the units (normal SD too large). So Z_max depends on how we z-score.

**Candidate strategies**
1. Data-based: per region, 95th (or 99th) percentile of z among clearly abnormal people
   (e.g. z >= 2 in that region, since tau status is unknown in real data); or mean of the
   highest GMM bump.
2. Cross-validation: try a grid (e.g. 12, 15, 20, 25, or the percentile values), compare
   out-of-sample log-likelihood / CVIC at fixed k (fast MCMC settings).
3. Sensitivity (always report): rerun with 2-3 Z_max values; compare subtype assignments
   (confusion matrix like paper Fig. 2b) and stages.

**Plan**
- Now: first SuStaIn run (S5) with fixed Z_max = 15.
- Later: add `z_max_method: fixed | percentile` to config.yaml; run Strategy 3 on toy data to
  answer Rosa's question "how much does the SuStaIn result depend on Z_max?"; then Strategy 2.
- Ask Jacob Vogel which Z_max he used.

## 2026-10-15 – S5: first SuStaIn run (fast test)

**What we did**
- src/sustain_model.py: `run_sustain(...)` creates pySuStaIn's ZscoreSustain and calls
  run_sustain_algorithm(plot=True). Each run type gets its own output folder because
  pySuStaIn reuses existing pickle files instead of refitting.
- config.yaml: `sustain.fast_test: {n_startpoints: 5, n_iterations_mcmc: 1000}`.
- steps/S5_run_sustain_fast.txt.

**Claude's test run (fast settings, k = 1..4)**: 573 s (~9.5 min).
- Most time goes to pySuStaIn's MCMC-tuning step (3 x 10,000 iterations per model,
  independent of n_iterations_mcmc) and to the start points.
- Stage 0: 304 people (truth 300; 299 of them true S0).
- Recovery vs answer key (curiosity): SuStaIn 0 = S1_limbic (66/69), 1 = S3_posterior (58/62),
  2 = S4_lateral_temporal (33/33), 3 = S2_MTL_sparing (33/36). Almost perfect.
  SuStaIn's subtype 2 also took 1 S0, 1 S1, 1 S2, 3 S3 people.

**Next**
- S6: look at results (PVD plots, subtype patterns, stages, probabilities).
- Later: full run (25 start points, 10,000 MCMC) - expect much longer; cross-validation for k.

## 2026-10-15 – S6 part 1: per-person results table

**What we did**
- Rosa ran S5 (fast test) herself: 4 pickle files + 5 figures in results/toy/sustain_fast_test/.
  PVD shows all 4 hidden "wave" orders recovered. MCMC trace flat; log-likelihood
  -15,900 / -12,100 / -10,200 / -9,280 for 1-4 subtypes.
- config.yaml: `results:` section (sustain_folder, n_subtypes 4, poor_fit_threshold 0.5).
- src/results.py: `load_sustain_result` and `subject_table` (stage 0 -> "S0"; poor_fit flag).
- steps/S6_results_table.txt (saves subject_results.csv).

**Findings**
- Counts: S0 304, Subtype 1 66, 2 58, 3 39, 4 33. Poor fit: 0 (fast MCMC is over-confident).
- Average z per subtype names them: 1 limbic, 2 posterior, 3 lateral temporal, 4 MTL-sparing.
- pySuStaIn gives stage-0 people a subtype number too (e.g. TOY_0002 subtype 1, p = 0.55):
  we must relabel them S0 ourselves - confirmed.
- PVD "n" in the figure title = f x all 500 people (includes stage 0), not the assigned count.

**Next**: S6 part 2 - clearer PVD figure with pySuStaIn's own plot function, bigger + readable.

## 2026-10-15 – S7: clearer PVD figure

**What we did**
- Rosa ran S6 herself.
- src/figures.py: `plot_pvd(...)` - calls pySuStaIn's ZscoreSustain.plot_positional_var with
  a bigger figure, larger fonts, stage ticks every 5, our titles, 150 dpi.
- config.yaml: `results.subtype_names: [limbic, posterior, lateral temporal, MTL-sparing]`
  (our interpretation of THIS run - re-check after every new run).
- steps/S7_figures.txt -> results/toy/sustain_fast_test/pvd_4subtypes_all-subtypes.png

**Observation**
- Order of red (z 2) events matches the hidden wave orders. The blue (z 10) events are placed
  later and more uncertain than in the truth (e.g. S1 L_MTL z 10 around stages 7-10 instead of 4),
  probably because our GMM z-scores are compressed (see S3) and Z_max = 15 is high for our units.
  Worth re-checking in the Z_max sensitivity analysis.

## 2026-10-15 – S8: stage-progression figure (paper Fig. 1e idea)

**What we did**
- src/figures.py: `plot_stage_progression(z, table, regions, subtype_titles, stage_edges, save_path)`:
  average z of real tau+ people per subtype x stage group (rows regions, columns stage groups).
- config.yaml: `results.stage_edges: [1, 7, 13, 19, 25, 31]` (groups of 6 stages; paper used 5).
- steps/S8_stage_progression.txt -> results/toy/sustain_fast_test/stage_progression.png

**Observation**: red spreads in the PVD order in every subtype (limbic MTL->temporal->parietal,
occipital last; posterior occipital first; lateral temporal left first; MTL-sparing MTL last).
Groups are small (n = 4-20), so late-stage columns are noisy.

## 2026-10-16 – Plan: Tau-typer app (Shiny for Python, 5 tabs)

**Decisions (Rosa)**
- Shiny for **Python** (same language as pySuStaIn and src/).
- 5 tabs: 1 upload + QC + data plots; 2 normalization (GMM; prepares SuStaIn input - the hard part);
  3 run SuStaIn (user picks hyperparameters, paper defaults; pySuStaIn never changed);
  4 all important/novel summary figures; 5 follow-up analyses comparing the subtypes.
- Build in small steps: empty 5-tab skeleton first, then tab 1, 2, ...
- Cross-validation will be built (needed to choose k).
- Z_max is chosen by the USER in the app (with options + a sensible default).
- The app should run on a server but must be SAFE: it works with the data but nobody can see
  or download other people's data.
- Design rule: the app is only a user interface over src/ functions (same results as step files).

**Privacy options (to decide - see chat 2026-10-16)**
- A. Browser-side app (Shinylive / WebAssembly): served from a server, but the code runs in the
  user's browser, so data never leaves their computer. Must test if pySuStaIn runs there
  (compiled helpers kde_ebm/awkde, pathos) and how slow SuStaIn is in a browser.
- B. Secure institutional server: HTTPS + login, each user sees only their own session,
  data kept in memory only, deleted at session end, never logged; GDPR / data-processing
  agreement needed (server admins could technically access it).
- C. Local install (Docker / package): strongest privacy, no server.

**Open**: which privacy option (A is the only one where the server truly never sees the data).

## 2026-10-16 – Git repository on GitHub

**What we did**
- git init (branch main); .gitignore now also excludes papers/ (copyright), Github/ (others'
  repos) and *.pickle. Still excluded: env/, data/ except data/toy/, results/.
- First commit 5f653ab (35 files). Merged GitHub's own initial commit fb15416 (README.md + MIT
  LICENSE) -> merge 9d5ec76. Pushed to git@github.com:rosaaghdam/tau-typer.git (PRIVATE).
- SSH: removed GitHub's old RSA host keys from ~/.ssh/known_hosts; created a new ED25519 key
  ~/.ssh/id_ed25519 (fingerprint SHA256:Mwa7cndvIkRzTCX75WZW9B0a5MC7GidOGA9BkcmOkoI, with a
  passphrase, stored in the macOS keychain) and added it to GitHub.

**Open**
- LICENSE is MIT (chosen by GitHub's template). Check with the lab before the repo is made public.
- README.md is only 2 lines - write a real one later.
- code/README.md still mentions run.sh (removed) - update.

## 2026-10-16 – App step 1: empty 5-tab skeleton

**What we did**
- environment.yml: added `shiny`; installed into ./env with conda (shiny 1.8.0).
  pySuStaIn still imports fine.
- app/app.py: Shiny for Python, `ui.page_navbar` with 5 tabs (1 Data & QC, 2 Normalization,
  3 Run SuStaIn, 4 Figures, 5 Subtype analysis), placeholder text only, empty server.
- steps/S10_run_app.txt: `shiny run app/app.py`, then open http://127.0.0.1:8000.
- Checked in the browser: title "Tau-typer", all 5 tabs, switching works.
- .claude/launch.json (Claude's browser-pane setting) kept out of git.

**Next**: tab 1 - upload a CSV, match columns, QC tables and plots.

## 2026-10-16 – App tab 1, piece 1: upload + Sample Data + Data Head

**Design decision (Rosa)**: follow her CMiNet app (https://cminet.wid.wisc.edu): each tab = sidebar
(upload, "Sample Data" download with help text, options with one-line help, one action button)
+ main area with small tabs (Data Head, Results). Possible 6th tab "About" (not decided yet).

**What we did**
- app/app.py tab 1: `ui.layout_sidebar` with `input_file` (CSV), Sample Data `download_link`
  (sends data/toy/toy_tau.csv), main `navset_tab` -> "Data Head" (status line + first 20 rows).
- Server: `uploaded_data()` reactive calc (pandas read_csv), `data_status`, `data_head`,
  `download_sample` (`@render.download_link` - `@render.download` is deprecated in shiny 1.8).
- Sidebar `open="always"` (it started folded away otherwise).
- Tested in the browser: Sample Data download = 501 lines; uploading it shows
  "500 rows and 15 columns" + 20-row table.

**Privacy note for later**: Shiny saves an uploaded file to a temporary file on the machine that
runs the app (`datapath`). For the server version we must delete it after reading / at session end.

**Tip**: `shiny run app/app.py --reload` restarts the app automatically when app.py changes.

## 2026-10-16 – Tab 1, piece 2a: guessing the column roles (src/qc.py)

**What we did**
- Committed + pushed the app so far (4ea4e5d).
- src/qc.py: `guess_columns(columns)` - first guess of ID, age, sex, diagnosis, amyloid columns
  (exact name, otherwise first column containing a hint word) and region columns (names starting
  with L_/R_/lh_/rh_/left_/right_). The user always checks the guess in the app.
- Tested: toy columns all correct; made-up names (RID, Age_at_scan, Gender, DX, AB_status,
  lh_temporal, Left-MTL) all correct; no match -> None / [].

**Known weaknesses**
- Partial matching can mislead (e.g. "ml_stage" contains "age" if no better age column exists).
- Desikan-Killiany names (ctx_lh_entorhinal) are not recognised as regions, and DK regions would
  first need averaging into lobes -> depends on the real file format (open question).

**Next**: piece 2b - dropdowns in the tab-1 sidebar, pre-filled with these guesses.

## 2026-10-17 – Tab 1, piece 2b: column-matching dropdowns

**What we did**
- app/app.py: sidebar placeholder `output_ui("column_matching")`; server `column_matching()` builds,
  after an upload, 5 dropdowns (Subject ID, Age, Sex, Diagnosis, Amyloid status; extra choice
  "(not in file)") and a multi-select "Region columns", all pre-filled with `guess_columns()`.
- The app adds the project folder to sys.path so it can import src/.
- Tested in the browser with the sample data: before upload the hint shows; after upload all
  5 roles + the 10 regions are pre-selected correctly.

**Next**: piece 3 - QC summary table (src/qc.py), using the user's column choices.

## 2026-10-17 – Tab 1, piece 3a: QC summary function

**What we did**
- Committed + pushed piece 2 (15d6d44).
- src/qc.py: `qc_summary(data, columns, regions, suvr_min, suvr_max)` -> table Check / Result /
  Status (OK or Warning): people, duplicate IDs, number of regions (>= 2), missing or non-numeric
  region values, values outside the plausible SUVR range, missing age/sex/diagnosis/amyloid
  ("not in file" if the column was not chosen).
- config.yaml: new `qc:` section, suvr_min 0.5 and suvr_max 4.0 (OUR choice, not from the paper).
- Tested: clean toy data all OK; broken copy (1 duplicate ID, 1 text value, 1 SUVR of 25,
  1 missing age, no amyloid column) -> exactly those 5 warnings.

**Next**: piece 3b - show the QC table in the app (new small tab "QC summary").

## 2026-10-17 – Tab 1, piece 3b: QC summary tab in the app

**What we did**
- app/app.py reads config.yaml once (CONFIG) and imports qc_summary.
- New small tab "QC summary": verdict line (green "All checks OK." / red "N warning(s)...")
  + checks table with warning rows highlighted in red (render.DataGrid styles).
- Reactive calcs: `chosen_columns()` (dropdown choices) and `qc_results()` (qc_summary) - the
  table updates by itself on upload or when a dropdown changes.
- Tested in the browser: sample data -> "All checks OK."; Amyloid set to "(not in file)" ->
  "1 warning(s)" and the Amyloid row in red.

**Small UI note**: the Check column wraps into narrow lines - widen later.

**Next**: piece 4 - region histograms and demographics.

## 2026-10-17 – Tab 1, piece 4a: region histograms

**What we did**
- Committed + pushed the QC summary (7f7ad00).
- src/figures.py: `plot_region_histograms(data, regions, n_cols=5)` - one histogram per region
  (raw values), unused panels hidden.
- app/app.py: new small tab "Region distributions" (`output_plot`, 520 px high), redrawn when the
  data or the region selection changes.
- Tested: the plot appears in the app (first draw ~10 s while matplotlib starts). The browser-pane
  screenshot tool did not capture it, so the same figure was also drawn to a file and checked.

**Observation (toy data)**: every region shows a tall normal bump (SUVR ~1.0-1.2) and a SEPARATE
second bump at ~2.0-2.6 (tau+ people). The second bump is an artefact of our toy design: after
z = 10 regions only creep towards Z_max, so many late-stage people pile up there. Real data will
probably show a smoother tail. Worth remembering when we judge the GMM in tab 2.

**Next**: piece 4b - demographics table.

## 2026-10-17 – Tab 1, piece 4b: demographics table

**What we did**
- Committed + pushed the histograms (d9d9303).
- src/qc.py: `demographics_table(data, columns)` - N, age mean (SD), % female, % amyloid+ for
  everyone ("All") and per diagnosis group; roles not in the file are left out.
- app/app.py: new small tab "Demographics".
- Tested: All 500 / AD 81 / CN 350 / MCI 69; age 66.7 (AD) - 74.9 (CN); amyloid+ 100% in MCI and
  AD, 35.4% in CN - same as the S1 checks. Without diagnosis/amyloid: only the "All" row, no
  amyloid column.

**Assumptions to check against the real file (open)**
- Sex: female = value starting with "F". Numeric codes (1/2) would need a setting.
- Amyloid: 1 = positive, 0 = negative. Text codes ("pos"/"neg") give an empty percentage.

**Speed note**: right after an upload the first plot takes ~10-15 s (matplotlib starts) and other
outputs wait. Polish later (e.g. import matplotlib when the app starts).

**Tab 1 status**: pieces 1-4 done. Left: piece 5 - the paper's inclusion rule (checkbox).

## 2026-10-17 – Tab 1, piece 5a: the paper's inclusion rule (function)

**What we did**
- Committed + pushed demographics (73bc6cb).
- src/qc.py: `apply_inclusion_rule(data, columns, min_age_cn, cn_labels, impaired_labels)` ->
  (kept data, reasons table). Keeps CN older than min_age_cn, and MCI/AD with amyloid == 1.
  Labels compared ignoring case. Missing diagnosis/age/amyloid column -> rule not applied, says why.
- config.yaml: `inclusion:` min_age_cn 40, cn_labels [CN, CU, HC, control],
  impaired_labels [MCI, AD, dementia].
- Tested: toy data keeps 500/500 (350 CN + 150 MCI/AD); edited copy removes exactly 3 (CN aged 38,
  AD amyloid-negative, diagnosis "FTD"), keeps an MCI aged 35 and a lower-case "cn";
  no amyloid column -> "Rule NOT applied".

**Open**: which diagnosis labels the real file uses (add them to config.yaml).

**Next**: piece 5b - checkbox in the app; QC, histograms and demographics then use the kept people.

## 2026-10-17 – Tab 1, piece 5b: inclusion-rule checkbox in the app

**What we did**
- Sidebar: checkbox "Apply the paper's inclusion rule (CN older than 40, or MCI/AD amyloid-positive)",
  unticked by default (see the whole file first).
- New reactive chain: uploaded_data() -> included() -> analysis_data(). QC summary, Region
  distributions and Demographics now use analysis_data(); Data Head still shows the raw file.
  Tab 2 will also start from analysis_data().
- QC summary tab: new "Inclusion rule" section (status line + reasons table).
- Tested in the browser with an edited sample (CN aged 38, AD amyloid-negative, diagnosis FTD):
  unticked -> 500 people; ticked -> "497 of 500 people kept", reasons 348/149/1/1/1, QC counts 497,
  Demographics All 497 (AD 80, all amyloid+); unticked again -> 500.

**Tab 1 is complete** (upload, sample data, column matching, QC summary, histograms,
demographics, inclusion rule). Still open for tab 1: widen QC "Check" column, speed of first plot,
real-file assumptions (sex/amyloid coding, diagnosis labels), maybe an "About" tab.

## 2026-10-17 – Tab 2, piece 2.1: sidebar, Normalize button, Normal reference table

**What we did**
- Committed + pushed tab 1 (6982a5a). Tab-2 plan saved in notes/app_tab2_plan.md.
- app/app.py tab 2: sidebar "Number of GMM components" (default from config, 2 = paper) + help text
  + Normalize button; small tab "Normal reference" (status line + normal mean/SD per region).
- Server: `normalized()` runs only on the Normalize click (reactive.event); uses analysis_data()
  from tab 1 and the chosen regions; people with missing region values are left out of the GMM
  (count shown); calls src/zscore.py `zscore_regions` (same as S3).
- Tested: sample data -> "Normalized 500 people in 10 regions (0 left out)"; reference identical to
  S3 (L_MTL 1.227/0.119, L_temporal 1.167/0.096, ...).

**Note**: results update only when Normalize is clicked again - if the data or tab-1 choices
change afterwards, the table is stale until the next click (to flag in the UI later).
