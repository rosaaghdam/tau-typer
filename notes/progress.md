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
