# Final fact-check - tracker-study release (README.md + tracker_study.md)

Report-only review, 2026-09-01. Sources checked against:
`analysis/tracker_study_raw.md`, `analysis/tracker_study_cross.md`,
`analysis/tracker_runs/matrix_run.log`, `analysis/fix_verification.md`, the
run CSVs, and the tracker internals where cited. Severity: **HIGH** (blocks
release), **MEDIUM** (fix before publishing), **LOW** (worth fixing),
**INFO** (aware, no action needed).

**Verdict: one HIGH (a broken link to the study itself), four MEDIUMs, and
a batch of LOW precision issues. Every matrix cell and almost every derived
number is correct; the problems are placement, attribution, and two
overstated claims.**

## 1. Number verification

### Verified correct (both documents, against the sources)

* **Results matrix, cell by cell (10 runs x 5 metrics)**: every value in
  tracker_study.md section 4 and the README's condensed table matches
  tracker_study_raw.md / matrix_run.log exactly (bytetrack 70/339/6/14.9
  and 18/1,818/8/9.9; botsort_gmc 70/323/6/10.1 and 21/1,658/11/7.3;
  botsort_reid 69/554/4/10.4 and 23/2,616/7/7.5; botsort_nogmc
  70/323/6/13.7 and 19/1,722/9/10.2; ocsort 70/302/8/12.2 and
  19/1,904/9/8.7).
* Determinism line (70 counted / 339 ids / 14,039 boxes), "41-cluster /
  24-multi", fragment excess "+14 to +18", identical counted table
  "(70, 38L/32R)" for all non-ReID configs - all match the cross-analysis.
* "1,818 distinct output ids / 17,309 allocations / 23.5s / median track 8
  frames / ~10x" (17,309 / 1,818 = 9.5) - matches the cross-analysis and
  the CSV.
* ReID: "+126 of the +231 extra tracks" (554 - 323 = 231; bucket deltas
  +126 +61 +37 +7 = 231), "median track 2 frames vs 9 in the same bucket",
  "~35x17px", "appearance_thresh 0.8", "only config that breaks the table
  (69)" - all match.
* Churn: 1,402 deaths; 39% / 4% / 58% split; 42% strict / 74% generous;
  medians 240 vs 330 px^2; "threshold-sensitive and reported as such" - all
  match the cross-analysis.
* "98 -> 62 switches, zero oscillation" (fix_verification.md s9); the
  "+15 / -17" naive-counter row; "97%" (70/72); "14,039 trajectory points
  across 339 tracks"; the two residual-miss descriptions (23px-vs-25px gate
  case c31; nose-to-tail convoy c32) - all match their sources.
* Percent claims that check out: "4-9%" GMC id reduction (1,658 vs 1,722 =
  -3.7%, vs 1,818 = -8.8%); "~18%" OC-SORT speed cost (12.2 vs 14.9 =
  -18.1%); "1.6-1.7x" ReID fragmentation (1.72 static, 1.58 zoom-out).
* Hypothesis verdicts H1 (all 69-70, within the predicted 68-72), H2
  (refuted: BoT-SORT had *more* fragment excess, +18 vs +15, spread 4), H4
  (refuted: numbers as stated), H5 - all consistent with the data.

### Findings

* **HIGH - the README's link to the study is broken.** README links
  [`analysis/tracker_study.md`](analysis/tracker_study.md), but the file
  sits at the **repo root** (`tracker_study.md`, untracked). Moving it to
  `analysis/` also fixes its own bare reference to
  `tracker_study_cross.md`, which resolves only from inside `analysis/`.
* **MEDIUM - "+22% over BoT-SORT" is mispaired** (tracker_study.md
  section 6). ByteTrack 14.9 FPS is +22% over **OC-SORT** (12.2); over
  BoT-SORT it is +48% (stock GMC, 10.1) or +9% (nogmc, 13.7). Either
  change the comparator to OC-SORT or the number to +48%.
* **MEDIUM - H6 is marked "Confirmed" but is only half-confirmed.**
  "ByteTrack is fastest" holds (14.9). "ReID pays the largest speed cost"
  is **contradicted by the data**: botsort_gmc is the slowest config in
  *both* arenas (10.1 / 7.3 FPS vs ReID's 10.4 / 7.5) - the big cost is
  GMC's sparse optical flow, not ReID (`model: auto` reuses detector
  features, nearly free). The verdict should read "half-confirmed" or
  split the two clauses. Related LOW: H3's "slowest with GMC" is ambiguous
  - as written it can be read as "ReID is slowest"; the two GMC-enabled
  configs being the two slowest is the accurate statement.
* **LOW - README "~11-15 FPS ... (tracking pass)"**: measured tracking FPS
  is 9.9-14.9 for the default ByteTrack and 7.3-14.9 across configs. The
  "11" floor matches nothing measured; suggest "10-15 FPS (ByteTrack;
  BoT-SORT variants slower)".
* **LOW - README "Roughly 40-74% of track deaths"**: the source range is
  **42%** (strict floor) to 74% (generous); 39% is the floor-only class.
  "40" matches neither; use 42-74%.
* **LOW - MOT17 "HOTA 63.7 vs 60.1 after per-tracker grid search" is
  uncited** and not verifiable from repo sources. The original papers
  report different numbers (ByteTrack ~63.1 MOT17 test); the 63.7 figure
  matches the Roboflow Trackers benchmark comparison
  (trackers.roboflow.com), which does per-tracker grid search - cite that
  source explicitly so it does not read as a misquote of the papers.
* **LOW - "26% speed cost" for GMC is the traffic3 figure only**; traffic2
  is 28% (10.2 -> 7.3). "26-28%" would cover both.
* **LOW - OC-SORT "longest median tracks"** holds on the static clip
  (8 > 6) but not the zoom-out (botsort_gmc 11 > ocsort 9); scope it to
  static or drop it.

## 2. GIF paths and demo.gif references

* All three README GIFs exist: `assets/demo_highway.gif` (2.6 MB),
  `assets/demo_intersection.gif` (2.6 MB), `assets/demo_zoomout.gif`
  (3.2 MB). `assets/demo.gif` is deleted in the working tree.
* Grep over all tracked files: the **only** remaining `demo.gif` reference
  is in `analysis/final_review.md` - the earlier audit report describing
  the repo state *at that time* ("assets/demo.gif is 12.9 MB"). INFO:
  acceptable as an audit-trail statement; optionally annotate it as
  historical.
* INFO: the three new GIFs total 8.4 MB - down from 12.9 MB but still the
  dominant clone weight.

## 3. Link and repo-state resolution

* **MEDIUM - `analysis/tracker_runs/` is NOT gitignored.** The premise
  that it is now ignored is false: `.gitignore` has no entry for it, and
  `git status` shows the whole directory (10 CSVs + matrix_run.log,
  **17 MB**) as untracked - a `git add .` commits all of it. Add
  `analysis/tracker_runs/` to `.gitignore` before committing.
* **MEDIUM - tracker_study_raw.md cites the run CSVs with no provenance
  note.** Every per-run section says "Trajectories:
  `analysis/tracker_runs/<config>_<video>.csv`" - for a repo visitor those
  files will not exist. It needs the same "generated locally by the
  scripts, deliberately not committed (re-run
  `scripts/run_tracker_matrix.py` to regenerate)" note the README uses for
  the other evidence. The cross-analysis report inherits the same issue
  via its header reference to `analysis/tracker_runs/`.
* Links that resolve to tracked files: `analysis/crossing_verification.md`,
  `analysis/fix_verification.md`, `scripts/verify_crossings.py`,
  `analysis/display_proof/` (INDEX + 5 jpgs, tracked). Links that resolve
  to **untracked-but-present** files (fine once committed):
  `analysis/tracker_study_raw.md`, `analysis/tracker_study_cross.md`; the
  Setup example needs `trackers/` (untracked) and the study text references
  `scripts/run_tracker_matrix.py` / `scripts/tracker_cross_analysis.py`
  (untracked). All must be included in the release commit.
* External links (GitHub repo, release tag v1.0 incl. the exact weights
  filename, profile) were verified live earlier today (HTTP 200); Pexels
  returns 403 to curl (bot-blocking) but loads in a browser.

## 4. Sanity checks

* `python -m py_compile track_count.py scripts/run_tracker_matrix.py
  scripts/tracker_cross_analysis.py` - **all pass**.
* All five `trackers/*.yaml` parse with PyYAML and carry the intended
  ablation fields (botsort_gmc: sparseOptFlow/False; botsort_reid:
  sparseOptFlow/True; botsort_nogmc: none/False; bytetrack and ocsort:
  stock).

## Summary

| Severity | Count | Items |
|---|---:|---|
| HIGH | 1 | README links `analysis/tracker_study.md`; file is at repo root |
| MEDIUM | 4 | tracker_runs/ not gitignored (17 MB); "+22% over BoT-SORT" mispaired; H6 marked Confirmed though its ReID-cost clause is contradicted; raw report lacks the generated-locally note |
| LOW | 6 | "~11-15 FPS" floor; "40-74%" vs 42-74%; uncited MOT17 numbers; "26%" is one-arena; H3 "slowest with GMC" ambiguity; OC-SORT "longest tracks" scope |
| INFO | 3 | historical demo.gif mention in the audit doc; 8.4 MB of GIFs; untracked files that must ship in the release commit |
