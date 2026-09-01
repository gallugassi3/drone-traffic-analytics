#!/usr/bin/env python
"""Cross-analysis of the tracker matrix runs (analysis/tracker_runs/).

Four questions, answered from the logged trajectories only (report-only):

1. Near-line fragmentation (H2): replay net crossings and cluster them with
   the ground-truth study's method (direction-aware, 80 px Chebyshev,
   15 frames, transitive) per traffic3 run; compare against the corrected
   human ground truth (72 crossings, 39 L / 33 R).
2. Metric reconciliation: distinct output ids vs the id-counter value
   ("16K+ ids" README claim) on traffic2.
3. ReID fragmentation diagnosis: where do botsort_reid's extra tracks live
   in box-size space, vs botsort_gmc on the same static video?
4. Churn attribution on traffic2 (bytetrack): track deaths at the detector's
   size floor vs mid-size losses.

Writes analysis/tracker_study_cross.md.

Usage:
    python scripts/tracker_cross_analysis.py
"""
from __future__ import annotations

import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_crossings import (ANALYSIS_DIR, Obs, cluster_events,
                              load_observations, net_crossings)
from run_tracker_matrix import replay_counts
from track_count import NAMES

RUNS_DIR = ANALYSIS_DIR / "tracker_runs"
REPORT_PATH = ANALYSIS_DIR / "tracker_study_cross.md"

CONFIGS = ["bytetrack", "botsort_gmc", "botsort_reid", "botsort_nogmc",
           "ocsort"]
GT_TOTAL, GT_L, GT_R = 72, 39, 33   # corrected human ground truth, traffic3
LINE_T3 = 640                        # traffic3: axis v, 0.5 * 1280

# Part 4 thresholds (rough split, stated in the report):
FLOOR_AREA = 200      # px^2 (~14x14): at/below this the detector is at its floor
SHRINK_RATIO = 0.5    # died at <= half its own max area (and small-ish)
SHRINK_CAP = 400      # px^2 cap for the "shrank toward the floor" class


def tracks_of(observations: list[Obs]) -> dict[int, list[Obs]]:
    tracks: dict[int, list[Obs]] = defaultdict(list)
    for o in observations:
        tracks[o.tid].append(o)
    return tracks


def part1() -> list[str]:
    lines = ["## 1. Near-line fragmentation on traffic3 (H2)", ""]
    lines.append(
        "Method: the ground-truth study's own estimator - per-track net "
        "crossings (zero-carrying side test), clustered direction-aware "
        "within 80 px Chebyshev / 15 frames, transitive "
        "(`verify_crossings.net_crossings` / `cluster_events`). Counted "
        "totals are the pipeline replay. Human ground truth (this video): "
        f"**{GT_TOTAL} crossings, {GT_L} L / {GT_R} R**. Caveat from the "
        "study: the clustering over-merges adjacent-lane co-crossings "
        "(its bytetrack estimate was 41 events vs 72 true), so treat "
        "cluster counts as a consistent comparator across configs, not an "
        "absolute truth; `net - 72` is the cleanest fragment-excess measure "
        "since all configs saw the same physical 72.")
    lines.append("")
    lines.append("| config | net crossings | excess vs GT 72 (fragments) | "
                 "clusters (study estimator) | multi-track clusters | "
                 "crossings merged by clustering | counted total (L/R) | "
                 "L1 vs GT (total, L, R) |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---:|")
    rows = []
    for cfg in CONFIGS:
        obs = load_observations(RUNS_DIR / f"{cfg}_traffic3.csv")
        tracks = tracks_of(obs)
        nets, jitter, _ = net_crossings(tracks, LINE_T3)
        clusters = cluster_events(nets)
        multi = sum(1 for c in clusters if len(c.members) > 1)
        counted = replay_counts(obs, "v", LINE_T3)
        tot = sum(counted.values())
        left = sum(n for (_, d), n in counted.items() if d == "left")
        right = tot - left
        l1 = abs(tot - GT_TOTAL) + abs(left - GT_L) + abs(right - GT_R)
        lines.append(
            f"| {cfg} | {len(nets)} | {len(nets) - GT_TOTAL:+d} | "
            f"{len(clusters)} | {multi} | {len(nets) - len(clusters)} | "
            f"{tot} ({left}/{right}) | {l1} |")
        rows.append((cfg, len(nets), len(clusters), multi, tot, left, right))
        if jitter:
            lines.append(f"| | | | | | jitter tracks: "
                         f"{[t for t, _ in jitter]} | | |")
    lines.append("")
    best = min(rows, key=lambda r: r[1] - GT_TOTAL)
    spread = (max(r[1] for r in rows) - min(r[1] for r in rows))
    lines.append(
        f"Least dedup work (fewest net crossings in excess of the physical "
        f"72): **{best[0]}** with {best[1] - GT_TOTAL:+d}. The spread across "
        f"configs is small ({spread} crossings), every config lands on the "
        f"same 41-cluster / 24-multi structure, and every config except "
        f"botsort_reid produces the identical counted table (70, 38/32) - "
        f"on the static clip the tracker choice barely moves the counting "
        f"pipeline.")
    lines.append("")
    return lines


def part2() -> list[str]:
    lines = ["## 2. Metric reconciliation: \"16K+ ids\" vs 1,818 unique ids "
             "(traffic2)", ""]
    for name in ("bytetrack_traffic2", "bytetrack_traffic3"):
        obs = load_observations(RUNS_DIR / f"{name}.csv")
        ids = {o.tid for o in obs}
        lines.append(f"* `{name}`: **{len(ids)} distinct output ids**, "
                     f"max id value **{max(ids)}**.")
    lines.append("")
    lines.append(
        "The two numbers measure different things. In ultralytics ByteTrack "
        "(`trackers/byte_tracker.py`), `STrack.activate()` draws a fresh id "
        "from the global counter for **every new candidate track** (line "
        "103), but sets `is_activated = True` only on frame 1 (lines "
        "108-109); a candidate born later becomes visible in the output only "
        "after it is re-associated in a subsequent frame (`re_activate` / "
        "`update`, lines 120/151). Every detection that spawns a candidate "
        "and never matches again burns an id invisibly. On the zoom-out clip "
        "that is ~15.5K of the 17.3K allocated ids; the annotated video's "
        "on-screen labels show the counter's *height* (e.g. `#17245`), which "
        "is where the README's \"16K+ ids in 23s\" reading comes from - it "
        "is the max id value, not the number of tracks that ever appeared.")
    lines.append("")
    lines.append(
        "**Which to quote:** distinct output ids (1,818 in 23.5 s, median "
        "track 8 frames) is the honest churn metric - it counts tracks a "
        "viewer can actually see. The id-counter value (17.3K) additionally "
        "counts one-frame candidates and is implementation-specific "
        "(ByteTrack's eager id allocation); if cited at all it should be "
        "labelled \"id allocations (including never-output candidates)\". "
        "The README's phrasing \"track identity churned (16K+ ids in 23s)\" "
        "overstates output churn by ~10x and should be restated as ~1.8K "
        "visible tracks / 17K id allocations (bytetrack).")
    lines.append("")
    return lines


def part3() -> list[str]:
    lines = ["## 3. ReID fragmentation diagnosis (botsort_reid, static "
             "video)", ""]
    per_cfg: dict[str, dict[int, list[Obs]]] = {}
    for cfg in ("botsort_gmc", "botsort_reid"):
        per_cfg[cfg] = tracks_of(load_observations(RUNS_DIR / f"{cfg}_traffic3.csv"))
    gmc, reid = per_cfg["botsort_gmc"], per_cfg["botsort_reid"]

    def track_stats(tracks: dict[int, list[Obs]]) -> list[tuple[float, int]]:
        return [(statistics.median(o.area for o in t), len(t))
                for t in tracks.values()]

    g, r = track_stats(gmc), track_stats(reid)
    # Area buckets from the GMC run's quartiles.
    areas = sorted(a for a, _ in g)
    q1, q2, q3 = (areas[len(areas) // 4], areas[len(areas) // 2],
                  areas[3 * len(areas) // 4])

    def bucket(stats: list[tuple[float, int]]) -> dict[str, tuple[int, float]]:
        out = {}
        for lbl, lo, hi in (("Q1 (smallest)", 0, q1), ("Q2", q1, q2),
                            ("Q3", q2, q3), ("Q4 (largest)", q3, 1e18)):
            sel = [n for a, n in stats if lo <= a < hi]
            out[lbl] = (len(sel), statistics.median(sel) if sel else 0)
        return out

    bg, br = bucket(g), bucket(r)
    lines.append("| median box area bucket (px^2, GMC quartiles) | "
                 "botsort_gmc tracks (median len) | botsort_reid tracks "
                 "(median len) | extra reid tracks |")
    lines.append("|---|---|---|---:|")
    for lbl in bg:
        (ng, lg), (nr, lr) = bg[lbl], br[lbl]
        lines.append(f"| {lbl} | {ng} ({lg:.0f}) | {nr} ({lr:.0f}) | "
                     f"{nr - ng:+d} |")
    lines.append("")
    med_area_all = statistics.median(a for a, _ in g)
    lines.append(
        f"On the *static* clip, with_reid + model:auto grows the track count "
        f"from {len(gmc)} to {len(reid)} and drops the median track from "
        f"{statistics.median(n for _, n in g):.0f} to "
        f"{statistics.median(n for _, n in r):.0f} frames - and the table "
        f"shows where: the excess concentrates in the smallest-box buckets. "
        f"Mechanism (config + data, not embedding inspection): with "
        f"`with_reid: True, model: auto`, BoT-SORT gates association on "
        f"appearance similarity (`appearance_thresh: 0.8`) computed from "
        f"detector features. The objects here are tiny, low-texture, "
        f"top-down vehicles (cars ~35x17 px; per-track median box area "
        f"~{med_area_all:.0f} px^2 with trucks included): embeddings from "
        f"such crops are unstable frame "
        f"to frame, so the appearance check keeps vetoing matches that IoU "
        f"alone would accept, splitting continuous physical tracks into new "
        f"ids. That is the opposite of what ReID is for, and it also costs "
        f"a broken counted table (69 vs 70). ReID at this object scale needs "
        f"a much lower appearance_thresh or a purpose-trained small-object "
        f"embedder - or simply staying off.")
    lines.append("")
    return lines


def part4() -> list[str]:
    lines = ["## 4. Churn attribution on traffic2 (bytetrack)", ""]
    obs = load_observations(RUNS_DIR / "bytetrack_traffic2.csv")
    tracks = tracks_of(obs)
    last_frame = max(o.frame for o in obs)
    deaths = floor_d = shrink_d = mid_d = short = 0
    final_areas = []
    death_frames = []
    for t in tracks.values():
        if len(t) < 3:
            short += 1
            continue
        if t[-1].frame >= last_frame - 2:
            continue  # survived to clip end
        deaths += 1
        death_frames.append(t[-1].frame)
        fin = statistics.median(o.area for o in t[-2:])
        peak = max(o.area for o in t)
        final_areas.append(fin)
        if fin <= FLOOR_AREA:
            floor_d += 1
        elif fin <= SHRINK_RATIO * peak and fin <= SHRINK_CAP:
            shrink_d += 1
        else:
            mid_d += 1
    det = floor_d + shrink_d
    half = last_frame // 2
    late = sum(1 for f in death_frames if f > half)
    all_areas = [o.area for o in obs]
    med_all = statistics.median(all_areas)
    sens_400 = sum(1 for a in final_areas if a <= 400)
    lines.append(
        f"Of {len(tracks)} bytetrack tracks, {short} have <3 observations "
        f"(excluded), and **{deaths} tracks die before the clip ends**. "
        f"Rough split by the box-area trajectory at death (thresholds "
        f"stated in `scripts/tracker_cross_analysis.py`):")
    lines.append("")
    lines.append(f"* **died at the size floor** (final area <= {FLOOR_AREA} "
                 f"px^2, ~14x14): {floor_d} ({floor_d / deaths:.0%})")
    lines.append(f"* **shrank toward the floor** (final <= "
                 f"{SHRINK_RATIO:.0%} of the track's own peak area and <= "
                 f"{SHRINK_CAP} px^2): {shrink_d} ({shrink_d / deaths:.0%})")
    lines.append(f"* **mid-size losses** (died while comfortably sized - "
                 f"association/occlusion territory): {mid_d} "
                 f"({mid_d / deaths:.0%})")
    lines.append("")
    lines.append(
        f"Detector-driven deaths (first two classes): **{det} of {deaths} "
        f"({det / deaths:.0%})**; {late / deaths:.0%} of deaths fall in the "
        f"second half of the clip (after the zoom-out has shrunk the "
        f"scene).")
    lines.append("")
    lines.append(
        f"Sensitivity note - this split is threshold-dependent and should "
        f"be read as a rough attribution, not a precise one: the median box "
        f"area *at death* is {statistics.median(final_areas):.0f} px^2, "
        f"versus {med_all:.0f} px^2 for all boxes in the clip - deaths skew "
        f"toward small boxes, and many \"mid-size\" deaths sit just above "
        f"the 200 px^2 floor cut. Widening the floor to 400 px^2 "
        f"reclassifies {sens_400 / deaths:.0%} as detector-driven, but that "
        f"bound is generous (400 px^2 exceeds the clip's median box, since "
        f"the zoom-out leaves *everything* small). So the fair statement "
        f"is: **roughly half of all track deaths (42% at a strict floor, "
        f"more under looser cuts) happen at or near the detector's size "
        f"floor**, which supports "
        f"\"the detector's small-object limit, not tracker association, is "
        f"the first-order bottleneck on this clip\" - but a substantial "
        f"minority of comfortable-size losses remains, so the tracker is "
        f"not blameless.")
    lines.append("")
    return lines


def main() -> None:
    lines = ["# Tracker study - cross-analysis", ""]
    lines.append(
        "Generated by `scripts/tracker_cross_analysis.py` on 2026-09-01 from "
        "the matrix runs in `analysis/tracker_runs/` "
        "(see `analysis/tracker_study_raw.md`). Report-only; no pipeline "
        "changes.")
    lines.append("")
    for part in (part1, part2, part3, part4):
        chunk = part()
        lines.extend(chunk)
        print("\n".join(chunk))
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
