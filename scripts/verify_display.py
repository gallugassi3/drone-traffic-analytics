#!/usr/bin/env python
"""Display-layer (class color/label) stability analysis for track_count.py.

Simulates display policies per track against the logged trajectories
(analysis/trajectories_traffic3.csv):

  layer 1 (counted):   window-majority over CLS_WINDOW frames (untouched)
  layer 2 (displayed): a display policy on top

and reports switch statistics (switches per track, age at switch, class
pairs, top flickering tracks) for the current policy and the proposed one,
plus diagnosis data (window-majority run lengths) for the worst tracks.
Also replays the counted table to prove the display change cannot move it.

Diagnosis result (2026-09-01): the window majority itself oscillates in runs
of 11-95 frames on ~50/50 lookalike tracks (car/van, truck/bus), so any
run-length gate (consecutive-disagreement) either passes the oscillation or
freezes genuine changes. The proposed policy therefore discriminates on
integrated vote share (long-half-life decayed mass + Schmitt margin) and
snaps the display to the counted class at the crossing moment.

Usage:
    python scripts/verify_display.py
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_crossings import ANALYSIS_DIR, Obs, load_observations
from track_count import (CLS_WINDOW, COUNT_CLASSES, DEDUP_ACROSS,
                         DEDUP_ALONG, DEDUP_WINDOW, DISP_DWELL,
                         DISP_HALF_LIFE, DISP_MARGIN, DISP_MIN_MASS, NAMES,
                         window_majority)

CSV_PATH = ANALYSIS_DIR / "trajectories_traffic3.csv"
LINE_POS = 640

# The replaced display policy (track_count.py's old CLS_STABLE), kept here so
# the before/after comparison in fix_verification.md section 9 stays
# reproducible.
CLS_STABLE = 10

# The ground-truth-verified counted table that must stay bit-identical.
EXPECTED_COUNTS: dict[tuple[str, str], int] = {
    ("bus", "right"): 2,
    ("car", "left"): 24, ("car", "right"): 14,
    ("motor", "right"): 1,
    ("truck", "left"): 8, ("truck", "right"): 11,
    ("van", "left"): 6, ("van", "right"): 4,
}


def majorities_of(raw: list[int]) -> list[int]:
    out: list[int] = []
    hist: list[int] = []
    for c in raw:
        hist.append(c)
        hist = hist[-CLS_WINDOW:]
        out.append(window_majority(hist))
    return out


# ---------------------------------------------------------------------------
# Display policies. Input: window majorities, raw classes, and the crossing
# moment (obs index, counted class) or None. Output: displayed class per obs.
# ---------------------------------------------------------------------------

def display_current(majorities: list[int], raw: list[int],
                    cross: tuple[int, int] | None) -> list[int]:
    """Shipped policy: switch after CLS_STABLE consecutive disagreements."""
    shown: int | None = None
    disagree = 0
    out: list[int] = []
    for m in majorities:
        if shown is None:
            shown = m
        elif m != shown:
            disagree += 1
            if disagree >= CLS_STABLE:
                shown = m
                disagree = 0
        else:
            disagree = 0
        out.append(shown)
    return out


def display_proposed(majorities: list[int], raw: list[int],
                     cross: tuple[int, int] | None) -> list[int]:
    """Schmitt trigger on long-horizon decayed vote mass + snap-on-count.

    The displayed class switches to the top class only when its decayed vote
    mass (half-life DISP_HALF_LIFE) exceeds DISP_MARGIN times the incumbent's
    AND an absolute floor (DISP_MIN_MASS) - a share-level criterion that a
    ~50/50 oscillation can never satisfy at this horizon, while a genuine
    sustained change does. After any switch a short dwell suppresses further
    switches. At the crossing moment the display snaps to the counted class,
    so the video always shows the label being counted.
    """
    decay = 0.5 ** (1.0 / DISP_HALF_LIFE)
    mass: dict[int, float] = defaultdict(float)
    shown: int | None = None
    dwell = 0
    out: list[int] = []
    for i, c in enumerate(raw):
        for k in mass:
            mass[k] *= decay
        mass[c] += 1.0
        if shown is None:
            shown = majorities[i]
        elif cross is not None and i == cross[0]:
            if shown != cross[1]:
                shown = cross[1]
                dwell = DISP_DWELL
        elif dwell > 0:
            dwell -= 1
        else:
            top = max(mass, key=lambda k: mass[k])
            if (top != shown and mass[top] >= DISP_MIN_MASS
                    and mass[top] >= DISP_MARGIN * mass.get(shown, 0.0)):
                shown = top
                dwell = DISP_DWELL
        out.append(shown)
    return out


# ---------------------------------------------------------------------------

def replay_counted(observations: list[Obs]
                   ) -> tuple[Counter, dict[int, tuple[int, int]]]:
    """Exact replay of the counting path. Returns the counted table and, per
    track, the crossing moment (frame, counted class) - for kept AND
    dedup-suppressed crossings (the display snap applies to both)."""
    cls_hist: dict[int, list[int]] = defaultdict(list)
    last_pos: dict[int, int] = {}
    counted: set[int] = set()
    counts: Counter = Counter()
    recent: list[tuple[int, int, int, str]] = []
    cross_moment: dict[int, tuple[int, int]] = {}
    for o in observations:
        cls_hist[o.tid].append(o.cls)
        cls_hist[o.tid] = cls_hist[o.tid][-CLS_WINDOW:]
        c_count = window_majority(cls_hist[o.tid])
        coord = o.cx
        if c_count in COUNT_CLASSES and o.tid in last_pos and o.tid not in counted:
            if (last_pos[o.tid] - LINE_POS) * (coord - LINE_POS) < 0:
                direction = "right" if coord > last_pos[o.tid] else "left"
                along, across = o.cx, o.cy
                dup = any(o.frame - f <= DEDUP_WINDOW and d == direction
                          and abs(across - pa) < DEDUP_ACROSS
                          and abs(along - pl) < DEDUP_ALONG
                          for f, pl, pa, d in recent)
                if not dup:
                    counts[(NAMES[c_count], direction)] += 1
                    recent.append((o.frame, along, across, direction))
                    recent[:] = [r for r in recent
                                 if o.frame - r[0] <= DEDUP_WINDOW]
                counted.add(o.tid)
                cross_moment[o.tid] = (o.frame, c_count)
        if coord != LINE_POS:
            last_pos[o.tid] = coord
    return counts, cross_moment


def majority_runs(majorities: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for m in majorities:
        if runs and runs[-1][0] == m:
            runs[-1] = (m, runs[-1][1] + 1)
        else:
            runs.append((m, 1))
    return runs


def analyse(name: str, tracks: dict[int, list[Obs]], policy,
            cross_moment: dict[int, tuple[int, int]]
            ) -> dict[int, list[tuple[int, int, int, int]]]:
    per_track: dict[int, list[tuple[int, int, int, int]]] = {}
    for tid, obs in tracks.items():
        raw = [o.cls for o in obs]
        frames = [o.frame for o in obs]
        cross = None
        if tid in cross_moment:
            cf, cc = cross_moment[tid]
            cross = (frames.index(cf), cc)
        shown = policy(majorities_of(raw), raw, cross)
        ev = [(i, frames[i], shown[i - 1], shown[i])
              for i in range(1, len(shown)) if shown[i] != shown[i - 1]]
        if ev:
            per_track[tid] = ev
    n_switches = sum(len(v) for v in per_track.values())
    pairs = Counter()
    ages = []
    for ev in per_track.values():
        for age, _, a, b in ev:
            pairs[tuple(sorted((NAMES[a], NAMES[b])))] += 1
            ages.append(age)
    print(f"\n=== {name} ===")
    print(f"total displayed-class switches: {n_switches} across "
          f"{len(per_track)} of {len(tracks)} tracks")
    print("switches-per-track histogram:",
          dict(sorted(Counter(len(v) for v in per_track.values()).items())))
    if ages:
        ages.sort()
        print(f"age at switch (obs index): min {ages[0]}, median "
              f"{ages[len(ages) // 2]}, max {ages[-1]}; "
              f"{sum(a < CLS_WINDOW for a in ages)} of {len(ages)} before the "
              f"window fills (age < {CLS_WINDOW})")
    print("class pairs:", dict(pairs.most_common()))
    top = sorted(per_track.items(), key=lambda kv: -len(kv[1]))[:10]
    print("top flickering tracks:", [(f"#{tid}", len(ev)) for tid, ev in top])
    return per_track


def main() -> None:
    observations = load_observations(CSV_PATH)
    tracks: dict[int, list[Obs]] = defaultdict(list)
    for o in observations:
        tracks[o.tid].append(o)

    counts, cross_moment = replay_counted(observations)
    if dict(counts) != EXPECTED_COUNTS:
        raise RuntimeError(f"counted table drifted: {dict(sorted(counts.items()))}")
    print(f"counted table bit-identical to verified table: True "
          f"(total {sum(counts.values())})")

    cur = analyse(f"CURRENT: consecutive-disagreement (CLS_STABLE={CLS_STABLE})",
                  tracks, display_current, cross_moment)
    analyse(f"PROPOSED: decayed-mass Schmitt (half-life {DISP_HALF_LIFE:.0f}, "
            f"margin {DISP_MARGIN}, floor {DISP_MIN_MASS}, dwell {DISP_DWELL})"
            f" + snap-on-count", tracks, display_proposed, cross_moment)

    print("\n=== diagnosis: window-majority run lengths (top current "
          "flickerers) ===")
    for tid, ev in sorted(cur.items(), key=lambda kv: -len(kv[1]))[:10]:
        raw = [o.cls for o in tracks[tid]]
        runs = majority_runs(majorities_of(raw))
        votes = Counter(NAMES[c] for c in raw)
        print(f"#{tid}: {len(raw)} obs, raw votes {dict(votes.most_common())}, "
              f"majority runs: {' '.join(f'{NAMES[m]}x{n}' for m, n in runs)}")

    # The #1619 requirement: displayed class at its crossing must be motor.
    for pname, policy in (("current", display_current),
                          ("proposed", display_proposed)):
        obs1619 = tracks[1619]
        frames = [o.frame for o in obs1619]
        cf, cc = cross_moment[1619]
        shown = policy(majorities_of([o.cls for o in obs1619]),
                       [o.cls for o in obs1619], (frames.index(cf), cc))
        at_cross = NAMES[shown[frames.index(723)]]
        print(f"#1619 displayed at crossing (f723), {pname}: {at_cross}; "
              f"labels shown over life: "
              f"{[NAMES[k] for k in dict.fromkeys(shown)]}")


if __name__ == "__main__":
    main()
