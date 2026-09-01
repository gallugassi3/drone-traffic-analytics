#!/usr/bin/env python
"""Independent verification of track_count.py's line-crossing counts.

Re-runs the exact same detector + tracker configuration on a video, logs every
track's center trajectory to a CSV, then - independently of track_count.py's
counting loop - detects line crossings, clusters near-coincident crossings
into physical crossing events, saves a crop image per event for human
spot-checking, and writes a markdown report comparing:

  * the no-dedup counts ("BEFORE": committed track_count.py behaviour),
  * the dedup counts    ("AFTER":  current working-tree behaviour),
  * the independently clustered physical events (this script).

Both BEFORE and AFTER are replayed on the logged trajectories with an exact
port of track_count.py's loop, so the user-reported BEFORE table doubles as a
determinism check for the re-run.

Report-only: never modifies track_count.py or its outputs.

Usage:
    python scripts/verify_crossings.py                  # full run (CPU: minutes)
    python scripts/verify_crossings.py --reuse-csv      # skip inference, reuse CSV
    python scripts/verify_crossings.py --max-frames 60  # quick smoke test
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import the operative constants from the code under test so the replay can
# never drift from it. Importing runs only module-level definitions.
from track_count import (CONF, COUNT_CLASSES, DEDUP_WINDOW,
                         IMGSZ, NAMES, TRACKER, WEIGHTS)

# The pre-fix dedup radius (track_count.py's old DEDUP_RADIUS, removed by the
# 2026-09-01 fixes). This script replays that historical algorithm and also
# uses the same value as its clustering radius, so the original study
# (analysis/crossing_verification.md, clusters c01-c41) stays reproducible.
LEGACY_RADIUS = 80

ANALYSIS_DIR = PROJECT_ROOT / "analysis"
CROPS_DIR = ANALYSIS_DIR / "crossing_crops"
REPORT_PATH = ANALYSIS_DIR / "crossing_verification.md"

# The BEFORE-dedup table reported from the original run of the committed code
# (traffic3.mp4, --axis v --line 0.5). Used as a determinism cross-check.
EXPECTED_BEFORE: dict[tuple[str, str], int] = {
    ("bus", "left"): 1, ("bus", "right"): 3,
    ("car", "left"): 20, ("car", "right"): 14,
    ("motor", "right"): 1,
    ("truck", "left"): 6, ("truck", "right"): 17,
    ("van", "left"): 5, ("van", "right"): 3,
}

NEAR_LINE_PX = 60      # birth/death within this distance of the line is suspicious
MIN_TRACK_OBS = 3      # ignore ultra-short tracks in the near-line diagnostics
MIN_DISPLACEMENT = 30  # px: ignore stationary tracks in the near-line diagnostics
CROP_PAD = 40          # px padding around the union of member boxes in crops
CROP_MIN_SIZE = 220    # px minimum crop side


@dataclass(frozen=True)
class Obs:
    """One tracked box in one frame (frame indices are 1-based)."""
    frame: int
    tid: int
    cls: int
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2

    @property
    def area(self) -> int:
        return max(self.x2 - self.x1, 0) * max(self.y2 - self.y1, 0)


@dataclass(frozen=True)
class Crossing:
    """A sign change of a track center across the counting line."""
    prev: Obs   # last observation on the near side
    obs: Obs    # first observation on the far side (the crossing moment)
    direction: str  # "left" | "right"


@dataclass(frozen=True)
class NetCross:
    """A track that ended up on the other side of the line from where it started."""
    tid: int
    cls: int          # majority class over the whole track
    rep: Crossing     # representative crossing event (the last raw crossing)
    n_raw: int        # number of raw sign changes for this track


@dataclass
class Cluster:
    """A group of net crossings judged to be one physical crossing event."""
    idx: int
    members: list[NetCross]

    @property
    def direction(self) -> str:
        return self.members[0].rep.direction

    @property
    def rep_frame(self) -> int:
        return min(m.rep.obs.frame for m in self.members)

    @property
    def cls_name(self) -> str:
        # A split vehicle's largest fragment is the best guess for its true class.
        biggest = max(self.members, key=lambda m: m.rep.obs.area)
        return NAMES[biggest.cls]

    @property
    def is_mixed_class(self) -> bool:
        return len({m.cls for m in self.members}) > 1


# ---------------------------------------------------------------------------
# Step 1: re-run the tracker and log every trajectory
# ---------------------------------------------------------------------------

def run_tracker(video: Path, csv_path: Path, max_frames: int) -> list[Obs]:
    from ultralytics import YOLO  # deferred: --reuse-csv skips model loading

    weights = PROJECT_ROOT / WEIGHTS
    if not weights.is_file():
        raise FileNotFoundError(f"model weights not found: {weights}")

    model = YOLO(str(weights))
    observations: list[Obs] = []
    t0 = time.time()
    n_frames = 0

    for n_frames, result in enumerate(
            model.track(source=str(video), imgsz=IMGSZ, conf=CONF,
                        tracker=TRACKER, persist=True, stream=True,
                        verbose=False), start=1):
        if result.boxes.id is not None:
            ids = result.boxes.id.int().tolist()
            clss = result.boxes.cls.int().tolist()
            confs = result.boxes.conf.tolist()
            for box, tid, c, cf in zip(result.boxes.xyxy, ids, clss, confs):
                x1, y1, x2, y2 = map(int, box)  # same truncation as track_count.py
                observations.append(Obs(n_frames, tid, c, float(cf), x1, y1, x2, y2))
        if n_frames % 100 == 0:
            fps = n_frames / (time.time() - t0)
            print(f"  frame {n_frames} ({fps:.1f} FPS, {len(observations)} boxes)",
                  flush=True)
        if max_frames and n_frames >= max_frames:
            break

    if not observations:
        raise RuntimeError(f"tracker produced zero tracked boxes on {video}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "track_id", "class_id", "class_name", "conf",
                     "x1", "y1", "x2", "y2", "cx", "cy"])
        for o in observations:
            wr.writerow([o.frame, o.tid, o.cls, NAMES[o.cls], f"{o.conf:.4f}",
                         o.x1, o.y1, o.x2, o.y2, o.cx, o.cy])
    print(f"Processed {n_frames} frames, {len(observations)} tracked boxes "
          f"-> {csv_path}")
    return observations


def load_observations(csv_path: Path) -> list[Obs]:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"--reuse-csv given but {csv_path} does not exist; run without it first")
    observations: list[Obs] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            observations.append(Obs(
                int(row["frame"]), int(row["track_id"]), int(row["class_id"]),
                float(row["conf"]), int(row["x1"]), int(row["y1"]),
                int(row["x2"]), int(row["y2"])))
    if not observations:
        raise RuntimeError(f"{csv_path} contains no observations")
    return observations


# ---------------------------------------------------------------------------
# Step 2a: exact replay of track_count.py's counting loop (BEFORE and AFTER)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimCrossing:
    obs: Obs
    direction: str


@dataclass(frozen=True)
class Suppressed:
    obs: Obs
    direction: str
    anchor: SimCrossing  # the earlier counted crossing that absorbed this one


def simulate_track_count(observations: list[Obs], line_pos: int, dedup: bool,
                         ) -> tuple[Counter, list[SimCrossing], list[Suppressed]]:
    """Exact port of track_count.py's per-box loop over the logged trajectories.

    `observations` must be in original order (frame ascending, tracker box
    order within each frame) - the dedup outcome depends on that order.
    """
    last_pos: dict[int, int] = {}
    counted: set[int] = set()
    counts: Counter = Counter()
    recent: list[SimCrossing] = []
    kept: list[SimCrossing] = []
    suppressed: list[Suppressed] = []

    for o in observations:
        coord = o.cx  # vertical line -> the x coordinate is the crossing axis
        if o.cls in COUNT_CLASSES and o.tid in last_pos and o.tid not in counted:
            if (last_pos[o.tid] - line_pos) * (coord - line_pos) < 0:
                direction = "right" if coord > last_pos[o.tid] else "left"
                anchor: SimCrossing | None = None
                if dedup:
                    for r in recent:
                        if (o.frame - r.obs.frame <= DEDUP_WINDOW
                                and abs(o.cx - r.obs.cx) < LEGACY_RADIUS
                                and abs(o.cy - r.obs.cy) < LEGACY_RADIUS):
                            anchor = r
                            break
                if anchor is None:
                    counts[(NAMES[o.cls], direction)] += 1
                    sc = SimCrossing(o, direction)
                    kept.append(sc)
                    recent.append(sc)
                    recent[:] = [r for r in recent
                                 if o.frame - r.obs.frame <= DEDUP_WINDOW]
                else:
                    suppressed.append(Suppressed(o, direction, anchor))
                counted.add(o.tid)
        last_pos[o.tid] = coord
    return counts, kept, suppressed


# ---------------------------------------------------------------------------
# Step 2b: independent crossing detection (zero-carrying side test)
# ---------------------------------------------------------------------------

def detect_track_crossings(track_obs: list[Obs], line_pos: int) -> list[Crossing]:
    """All sign changes of the track center across the line.

    Unlike track_count.py's strict `< 0` product test, a center that lands
    exactly ON the line keeps the previous side, so such crossings are still
    detected on the next off-line observation.
    """
    crossings: list[Crossing] = []
    prev_side = 0
    prev_obs: Obs | None = None
    for o in track_obs:
        side = (o.cx > line_pos) - (o.cx < line_pos)
        if side != 0:
            if prev_side != 0 and side != prev_side:
                assert prev_obs is not None
                direction = "right" if side > 0 else "left"
                crossings.append(Crossing(prev_obs, o, direction))
            prev_side = side
            prev_obs = o
    return crossings


def net_crossings(tracks: dict[int, list[Obs]], line_pos: int,
                  ) -> tuple[list[NetCross], list[tuple[int, int]], int]:
    """Per-track net outcome.

    Returns (net crossings, jitter tracks as (tid, n_raw), total raw crossings).
    Raw crossing directions alternate, so an odd count means the track really
    ended up on the other side (a net crossing, direction = the last raw one);
    an even count means it wobbled across and came back (jitter, no net cross).
    Only tracks whose majority class is in COUNT_CLASSES are considered.
    """
    nets: list[NetCross] = []
    jitter: list[tuple[int, int]] = []
    total_raw = 0
    for tid, track_obs in tracks.items():
        majority_cls = Counter(o.cls for o in track_obs).most_common(1)[0][0]
        if majority_cls not in COUNT_CLASSES:
            continue
        crossings = detect_track_crossings(track_obs, line_pos)
        total_raw += len(crossings)
        if not crossings:
            continue
        if len(crossings) % 2 == 1:
            nets.append(NetCross(tid, majority_cls, crossings[-1], len(crossings)))
        else:
            jitter.append((tid, len(crossings)))
    return nets, jitter, total_raw


# ---------------------------------------------------------------------------
# Step 3: cluster net crossings into physical events
# ---------------------------------------------------------------------------

def cluster_events(nets: list[NetCross]) -> list[Cluster]:
    """Connected components under: same direction, |dframe| <= DEDUP_WINDOW,
    Chebyshev distance < LEGACY_RADIUS (the same metric track_count.py uses,
    but transitive and direction-aware)."""
    n = len(nets)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = nets[i].rep.obs, nets[j].rep.obs
            if (nets[i].rep.direction == nets[j].rep.direction
                    and abs(a.frame - b.frame) <= DEDUP_WINDOW
                    and abs(a.cx - b.cx) < LEGACY_RADIUS
                    and abs(a.cy - b.cy) < LEGACY_RADIUS):
                parent[find(i)] = find(j)

    groups: dict[int, list[NetCross]] = defaultdict(list)
    for i, nc in enumerate(nets):
        groups[find(i)].append(nc)
    clusters = [Cluster(0, sorted(ms, key=lambda m: m.rep.obs.frame))
                for ms in groups.values()]
    clusters.sort(key=lambda c: c.rep_frame)
    for k, c in enumerate(clusters, start=1):
        c.idx = k
    return clusters


# ---------------------------------------------------------------------------
# Step 4: crop images for human spot-checking
# ---------------------------------------------------------------------------

def save_crops(clusters: list[Cluster], video: Path, line_pos: int,
               obs_index: dict[tuple[int, int], Obs]) -> dict[int, Path]:
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    for stale in CROPS_DIR.glob("*.jpg"):
        stale.unlink()

    by_frame: dict[int, list[Cluster]] = defaultdict(list)
    for c in clusters:
        by_frame[c.rep_frame].append(c)

    paths: dict[int, Path] = {}
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video for crop extraction: {video}")
    frame_idx = 0
    pending = set(by_frame)
    while pending:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(
                f"video ended at frame {frame_idx} but crops still pending "
                f"for frames {sorted(pending)}")
        frame_idx += 1
        if frame_idx not in pending:
            continue
        pending.discard(frame_idx)
        h, w = frame.shape[:2]
        for c in by_frame[frame_idx]:
            boxes = [obs_index[(frame_idx, m.tid)] for m in c.members
                     if (frame_idx, m.tid) in obs_index]
            if not boxes:  # member boxes live in a neighbouring frame
                boxes = [c.members[0].rep.obs]
            x1 = min(b.x1 for b in boxes) - CROP_PAD
            y1 = min(b.y1 for b in boxes) - CROP_PAD
            x2 = max(b.x2 for b in boxes) + CROP_PAD
            y2 = max(b.y2 for b in boxes) + CROP_PAD
            if x2 - x1 < CROP_MIN_SIZE:
                cx = (x1 + x2) // 2
                x1, x2 = cx - CROP_MIN_SIZE // 2, cx + CROP_MIN_SIZE // 2
            if y2 - y1 < CROP_MIN_SIZE:
                cy = (y1 + y2) // 2
                y1, y2 = cy - CROP_MIN_SIZE // 2, cy + CROP_MIN_SIZE // 2
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w), min(y2, h)

            vis = frame.copy()
            if x1 <= line_pos < x2:
                cv2.line(vis, (line_pos, y1), (line_pos, y2), (255, 255, 255), 1)
            for b in boxes:
                cv2.rectangle(vis, (b.x1, b.y1), (b.x2, b.y2), (80, 220, 80), 2)
                cv2.putText(vis, f"#{b.tid} {NAMES[b.cls]}",
                            (b.x1, max(b.y1 - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 220, 80), 1,
                            cv2.LINE_AA)
            crop = vis[y1:y2, x1:x2]
            tids = "-".join(str(m.tid) for m in c.members)
            multi = "_MULTI" if len(c.members) > 1 else ""
            name = (f"c{c.idx:02d}_f{frame_idx:04d}_{c.direction}_"
                    f"{c.cls_name}_tids{tids}{multi}.jpg")
            path = CROPS_DIR / name
            if not cv2.imwrite(str(path), crop):
                raise RuntimeError(f"failed to write crop {path}")
            paths[c.idx] = path
    cap.release()
    return paths


# ---------------------------------------------------------------------------
# Step 5: report
# ---------------------------------------------------------------------------

def fmt_counts(counts: dict[tuple[str, str], int]) -> list[str]:
    lines = ["| class | direction | count |", "|---|---|---:|"]
    for (name, direction), n in sorted(counts.items()):
        lines.append(f"| {name} | {direction} | {n} |")
    lines.append(f"| **total** | | **{sum(counts.values())}** |")
    return lines


def l1_distance(a: dict[tuple[str, str], int], b: dict[tuple[str, str], int]) -> int:
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)


def write_report(args: argparse.Namespace, line_pos: int, n_obs: int,
                 before: Counter, after: Counter,
                 suppressed: list[Suppressed],
                 nets: list[NetCross], jitter: list[tuple[int, int]],
                 total_raw: int, clusters: list[Cluster],
                 crop_paths: dict[int, Path],
                 missed_by_lt0: list[int],
                 born_near: list[tuple[int, str, int, bool]],
                 died_near: list[tuple[int, str, int, bool]]) -> None:
    physical: Counter = Counter()
    for c in clusters:
        physical[(c.cls_name, c.direction)] += 1

    before_matches = dict(before) == EXPECTED_BEFORE
    l1_b = l1_distance(before, physical)
    l1_a = l1_distance(after, physical)

    lines: list[str] = []
    add = lines.append
    add("# Line-crossing count verification - traffic3.mp4")
    add("")
    add(f"Generated by `scripts/verify_crossings.py` on 2026-09-01. "
        f"Video `{args.video}`, vertical line at x = {line_pos} "
        f"(fraction {args.line}), model `{WEIGHTS}`, imgsz {IMGSZ}, "
        f"conf {CONF}, tracker `{TRACKER}`. {n_obs} tracked boxes logged to "
        f"`{args.csv_name}`.")
    add("")

    add("## 1. Determinism check: replayed BEFORE table vs reported BEFORE table")
    add("")
    add("track_count.py's exact counting loop (no dedup) replayed over the "
        "logged trajectories:")
    add("")
    lines += fmt_counts(before)
    add("")
    if before_matches:
        add("This **matches the reported BEFORE table exactly** (bus L1/R3, "
            "car L20/R14, motor R1, truck L6/R17, van L5/R3, total 70), so the "
            "re-run reproduces the original tracker output and the analysis "
            "below applies to the reported numbers.")
    else:
        diff = l1_distance(before, EXPECTED_BEFORE)
        add(f"**WARNING: this does NOT exactly match the reported BEFORE table** "
            f"(L1 difference {diff}). The tracker re-run is not bit-identical to "
            f"the original run; comparisons below are against the replayed "
            f"numbers.")
    add("")

    add("## 2. Replayed AFTER table (current working-tree dedup)")
    add("")
    lines += fmt_counts(after)
    add("")
    add(f"{len(suppressed)} raw crossings were suppressed by the dedup:")
    add("")
    add("| suppressed track | class | direction | frame | absorbed by track | "
        "anchor class | anchor direction | anchor frame | cross-direction merge? |")
    add("|---|---|---|---|---|---|---|---|---|")
    for s in suppressed:
        cross_dir = "**YES - likely undercount**" if s.direction != s.anchor.direction else "no"
        add(f"| #{s.obs.tid} | {NAMES[s.obs.cls]} | {s.direction} | "
            f"{s.obs.frame} | #{s.anchor.obs.tid} | {NAMES[s.anchor.obs.cls]} | "
            f"{s.anchor.direction} | {s.anchor.obs.frame} | {cross_dir} |")
    add("")

    add("## 3. Independent physical crossing events (this script)")
    add("")
    add(f"Independent detection over all logged trajectories: {total_raw} raw "
        f"sign changes; {len(jitter)} tracks wobbled across the line and came "
        f"back (even number of crossings, no net crossing); {len(nets)} tracks "
        f"genuinely ended on the other side (net crossings). Net crossings "
        f"were clustered into physical events (same direction, <= "
        f"{DEDUP_WINDOW} frames apart, < {LEGACY_RADIUS} px Chebyshev - the "
        f"same radius/window as the dedup, but transitive and "
        f"direction-aware). **{len(nets)} net crossings collapsed into "
        f"{len(clusters)} physical events ({len(nets) - len(clusters)} "
        f"merged).**")
    add("")
    lines += fmt_counts(physical)
    add("")
    if jitter:
        add("Jitter tracks (crossed and came back; the counter counts these "
            "once even though there is no net crossing): "
            + ", ".join(f"#{tid} ({n} crossings)" for tid, n in jitter))
        add("")
    if missed_by_lt0:
        add("Crossings **missed by track_count.py's strict `< 0` test** (the "
            "center landed exactly on the line for one frame): tracks "
            + ", ".join(f"#{t}" for t in missed_by_lt0))
        add("")

    add("### Multi-track clusters (split vehicles / ID switches)")
    add("")
    multi = [c for c in clusters if len(c.members) > 1]
    if multi:
        add("| event | frame | direction | assigned class | members (track: "
            "class, raw crossings) | mixed class? | crop |")
        add("|---|---|---|---|---|---|---|")
        for c in multi:
            members = "; ".join(f"#{m.tid}: {NAMES[m.cls]}, {m.n_raw}x"
                                for m in c.members)
            crop = crop_paths[c.idx].name if c.idx in crop_paths else "-"
            add(f"| c{c.idx:02d} | {c.rep_frame} | {c.direction} | "
                f"{c.cls_name} | {members} | "
                f"{'yes' if c.is_mixed_class else 'no'} | `{crop}` |")
    else:
        add("None - every physical event was a single track.")
    add("")

    add("### All events")
    add("")
    add("| event | frame | direction | class | tracks | crop |")
    add("|---|---|---|---|---|---|")
    for c in clusters:
        tids = ", ".join(f"#{m.tid}" for m in c.members)
        crop = crop_paths[c.idx].name if c.idx in crop_paths else "-"
        add(f"| c{c.idx:02d} | {c.rep_frame} | {c.direction} | {c.cls_name} | "
            f"{tids} | `{crop}` |")
    add("")
    add(f"Crop images: `analysis/crossing_crops/` (one per event; `_MULTI` "
        f"suffix marks multi-track events).")
    add("")

    add("## 4. Near-line track births/deaths (possible ID switches at the line)")
    add("")
    add(f"Moving tracks (>= {MIN_TRACK_OBS} observations, >= {MIN_DISPLACEMENT} px "
        f"total displacement) that appear or disappear within {NEAR_LINE_PX} px "
        f"of the line. A death immediately left of the line paired with a birth "
        f"right of it is one vehicle counted zero or two times.")
    add("")
    if born_near or died_near:
        add("| track | class | event | frame | netted a crossing? |")
        add("|---|---|---|---|---|")
        for tid, name, frame, netted in died_near:
            add(f"| #{tid} | {name} | died near line | {frame} | "
                f"{'yes' if netted else 'no'} |")
        for tid, name, frame, netted in born_near:
            add(f"| #{tid} | {name} | born near line | {frame} | "
                f"{'yes' if netted else 'no'} |")
    else:
        add("None found.")
    add("")

    add("## 5. Verdict")
    add("")
    add("| table | L1 distance to independent physical events |")
    add("|---|---:|")
    add(f"| BEFORE (no dedup), total {sum(before.values())} | {l1_b} |")
    add(f"| AFTER (dedup), total {sum(after.values())} | {l1_a} |")
    add(f"| physical events, total {sum(physical.values())} | 0 (reference) |")
    add("")
    closer = ("the AFTER (dedup) table" if l1_a < l1_b
              else "the BEFORE (no dedup) table" if l1_b < l1_a
              else "neither table (they are equally far)")
    add(f"By this measure **{closer} is closer to the ground truth** as "
        f"estimated by independent clustering. Caveats: cluster class labels "
        f"use the majority class of the largest fragment, which can differ "
        f"from the class the counter saw at the crossing frame; the final word "
        f"belongs to the human spot-check of the crops.")
    add("")
    add("## 6. Human spot-check")
    add("")
    add("_Pending review of `analysis/crossing_crops/`._")
    add("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default="videos/traffic3.mp4")
    ap.add_argument("--line", type=float, default=0.5,
                    help="vertical line position as fraction of frame width")
    ap.add_argument("--reuse-csv", action="store_true",
                    help="skip inference and reuse the previously logged CSV")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after N frames (0 = whole video; smoke tests only)")
    args = ap.parse_args()

    video = PROJECT_ROOT / args.video
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    line_pos = int(width * args.line)  # same truncation as track_count.py
    print(f"Video {video.name}, width {width}, counting line at x={line_pos}")

    csv_path = ANALYSIS_DIR / f"trajectories_{video.stem}.csv"
    args.csv_name = f"analysis/{csv_path.name}"
    if args.reuse_csv:
        observations = load_observations(csv_path)
        print(f"Reusing {csv_path} ({len(observations)} boxes)")
    else:
        observations = run_tracker(video, csv_path, args.max_frames)

    # Exact replay of the code under test, with and without dedup.
    before, _, _ = simulate_track_count(observations, line_pos, dedup=False)
    after, _, suppressed = simulate_track_count(observations, line_pos, dedup=True)

    # Independent detection and clustering.
    tracks: dict[int, list[Obs]] = defaultdict(list)
    for o in observations:  # already frame-ordered
        tracks[o.tid].append(o)
    nets, jitter, total_raw = net_crossings(tracks, line_pos)
    if not nets:
        raise RuntimeError("independent detection found zero net crossings; "
                           "wrong line position or broken trajectories?")
    clusters = cluster_events(nets)

    # Tracks the strict `< 0` test missed (center landed exactly on the line).
    counted_by_sim = {s.obs.tid for s in
                      simulate_track_count(observations, line_pos, False)[1]}
    missed_by_lt0 = sorted(nc.tid for nc in nets if nc.tid not in counted_by_sim)

    # Near-line birth/death diagnostics. Births at the first frame and deaths
    # at the last are clip boundaries, not ID switches - exclude them.
    last_frame = max(o.frame for o in observations)
    netted_tids = {nc.tid for nc in nets}
    born_near: list[tuple[int, str, int, bool]] = []
    died_near: list[tuple[int, str, int, bool]] = []
    for tid, track_obs in tracks.items():
        majority_cls = Counter(o.cls for o in track_obs).most_common(1)[0][0]
        if majority_cls not in COUNT_CLASSES or len(track_obs) < MIN_TRACK_OBS:
            continue
        first, last = track_obs[0], track_obs[-1]
        if max(abs(last.cx - first.cx), abs(last.cy - first.cy)) < MIN_DISPLACEMENT:
            continue
        name = NAMES[majority_cls]
        if first.frame > 1 and abs(first.cx - line_pos) < NEAR_LINE_PX:
            born_near.append((tid, name, first.frame, tid in netted_tids))
        if last.frame < last_frame and abs(last.cx - line_pos) < NEAR_LINE_PX:
            died_near.append((tid, name, last.frame, tid in netted_tids))

    obs_index = {(o.frame, o.tid): o for o in observations}
    crop_paths = save_crops(clusters, video, line_pos, obs_index)
    print(f"{len(crop_paths)} crop images written to {CROPS_DIR}")

    write_report(args, line_pos, len(observations), before, after, suppressed,
                 nets, jitter, total_raw, clusters, crop_paths, missed_by_lt0,
                 born_near, died_near)

    # Console summary.
    physical: Counter = Counter()
    for c in clusters:
        physical[(c.cls_name, c.direction)] += 1
    print(f"\nBEFORE (replayed): {sum(before.values())}  "
          f"AFTER (replayed): {sum(after.values())}  "
          f"physical events: {sum(physical.values())}")
    print(f"BEFORE matches reported table: {dict(before) == EXPECTED_BEFORE}")


if __name__ == "__main__":
    main()
