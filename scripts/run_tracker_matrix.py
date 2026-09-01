#!/usr/bin/env python
"""Tracker comparison harness: run every tracker config on both study videos.

For each (config, video) pair this runs one tracking pass (same detector,
weights, imgsz, conf as track_count.py; no drawing/encode), logs every
track's trajectory to analysis/tracker_runs/<config>_<video>.csv (same
format as analysis/trajectories_traffic3.csv), and collects:

  * the counts table - an exact axis-generalized replay of track_count.py's
    counting logic (window-majority class, on-line fix, direction-aware
    anisotropic dedup) on the logged trajectories,
  * total unique track ids,
  * median track length (frames observed per id),
  * tracking FPS (detector + tracker + logging; excludes drawing/encode).

Prints a summary matrix and writes the raw results (no conclusions) to
analysis/tracker_study_raw.md.

CPU-only torch: expect ~1-3 minutes per run, ~15+ minutes total.

Usage:
    python scripts/run_tracker_matrix.py
"""
from __future__ import annotations

import csv
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2

SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

from verify_crossings import Obs
from track_count import (CLS_WINDOW, CONF, COUNT_CLASSES, DEDUP_ACROSS,
                         DEDUP_ALONG, DEDUP_WINDOW, IMGSZ, NAMES, WEIGHTS,
                         window_majority)

RUNS_DIR = PROJECT_ROOT / "analysis" / "tracker_runs"
REPORT_PATH = PROJECT_ROOT / "analysis" / "tracker_study_raw.md"

CONFIGS: list[tuple[str, Path]] = [
    ("bytetrack", PROJECT_ROOT / "trackers" / "bytetrack.yaml"),
    ("botsort_gmc", PROJECT_ROOT / "trackers" / "botsort_gmc.yaml"),
    ("botsort_reid", PROJECT_ROOT / "trackers" / "botsort_reid.yaml"),
    ("botsort_nogmc", PROJECT_ROOT / "trackers" / "botsort_nogmc.yaml"),
    ("ocsort", PROJECT_ROOT / "trackers" / "ocsort.yaml"),
]
# (name, path, axis, line fraction)
VIDEOS: list[tuple[str, Path, str, float]] = [
    ("traffic3", PROJECT_ROOT / "videos" / "traffic3.mp4", "v", 0.5),
    ("traffic2", PROJECT_ROOT / "videos" / "traffic2.mp4", "h", 0.75),
]


@dataclass
class RunResult:
    config: str
    video: str
    frames: int
    elapsed: float
    n_boxes: int
    n_ids: int
    median_len: float
    counts: Counter
    csv_path: Path

    @property
    def fps(self) -> float:
        return self.frames / self.elapsed

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def track_once(video: Path, tracker_yaml: Path) -> tuple[list[Obs], int, float]:
    """One tracking pass; fresh model per run so no tracker state leaks."""
    from ultralytics import YOLO

    model = YOLO(str(PROJECT_ROOT / WEIGHTS))
    observations: list[Obs] = []
    t0 = time.time()
    n_frames = 0
    for n_frames, result in enumerate(
            model.track(source=str(video), imgsz=IMGSZ, conf=CONF,
                        tracker=str(tracker_yaml), persist=True, stream=True,
                        verbose=False), start=1):
        if result.boxes.id is not None:
            ids = result.boxes.id.int().tolist()
            clss = result.boxes.cls.int().tolist()
            confs = result.boxes.conf.tolist()
            for box, tid, c, cf in zip(result.boxes.xyxy, ids, clss, confs):
                x1, y1, x2, y2 = map(int, box)
                observations.append(Obs(n_frames, tid, c, float(cf),
                                        x1, y1, x2, y2))
        if n_frames % 200 == 0:
            print(f"    frame {n_frames} "
                  f"({n_frames / (time.time() - t0):.1f} FPS)", flush=True)
    elapsed = time.time() - t0
    if not observations:
        raise RuntimeError(f"zero tracked boxes: {tracker_yaml.name} on {video.name}")
    return observations, n_frames, elapsed


def write_csv(observations: list[Obs], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "track_id", "class_id", "class_name", "conf",
                     "x1", "y1", "x2", "y2", "cx", "cy"])
        for o in observations:
            wr.writerow([o.frame, o.tid, o.cls, NAMES[o.cls], f"{o.conf:.4f}",
                         o.x1, o.y1, o.x2, o.y2, o.cx, o.cy])


def replay_counts(observations: list[Obs], axis: str, line_pos: int) -> Counter:
    """Axis-generalized exact replay of track_count.py's counting loop."""
    cls_hist: dict[int, list[int]] = defaultdict(list)
    last_pos: dict[int, int] = {}
    counted: set[int] = set()
    counts: Counter = Counter()
    recent: list[tuple[int, int, int, str]] = []
    for o in observations:
        cls_hist[o.tid].append(o.cls)
        cls_hist[o.tid] = cls_hist[o.tid][-CLS_WINDOW:]
        c_count = window_majority(cls_hist[o.tid])
        coord = o.cy if axis == "h" else o.cx
        if c_count in COUNT_CLASSES and o.tid in last_pos and o.tid not in counted:
            if (last_pos[o.tid] - line_pos) * (coord - line_pos) < 0:
                if axis == "h":
                    direction = "down" if coord > last_pos[o.tid] else "up"
                    along, across = o.cy, o.cx
                else:
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
        if coord != line_pos:
            last_pos[o.tid] = coord
    return counts


def line_position(video: Path, axis: str, frac: float) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return int((h if axis == "h" else w) * frac)


def fmt_counts(counts: Counter) -> list[str]:
    lines = ["| class | direction | count |", "|---|---|---:|"]
    for (name, direction), n in sorted(counts.items()):
        lines.append(f"| {name} | {direction} | {n} |")
    lines.append(f"| **total** | | **{sum(counts.values())}** |")
    return lines


def main() -> None:
    for _, yaml_path in CONFIGS:
        if not yaml_path.is_file():
            raise FileNotFoundError(f"tracker config missing: {yaml_path}")
    for _, video, _, _ in VIDEOS:
        if not video.is_file():
            raise FileNotFoundError(f"video missing: {video}")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    import torch
    import ultralytics
    env = (f"ultralytics {ultralytics.__version__}, torch "
           f"{torch.__version__} (CUDA {torch.cuda.is_available()})")
    print(f"Environment: {env}")

    results: list[RunResult] = []
    for vname, video, axis, frac in VIDEOS:
        line_pos = line_position(video, axis, frac)
        for cname, yaml_path in CONFIGS:
            print(f"\n[{len(results) + 1}/{len(CONFIGS) * len(VIDEOS)}] "
                  f"{cname} on {vname} (axis {axis}, line {frac} -> "
                  f"{line_pos}px)", flush=True)
            observations, n_frames, elapsed = track_once(video, yaml_path)
            csv_path = RUNS_DIR / f"{cname}_{vname}.csv"
            write_csv(observations, csv_path)
            per_id = Counter(o.tid for o in observations)
            counts = replay_counts(observations, axis, line_pos)
            r = RunResult(cname, vname, n_frames, elapsed, len(observations),
                          len(per_id),
                          float(statistics.median(per_id.values())),
                          counts, csv_path)
            results.append(r)
            print(f"    -> {r.n_boxes} boxes, {r.n_ids} ids, median track "
                  f"{r.median_len:.0f} frames, {r.fps:.1f} FPS, counted "
                  f"{r.total}", flush=True)

    # Summary matrix.
    print("\n=== SUMMARY (counted total | unique ids | median track len | "
          "tracking FPS) ===")
    header = f"{'config':>14}" + "".join(f"{v[0]:>34}" for v in VIDEOS)
    print(header)
    for cname, _ in CONFIGS:
        row = f"{cname:>14}"
        for vname, *_ in VIDEOS:
            r = next(x for x in results if x.config == cname and x.video == vname)
            row += f"{r.total:>8} | {r.n_ids:>5} | {r.median_len:>4.0f} | {r.fps:>5.1f}"
        print(row)

    # Raw report.
    lines: list[str] = []
    add = lines.append
    add("# Tracker comparison study - raw results")
    add("")
    add(f"Generated by `scripts/run_tracker_matrix.py` on 2026-09-01. "
        f"{env}; CPU-only inference. Detector: `{WEIGHTS}`, imgsz {IMGSZ}, "
        f"conf {CONF}. Counting: exact replay of track_count.py's verified "
        f"logic (window-{CLS_WINDOW} majority class, on-line fix, "
        f"direction-aware anisotropic dedup {DEDUP_ACROSS}/{DEDUP_ALONG}px, "
        f"{DEDUP_WINDOW} frames) on the logged trajectories. FPS is the "
        f"tracking pass only (no drawing/video encode). Raw numbers only - "
        f"analysis against pre-registered hypotheses comes separately.")
    add("")
    add("## Tracker availability (installed ultralytics)")
    add("")
    add("`ultralytics/cfg/trackers/` ships: botsort, bytetrack, deepocsort, "
        "fasttrack, ocsort, tracktrack. This study runs bytetrack (current "
        "pipeline default), three botsort variants (stock GMC, +ReID, "
        "GMC ablation), and ocsort; configs in `trackers/`.")
    add("")
    add("## Videos")
    add("")
    add("| video | frames | axis | line | notes |")
    add("|---|---|---|---|---|")
    add("| traffic3.mp4 | 921 | v | 0.5 (x=640) | static camera, highway; "
        "the ground-truth-verified study clip |")
    add("| traffic2.mp4 | 564 | h | 0.75 | camera zoom-out mid-clip; the "
        "known tracking-churn clip |")
    add("")
    add("## Summary matrix")
    add("")
    add("| config | video | counted total | unique ids | median track len "
        "(frames) | tracking FPS | boxes logged |")
    add("|---|---|---:|---:|---:|---:|---:|")
    for r in results:
        add(f"| {r.config} | {r.video} | {r.total} | {r.n_ids} | "
            f"{r.median_len:.0f} | {r.fps:.1f} | {r.n_boxes} |")
    add("")
    add("## Per-run counts tables")
    add("")
    for r in results:
        add(f"### {r.config} on {r.video}")
        add("")
        lines_c = fmt_counts(r.counts)
        lines.extend(lines_c)
        add("")
        add(f"Trajectories: `analysis/tracker_runs/{r.csv_path.name}`")
        add("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRaw report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
