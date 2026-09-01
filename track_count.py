"""Drone-view traffic tracking and directional line-crossing counter.

Runs the VisDrone-trained YOLO11n detector (project 1's released weights)
with ByteTrack multi-object tracking, draws per-object trails and IDs,
and counts objects crossing a virtual line, per class and per direction.

The counting logic is deliberately simple and inspectable: an object is
counted when its box center crosses the counting line between consecutive
frames of the same track id. The line can be horizontal (counts vertical
flow, directions up/down) or vertical (counts horizontal flow, directions
left/right).

Class handling is two-layered, because measurement and presentation have
different needs (verified in analysis/fix_verification.md, section 9):
- COUNTED class = majority vote over the last CLS_WINDOW frames at
  crossing time. Local evidence wins: the most discriminative views are
  near the line, and the window majority was verified against human
  ground truth (it labels the crossing motorcycle and the 14:1 bus
  correctly, where accumulate-forever hysteresis locked onto early
  frame-edge misreads).
- DISPLAYED class (box color/label) = display-only Schmitt trigger on
  long-horizon decayed vote mass: the label switches only when a
  challenger's decayed mass clearly outweighs the incumbent's
  (DISP_MARGIN, DISP_MIN_MASS), with a short post-switch dwell - a
  share-level criterion, because the window majority itself oscillates
  in runs of 11-95 frames on ~50/50 lookalikes (car/van, truck/bus), so
  no run-length gate can separate that flicker from a genuine change.
  At the crossing moment the display snaps to the counted class, so the
  video always shows the label being counted. Nothing is locked
  forever: a genuinely sustained majority change still switches.

Other verified behaviors baked in:
- A center landing exactly on the line for one frame used to zero the
  sign-change product and silently drop the crossing; last_pos is
  therefore only updated off the line.
- Split long vehicles (cab + trailer) cross as two tracks; dedup merges
  them, direction-aware and lane-tight (anisotropic radius), so
  adjacent-lane and opposite-direction vehicles are never merged.

Dedup gate constants (DEDUP_ACROSS / DEDUP_ALONG / DEDUP_WINDOW) are
calibrated for 1280x720 @ 30fps footage; rescale them for other
resolutions or frame rates.

Usage:
    python track_count.py videos/traffic.mp4
    python track_count.py videos/traffic.mp4 --axis v --line 0.5 --show
"""
import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

WEIGHTS = "weights/yolo11n_visdrone_1024.pt"
IMGSZ = 1024              # the resolution the model was trained at; project 1's core finding
CONF = 0.25               # operating point validated in project 1 (near F1 peak)
TRACKER = "bytetrack.yaml"
TRAIL_LEN = 30            # frames of trail history to draw per track
OUT_DIR = Path("output")

# Spatial dedup for split long vehicles (cab + trailer counted as one).
# Anisotropic and direction-aware, per the verification study: tight across
# lanes (lanes are ~60px, cars ~35px in this footage), wide along travel
# (cab-to-trailer distance), and opposite directions are never merged.
DEDUP_ACROSS = 25         # px perpendicular to travel direction (lane gating)
DEDUP_ALONG = 120         # px along travel direction
DEDUP_WINDOW = 15         # frames

# Two-layer class handling (see module docstring):
CLS_WINDOW = 15           # frames in the majority-vote window (counted class)
# Display-only stability (verified in analysis/fix_verification.md section 9):
DISP_HALF_LIFE = 40.0     # frames: horizon of the decayed per-class vote mass
DISP_MARGIN = 2.0         # challenger needs 2x the incumbent's decayed mass
DISP_MIN_MASS = 8.0       # ... and at least this much absolute mass
DISP_DWELL = 25           # frames: no further display switch right after one
DISP_DECAY = 0.5 ** (1.0 / DISP_HALF_LIFE)

# VisDrone class names (model was trained on these ids)
NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
         "tricycle", "awning-tricycle", "bus", "motor"]
# Count only road users that matter for traffic analytics; ignore noisy rares
COUNT_CLASSES = {0, 3, 4, 5, 8, 9}  # pedestrian, car, van, truck, bus, motor

COLORS = [(66, 135, 245), (52, 195, 235), (99, 220, 120), (60, 76, 231),
          (180, 130, 70), (30, 105, 210), (190, 100, 220), (128, 128, 128),
          (0, 165, 255), (203, 70, 250)]


def window_majority(history: list[int]) -> int:
    """Most frequent class in the window; ties break toward the most recent."""
    counts: dict[int, int] = {}
    for cls in history:
        counts[cls] = counts.get(cls, 0) + 1
    best_cls, best_key = history[-1], (-1, -1)
    for recency, cls in enumerate(reversed(history)):
        key = (counts[cls], -recency)   # higher count wins; then more recent
        if key > best_key:
            best_cls, best_key = cls, key
    return best_cls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="path to input video")
    ap.add_argument("--line", type=float, default=0.5,
                    help="counting line position as fraction of frame (0-1)")
    ap.add_argument("--axis", choices=["h", "v"], default="h",
                    help="h: horizontal line counts vertical flow (up/down); "
                         "v: vertical line counts horizontal flow (left/right)")
    ap.add_argument("--tracker", default=TRACKER,
                    help="tracker yaml: a bundled name (bytetrack.yaml, "
                         "botsort.yaml, ...) or a path to a custom config")
    ap.add_argument("--show", action="store_true", help="display live window")
    args = ap.parse_args()

    model = YOLO(WEIGHTS)
    cap = cv2.VideoCapture(args.video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    line_pos = int((h if args.axis == "h" else w) * args.line)
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / (Path(args.video).stem + "_tracked.mp4")
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps_in, (w, h))

    trails = defaultdict(list)          # id -> recent centers
    cls_hist = defaultdict(list)         # id -> recent raw class votes (window)
    cls_mass = defaultdict(lambda: defaultdict(float))  # id -> decayed vote mass per class
    shown_cls = {}                       # id -> class currently displayed
    disp_dwell = defaultdict(int)        # id -> frames left before display may switch again
    last_pos = {}                        # id -> previous off-line center coord on the axis
    counted = set()                      # ids already counted (count each track once)
    counts = defaultdict(int)            # (class_name, direction) -> n
    recent_crossings = []                # (frame_idx, along, across, direction)
    t0, frames = time.time(), 0

    # stream=True processes frame by frame; persist=True keeps track ids across frames
    for result in model.track(source=args.video, imgsz=IMGSZ, conf=CONF,
                              tracker=args.tracker, persist=True, stream=True,
                              verbose=False):
        frame = result.orig_img
        frames += 1

        if result.boxes.id is not None:
            ids = result.boxes.id.int().tolist()
            clss = result.boxes.cls.int().tolist()
            for box, tid, c in zip(result.boxes.xyxy, ids, clss):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Layer 1 - measurement: window majority (the counted class)
                cls_hist[tid].append(c)
                cls_hist[tid] = cls_hist[tid][-CLS_WINDOW:]
                c_count = window_majority(cls_hist[tid])

                # Crossing test: center moved from one side of the line to the
                # other (sign change). last_pos never stores an on-line value,
                # so a frame exactly on the line cannot zero the product and
                # hide the crossing. (Evaluated before drawing so the display
                # can snap to the counted class on the crossing frame itself.)
                coord = cy if args.axis == "h" else cx
                crossed_now = False
                if c_count in COUNT_CLASSES and tid in last_pos and tid not in counted:
                    if (last_pos[tid] - line_pos) * (coord - line_pos) < 0:
                        if args.axis == "h":
                            direction = "down" if coord > last_pos[tid] else "up"
                            along, across = cy, cx
                        else:
                            direction = "right" if coord > last_pos[tid] else "left"
                            along, across = cx, cy
                        # Dedup: same direction, same lane band, close along
                        # travel, within the time window = split vehicle
                        dup = any(frames - f <= DEDUP_WINDOW and d == direction and
                                  abs(across - pa) < DEDUP_ACROSS and
                                  abs(along - pl) < DEDUP_ALONG
                                  for f, pl, pa, d in recent_crossings)
                        if not dup:
                            counts[(NAMES[c_count], direction)] += 1
                            recent_crossings.append((frames, along, across, direction))
                            recent_crossings[:] = [
                                (f, pl, pa, d) for f, pl, pa, d in recent_crossings
                                if frames - f <= DEDUP_WINDOW
                            ]
                        counted.add(tid)
                        crossed_now = True
                if coord != line_pos:
                    last_pos[tid] = coord

                # Layer 2 - presentation: Schmitt trigger on decayed vote
                # mass, with snap-to-counted-class at the crossing moment
                # (see module docstring; verified in fix_verification.md s9)
                mass = cls_mass[tid]
                for k in mass:
                    mass[k] *= DISP_DECAY
                mass[c] += 1.0
                if tid not in shown_cls:
                    shown_cls[tid] = c_count
                elif crossed_now:
                    if shown_cls[tid] != c_count:
                        shown_cls[tid] = c_count
                        disp_dwell[tid] = DISP_DWELL
                elif disp_dwell[tid] > 0:
                    disp_dwell[tid] -= 1
                else:
                    top = max(mass, key=lambda k: mass[k])
                    if (top != shown_cls[tid] and mass[top] >= DISP_MIN_MASS and
                            mass[top] >= DISP_MARGIN * mass.get(shown_cls[tid], 0.0)):
                        shown_cls[tid] = top
                        disp_dwell[tid] = DISP_DWELL
                c_disp = shown_cls[tid]
                color = COLORS[c_disp % len(COLORS)]

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{NAMES[c_disp]} #{tid}", (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

                trails[tid].append((cx, cy))
                trails[tid] = trails[tid][-TRAIL_LEN:]
                for p, q in zip(trails[tid], trails[tid][1:]):
                    cv2.line(frame, p, q, color, 2)

        # HUD: counting line + live totals
        if args.axis == "h":
            cv2.line(frame, (0, line_pos), (w, line_pos), (255, 255, 255), 2)
        else:
            cv2.line(frame, (line_pos, 0), (line_pos, h), (255, 255, 255), 2)
        total = sum(counts.values())
        cv2.putText(frame, f"crossed: {total}", (14, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(frame)
        if args.show:
            cv2.imshow("tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f"\nProcessed {frames} frames in {elapsed:.1f}s "
          f"({frames / elapsed:.1f} FPS end-to-end)")
    print(f"Annotated video: {out_path}\n")
    print(f"{'class':>12} {'direction':>9} {'count':>6}")
    print("-" * 32)
    csv_path = OUT_DIR / "counts.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["class", "direction", "count"])
        for (name, direction), n in sorted(counts.items()):
            print(f"{name:>12} {direction:>9} {n:>6}")
            wr.writerow([name, direction, n])
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()