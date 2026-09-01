"""Drone-view traffic tracking and directional line-crossing counter.

Runs the VisDrone-trained YOLO11n detector (project 1's released weights)
with ByteTrack multi-object tracking, draws per-object trails and IDs,
and counts objects crossing a virtual line, per class and per direction.

The counting logic is deliberately simple and inspectable: an object is
counted when its box center crosses the counting line between consecutive
frames of the same track id. The line can be horizontal (counts vertical
flow, directions up/down) or vertical (counts horizontal flow, directions
left/right) - real intersections taught us the dominant flow axis matters.

Usage:
    python track_count.py videos/traffic.mp4
    python track_count.py videos/traffic.mp4 --axis v --line 0.78 --show
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

# VisDrone class names (model was trained on these ids)
NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
         "tricycle", "awning-tricycle", "bus", "motor"]
# Count only road users that matter for traffic analytics; ignore noisy rares
COUNT_CLASSES = {0, 3, 4, 5, 8, 9}  # pedestrian, car, van, truck, bus, motor

COLORS = [(66, 135, 245), (52, 195, 235), (99, 220, 120), (60, 76, 231),
          (180, 130, 70), (30, 105, 210), (190, 100, 220), (128, 128, 128),
          (0, 165, 255), (203, 70, 250)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="path to input video")
    ap.add_argument("--line", type=float, default=0.5,
                    help="counting line position as fraction of frame (0-1)")
    ap.add_argument("--axis", choices=["h", "v"], default="h",
                    help="h: horizontal line counts vertical flow (up/down); "
                         "v: vertical line counts horizontal flow (left/right)")
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
    last_pos = {}                        # id -> previous center coord on the counting axis
    counted = set()                      # ids already counted (count each track once)
    counts = defaultdict(int)            # (class_name, direction) -> n
    t0, frames = time.time(), 0

    # stream=True processes frame by frame; persist=True keeps track ids across frames
    for result in model.track(source=args.video, imgsz=IMGSZ, conf=CONF,
                              tracker=TRACKER, persist=True, stream=True,
                              verbose=False):
        frame = result.orig_img
        frames += 1

        if result.boxes.id is not None:
            ids = result.boxes.id.int().tolist()
            clss = result.boxes.cls.int().tolist()
            for box, tid, c in zip(result.boxes.xyxy, ids, clss):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                color = COLORS[c % len(COLORS)]

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{NAMES[c]} #{tid}", (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

                trails[tid].append((cx, cy))
                trails[tid] = trails[tid][-TRAIL_LEN:]
                for p, q in zip(trails[tid], trails[tid][1:]):
                    cv2.line(frame, p, q, color, 2)

                # Crossing test: center moved from one side of the line to the
                # other between consecutive frames (sign change of the offset)
                coord = cy if args.axis == "h" else cx
                if c in COUNT_CLASSES and tid in last_pos and tid not in counted:
                    if (last_pos[tid] - line_pos) * (coord - line_pos) < 0:
                        if args.axis == "h":
                            direction = "down" if coord > last_pos[tid] else "up"
                        else:
                            direction = "right" if coord > last_pos[tid] else "left"
                        counts[(NAMES[c], direction)] += 1
                        counted.add(tid)
                last_pos[tid] = coord

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