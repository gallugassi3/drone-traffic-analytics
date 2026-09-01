# Drone Traffic Analytics

![Demo](assets/demo.gif)

Directional traffic counting from drone footage: **YOLO11n trained on VisDrone**
(my [previous project's](https://github.com/gallugassi3/visdrone-small-object-detection)
released weights) + **ByteTrack** multi-object tracking + an inspectable
line-crossing counter with per-class, per-direction totals.

*Highway and intersection demo clips from [Pexels](https://www.pexels.com)
(intersection footage by Siarhei Dalivelia).*

## What it does

- Tracks every road user (car, truck, van, bus, motorcycle, pedestrian) with
  persistent ids and motion trails, at the detector's native 1024px resolution.
- Counts each track **once** when its center crosses a virtual line - horizontal
  or vertical (`--axis`), because real intersections taught me the dominant flow
  axis matters.
- Writes an annotated MP4, a live HUD, and a per-class/direction CSV.

```bash
python track_count.py videos/traffic.mp4 --axis v --line 0.5 --show
```

## Counts you can trust (the interesting part)

The first counting run looked right and was **right by coincidence** - an
independent verification study (re-running the tracker, logging 14,039 trajectory
points across 339 tracks, clustering crossings, and human-reviewing crop images of
every multi-track event) found two cancelling bugs:

| Version | Total counted | vs ground truth (72 crossings, 39 left / 33 right) |
|---|---|---|
| Naive counter | 70 | +15 duplicate crossings from split trucks and id switches, −17 crossings silently dropped by an exact-on-the-line zero-product edge case |
| First dedup (80px radius) | 41 | merged real vehicles in adjacent lanes and even opposite directions - **over-correction, measured** |
| Final (direction-aware, lane-tight anisotropic dedup + edge-case fix) | **70** | **38 left / 32 right - 97% of ground truth with correct direction balance; the two residual misses are localized and analyzed in the fix report** |

Full study: [`analysis/crossing_verification.md`](analysis/crossing_verification.md) ·
fix verification (replayed against logged trajectories):
[`analysis/fix_verification.md`](analysis/fix_verification.md) ·
tooling: [`scripts/verify_crossings.py`](scripts/verify_crossings.py)

> The studies reference raw evidence (a trajectory CSV and per-crossing crop
> images) that is generated locally by the scripts and deliberately not committed;
> run `scripts/verify_crossings.py` on your own clip to reproduce the pipeline.
> `scripts/verify_fixes.py` hard-asserts this study's exact data, so it is
> expected to fail on other footage - it documents the audit, it isn't a tool.

## Design choices

- **Weights are consumed from the detector project's Release v1.0** (not retrained):
  research repo trains and publishes; this repo builds a product on top.
- `imgsz=1024` and `conf=0.25` come straight from that project's measured findings
  (the resolution experiment and the F1-optimal operating point).
- Counting is per-track-once with a sign-change test - simple enough to verify by
  eye, which is exactly what the verification study did.

## Known limitations (measured, not guessed)

- **Split long vehicles:** the detector occasionally splits a truck into
  cab + trailer; the dedup merges those crossings (verified against human ground
  truth on all multi-track events). Two residual misses remain in this footage,
  both localized in the fix report - one marginal 23px-vs-25px gate case, and one
  genuine nose-to-tail convoy that is geometrically indistinguishable from a
  cab/trailer split at the line.
- **Class flicker on lookalikes:** box color follows the per-frame class, so a
  vehicle on the model's car/van decision boundary visibly flips between the two -
  the twin-class confusion measured in the detector project, live on video. The
  counter is unaffected (crossing test is class-agnostic; the class logged is the
  one at crossing time).
- **Camera motion breaks tracking:** on a zoom-out clip, detection stayed solid
  but track identity churned (16K+ ids in 23s). No motion compensation - documented
  as the next step, not hidden.
- **Throughput: ~11-13 FPS end-to-end measured on CPU-only torch** (inference +
  tracking + drawing + video encode); a CUDA build runs substantially faster. This
  is an offline analytics tool, not a real-time system.
- Dedup gates (25px across / 120px along / 15 frames) are calibrated for
  1280x720 @ 30fps footage; other resolutions or frame rates need rescaling.

## Setup

```bash
git clone https://github.com/gallugassi3/drone-traffic-analytics.git
cd drone-traffic-analytics
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows (use: source .venv/bin/activate on Linux)
pip install -r requirements.txt
# weights: download yolo11n_visdrone_1024.pt from the detector project's Releases
#   https://github.com/gallugassi3/visdrone-small-object-detection/releases/tag/v1.0
#   and place it in weights/
# video: any top-down drone traffic clip (Pexels has plenty) -> videos/
python track_count.py videos/your_clip.mp4 --axis v --line 0.5 --show
```

---

*Built by [Gal Lugassi](https://github.com/gallugassi3) - Sep 2026 - the detector
behind this: [visdrone-small-object-detection](https://github.com/gallugassi3/visdrone-small-object-detection)*