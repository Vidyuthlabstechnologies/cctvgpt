# Phase 1 feasibility test protocol

This follows PRD section 8 and the Phase 1 roadmap gate: *would at least one practitioner test the output?*

## Setup

* **Scene:** corridor (default) or parking lot.
* **Event:** fall (default) or intrusion.
* **Safeguards SR-1 to SR-3** are applied from day one. The script uses the full pipeline.

Generate the set:

```bash
python scripts/feasibility_set.py --scene corridor --event fall --seeds 2 --out out/feasibility
```

This writes one clip for each combination of time of day, weather (outdoor scenes only), camera tilt and seed. Each clip comes with its annotations and provenance, and the set comes with `feasibility_summary.csv`.

## Automatic metrics (in the CSV)

| Metric | Meaning |
|---|---|
| `fixed_view_stability` | 1 − mean absolute background change relative to the first frame, excluding annotated objects and overlays. Values near 1 mean a stable view (FR-8). |
| `event_occurred`, `event_start_s` | Whether the scripted event happened within the clip, and when |
| `label_ok` | Score for the visible label in the encoded file (≥ 0.5 required) |
| `watermark_confidence` | The weakest payload bit's t-statistic (≥ 3 required) |

## Practitioner review (blank columns in the CSV)

A video-analytics practitioner, ideally from a target tier 1 customer, fills in:

* `reviewer_realism_1to5`: does it look like CCTV footage?
* `reviewer_event_correct_yn`: is the event clearly what the label says?
* `reviewer_would_use_yn`: would they use clips like this for testing or augmentation?
* `reviewer_notes`

## Utility test (the most important proof point, PRD section 9)

Use a partner's detector:

1. Measure a baseline on their real validation set.
2. Fine-tune or augment with synthetic clips from `POST /v1/batches` sweeps.
3. Re-measure on the same real validation set.

Report the change in recall and precision for the rare event. The ground-truth tracks (`*.tracks.csv`) load directly into most MOT-style tooling.

## Decision for the technical lead

If practitioners reject the realism of the simulation renderer, the options in PRD section 8 are:

* add an AI realism pass (image-to-image or video-to-video) on top of the simulation, which keeps its geometry and labels; or
* evaluate a foundation model (Option A) through the `cctvgpt/backends` adapter.

Either way the safeguard pipeline is unchanged. Model and dataset licences must be checked before adoption (PRD section 7).
