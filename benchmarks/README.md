# Benchmarks: testing CCTVGPT against real data

These experiments test the PRD section 9 claims against real footage and an independent detector. They report what worked and what did not. Results JSON is in `results/`.

## 1. Does synthetic data help detect *real* falls? (`fall_sim2real.py`)

**Real data.** UR Fall Detection Dataset (URFD), camera 0 RGB: 30 staged falls and 40 activities of daily living (ADL), which include walking, sitting, crouching, picking things up and lying on a bed. Licence: CC BY-NC-SA 4.0, used here as an internal, non-commercial evaluation set. Download it with `download_urfd.py`; it is never committed.

**Pipeline, identical for both domains:**
1. A COCO-pretrained person detector (torchvision Faster R-CNN MobileNetV3) runs on the real clips at 10 fps.
2. The main person is tracked.
3. Scale-invariant motion features are computed: height drop relative to recent standing height, aspect-ratio change, descent speed, time from standing to low, and end state.
4. A logistic-regression classifier labels each clip as fall or not.

**Synthetic training data:** 150 fall clips plus 50 each of walk, crouch and lie-down. Viewpoint is randomised (camera height 2–3.5 m, tilt 10–35°, FOV 55–80°, event at 2.5–8 m), as are scene, lighting and compression. Labels are the renderer's exact ground-truth boxes with 4% detector-like jitter. Rendering takes about 80 s on 4 CPU cores.

### Results (logistic regression, threshold 0.5, all 70 real clips unless noted)

| Training data | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|
| Real only, 5-fold CV ×10 (in-domain upper bound) | 0.997 | 0.933 | 0.962 | 0.995 |
| **Synthetic only, zero real labels (v2 generator)** | **1.000** | **0.833** | **0.909** | **0.988** |
| Synthetic only, v1 generator (fast trips only) | 1.000 | 0.633 | 0.776 | 0.982 |
| Hand-tuned rule (aspect ratio > 1 and drop > 35%) | 0.800 | 0.800 | 0.800 | 0.825 |

**Few-shot (k real clips per class; mean of 50 random draws; tested on the remaining real clips):**

| k | Real only, F1 / AUROC | Real + synthetic (v2), F1 / AUROC |
|---|---|---|
| 1 | 0.682 / 0.747 | **0.873 / 0.981** |
| 2 | 0.863 / 0.940 | **0.883 / 0.983** |
| 3 | **0.892** / 0.962 | 0.874 / **0.982** |
| 5 | **0.929** / 0.979 | 0.890 / **0.984** |
| 10 | **0.935** / 0.987 | 0.902 / **0.989** |

### What this shows

* **Zero-shot transfer works for this task.** A classifier that never saw a real fall reaches F1 0.91 on real falls with no false alarms, beating the rule baseline and approaching in-domain training.
* **Synthetic data is most valuable when real examples are scarce**, which is the rare-event case the PRD targets. With 1–2 real examples per class it lifts F1 by 0.02–0.19 and AUROC by 0.04–0.23.
* **With 3 or more real examples per class, real data alone gives higher F1** at a fixed threshold. Synthetic data keeps AUROC and precision higher but costs recall, because the decision threshold is calibrated to synthetic motion. Re-calibrating the threshold on the few real clips is the obvious next step.
* **Measuring on real data improved the product.** v1 rendered only fast trips. Real falls took about 1.1 s from standing to low, against our 0.5 s. Adding knee-buckle collapses and slow slumps (renderer `sim-1.1.0`) raised zero-shot F1 from 0.78 to 0.91.
* **The hard-negative ablation is inconclusive.** Removing the confuser events helped logistic regression (F1 0.947) but hurt random forest (0.763 vs 0.833). URFD's "lying on a bed" ADLs are elevated and slow, which our floor-level lie-down does not model.

### Limitations

These are *not* product claims:
- 70 clips, one lab, healthy young actors, staged falls.
- Clip-level classification, not real-time detection.
- Hand-crafted motion features.
- Several URFD falls start from a chair, a pose our library lacks.

The confidence intervals are wide: with 30 falls, one fall is 3.3 points of recall. Next steps:
- Re-run on larger multi-dataset benchmarks (OmniFall's staged and in-the-wild splits, licence permitting).
- Use frame-level metrics.
- Compare against OmniFall's synthetic set, which comes from a diffusion model.

## 2. Does a real-world detector recognise our synthetic people? (`stress_test_detector.py`)

The same COCO detector, never trained on our data, was run on a 60-clip suite: 3 scenes × day/night × occlusion × weather × {fall, crowd surge, lie-down}, with 15,611 ground-truth people. It was scored with the acceptance-testing module (`cctvgpt/evaluation.py`).

| Metric | Value |
|---|---|
| Person recall (score ≥ 0.5) | **0.088** |
| Precision | 0.784 |
| AP50 | 0.104 |
| Recall by box height | <32 px: 0.001 · 32–64 px: 0.019 · 64–128 px: 0.191 · ≥128 px: 0.140 |
| Recall by state | walking 0.10 · crouching 0.31 · falling 0.24 · fallen/lying: 0.00 |
| Recall by time of day | day 0.13 · night 0.05 |

In the single-person close-range clips of experiment 1, recall was 0.22 (falls) to 0.57 (walking).

### What this shows

* **Today's renderer has a large appearance gap.** Its people are faceless articulated figures, which real-photo detectors mostly do not recognise. Experiment 1 works because it learns from *motion* using exact ground-truth boxes, not from pixels.
* **Training an appearance-based detector on these frames would need domain adaptation or a realism pass.** This is consistent with the literature: MOTSynth needed GTA-V-level visuals, and UBnormal used CycleGAN to transfer to real benchmarks.
* **Acceptance testing (Certify) must not be sold on this renderer.** Scorecards would mostly measure our gap, not the vendor's detector. A proposed **realism gate**: a reference COCO detector must reach recall on our suite within an agreed margin of its recall on comparable real footage before the suite is used for certification.
* **The scorecard machinery itself works.** It surfaces exactly the breakdowns a buyer needs: size, state (fallen people are missed), and lighting.

## 3. Watermark robustness (`../scripts/eval_watermark.py`)

See [docs/SAFEGUARDS.md](../docs/SAFEGUARDS.md). Detection is 100% under common re-encodes, resizes, frame-rate changes and colour edits; it degrades under heavy compression and fails under cropping and rotation.

## Reproduce

Download the real dataset:

```bash
python benchmarks/download_urfd.py --out external_data/urfd
```

Install the CPU build of PyTorch and torchvision, plus scikit-learn:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install scikit-learn
```

Run the fall experiment (about 12 minutes on 4 CPU cores the first time; real-clip features are cached afterwards):

```bash
python benchmarks/fall_sim2real.py --out benchmarks/results
```

Run the stress test (about 10 minutes):

```bash
python benchmarks/stress_test_detector.py --out benchmarks/results
```
