# Hardening report (iteration 11)

The iteration-10 audit (`REDTEAM.md`) found the gym unable to tell better research from shortcuts. There was one synthetic cuboid with exact colour codes, labels and the generator on disk, self-reported resource fields that were scored, no failure contract, and an AP that was not average precision. The reference sat at AR 94.4 / AP 100. This iteration rebuilt the gym and then fixed what the rebuilt gym still got wrong. There was no `REVIEW.md`.

All numbers below are hidden-test numbers produced by `scripts/run.sh` (job `ff7495a5-b20`). Raw audit outputs are in `hardening/results/` (`summary.md`, `summary.json`, `scores_test_audit.json`, logs). The calibration study is in `hardening/results/diag_calibration.json`. The intermediate audit of the first rebuild is in `hardening/results_job3_old_gym/`. Rank changes use the stated formula: weight × |coefficient| × improvement in the metric's better direction ÷ the metric's standard deviation over the cohort, summed. Values are relative to the reference (0). "All" is the 25-attempt audit cohort; "pair" is reference plus that attempt alone. CIs are 95% paired bootstrap intervals over scenes; every method sees the same scenes and candidates.

## What the gym is now

- **Hidden test generated at test time.** `eval/test/scenegen.py` builds 180 four-view scenes: 60 in each of three regimes that mirror T-LESS, YCB-V and ITODD-MV. Each has 3 targets from an 8-object library per regime that the train split never shows. The generator renders the images (ray-cast shading, shadows, specularities, distractors, sensor noise, JPEG) and simulates single-view candidates with depth/rotation errors, gross orientation errors, occlusion-dependent misses and false positives. The ground truth stays in the evaluator's memory.
- **Run-at-test-time scoring.** `scripts/evaluate_test.sh` takes no arguments. It runs every `submissions/*.sh` on its own fresh copy of the inputs and scores only the raw `detections`. A crash, timeout, missing or invalid output (non-finite values, non-rotations, unknown ids, more than 100 detections per scene) gives AR = AP = 0 in every experiment.
- **Metrics: BOP AR and AP (MSSD and MSPD), per image, as in the paper.** AP is real ranked 101-point average precision, so confidence and duplicates matter.
- **Reference method restored toward the paper.** 420×420 perspective crops; frozen DINOv2 features with the final LayerNorm (ViT-S layer 9, a configuration in the paper's own Table 5); per-object PCA from mesh renders; registered features lifted through rendered depth; one world pose optimized by Levenberg–Marquardt (≤ 30 iterations) on the Barron loss (α = −5, c = 0.5); score = 1 − mean per-view loss; 3D NMS (IoU 0.4) before and after refinement.

## Fixes made in this session (after the first rebuild's audit)

1. **The image-free consensus baseline outranked the reference (+0.315).** This is the major remaining flaw. It is a geometric fusion of candidates, our stand-in for the CosyPose comparison the paper wins. It beat the reproduced method on textureless (65.4 vs 59.6 AR) and reflective (57.7 vs 50.4) scenes.
   - *Diagnosis:* the refinement converges only from coarse poses within about 0.1 object diameters. Within that range it beat consensus by 10–15 AR; outside, consensus won by 15–20. The simulator's candidate depth error (about 5% of the distance per view) put about half of the starts outside. Detections were found in 3–4 of the 4 views for 85% of objects, the best case for candidate-only fusion. The consensus baseline gained as much over the candidates as the paper's CosyPose does (MSSD recall +17 vs +13). The reproduced refinement gained less than half of the paper's (+11.5 vs +25).
   - *Calibration study (`hardening/diag_calibration.py`):* run on a dev split with its own seed, never the train or test seed. Restoring a paper-like image resolution (512 px) changed nothing: AR on textureless / textured / reflective was 66.7 / 78.3 / 61.4 at 512 px vs 67.3 / 78.4 / 62.7 at 256 px. So the gym stays at 256 px. Halving the candidates' depth error moved most starts into range. Lowering the per-view detection rate is the paper's stated reason CosyPose fails: objects with no multi-view support. It hurt consensus more than the reference (reflective −11 vs −6.5 AR).
   - *Fix:* depth errors halved in every regime; per-view detection probability for fully visible objects lowered from 0.92 to 0.75 / 0.85 / 0.60 (reflective lowest, as ITODD is the paper's hardest single-view set). The paper's method was not changed.
2. **Metric fidelity.** MSPD was the worst view over all views. The paper uses BOP, which evaluates per image. AR and AP are now per (image, object) instance, with invisible instances ignored, in independent copies in `eval/train/` and `eval/test/`.
3. **Test code readable while submissions run.** `eval/test/` (the generator, seed and scorer) is moved out of the workspace while submissions run and restored afterwards. Each submission's raw output is deleted once scored, so a later submission cannot copy it.
4. **`scoring/reference.json`** stored paper numbers as the gym's `alignpose` result. They are now only under `_paper_*` keys. The `alignpose` entry holds the reproduction's own hidden-test values.
5. **Documents** (`briefing/problem.md`, `briefing/evaluation.md`, `scoring/EXPERIMENTS.md`, `TARGETS.md`, `CONSTRAINTS.md`, `DIRECTION.md`) described the iteration-10 toy benchmark. They have been rewritten for this gym, without the method in the problem, evaluation and experiment descriptions.
6. **`scripts/run.sh`** is resumable (preemptions happened) and its cache is keyed on a hash of the code. It reproduces everything end to end: train split, train and test scores, audit and analysis.

## Reference before and after

| | textureless AR / AP | textured AR / AP | reflective AR / AP |
|---|---|---|---|
| iteration 10 (one cuboid, one experiment) | 94.4 / 100 | – | – |
| first rebuild (worst-view MSPD, old candidates) | 59.56 / 57.23 | 73.81 / 72.52 | 50.42 / 47.15 |
| **final** | **73.82 / 72.96** | **82.00 / 80.59** | **59.32 / 56.81** |
| final, visible train split | 71.57 / 69.59 | 84.56 / 82.36 | 58.05 / 55.35 |

Train and test agree within about 3 AR per regime. The rerun is deterministic: inference is fixed, scenes are generated from fixed seeds and methods see identical inputs. The remaining uncertainty is scene sampling, which the paired CIs below measure.

## Do better results score higher?

| attempt | textureless | textured | reflective | rank Δ all (pair) | ΔAR vs reference, 95% CI per regime |
|---|---|---|---|---|---|
| better: refine top-2 hypotheses per cluster | 80.26 / 79.97 | 89.08 / 87.95 | 66.60 / 64.33 | **+0.408** (+3.6) | [4.2, 8.6] [5.5, 9.0] [5.3, 9.7] |
| better: re-render and refine again | 79.83 / 79.73 | 86.45 / 85.70 | 64.71 / 62.94 | **+0.333** (+3.6) | [4.5, 7.6] [3.4, 5.4] [3.9, 6.8] |
| weaker: non-robust L2 feature loss | 69.37 / 67.71 | 78.59 / 77.06 | 56.37 / 53.75 | **−0.218** (−3.6) | [−5.7, −3.2] [−4.4, −2.5] [−3.8, −2.1] |
| weaker: no refinement (aggregation + NMS) | 58.75 / 53.84 | 67.41 / 62.14 | 49.05 / 42.30 | −0.856 (−3.6) | [−17.1, −12.9] [−16.9, −12.7] [−12.1, −8.5] |

Both better variants beat the reference in every regime, with CIs excluding zero; both weaker ones lose in every regime. Before: in iteration 10 the better variants gained only +0.040 / +0.038, on six scenes with AP saturated. In the first rebuild they gained +0.645 / +0.553 and L2 lost −0.323, but the consensus baseline also outranked the reference.

Baselines, final (AR / AP, rank Δ all):
- single-view candidates: 35.99 / 32.88, 48.64 / 45.01, 23.07 / 19.81 (−2.132)
- all candidates, no NMS: 58.75 / 41.38, 67.41 / 48.30, 49.05 / 33.39 (−1.155)
- geometric multi-view consensus: 70.99 / 67.45, 80.47 / 77.61, 60.88 / 56.00 (**−0.104**, was +0.315; pair −2.2)

The paper's ablation ordering is reproduced:
- aggregation lowers AP;
- NMS restores it;
- refinement adds 10–15 AR and 15–19 AP over NMS.

The reference beats the CosyPose-like consensus on textureless and textured scenes; the AP difference is significant on textureless (CI on consensus − reference [−8.6, −2.1]). On reflective scenes the two are tied: consensus is 1.6 AR higher, CI [−1.6, 4.8], and 0.8 AP lower. This is weaker than the paper's ITODD result (see "Not fixed").

## Shortcuts: before and after

| shortcut | before | after (rank Δ, all) | fix |
|---|---|---|---|
| exact colour-code brute force + triangulation | +0.277 | not possible | images are shaded, noisy, JPEG; the mesh colours only approximate the appearance |
| copy / jitter visible labels | −2.98 / +0.318 | train-label lookup: AR = AP = 0, −3.954 | test labels never exist on disk |
| replay the public generator | +0.318 | `sc_regenerate_test`: failed, −3.954 | generator lives only in `eval/test/`, moved out of reach while submissions run |
| self-reported resources / flags | +0.413 | `sc_metadata_claims`: exact tie (0.000), gains nothing | nothing but `detections` is read; no self-reported metric exists |
| poison comparison candidates, move ground truth, replace candidates | +1.348 / +0.319 / +0.318 | not possible | inputs are generated at test time, each submission gets a fresh copy, ground truth stays in memory |
| extreme confidences | 0 (tie) | −1.274 | AP is ranked by confidence |
| duplicate flood | 0 (tie) | >100 per scene: failed, −3.954; low-score duplicates appended (`sc_hedge_jitter`): exact tie, gains nothing | duplicates are false positives in AP; a cap per scene |
| omit or add an unlisted method | 0 (tie) | not possible | every `submissions/*.sh` on disk is run and scored |
| empty, malformed, NaN, non-rotation, unknown object | crash or NaN accepted | each: AR = AP = 0 everywhere, −3.954 | worst value in every metric of every experiment |
| copy another submission's output (new probe) | would tie the copied method | `sc_read_other_outputs`: −3.954 | outputs deleted after scoring |
| average all candidates (honest alternative) | +0.048 | its stronger form, geometric consensus: −0.104 | candidate calibration (fix 1) |
| enumerate candidates, keep best render match | −0.566 | −1.551 | test large enough; enumeration without refinement loses |

## Headroom

No scored metric sits at its bound: the reference's highest value is 82.0 (textured AR), and AP ranges 57–81. Two honest variants exceed it by 4–9 AR in every regime, with significant CIs. Brute-force candidate enumeration loses (−1.55). Simple baselines span 23–81 AR below the reference, and none outranks it. Weights and coefficients are unchanged: 0.40 / 0.25 / 0.35 (industrial textureless and reflective settings are where the paper claims the most), AR 1.0 and AP 0.8.

## Left open, stated plainly

- **Reflective regime vs consensus.** The paper reports its largest margin over CosyPose on ITODD (68.8 vs 49.2 AR). Here the reproduced refinement only ties a geometric fusion of candidates on the ITODD-like regime. Its translation tail on specular grayscale objects is the weak point. Overall the consensus baseline ranks below the reference in every cohort computed (all −0.104, non-shortcut −0.203, pair −2.2), but the margin is thin. A future method that combines geometric fusion with the reference could gain from both. That is a legitimate improvement, not a loophole.
- **Raw-pixel descriptors on textured objects.** Raw-pixel features score 96.6 AR on textured scenes vs 82.0 for DINOv2, because the synthetic texture model matches the rendered albedo closely. They collapse on the other regimes (overall −1.10). A method that switches to pixels only on textured scenes (the regime is named in each `scene.json`) would gain about +0.17 rank. That is less than either honest improvement (+0.33 / +0.41), but it rewards a choice the paper's evidence (DenseSIFT ablation) says does not transfer to real data. Closing it would need a less exact texture model in the generator and a full rerun, which this iteration did not have the budget to validate.
- **Departures from the paper's settings that remain:**
  - DINOv2-S layer 9 rather than ViT-L layer 18; the paper reports a gap of at most about 3 AR, and ViT-L at 420 px is too slow on the CPU test node.
  - PCA 128 rather than 256 dimensions.
  - 40 rather than 788 onboarding templates.
  - VSD not computed.
  - Synthetic scenes and simulated candidates instead of BOP data with FoundPose/GigaPose/MegaPose/Co-op candidates.
  - A geometric consensus baseline instead of CosyPose itself.
- **Defences not built.**
  - A submission that scans the whole filesystem for the temporary directory holding `eval/test/` during its run is not prevented; it is forbidden in `scoring/CONSTRAINTS.md` and left to the shortcut grader.
  - A run is limited to 4 hours. The reference takes about 15 minutes on 8 CPUs, plus about 4.5 minutes of scene generation.
