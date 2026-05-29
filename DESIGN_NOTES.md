Train/Test Evaluation Design Notes

Checked:
- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. The train script calls `eval.train.evaluate` and writes `scoring/scores_train.json`; the test script calls `eval.test.evaluate` and writes `scoring/scores_test.json`.
- Confirmed both split evaluators discover `*_predictions.json` artifacts from `outputs/` instead of hardcoding a single method name.
- Grepped for train/test cross-imports and forbidden paper-agreement metric names in the split evaluators, split scripts, and `scoring/reference.json`.
- Confirmed no `scoring/scores_train.json` or `scoring/scores_test.json` stub exists.
- Ran `bash -n` on both split scripts and checked `scoring/reference.json` parses as JSON.

Changed:
- Fixed the missing-prediction shortcut in both `eval/train/evaluate.py` and `eval/test/evaluate.py`. Missing predictions now count as failed scenes in AR/AP/error aggregation instead of being silently skipped, and `complete_prediction_rate` is emitted as an explicit metric.
- Added `candidate_add_gain_pct`, measuring ADD improvement over the best supplied single-view candidate. This better targets the paper's core scientific claim that multi-view aggregation/refinement improves over noisy single-view candidates, not just absolute pose accuracy.
- Added `exact_pose_copy_rate` as a leakage sentinel. The current synthetic benchmark JSON contains `T_WO_gt`, so a future method could copy labels into its prediction artifact. The metric penalizes exact numerical copies while allowing normal high-quality estimates.
- Added required metric descriptions in `scoring/reference.json` and included the new metrics under the existing `tless_coop_ablation` experiment.
- Changed `uses_dinov2_features` from a small positive objective to an informational zero-weight metric. Rewarding a specific backbone would favor reproducing this implementation over genuine improvement.

Runtime concerns for the next iteration:
- The benchmark data file still contains ground-truth poses. The exact-copy sentinel catches direct JSON copying, but it cannot catch a method that uses ground truth after small numerical perturbation. A stronger future pass should move labels out of method-visible data or generate held-out test labels outside the method input path.
- The split remains small: the current tracked synthetic benchmark has 12 scenes, yielding 6 train and 6 test scenes under the existing parity split. Scores may have high variance and AP may saturate for strong methods.
- The split is a scene partition over a single synthetic object and camera regime. It distinguishes omission/overfit shortcuts better after this pass, but it does not yet test a held-out object, material category, or camera distribution.
- `eval/test/evaluate.py` optionally reads `scoring/scores_train.json` to keep method keys aligned with the train slice. This is intentional for final scoring consistency, but the runner should make sure train scores are created before test scoring in workflows that use that path.
