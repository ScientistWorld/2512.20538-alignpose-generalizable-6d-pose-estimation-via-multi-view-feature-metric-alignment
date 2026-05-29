# Evaluation Targets

## tless_coop_ablation

Tests the core claim on the runnable reduced benchmark: a calibrated four-view
pose task with noisy single-view candidates, where the full multi-view pipeline
should improve over the unrefined candidate baseline.

- **Benefit metrics**: `AR`, `AP` (higher is better)
- **Constraint metrics**: `num_views`, `uses_depth_sensor`,
  `object_specific_training`, and `mean_final_candidates`. These audit that
  the score comes from the intended calibrated RGB-only, no-training setting
  rather than extra sensors, extra labels, or unbounded duplicate predictions.
- **How to run evaluation**: run `scripts/reproduce.sh`, or run
  `scripts/baseline.sh`, `scripts/method.sh`, then `scripts/evaluate.sh`.

The paper's full BOP targets for YCB-V, T-LESS, ITODD-MV, and HouseCat6D are
kept in `scoring/reference.json` for context, but the currently executable gym
target is this reduced T-LESS-style ablation substitute.
