# Evaluation Targets

Three scored experiments, one per scene regime, each with AR and AP on the hidden test (`scripts/evaluate_test.sh`, no arguments; results in `scoring/scores_test.json`). The visible train split is scored the same way by `scripts/evaluate_train.sh` (`scoring/scores_train.json`).

| experiment | weight | metrics (coefficient) | paper analogue |
|---|---|---|---|
| `textureless_clutter` | 0.40 | AR (1.0), AP (0.8) | T-LESS, Tables 1 and 4 |
| `textured_household` | 0.25 | AR (1.0), AP (0.8) | YCB-V, Tables 1 and 4 |
| `reflective_industrial` | 0.35 | AR (1.0), AP (0.8) | ITODD-MV, Tables 1 and 8 |

- **Benefit metric**: `AR` (per-image BOP average recall over MSSD and MSPD thresholds), higher is better.
- **Constraint metric**: `AP` (per-image BOP/COCO average precision over the same thresholds), higher is better. It guards the paper's claim that accuracy is gained without giving up precision: redundant or spurious hypotheses and miscalibrated confidences lower AP even when AR is unchanged.
- **Protocol constraints (enforced by the evaluator)**: RGB images, calibration, meshes and candidates only (no depth in the inputs); objects at test time are unseen; invalid or impossible output fails the whole run with AR = AP = 0 everywhere.

Reference (reproduced method, hidden test): textureless 73.82 / 72.96, textured 82.00 / 80.59, reflective 59.32 / 56.81 (AR / AP). The paper's real-data numbers are kept in `scoring/reference.json` under `_paper_*` keys for context only.

How to run: `scripts/download.sh` (weights), then `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh`; `scripts/reproduce.sh` does all three.
