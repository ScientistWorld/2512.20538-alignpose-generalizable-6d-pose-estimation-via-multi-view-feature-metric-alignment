# Experiment Overview

## tless_coop_ablation

This experiment checks whether a multi-view pose estimator improves over noisy
single-view candidates when all methods receive the same calibrated RGB views
and object mesh. It is a reduced, synthetic substitute for the paper's T-LESS
Co-op ablation row, designed to keep the core evaluation non-trivial without
requiring the full BOP candidate archive.

The reported metrics summarize pose correctness and detection-style ranking
from prediction artifacts. The evaluator does not call method code, so future
agents can replace the method as long as they write the same prediction format.
