Audit starting revision: 1152a87a0c98a692cd175162ce8788be5c18b963.

All trial payloads use only the scientist-visible method, baseline, data, and
visible train interface. The hidden test is invoked unchanged as a subprocess
with no arguments. No evaluator, interpreter, import path, environment package,
or surviving process is modified. `run_case.py` saves and restores mutable
workspace artifacts around each trial. Reference predictions may be reused for
trials that only change artifact fields, as permitted by the audit request.

Plan: three seeded reference reruns; denser registered features and a proper
rendered-depth lift / robust-objective fix; no-refinement ablation; candidate
averaging and exhaustive candidate rendering / palette matching controls;
visible-label copying and generator replay; self-reported field falsification;
confidence/duplicate omissions; changed comparison candidates and changed
labels; empty, nonfinite, and malformed predictions; train-manifest filtering.

The external ranking implementation is not in the repository. `rank.py` will
apply the user-specified magnitude/direction/std-normalized formula to one
explicit audit cohort; raw hidden scores are the authoritative evidence.
