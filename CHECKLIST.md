# Reproduction Checklist

Check off items as you complete them. Order does not matter.

## Briefing

- [x] `briefing/problem.md` (method-agnostic)
- [x] `briefing/evaluation.md` (method-agnostic)
- [x] `briefing/method.md`
- [x] `briefing/overview.md`

## Scoring

- [x] `scoring/reference.json` - paper's reported numbers
- [ ] Workspace validated via `python validate.py`
- [ ] `scoring/scores.json` - reproduced numbers matching `reference.json` structure
- [x] `scoring/EXPERIMENTS.md` - high-level purpose of each reference experiment
- [x] `scoring/TARGETS.md` - primary, constraint, and ablation targets
- [x] `scoring/CONSTRAINTS.md` - what a scientist must hold fixed
- [x] `scoring/DIRECTION.md` - research direction scope for scientist

## Code

- [ ] Evaluation code in `eval/`
- [ ] Method implementation in `method/`
- [ ] Baseline implementation in `baseline/` (if applicable)

## Scripts

- [ ] `scripts/evaluate.sh`
- [ ] `scripts/reproduce.sh`
- [ ] `scripts/download.sh` (idempotent, size comment at top)
- [ ] `scripts/baseline.sh` (if applicable)

## Environment

- [ ] `environment/container.def` + `environment/setup.sh`

## Data

- [ ] Dataset acquired (shared dir or `data/`)
- [ ] Pretrained models downloaded if needed

## Milestones

- [ ] `method_runs` - executes end-to-end without errors
- [ ] `core_claim` - minimum experiment supports central claim
- [ ] `core_claim_plus` - additional settings
- [ ] `secondary_claims`
- [ ] `majority`
- [ ] `near_complete`
- [ ] `full`
