# AlignPose: Generalizable 6D Pose Estimation via Multi-view Feature-metric Alignment

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2512.20538"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2512.20538-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2512.20538](https://arxiv.org/abs/2512.20538) |
|:---|:---|
| **Domain** | Computer Vision / 6D Object Pose Estimation |
| **TL;DR** | The paper targets RGB-only, multi-view 6D pose estimation for novel rigid objects using calibrated cameras and known object meshes. |

## Summary

This paper studies model-based 6D object pose estimation from multiple calibrated RGB views. The target setting is challenging because the object instances may be unseen during training, single-view pose candidates can be noisy or ambiguous, and scenes may contain clutter, occlusion, textureless surfaces, metallic objects, or transparent objects. Evaluation follows BOP-style localization and detection metrics on YCB-V, T-LESS, ITODD-MV, and HouseCat6D. The reported results show large gains over single-view estimates and the published CosyPose multi-view baseline, especially on industrial and difficult-material settings.

## Problem

Can a system accurately estimate 6D poses for previously unseen objects from calibrated multi-view RGB images, without relying on object-specific training or depth input, and outperform strong single-view and multi-view baselines under BOP-style evaluation?

## The paper's method

The paper introduces AlignPose, a generalizable RGB-only multi-view 6D pose estimation pipeline for known object meshes and calibrated cameras. Its core contribution is a multi-view feature-metric refinement objective that optimizes a single world-frame object pose against image evidence from all available views.

## Key results of the paper

- On unseen-object multi-view evaluation, the paper reports higher AR and AP than both single-view inputs and CosyPose multi-view refinement across YCB-V, T-LESS, ITODD-MV, and HouseCat6D.
- On HouseCat6D glass and metallic subsets, the reported gains are largest for transparent glass objects and meaningful but harder for thin metallic cutlery.
- On seen-object T-LESS evaluation, the reported system remains competitive with methods trained on the test objects while using the multi-view RGB evaluation setup.

## Run it

```bash
pip install sciworld
sw research 2512.20538 --mode innovate --agent claude-code
sw evaluate 2512.20538
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

