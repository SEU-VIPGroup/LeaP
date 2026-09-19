# Release validation

## Review fixes (2026-09-19)

- Twelve CPU unittest cases passed, including the new regression comparing
  three observation frames with an eight-frame training window. With the same
  first three frames and random seed, both inputs now return identical six-step
  actions and identical full predictions. The original implementation returned
  only one executable action for the full window.
- The actual release ZIP builder was exercised: citation metadata, the GitHub
  workflow, website assets and dependency locks are included; local data/cache
  exclusions and refusal to overwrite an existing archive remain in place.
- The installation wrapper checks the official RoboTwin revision and installer,
  pins cuRobo to v0.7.8 (`d64c4b005459db10c5dd867d8b30a87d5bda9bdb`),
  prepares build tools and disables PyTorch3D's isolated build so it can import
  the installed PyTorch. An executable local Git fixture verifies that a newer,
  incompatible default branch does not replace the locked API.

The unit suite is complemented by the fresh-environment checks below. The
original publication and research checks follow as historical results.

## Fresh environment installation (2026-09-19)

A new isolated Conda environment was created on an existing Linux/NVIDIA host,
with user site packages disabled. It used Python 3.10.21, PyTorch 2.4.1+cu121,
torchvision 0.19.1+cu121 and a CUDA 12.1 development toolkit, including the CUDA
library headers. No packages from the existing research environment were used.

The revised `scripts/install_robotwin.py` completed against the locked RoboTwin
checkout. PyTorch3D 0.7.8 (upstream `stable` commit
`75ebeeaea0908c5527e7b1e305fbc7681382db47`) and cuRobo 0.7.8 were compiled from
source. The initial clean-environment attempt exposed PyTorch3D's isolated-build
`ModuleNotFoundError: torch`; the wrapper now prepares build tools and uses
`--no-build-isolation`. The local toolkit also needed its development library
headers, as now stated in the README.

The following checks passed:

- All 12 LeaP CPU tests in the new environment.
- Imports of `MotionGen`, `MotionGenConfig`, `MotionGenPlanConfig`,
  `PoseCostMetric`, `Pose` and `JointState`, as required by RoboTwin's planner.
- PyTorch3D farthest-point sampling on CUDA: 128 input points, 16 sampled points.
- cuRobo Franka forward kinematics on CUDA: two joint configurations produced
  finite end-effector positions and quaternions.
- The pinned official `script/test_render.py` reported `Render Well`
  (renderer/scene initialization).
- `python -m pip check` reported no broken requirements. The tracked files in
  both external checkouts remained unchanged.

[Machine-readable versions and results](install-validation.json) accompany
this record. Benchmark assets were not downloaded for this check; no complete
task collection, training run, motion-planning benchmark or 100-episode policy
evaluation was repeated. This verifies a fresh Python dependency installation
on the tested host, not installation on every clean machine or GPU.

## Initial publication checks (commit 55fede3)

On 2026-09-19, all eight bundled unittest cases passed again on CPU for this
public repository. This check used Python 3.10, PyTorch 2.3.1+cu121,
torchvision 0.18.1+cu121, NumPy 1.26.4, Zarr 2.12.0, numcodecs 0.13.1 and
h5py 3.16.0. The policy and training implementation was kept unchanged from
the provided release archive. GitHub Actions runs the same tests with the
documented PyTorch 2.4.1 / torchvision 0.19.1 combination on CPU.

These publication checks do not repeat simulator evaluation or establish
paper success-rate reproduction. The original release validation record is
preserved below.

## Original release validation

Validated on 2026-09-19 in an existing Linux RoboTwin environment: Python 3.10,
PyTorch 2.4.1+cu121, torchvision 0.19.1+cu121, NumPy 1.26.4, Zarr 2.12.0,
numcodecs 0.13.1, SAPIEN 3.0.0b1 and mplib 0.2.1.

## Automated checks

- Eight unittest cases pass: learned mean/standard-deviation sampling,
  reproducibility, gradients, data alignment, checkpoint reload, deployment,
  official evaluation launch and rejection of a modified official entry point.
- The research comparison checks matched weights against
  `structureflowpolicy_fixedattn_film_state_a2afm`: training/evaluation losses,
  parameter gradients, AdamW updates, EMA warmup and stochastic inference.
- Dataset windows, validation split and normalization are compared against
  the research implementation using the 50-demonstration
  `beat_block_hammer-demo_clean` dataset.

Reproduce these checks with the commands in the README. Numerical parity uses
explicit weight mapping and synchronized RNG states; it does not establish
identical independently initialized full training runs.

## Real simulator integration

The official RoboTwin checkout was pinned to
`9b5491d5f1d838685da126d5378dca060103dede`. Its collection and evaluation entry
points and environment code were unmodified. Existing local simulator assets
were reused; downloading and installing everything into a fresh environment
was not tested.

1. Official collection produced one successful `beat_block_hammer`
   demonstration (115 frames).
2. Conversion produced 114 next-frame action/state pairs in Zarr.
3. A full-size policy completed three training updates and saved checkpoints.
4. The separate one-episode smoke harness loaded a checkpoint and completed
   all 400 steps through the official evaluation function, including official
   expert validity checking. The harness reported `smoke_test: passed`.

The nearly untrained policy achieved **0/1 task successes**. Episode completion
is an integration check, not a performance result. This smoke run used the
initial three-update checkpoint; later training-default alignment is covered
by the automated checks, not by a second full simulator rollout.

## Not established

- Full 3001-epoch training or paper success-rate reproduction.
- A full official 100-episode evaluation.
- A clean-machine installation or compatibility with arbitrary RoboTwin revisions.

The normal evaluation launcher retains the official 100-episode protocol.
The one-episode smoke helper is separate and does not write a benchmark score.
No checkpoints, demonstration data, simulator assets or local logs are included
in the source release.
