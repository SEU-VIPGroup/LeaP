# Release validation

## Publication checks

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
