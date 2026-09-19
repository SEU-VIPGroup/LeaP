# External benchmark

Run `python scripts/fetch_robotwin.py` from the LeaP root. The official RoboTwin
checkout is fixed to `robotwin.lock.json`, ignored by Git and excluded from
release archives. Download assets with the official RoboTwin installer.

Install simulator dependencies via `python scripts/install_robotwin.py --robotwin
third_party/RoboTwin` from the LeaP root. This runs the verified upstream installer
with cuRobo locked to the v1-compatible commit in `curobo.lock.json`, instead of
the incompatible current default branch. The official tracked scripts remain
unchanged; a failing command stops installation. cuRobo is downloaded externally
under its own license and is not bundled in this source archive.

This is a pinned external checkout, not a Git submodule, so it works from a ZIP.
The pin uses the legacy RoboTwin 2.0 `script/eval_policy.py` API. Current upstream
main has a different deployment interface. The launcher checks the revision and
official entry script contents before collection/evaluation.
