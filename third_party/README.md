# External benchmark

Run `python scripts/fetch_robotwin.py` from the LeaP root. The official RoboTwin
checkout is fixed to `robotwin.lock.json`, ignored by Git and excluded from
release archives. Download assets with the official RoboTwin installer.

This is a pinned external checkout, not a Git submodule, so it works from a ZIP.
The pin uses the legacy RoboTwin 2.0 `script/eval_policy.py` API. Current upstream
main has a different deployment interface. The launcher checks the revision and
official entry script contents before collection/evaluation.
