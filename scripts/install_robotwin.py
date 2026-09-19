"""Run the pinned RoboTwin installer with a pinned, v1-compatible cuRobo."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from robotwin import ROOT, verified_root


def pinned_installer(script, lock):
    clone = "git clone https://github.com/NVlabs/curobo.git"
    if script.count(clone) != 1 or not re.fullmatch(r"[0-9a-f]{40}", lock["commit"]):
        raise ValueError("Unrecognized RoboTwin installer or invalid cuRobo commit")
    replacement = (
        f"git clone --filter=blob:none --no-checkout {shlex.quote(lock['url'])} curobo\n"
        f"git -C curobo checkout --detach {lock['commit']}"
    )
    # Use this interpreter even when the shell's `pip` belongs to another env.
    script = script.replace(clone, replacement)
    return re.sub(r"\bpip (?=install|show)",
                  lambda _: f"{shlex.quote(sys.executable)} -m pip ", script)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin", type=Path, required=True)
    args = parser.parse_args()
    root = verified_root(args.robotwin, "script/_install.sh")
    destination = root / "envs/curobo"
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite {destination}; use a fresh RoboTwin checkout")
    lock = json.loads((ROOT / "third_party/curobo.lock.json").read_text())
    script = pinned_installer((root / "script/_install.sh").read_text(), lock)
    pytorch3d = '"git+https://github.com/facebookresearch/pytorch3d.git@stable"'
    if script.count(pytorch3d) != 1:
        raise ValueError("Unrecognized PyTorch3D installation command")
    # Its setup imports torch; isolated PEP 517 builds cannot see the torch
    # already installed in this environment.
    script = script.replace(pytorch3d, pytorch3d + " --no-build-isolation")
    env = dict(os.environ, PIP_CONSTRAINT=str(ROOT / "constraints-robotwin.txt"))
    subprocess.run([sys.executable, "-m", "pip", "install", "setuptools<81",
                    "wheel", "ninja", "setuptools_scm"], env=env, check=True)
    # Execute in memory: official benchmark files remain unchanged. Fail on
    # errors instead of continuing to report a successful installation.
    subprocess.run(["bash", "-e", "-c", script], cwd=root, env=env, check=True)
    subprocess.run([sys.executable, "-c",
                    "from curobo.types.math import Pose; "
                    "from curobo.types.robot import JointState; "
                    "from curobo.wrap.reacher.motion_gen import "
                    "MotionGen, MotionGenConfig, MotionGenPlanConfig, PoseCostMetric; "
                    "print('cuRobo v1 motion-generation API: OK')"], check=True)


if __name__ == "__main__":
    main()
