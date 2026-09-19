"""Fetch the pinned official benchmark without embedding simulator assets in LeaP."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=root / "third_party/RoboTwin")
    args = parser.parse_args()
    lock = json.loads((root / "third_party/robotwin.lock.json").read_text())
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Destination already exists; refusing to change it: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", lock["url"],
                    str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", lock["commit"]], check=True)
    print(f"RoboTwin ready at {destination}; install simulator dependencies and assets next.")


if __name__ == "__main__":
    main()
