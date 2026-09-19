"""Launch official RoboTwin collection/evaluation and convert demonstrations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verified_root(path, script):
    root = path.resolve()
    lock = json.loads((ROOT / "third_party/robotwin.lock.json").read_text())
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != lock["commit"]:
        raise RuntimeError("RoboTwin version mismatch; use scripts/fetch_robotwin.py")
    original = subprocess.check_output(["git", "-C", str(root), "show", f"HEAD:{script}"])
    if (root / script).read_bytes() != original:
        raise RuntimeError(f"{script} differs from official RoboTwin; restore a clean checkout")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Create a separate collection task config")
    prepare.add_argument("--robotwin", type=Path, required=True)
    prepare.add_argument("--name", default="leap_clean")
    prepare.add_argument("--episodes", type=int, default=50)
    prepare.add_argument("--save-path", type=Path, required=True)
    for command in ("collect", "eval"):
        sub = commands.add_parser(command)
        sub.add_argument("--robotwin", type=Path, required=True)
        sub.add_argument("--task", required=True)
        sub.add_argument("--task-config", default="leap_clean")
        sub.add_argument("--gpu", type=int, required=True)
        if command == "eval":
            sub.add_argument("--checkpoint", type=Path, required=True)
            sub.add_argument("--seed", type=int, default=0, help="Official RoboTwin evaluation seed")
    convert = commands.add_parser("convert")
    convert.add_argument("--input", type=Path, required=True, help="Directory containing episode*.hdf5")
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    if args.command == "convert":
        from leap.robotwin.data import convert_episodes
        print(json.dumps(convert_episodes(args.input, args.output, args.episodes), indent=2))
        return
    if args.command == "prepare":
        root = verified_root(args.robotwin, "script/collect_data.py")
        if not args.name.replace("_", "").isalnum() or args.episodes < 1:
            parser.error("Use an alphanumeric config name and a positive episode count")
        config = yaml.safe_load((root / "task_config/demo_clean.yml").read_text())
        config.update(episode_num=args.episodes, save_path=str(args.save_path.resolve()),
                      use_seed=False, collect_data=True, render_freq=0, eval_video_log=False)
        config["data_type"]["pointcloud"] = True
        config["pcd_down_sample_num"] = 1024
        config["pcd_crop"] = True
        path = root / "task_config" / f"{args.name}.yml"
        with path.open("x") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        print(f"Created {path}")
        return
    script = f"script/{'collect_data' if args.command == 'collect' else 'eval_policy'}.py"
    root = verified_root(args.robotwin, script)
    # No local seed-list, scene filtering or expert-check bypass hooks are used.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if args.command == "collect":
        subprocess.run([sys.executable, script, args.task, args.task_config], cwd=root, env=env, check=True)
    else:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        deploy = {"policy_name": "leap.robotwin.deploy", "task_name": args.task,
                  "task_config": args.task_config, "ckpt_setting": "leap",
                  "ckpt_path": str(args.checkpoint.resolve()), "seed": args.seed,
                  "instruction_type": "unseen", "device": "cuda:0"}
        with tempfile.TemporaryDirectory(prefix="leap-eval-") as directory:
            config_path = Path(directory) / "deploy.yml"
            config_path.write_text(yaml.safe_dump(deploy))
            subprocess.run([sys.executable, script, "--config", str(config_path)],
                           cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
