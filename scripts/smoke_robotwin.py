"""One-episode integration test using the official evaluator (not a benchmark score).

The official main() constructs the environment and model. This harness calls its
unchanged eval_policy() with test_num=1, then exits before main() writes a score
using its hard-coded 100-episode denominator. Scene generation, expert validation,
termination and policy actions are all executed by the official evaluator.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

from robotwin import verified_root, ROOT


class SmokeComplete(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-config", default="leap_clean")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = verified_root(args.robotwin, "script/eval_policy.py")
    checkpoint = args.checkpoint.resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.chdir(root)
    sys.path[:0] = [str(ROOT), str(root), str(root / "script")]
    spec = importlib.util.spec_from_file_location("official_eval", root / "script/eval_policy.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    from test_render import Sapien_TEST
    Sapien_TEST()
    evaluate = official.eval_policy

    def one_episode(*positional, **kwargs):
        kwargs["test_num"] = 1
        next_seed, successes = evaluate(*positional, **kwargs)
        print(json.dumps({"smoke_test": "passed", "completed_episodes": 1,
                          "policy_successes": successes, "next_seed": next_seed,
                          "benchmark_score": False}), flush=True)
        raise SmokeComplete

    official.eval_policy = one_episode
    try:
        official.main({"policy_name": "leap.robotwin.deploy", "task_name": args.task,
                       "task_config": args.task_config, "ckpt_setting": "smoke-only",
                       "ckpt_path": str(checkpoint), "seed": args.seed,
                       "instruction_type": "unseen", "device": "cuda:0"})
    except SmokeComplete:
        pass


if __name__ == "__main__":
    main()
