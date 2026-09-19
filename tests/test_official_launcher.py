"""Ensure benchmark launch delegates to the unchanged pinned official evaluator."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("leap_launcher", ROOT / "scripts/robotwin.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class OfficialLauncherTest(unittest.TestCase):
    def test_official_eval_command(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "latest.ckpt"
            checkpoint.touch()
            with patch.object(launcher, "verified_root", return_value=Path(temp)), \
                 patch.object(launcher.subprocess, "run") as run, \
                 patch.object(sys, "argv", ["robotwin.py", "eval", "--robotwin", temp,
                                           "--task", "beat_block_hammer", "--gpu", "2",
                                           "--checkpoint", str(checkpoint), "--seed", "3"]):
                def inspect(cmd, **kwargs):
                    self.assertEqual(cmd[:3], [sys.executable, "script/eval_policy.py", "--config"])
                    config = yaml.safe_load(Path(cmd[3]).read_text())
                    self.assertEqual(config["policy_name"], "leap.robotwin.deploy")
                    self.assertEqual(config["seed"], 3)
                    self.assertNotIn("test_num_override", config)
                    self.assertNotIn("seed_list", config)
                    self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2")
                    self.assertEqual(kwargs["cwd"], Path(temp))
                run.side_effect = inspect
                launcher.main()
                run.assert_called_once()

    def test_modified_official_entry_rejected(self):
        import json
        commit = json.loads((ROOT / "third_party/robotwin.lock.json").read_text())["commit"]
        with tempfile.TemporaryDirectory() as temp:
            entry = Path(temp) / "script/eval_policy.py"
            entry.parent.mkdir()
            entry.write_text("modified")
            with patch.object(launcher.subprocess, "check_output", side_effect=[commit + "\n", b"official"]):
                with self.assertRaisesRegex(RuntimeError, "differs from official"):
                    launcher.verified_root(Path(temp), "script/eval_policy.py")


if __name__ == "__main__":
    unittest.main()
